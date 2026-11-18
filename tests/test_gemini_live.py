"""Gemini Live call engine, driven with canned Gemini messages (no network)."""

import asyncio
import base64
import json

import numpy as np

from app.streaming.gemini_live_session import Downsampler, GeminiLiveSession


def make_session():
    session = GeminiLiveSession("key", "Book a site visit.", "te", model="gemini-3.8-live", caller_name="Ravi")
    to_gemini: list[dict] = []
    to_phone: list[dict] = []
    ended: list[bool] = []

    async def capture(message: dict) -> None:
        to_gemini.append(message)

    async def send(message: str) -> None:
        to_phone.append(json.loads(message))

    async def end() -> None:
        ended.append(True)

    session._send_json = capture
    return session, to_gemini, to_phone, send, end, ended


def speech(seconds: float) -> dict:
    pcm = bytes(int(24000 * seconds) * 2)
    return {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
                                                                      "data": base64.b64encode(pcm).decode()}}]}}}


def tool_call(name: str, call_id: str) -> dict:
    return {"toolCall": {"functionCalls": [{"id": call_id, "name": name, "args": {}}]}}


TURN_DONE = {"serverContent": {"turnComplete": True}}


def test_setup_asks_for_telugu_audio_with_the_agent_prompt_and_tools():
    session, *_ = make_session()
    setup = session._setup_message()["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["generationConfig"]["speechConfig"]["languageCode"] == "te-IN"
    assert setup["inputAudioTranscription"] == {"languageCodes": ["te-IN"]}
    assert "Book a site visit." in setup["systemInstruction"]["parts"][0]["text"]
    assert {f["name"] for f in setup["tools"][0]["functionDeclarations"]} == {"book_appointment", "end_call"}


def test_greeting_made_before_pickup_plays_at_8_khz_when_the_phone_attaches():
    async def scenario():
        session, to_gemini, to_phone, send, end, _ = make_session()
        await session._on_message({"setupComplete": {}})
        assert "Ravi" in to_gemini[-1]["clientContent"]["turns"][0]["parts"][0]["text"]

        await session._on_message(speech(0.5))
        assert not to_phone  # nobody on the line yet
        await session.attach("sid", send, end)
        pcm = base64.b64decode(to_phone[0]["media"]["payload"])
        assert to_phone[0]["event"] == "playAudio" and len(pcm) == 4000 * 2  # 0.5 s at 8 kHz

    asyncio.run(scenario())


def test_transcripts_become_the_call_record_in_order():
    async def scenario():
        session, *_ = make_session()
        for message in (
            {"serverContent": {"outputTranscription": {"text": "నమస్కారం రవి గారు,"}}},
            {"serverContent": {"outputTranscription": {"text": " నేను సుమ."}}},
            {"serverContent": {"inputTranscription": {"text": "ఆ ప్లాట్ రేట్"}}},
            {"serverContent": {"inputTranscription": {"text": " ఎంత ఉంటుంది?"}}},
            {"serverContent": {"outputTranscription": {"text": "పద్దెనిమిది లక్షల నుంచి మొదలవుతుంది."}}},
        ):
            await session._on_message(message)
        assert session.turns == [
            ("agent", "నమస్కారం రవి గారు, నేను సుమ."),
            ("caller", "ఆ ప్లాట్ రేట్ ఎంత ఉంటుంది?"),
            ("agent", "పద్దెనిమిది లక్షల నుంచి మొదలవుతుంది."),
        ]

    asyncio.run(scenario())


def test_caller_talking_over_the_agent_clears_queued_phone_audio():
    async def scenario():
        session, _, to_phone, send, end, _ = make_session()
        await session.attach("sid", send, end)
        await session._on_message(speech(2.0))
        await session._on_message({"serverContent": {"interrupted": True}})
        assert [m["event"] for m in to_phone] == ["playAudio", "clearAudio"]

    asyncio.run(scenario())


def test_booking_is_answered_and_hang_up_waits_for_the_goodbye_turn():
    async def scenario():
        session, to_gemini, _, send, end, ended = make_session()
        await session.attach("sid", send, end)
        await session._on_message(tool_call("book_appointment", "c1"))
        assert to_gemini[-1]["toolResponse"]["functionResponses"][0]["id"] == "c1"
        assert session.outcome() == "appointment_booked"

        await session._on_message(speech(0.1))
        await session._on_message(tool_call("end_call", "c2"))
        await asyncio.sleep(0.4)
        assert not ended  # still mid-turn
        await session._on_message(TURN_DONE)
        await asyncio.sleep(0.6)
        assert ended and session.ended

    asyncio.run(scenario())


def test_silent_hang_up_says_goodbye_first():
    async def scenario():
        session, to_gemini, _, send, end, ended = make_session()
        await session.attach("sid", send, end)
        await session._on_message({"serverContent": {"inputTranscription": {"text": "ఓకే బై"}}})
        await session._on_message(tool_call("end_call", "c1"))
        assert "goodbye" in to_gemini[-1]["clientContent"]["turns"][0]["parts"][0]["text"]
        await asyncio.sleep(0.4)
        assert not ended

        await session._on_message(speech(0.1))
        await session._on_message(TURN_DONE)
        await asyncio.sleep(0.6)
        assert ended

    asyncio.run(scenario())


def test_cost_adds_every_turn_at_gemini_prices():
    async def scenario():
        session, *_ = make_session()
        usage = {"promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1000}, {"modality": "AUDIO", "tokenCount": 300}],
                 "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 100}]}
        for _ in range(2):
            await session._on_message({"usageMetadata": usage})
        expected = 2 * (1000 * 0.75 + 300 * 3.00 + 100 * 12.00) / 1_000_000
        assert abs(session.cost_usd() - expected) < 1e-12
        assert session.usage_report().startswith("gemini-3.8-live $0.0057") and "2 replies" in session.usage_report()

    asyncio.run(scenario())


def test_downsampling_keeps_every_third_sample_across_uneven_chunks():
    tone = (np.sin(np.linspace(0, 400, 2400)) * 8000).astype("<i2").tobytes()
    down = Downsampler()
    out = b"".join(down(tone[i:i + size]) for i, size in ((0, 1000), (1000, 2002), (3002, 1798)))
    assert len(out) == 800 * 2  # 2400 samples at 24 kHz -> 800 at 8 kHz
