"""Realtime speech-to-speech session, driven with canned OpenAI events (no network)."""

import asyncio
import base64
import json

import numpy as np

from app.streaming.realtime_session import RealtimeSession, pcm_to_ulaw, ulaw_to_pcm


def agent_audio(item_id: str, seconds: float) -> dict:
    ulaw = pcm_to_ulaw(bytes(int(8000 * seconds) * 2))
    return {"type": "response.output_audio.delta", "item_id": item_id, "delta": base64.b64encode(ulaw).decode()}


def make_session(**kwargs):
    session = RealtimeSession("key", "Book a site visit.", "te", **kwargs)
    to_openai: list[dict] = []
    to_phone: list[dict] = []
    ended: list[bool] = []

    async def capture(event: dict) -> None:
        to_openai.append(event)

    async def send(message: str) -> None:
        to_phone.append(json.loads(message))

    async def end() -> None:
        ended.append(True)

    session._to_openai = capture
    return session, to_openai, to_phone, send, end, ended


def test_ulaw_round_trip_stays_close_to_the_original():
    pcm = (np.sin(np.linspace(0, 200, 8000)) * 12000).astype("<i2")
    decoded = np.frombuffer(ulaw_to_pcm(pcm_to_ulaw(pcm.tobytes())), dtype="<i2")
    assert len(decoded) == len(pcm)
    assert np.abs(decoded.astype(int) - pcm.astype(int)).max() < 600


def test_greeting_made_before_pickup_plays_the_moment_the_phone_attaches():
    async def scenario():
        session, to_openai, to_phone, send, end, _ = make_session(greeting="నమస్కారం అండి")
        await session._on_event({"type": "session.updated"})
        assert to_openai[-1]["type"] == "response.create"
        assert "నమస్కారం అండి" in to_openai[-1]["response"]["instructions"]

        await session._on_event(agent_audio("greet", 1.0))
        assert to_phone == []  # nobody on the line yet

        await session.attach("sid", send, end)
        assert [m["event"] for m in to_phone] == ["playAudio"]

    asyncio.run(scenario())


def test_barge_in_clears_phone_audio_and_truncates_what_was_not_heard():
    async def scenario():
        session, to_openai, to_phone, send, end, _ = make_session()
        await session.attach("sid", send, end)
        await session._on_event(agent_audio("a1", 5.0))
        await session._on_event({"type": "input_audio_buffer.speech_started"})

        assert to_phone[-1] == {"event": "clearAudio", "streamId": "sid"}
        truncate = to_openai[-1]
        assert truncate["type"] == "conversation.item.truncate" and truncate["item_id"] == "a1"
        assert truncate["audio_end_ms"] < 1000

        await session._on_event(agent_audio("a1", 1.0))  # late audio from the cancelled reply
        assert to_phone[-1]["event"] == "clearAudio"

    asyncio.run(scenario())


def test_transcript_merges_split_caller_speech_and_drops_unheard_replies():
    async def scenario():
        session, _, _, send, end, _ = make_session()
        await session.attach("sid", send, end)
        await session._on_event(agent_audio("a1", 0.2))
        await session._on_event({"type": "response.output_audio_transcript.done", "item_id": "a1", "transcript": "నమస్తే"})
        await session._on_event({"type": "input_audio_buffer.committed", "item_id": "u1"})
        await session._on_event({"type": "response.output_audio_transcript.delta", "item_id": "a2", "delta": "ధ"})
        await session._on_event({"type": "input_audio_buffer.committed", "item_id": "u2"})
        await session._on_event({"type": "conversation.item.input_audio_transcription.completed", "item_id": "u2", "transcript": "రేటు ఎంత?"})
        await session._on_event({"type": "conversation.item.input_audio_transcription.completed", "item_id": "u1", "transcript": "అవునా,"})
        assert session.turns == [("agent", "నమస్తే"), ("caller", "అవునా, రేటు ఎంత?")]

    asyncio.run(scenario())


def test_booking_is_recorded_and_end_call_hangs_up_after_the_goodbye():
    async def scenario():
        session, to_openai, _, send, end, ended = make_session()
        await session.attach("sid", send, end)

        book = {"type": "function_call", "name": "book_appointment", "call_id": "c1"}
        await session._on_event({"type": "response.done", "response": {"output": [book]}})
        assert [e["type"] for e in to_openai[-2:]] == ["conversation.item.create", "response.create"]
        assert session.outcome() == "appointment_booked"

        await session._on_event(agent_audio("bye", 0.1))
        hangup = {"type": "function_call", "name": "end_call", "call_id": "c2"}
        await session._on_event({"type": "response.done", "response": {"output": [hangup]}})
        assert not ended
        await asyncio.sleep(0.6)
        assert ended and session.ended

    asyncio.run(scenario())
