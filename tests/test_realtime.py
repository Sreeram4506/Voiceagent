"""Realtime speech-to-speech session, driven with canned OpenAI events (no network)."""

import asyncio
import base64
import json

import numpy as np

from app.streaming.realtime_session import RealtimeSession, pcm_to_ulaw, ulaw_to_pcm


def agent_audio(item_id: str, seconds: float) -> dict:
    ulaw = pcm_to_ulaw(bytes(int(8000 * seconds) * 2))
    return {"type": "response.output_audio.delta", "item_id": item_id, "delta": base64.b64encode(ulaw).decode()}


# Language-style rules apply to the full models; mini models get a short plain-Telugu brief instead.
FULL_MODEL = "gpt-realtime-1.5"


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
        await session._configure()  # settings, then the greeting request right behind them
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


def test_silent_hang_up_says_goodbye_first_then_ends_the_call():
    async def scenario():
        session, to_openai, _, send, end, ended = make_session()
        await session.attach("sid", send, end)
        await session._on_event({"type": "input_audio_buffer.committed", "item_id": "c-bye"})

        hangup = {"type": "function_call", "name": "end_call", "call_id": "c1"}
        await session._on_event({"type": "response.done", "response": {"status": "completed", "output": [hangup]}})
        assert to_openai[-1]["type"] == "response.create"
        assert "goodbye" in to_openai[-1]["response"]["instructions"]
        await asyncio.sleep(0.4)
        assert not ended  # no hanging up before the goodbye is said

        goodbye = {"type": "message", "role": "assistant"}
        await session._on_event(agent_audio("bye", 0.1))
        await session._on_event({"type": "response.done", "response": {"status": "completed", "output": [goodbye]}})
        await asyncio.sleep(0.6)
        assert ended and session.ended

    asyncio.run(scenario())


def test_reply_the_caller_talked_over_never_plays_on_the_line():
    async def scenario():
        session, _, to_phone, send, end, _ = make_session()
        await session.attach("sid", send, end)
        await session._on_event({"type": "response.created", "response": {"id": "resp_early"}})
        await session._on_event({"type": "input_audio_buffer.speech_started"})  # caller kept talking

        await session._on_event(agent_audio("early", 0.5) | {"response_id": "resp_early"})
        assert to_phone == []

        await session._on_event({"type": "response.created", "response": {"id": "resp_real"}})
        await session._on_event(agent_audio("real", 0.5) | {"response_id": "resp_real"})
        assert [m["event"] for m in to_phone] == ["playAudio"]

    asyncio.run(scenario())


def test_agent_knows_who_it_is_calling_and_what_time_it_is_there():
    from datetime import datetime

    session, *_ = make_session(caller_name="Ravi  Kumar", caller_notes="Asked about east-facing plots",
                               local_time=datetime(2026, 11, 16, 18, 30))
    assert "You are speaking with Ravi Kumar" in session._instructions
    assert "Monday evening" in session._instructions
    assert "east-facing plots" in session._instructions
    assert "speaking with Ravi Kumar" in session._greeting_request()["response"]["instructions"]

    unnamed, *_ = make_session()
    assert "never guess one" in unnamed._instructions
    assert "# This call" not in unnamed._instructions


def test_silent_line_gets_two_check_ins_and_then_a_polite_goodbye():
    async def scenario():
        import time

        from app.streaming.realtime_session import MAX_SILENCE_CHECKS, SILENCE_CHECK_SECONDS

        session, to_openai, _, send, end, ended = make_session()
        await session.attach("sid", send, end)
        now = time.monotonic()
        assert session._silent_for(now + SILENCE_CHECK_SECONDS + 1) > SILENCE_CHECK_SECONDS

        await session._on_event({"type": "input_audio_buffer.speech_started"})
        assert session._silent_for(now + 60) == 0  # the caller is talking
        await session._on_event({"type": "input_audio_buffer.speech_stopped"})

        for _ in range(MAX_SILENCE_CHECKS):
            await session._check_in()
            assert "still there" in to_openai[-1]["response"]["instructions"]
            assert session._silent_for(time.monotonic() + 60) == 0  # waiting for the agent to finish asking
            await session._on_event({"type": "response.done", "response": {"status": "completed", "output": []}})
        assert not session._giving_up

        await session._check_in()
        assert "say goodbye" in to_openai[-1]["response"]["instructions"]
        await session._on_event({"type": "response.done", "response": {"status": "completed", "output": []}})
        await asyncio.sleep(0.5)
        assert ended and session.ended
        await session.cancel()

    asyncio.run(scenario())


def test_telugu_natural_mix_uses_swap_list_and_examples():
    session, *_ = make_session(language_style="natural_mix", model=FULL_MODEL)
    assert "టైమ్" in session._instructions
    assert "రేట్" in session._instructions
    assert "సైట్ విజిట్" in session._instructions
    assert "కన్ఫర్మ్" in session._instructions
    assert "ఒక్క నిమిషం టైమ్ ఉందా అండి" in session._instructions
    assert "Never translate an English word into formal Telugu" in session._instructions


def test_telugu_pure_style_avoids_english():
    session, *_ = make_session(language_style="pure", model=FULL_MODEL)
    assert "pure, natural Telugu" in session._instructions
    assert "టైమ్" not in session._instructions


def test_telugu_english_heavy_style():
    session, *_ = make_session(language_style="english_heavy", model=FULL_MODEL)
    assert "lean heavily on English" in session._instructions


def test_hindi_natural_mix_uses_hindi_swap_list():
    from app.streaming.realtime_session import RealtimeSession
    session = RealtimeSession("key", "Book visits.", "hi", language_style="natural_mix")
    assert "टाइम" in session._instructions
    assert "रेट" in session._instructions
    assert "Delhi" in session._instructions


def test_english_language_ignores_language_style():
    from app.streaming.realtime_session import RealtimeSession
    session = RealtimeSession("key", "Book visits.", "en", language_style="natural_mix")
    assert "Indian English" in session._instructions
    assert "టైమ్" not in session._instructions


def test_caller_speaking_again_resets_the_silence_checks():
    async def scenario():
        session, _, _, send, end, ended = make_session()
        await session.attach("sid", send, end)
        await session._check_in()
        await session._check_in()
        await session._on_event({"type": "input_audio_buffer.committed", "item_id": "u1"})
        assert session._silence_checks == 0
        await session.cancel()
        assert not ended

    asyncio.run(scenario())


REPLY_USAGE = {
    "input_token_details": {"text_tokens": 2000, "audio_tokens": 500,
                            "cached_tokens_details": {"text_tokens": 1500, "audio_tokens": 100}},
    "output_token_details": {"text_tokens": 50, "audio_tokens": 300},
}


def test_call_cost_adds_every_reply_and_the_caller_transcription():
    async def scenario():
        session, *_ = make_session(model="gpt-realtime-1.5")
        for _ in range(2):
            await session._on_event({"type": "response.done", "response": {"status": "completed", "usage": REPLY_USAGE}})
        await session._on_event({"type": "conversation.item.input_audio_transcription.completed", "item_id": "c1",
                                 "transcript": "హలో", "usage": {"type": "tokens", "input_tokens": 40, "output_tokens": 10}})
        # per reply: 500 text + 1500 cached text + 50 text out + 400 audio + 100 cached audio + 300 audio out
        reply = 500 * 4.00 + 1500 * 0.40 + 50 * 16.00 + 400 * 32.00 + 100 * 0.40 + 300 * 64.00
        transcription = 40 * 2.50 + 10 * 10.00
        assert abs(session.cost_usd() - (2 * reply + transcription) / 1_000_000) < 1e-12
        report = session.usage_report()
        assert report.startswith("gpt-realtime-1.5 $0.0711") and "2 replies" in report

    asyncio.run(scenario())


def test_model_without_a_known_price_reports_tokens_but_no_cost():
    async def scenario():
        session, *_ = make_session(model="gpt-realtime-future")
        await session._on_event({"type": "response.done", "response": {"status": "completed", "usage": REPLY_USAGE}})
        assert session.cost_usd() is None
        assert "no price known" in session.usage_report() and "audio out 300" in session.usage_report()

    asyncio.run(scenario())


def _goodbye_reply(session, text: str):
    async def play():
        await session._on_event(agent_audio("bye", 0.1))
        await session._on_event({"type": "response.output_audio_transcript.done", "item_id": "bye", "transcript": text})
        await session._on_event({"type": "response.done", "response": {
            "status": "completed", "output": [{"type": "message", "role": "assistant"}]}})
    return play()


def test_agent_hangs_up_after_its_own_goodbye_even_without_the_tool():
    async def scenario():
        session, _, _, send, end, ended = make_session()
        await session.attach("sid", send, end)
        await _goodbye_reply(session, "సరే అండి, థాంక్యూ, bye!")
        await asyncio.sleep(0.6)
        assert ended and session.ended

    asyncio.run(scenario())


def test_bye_in_the_middle_of_a_reply_does_not_hang_up():
    async def scenario():
        session, _, _, send, end, ended = make_session()
        await session.attach("sid", send, end)
        await _goodbye_reply(session, "Bye చెప్పే ముందు, ఒక్క విషయం అండి, Saturday ఓకేనా?")
        await asyncio.sleep(0.6)
        assert not ended
        await session.cancel()

    asyncio.run(scenario())


def speech(level: float, seconds: float = 0.02) -> bytes:
    t = np.arange(int(8000 * seconds))
    return (np.sin(t * 0.3) * level * np.sqrt(2)).astype("<i2").tobytes()


def agent_speech(item_id: str, seconds: float, level: float = 3000.0) -> dict:
    ulaw = pcm_to_ulaw(speech(level, seconds))
    return {"type": "response.output_audio.delta", "item_id": item_id, "delta": base64.b64encode(ulaw).decode()}


def sent_audio(to_openai: list[dict]) -> list[bytes]:
    return [ulaw_to_pcm(base64.b64decode(e["audio"])) for e in to_openai if e["type"] == "input_audio_buffer.append"]


def test_echo_of_the_agents_own_voice_reaches_openai_as_silence():
    async def scenario():
        session, to_openai, _, send, end, _ = make_session()
        session._ws = object()
        await session.attach("sid", send, end)
        session._guard_until = 0.0
        await session._on_event(agent_speech("a1", 5.0))
        await session.on_audio(speech(3000 * 0.15))  # the phone leaking the agent back
        heard = np.frombuffer(sent_audio(to_openai)[-1], dtype="<i2")
        assert np.abs(heard).max() < 50

    asyncio.run(scenario())


def test_caller_talking_over_the_agent_still_reaches_openai():
    async def scenario():
        session, to_openai, _, send, end, _ = make_session()
        session._ws = object()
        await session.attach("sid", send, end)
        session._guard_until = 0.0
        await session._on_event(agent_speech("a1", 5.0))
        for _ in range(3):
            await session.on_audio(speech(2500))
        heard = np.frombuffer(sent_audio(to_openai)[-1], dtype="<i2")
        assert np.abs(heard).max() > 1500

    asyncio.run(scenario())


def test_quiet_caller_is_heard_once_the_agent_has_finished():
    async def scenario():
        session, to_openai, _, send, end, _ = make_session()
        session._ws = object()
        await session.attach("sid", send, end)
        session._guard_until = 0.0
        await session.on_audio(speech(300))  # nothing playing: even a soft "హా" goes through
        heard = np.frombuffer(sent_audio(to_openai)[-1], dtype="<i2")
        assert np.abs(heard).max() > 200

    asyncio.run(scenario())
