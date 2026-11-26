"""A recorded-audio call asks a yes/no question at the end and saves the caller's answer."""

import asyncio
import json

import numpy as np

from app.streaming.protocol import parse_event
from app.streaming.recorded import RecordedAudioSession
from app.streaming.yes_no import AnswerListener, answer_from_digit, classify_reply

FRAME = 320  # 20 ms of 8 kHz 16-bit audio


def voice(level: int = 3000) -> bytes:
    return (np.sin(np.linspace(0, 60, FRAME // 2)) * level).astype("<i2").tobytes()


SILENCE = bytes(FRAME)


def test_spoken_replies_in_telugu_hindi_and_english():
    for text in ("అవును అండి", "హా సరే", "Yes, interested", "ok", "హాఁ జీ", "కావాలి", "చెప్పండి ఓకే"):
        assert classify_reply(text) == "yes", text
    for text in ("లేదండి", "వద్దు", "No thanks", "not interested", "नहीं", "అవసరం లేదు"):
        assert classify_reply(text) == "no", text
    for text in ("", "హలో", "who is this?", "తర్వాత చెప్తా"):
        assert classify_reply(text) is None, text


def test_keypad_one_is_yes_and_two_is_no():
    assert answer_from_digit("1") == "yes"
    assert answer_from_digit("2") == "no"
    assert answer_from_digit("5") is None


def test_listener_waits_for_the_caller_to_finish_speaking():
    listener = AnswerListener()
    for _ in range(10):
        assert not listener.feed(SILENCE)  # nothing said yet
    for _ in range(25):
        assert not listener.feed(voice())  # half a second of speech
    done = False
    for _ in range(60):
        done = listener.feed(SILENCE) or done
    assert done and listener.spoke
    assert len(listener.audio()) >= 25 * FRAME


def test_listener_ignores_line_hiss():
    listener = AnswerListener()
    for _ in range(100):
        listener.feed(voice(level=150))
    assert not listener.spoke


def test_dtmf_events_are_parsed_from_the_stream():
    event = parse_event(json.dumps({"event": "dtmf", "streamId": "s1", "dtmf": {"digit": "1"}}))
    assert event.kind == "dtmf" and event.digit == "1"
    flat = parse_event(json.dumps({"event": "dtmf", "digit": "2"}))
    assert flat.digit == "2"


def make_session(transcript: str = "", **timing):
    sent: list[dict] = []
    ended: list[bool] = []
    heard: list[bytes] = []

    async def send(message: str) -> None:
        sent.append(json.loads(message))

    async def end() -> None:
        ended.append(True)

    async def transcribe(wav: bytes) -> str:
        heard.append(wav)
        return transcript

    session = RecordedAudioSession(pcm=bytes(8000), stream_sid="s1", send=send, on_end=end,
                                   transcribe=transcribe, **timing)
    return session, sent, ended, heard


def test_pressing_one_during_the_message_saves_yes_and_hangs_up():
    async def scenario():
        session, _, ended, _ = make_session()
        await session.start()
        await asyncio.sleep(0.05)
        await session.on_dtmf("1")
        await asyncio.sleep(0.1)
        return session, ended

    session, ended = asyncio.run(scenario())
    assert session.outcome() == "said_yes"
    assert ("caller", "(pressed 1: yes)") in session.turns
    assert ended


def test_spoken_no_after_the_message_is_transcribed_and_saved():
    async def scenario():
        session, _, ended, heard = make_session(transcript="వద్దు అండి", answer_wait=1.0)
        await session.start()
        await asyncio.sleep(1.1)  # the half-second message and the pause after it
        for _ in range(25):
            await session.on_audio(voice())
        for _ in range(60):
            await session.on_audio(SILENCE)
        await asyncio.sleep(0.2)
        return session, ended, heard

    session, ended, heard = asyncio.run(scenario())
    assert heard, "the reply was sent for transcription"
    assert session.outcome() == "said_no"
    assert ("caller", "వద్దు అండి") in session.turns
    assert ended


def test_unclear_reply_is_kept_for_the_team_to_hear():
    async def scenario():
        session, _, _, _ = make_session(transcript="ఎవరు మాట్లాడుతున్నారు?", answer_wait=1.0)
        await session.start()
        await asyncio.sleep(1.1)
        for _ in range(25):
            await session.on_audio(voice())
        for _ in range(60):
            await session.on_audio(SILENCE)
        await asyncio.sleep(0.2)
        return session

    session = asyncio.run(scenario())
    assert session.outcome() == "unclear_reply"
    assert ("caller", "ఎవరు మాట్లాడుతున్నారు?") in session.turns


def test_silence_after_the_message_is_no_reply():
    async def scenario():
        session, _, ended, heard = make_session(answer_wait=0.3)
        await session.start()
        await asyncio.sleep(1.6)
        return session, ended, heard

    session, ended, heard = asyncio.run(scenario())
    assert session.outcome() == "no_reply"
    assert ended and not heard


def test_caller_who_hangs_up_during_the_message_has_no_resolution():
    async def scenario():
        session, _, _, _ = make_session()
        await session.start()
        await session.cancel()
        return session

    assert asyncio.run(scenario()).outcome() == "no_resolution"


def test_recorded_call_is_recorded_with_the_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("VA_RECORDINGS_DIR", str(tmp_path))

    async def scenario():
        session, _, _, _ = make_session(transcript="అవును", answer_wait=1.0)
        await session.start()
        await asyncio.sleep(1.1)
        for _ in range(25):
            await session.on_audio(voice())
        for _ in range(60):
            await session.on_audio(SILENCE)
        await asyncio.sleep(0.2)
        return session

    session = asyncio.run(scenario())
    assert session.outcome() == "said_yes"
    assert session.recorder is not None and session.recorder.has_audio()
