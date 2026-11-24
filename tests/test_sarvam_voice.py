"""Sarvam's Telugu voice on an agent: the voice adapter, the pipeline switch and the prompt it runs with."""

import asyncio
import json
import struct

import httpx
import pytest

from app.streaming import factory
from app.streaming.sarvam_tts import SARVAM_TTS_URL, SarvamTTS, SarvamTTSError

SARVAM = {"voice_engine": "sarvam", "voice_gender": "female", "speech_rate": 0.9}


def pcm(samples: int, value: int = 1000) -> bytes:
    return struct.pack(f"<{samples}h", *([value] * samples))


def wav(data: bytes) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
    return header + b"data" + struct.pack("<I", len(data)) + data


def recording_transport(chunks: list[bytes], status: int = 200, seen: list | None = None) -> httpx.MockTransport:
    async def stream():
        for chunk in chunks:
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "Invalid API key", "code": "invalid_api_key_error"}})
        return httpx.Response(200, content=stream(), headers={"content-type": "audio/pcm"})

    return httpx.MockTransport(handler)


def collect(tts: SarvamTTS, text: str = "హలో అండి") -> list[bytes]:
    async def run():
        return [chunk async for chunk in tts.stream_synthesize(text, "te", "premium")]
    return asyncio.run(run())


def test_asks_sarvam_for_telugu_phone_audio_with_the_agents_voice_and_pace():
    seen: list[httpx.Request] = []
    tts = SarvamTTS("key-1", gender="female", pace=0.9, transport=recording_transport([pcm(800)], seen=seen))
    collect(tts)
    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == SARVAM_TTS_URL
    assert request.headers["api-subscription-key"] == "key-1"
    assert body["language_code"] == "te-IN" and body["text"] == "హలో అండి"
    assert body["speech_sample_rate"] == 8000 and body["output_audio_codec"] == "linear16"
    assert body["speaker"].endswith("_te_conversation") and body["pace"] == 0.9


def test_streams_audio_as_it_arrives_in_whole_samples():
    tts = SarvamTTS("key", transport=recording_transport([pcm(100) + b"\x01", b"\x02" + pcm(100)]))
    chunks = collect(tts)
    assert all(len(chunk) % 2 == 0 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == len(pcm(201))


def test_a_wav_header_in_front_of_the_audio_is_not_played():
    tts = SarvamTTS("key", transport=recording_transport([wav(pcm(400))]))
    assert b"".join(collect(tts)) == pcm(400)


def test_full_reply_audio_for_the_pipeline():
    tts = SarvamTTS("key", transport=recording_transport([pcm(300), pcm(300)]))
    audio = asyncio.run(tts.synthesize("సరే అండి", "te", "premium"))
    assert audio == pcm(600)


def test_pace_stays_inside_what_sarvam_accepts():
    assert SarvamTTS("key", pace=5.0).pace == 2.0
    assert SarvamTTS("key", pace=0.1).pace == 0.5


def test_a_rejected_key_says_so_clearly():
    tts = SarvamTTS("bad", transport=recording_transport([], status=403))
    with pytest.raises(SarvamTTSError, match="403"):
        collect(tts)


def test_agent_set_to_sarvam_gets_sarvams_voice(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "key")
    employee = type("E", (), {"language": "te", "voice_tier": "premium"})()
    tts = factory._build_tts(employee, "openai-key", SARVAM)
    assert isinstance(tts, SarvamTTS) and tts.pace == 0.9


def test_without_a_sarvam_key_the_agent_keeps_the_realtime_voice(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert not factory.sarvam_voice(SARVAM)
    employee = type("E", (), {"language": "te", "voice_tier": "premium"})()
    assert not isinstance(factory._build_tts(employee, "openai-key", SARVAM), SarvamTTS)


def test_sarvam_agents_skip_realtime_and_others_keep_it(monkeypatch):
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("SARVAM_API_KEY", "key")
    assert factory.use_realtime("premium", "ai_agent", SARVAM) is False
    assert factory.use_realtime("premium", "ai_agent", {"voice_engine": "openai"}) is True
    assert factory.use_realtime("premium", "ai_agent") is True


def test_sarvam_pipeline_replies_with_the_tuned_prompt_not_the_old_rules(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    from app.streaming.openai_llm import OpenAILLM
    employee = type("E", (), {"language": "te", "voice_tier": "premium"})()
    _, llm, _ = factory.build_providers(employee, extras=SARVAM)
    assert isinstance(llm, OpenAILLM) and llm.rules == ""
    _, default_llm, _ = factory.build_providers(employee, extras={})
    assert default_llm.rules.startswith("VOICE CALL RULES")


def sarvam_context(greeting: str = ""):
    from app.models import VoiceTier
    from app.streaming.ws import CallContext
    employee = type("E", (), {"name": "Suma", "language": "te", "voice_tier": VoiceTier.PREMIUM,
                              "script": "You are Suma from Indraam, calling about our AI voice agent."})()
    return CallContext(employee=employee, extras={**SARVAM, "speaking_style": "professional"}, mode="ai_agent",
                       greeting=greeting, caller_name="Sreeram")


def test_sarvam_call_runs_on_the_tuned_prompt(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "key")
    from app.streaming.ws import _classic_script
    script = _classic_script(sarvam_context())
    assert script.startswith("You are Suma from Indraam")
    assert "NEVER ASK IF THEY WANT TO HEAR MORE" in script and "# Pronunciation" in script
    assert "Sreeram" in script


def test_sarvam_call_opens_by_asking_if_they_are_free(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "key")
    from app.streaming.ws import _opening_request
    assert "ఇప్పుడు free గా ఉన్నారా?" in _opening_request(sarvam_context())
    assert "నమస్కారం" in _opening_request(sarvam_context(greeting="నమస్కారం అండి"))


def test_other_agents_keep_the_relay_script_and_opening(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    from app.streaming.ws import _classic_script, _opening_request
    ctx = sarvam_context()
    assert "# Flow" not in _classic_script(ctx)
    assert _opening_request(ctx) == ""


def test_dashboard_saves_the_voice_engine_and_rejects_unknown_ones():
    from app.dashboard_ops import _agent_extras
    assert _agent_extras({"voice_engine": "sarvam"}) == {"voice_engine": "sarvam"}
    assert _agent_extras({"voice_engine": "elevenlabs"}) == {"voice_engine": "openai"}


def classic_session():
    from app.streaming.mock_providers import MockAsyncLLM, MockAsyncSTT, MockAsyncTTS
    from app.streaming.session import StreamSession
    sent: list[str] = []

    async def send(message: str) -> None:
        sent.append(message)

    session = StreamSession(stt=MockAsyncSTT(), llm=MockAsyncLLM(), tts=MockAsyncTTS(), script="", language="te",
                            voice_tier="premium", stream_sid="sid", send=send)
    return session, sent


def tone(level: float, samples: int = 160) -> bytes:
    import math
    return struct.pack(f"<{samples}h", *[int(math.sin(i * 0.3) * level * 1.414) for i in range(samples)])


def test_relay_hears_its_own_echo_as_silence_while_it_talks():
    async def scenario():
        session, _ = classic_session()
        await session._send_agent_audio(tone(3000, 8000 * 3))  # three seconds of the agent talking
        assert session._without_echo(tone(450)) == bytes(320)

    asyncio.run(scenario())


def test_relay_still_hears_a_caller_who_talks_over_it():
    async def scenario():
        session, _ = classic_session()
        await session._send_agent_audio(tone(3000, 8000 * 3))
        loud = tone(2500)
        session._without_echo(loud)
        assert session._without_echo(loud) == loud

    asyncio.run(scenario())


def test_relay_hears_everything_once_it_has_stopped_talking():
    session, _ = classic_session()
    quiet = tone(300)
    assert session._without_echo(quiet) == quiet
