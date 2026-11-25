"""Which speech-to-speech service runs a call."""

import pytest

from app.streaming import factory


@pytest.fixture
def env(monkeypatch):
    for name in ("VA_VOICE_ENGINE", "GEMINI_API_KEY", "OPENAI_API_KEY", "TESTING", "VA_VOICE_PIPELINE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_gemini_runs_calls_by_default_when_its_key_is_set(env):
    env.setenv("GEMINI_API_KEY", "g")
    assert factory.voice_engine() == "gemini"


def test_openai_runs_calls_when_there_is_no_gemini_key(env):
    env.setenv("OPENAI_API_KEY", "o")
    assert factory.voice_engine() == "openai"
    assert factory.use_realtime("standard", "outbound")


def test_va_voice_engine_openai_overrides_the_gemini_default(env):
    env.setenv("GEMINI_API_KEY", "g")
    env.setenv("VA_VOICE_ENGINE", "OpenAI")
    assert factory.voice_engine() == "openai"


def test_gemini_default_uses_the_live_voice_not_the_relay(env):
    env.setenv("GEMINI_API_KEY", "g")
    assert factory.use_realtime("standard", "outbound")
