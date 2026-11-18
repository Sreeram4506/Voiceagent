"""Chooses speech and LLM providers for a call.

Default: OpenAI Realtime speech-to-speech (see realtime_session.py). The STT/LLM/TTS
relay below is the fallback when Realtime is unavailable, serves voice_clone agents,
and can be forced with VA_VOICE_PIPELINE=classic to save cost.

TTS priority:
  voice_clone tier       → XTTS v2 (zero-shot voice cloning, ~1-3s on MPS)
  te / hi / en / ta / kn → EdgeTTS (Microsoft Neural, free, native pronunciation)
  Other languages        → OpenAI tts-1 → Piper (offline) → Mock

voice_clone: set voice_tier="voice_clone" on the employee and place a 10-30s
  WAV at voice_profiles/reference.wav (or set XTTS_REFERENCE_WAV env var).

STT: OpenAI gpt-4o-mini-transcribe
LLM: OpenAI GPT-4o-mini → Claude Haiku → Mock
"""

import os

from app.streaming.mock_providers import MockAsyncLLM, MockAsyncSTT, MockAsyncTTS


def _speech_rate(extras: dict) -> float:
    """The agent's saved speech rate as a multiplier (1.0 = normal)."""
    try:
        return float(extras.get("speech_rate") or 1.0)
    except (TypeError, ValueError):
        return 1.0


def _build_tts(employee, openai_key, extras=None):
    lang = getattr(employee, "language", "te") if employee else "te"
    voice_tier = getattr(employee, "voice_tier", "standard") if employee else "standard"
    extras = extras or {}
    gender = extras.get("voice_gender", "female")
    speech_rate = _speech_rate(extras)

    # Voice clone tier: Coqui XTTS v2 (zero-shot cloning from reference WAV)
    if str(voice_tier) == "voice_clone":
        try:
            from app.streaming.xtts_provider import XTTS
            ref = os.getenv("XTTS_REFERENCE_WAV")
            provider = XTTS(reference_wav=ref)
            print(f"[Factory] XTTS v2 voice clone lang={lang}")
            return provider
        except Exception as e:
            print(f"[Factory] XTTS unavailable ({e}), falling back to EdgeTTS")

    # Microsoft Edge Neural voices for every tier: free, native Indian-language pronunciation
    try:
        from app.streaming.edge_tts_provider import EDGE_LANGUAGES, EdgeTTS
        if lang in EDGE_LANGUAGES:
            rate = f"{round((speech_rate - 1) * 100):+d}%"
            print(f"[Factory] EdgeTTS lang={lang} gender={gender} rate={rate}")
            return EdgeTTS(gender=gender, rate=rate)
    except Exception as e:
        print(f"[Factory] EdgeTTS unavailable ({e}), falling back")

    if openai_key:
        from app.streaming.openai_tts import OpenAITTS
        print(f"[Factory] OpenAI TTS lang={lang} speed={speech_rate}")
        return OpenAITTS(openai_key, voice="nova", speed=speech_rate)

    # Piper — local offline fallback
    try:
        from app.streaming.piper_tts import PiperTTS
        print("[Factory] Piper TTS (offline fallback)")
        return PiperTTS()
    except Exception as e:
        print(f"[Factory] Piper TTS unavailable ({e})")

    return MockAsyncTTS()


def use_realtime(voice_tier: str, mode: str) -> bool:
    """Speech-to-speech over OpenAI Realtime is the default. Set VA_VOICE_PIPELINE=classic for the cheaper relay."""
    if os.getenv("TESTING") == "1" or os.getenv("VA_VOICE_PIPELINE", "realtime") == "classic":
        return False
    key = "GEMINI_API_KEY" if voice_engine() == "gemini" else "OPENAI_API_KEY"
    return bool(os.getenv(key)) and mode != "broadcast" and voice_tier != "voice_clone"


def voice_engine() -> str:
    """Which speech-to-speech service runs calls: "openai" (default) or "gemini" (VA_VOICE_ENGINE=gemini)."""
    return os.getenv("VA_VOICE_ENGINE", "openai").strip().lower()


def build_providers(employee, extras=None):
    openai_key = os.getenv("OPENAI_API_KEY")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    extras = extras or {}

    tts = _build_tts(employee, openai_key, extras)
    temperature = float(extras.get("temperature", 0.85))

    if openai_key:
        from app.streaming.openai_llm import OpenAILLM
        from app.streaming.openai_stt import OpenAISTT
        return OpenAISTT(openai_key), OpenAILLM(openai_key, temperature=temperature), tts

    if deepgram_key and anthropic_key:
        from app.streaming.claude_llm import ClaudeLLM
        from app.streaming.deepgram_stt import DeepgramSTT
        return DeepgramSTT(deepgram_key), ClaudeLLM(anthropic_key), tts

    return MockAsyncSTT(), MockAsyncLLM(), tts
