"""Chooses speech and LLM providers for a call.

TTS priority:
  voice_clone tier → XTTS v2 (zero-shot voice cloning, ~1-3s on MPS)
  Telugu/Hindi     → EdgeTTS (Microsoft Neural, ~0.9s, native pronunciation)
  English          → OpenAI tts-1 (best English quality)
  Fallback         → Piper (offline) → Mock

voice_clone: set voice_tier="voice_clone" on the employee and place a 10-30s
  WAV at voice_profiles/reference.wav (or set XTTS_REFERENCE_WAV env var).

STT: OpenAI Whisper
LLM: OpenAI GPT-4o-mini → Claude Haiku → Mock
"""

import os

from app.streaming.mock_providers import MockAsyncLLM, MockAsyncSTT, MockAsyncTTS


def _build_tts(employee, openai_key, extras=None):
    lang = getattr(employee, "language", "te") if employee else "te"
    voice_tier = getattr(employee, "voice_tier", "standard") if employee else "standard"
    extras = extras or {}
    gender = extras.get("voice_gender", "female")
    speech_rate = extras.get("speech_rate", "+0%")

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

    # Premium tier: OpenAI tts-1 — higher quality for any language
    if str(voice_tier) == "premium" and openai_key:
        from app.streaming.openai_tts import OpenAITTS
        speed = 0.9 if lang in ("te", "hi") else 1.0
        print(f"[Factory] OpenAI TTS (premium) lang={lang} speed={speed}")
        return OpenAITTS(openai_key, voice="nova", speed=speed)

    # Telugu & Hindi standard: Microsoft Edge Neural voices — free, native pronunciation
    if lang in ("te", "hi"):
        try:
            from app.streaming.edge_tts_provider import EdgeTTS
            rate = speech_rate if speech_rate else "-5%"
            print(f"[Factory] EdgeTTS lang={lang} gender={gender} rate={rate}")
            return EdgeTTS(gender=gender, rate=rate)
        except Exception as e:
            print(f"[Factory] EdgeTTS unavailable ({e}), falling back")

    # English standard: OpenAI tts-1
    if lang == "en" and openai_key:
        from app.streaming.openai_tts import OpenAITTS
        print("[Factory] OpenAI TTS (English)")
        return OpenAITTS(openai_key, voice="nova", speed=1.0)

    # Piper — local offline fallback
    try:
        from app.streaming.piper_tts import PiperTTS
        print("[Factory] Piper TTS (offline fallback)")
        return PiperTTS()
    except Exception as e:
        print(f"[Factory] Piper TTS unavailable ({e})")

    return MockAsyncTTS()


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
