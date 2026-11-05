"""Chooses the speech and LLM providers for a call.

Priority: Piper TTS (local offline Telugu voice, ~50ms synthesis).
OpenAI (Whisper STT + GPT LLM) for speech recognition and language model.
"""

import os

from app.streaming.mock_providers import MockAsyncLLM, MockAsyncSTT, MockAsyncTTS


def _build_tts(employee, elevenlabs_key, deepgram_key):
    lang = getattr(employee, "language", "te") if employee else "te"

    # For English-specific agents, prefer high-quality multilingual engines
    if lang == "en":
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            from app.streaming.openai_tts import OpenAITTS
            print("[Factory] Using OpenAI TTS for English agent")
            return OpenAITTS(openai_key, voice="nova")
        if elevenlabs_key:
            try:
                from app.streaming.elevenlabs_tts import ElevenLabsTTS
                print("[Factory] Using ElevenLabs TTS for English agent")
                return ElevenLabsTTS(elevenlabs_key)
            except Exception:
                pass

    # Piper TTS — local offline Telugu voice, RTF=0.02x (~50ms for 2.5s audio)
    try:
        from app.streaming.piper_tts import PiperTTS
        print("[Factory] Using Piper TTS (fast Telugu & Kanglish voice)")
        return PiperTTS()
    except Exception as e:
        print(f"[Factory] Piper TTS unavailable ({e}), trying fallback")

    # OpenAI TTS — supports Telugu natively, fallback
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from app.streaming.openai_tts import OpenAITTS
        print("[Factory] Using OpenAI TTS (fallback)")
        return OpenAITTS(openai_key)

    if elevenlabs_key:
        try:
            from app.streaming.elevenlabs_tts import ElevenLabsTTS
            print("[Factory] Using ElevenLabs TTS")
            return ElevenLabsTTS(elevenlabs_key)
        except Exception as e:
            print(f"[Factory] ElevenLabs TTS unavailable ({e}), trying fallback")

    if deepgram_key:
        from app.streaming.elevenlabs_tts import DeepgramTTS
        print("[Factory] Falling back to Deepgram TTS")
        return DeepgramTTS(deepgram_key)
    return MockAsyncTTS()


def build_providers(employee):
    openai_key = os.getenv("OPENAI_API_KEY")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

    if openai_key:
        from app.streaming.openai_llm import OpenAILLM
        from app.streaming.openai_stt import OpenAISTT
        return OpenAISTT(openai_key), OpenAILLM(openai_key), _build_tts(employee, elevenlabs_key, deepgram_key)

    if deepgram_key and anthropic_key:
        from app.streaming.claude_llm import ClaudeLLM
        from app.streaming.deepgram_stt import DeepgramSTT
        return DeepgramSTT(deepgram_key), ClaudeLLM(anthropic_key), _build_tts(employee, elevenlabs_key, deepgram_key)

    return MockAsyncSTT(), MockAsyncLLM(), MockAsyncTTS()
