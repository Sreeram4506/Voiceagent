"""OpenAI TTS adapter — uses OpenAI's TTS API for Telugu speech synthesis.

Supports Telugu natively (and all other languages). Outputs 8kHz 16-bit mono PCM.
Model: tts-1 (fast) or tts-1-hd (higher quality)
Voice: alloy, echo, fable, onyx, nova, shimmer
"""

import io
import os
from math import gcd

import numpy as np

OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "nova")
TELEPHONY_SAMPLE_RATE = 8000
OPENAI_TTS_SAMPLE_RATE = 24000  # OpenAI PCM output is 24kHz


def _pcm24k_to_8k(pcm24k: bytes) -> bytes:
    """Downsample 24kHz 16-bit mono PCM to 8kHz with normalization."""
    from scipy.signal import resample_poly

    audio = np.frombuffer(pcm24k, dtype=np.int16).astype(np.float32) / 32768.0

    if audio.size == 0:
        return b""

    g = gcd(OPENAI_TTS_SAMPLE_RATE, TELEPHONY_SAMPLE_RATE)
    audio = resample_poly(audio, TELEPHONY_SAMPLE_RATE // g, OPENAI_TTS_SAMPLE_RATE // g)

    # Normalize to -20 dBFS, cap boost at +20 dB
    peak = float(np.abs(audio).max())
    if peak < 1e-6:
        return b""

    target_peak = 10 ** (-20.0 / 20)
    gain = min(target_peak / peak, 10 ** (20 / 20))
    audio = np.clip(audio * gain, -1.0, 1.0)

    return (audio * 32767).astype(np.int16).tobytes()


class OpenAITTS:
    """Async TTS adapter using OpenAI TTS API. Outputs 8kHz 16-bit mono PCM."""

    def __init__(self, api_key: str, model: str = OPENAI_TTS_MODEL, voice: str = OPENAI_TTS_VOICE):
        self.api_key = api_key
        self.model = model
        self.voice = voice

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        """Generate speech and return 8kHz 16-bit mono PCM bytes."""
        import asyncio
        import time
        from openai import OpenAI

        loop = asyncio.get_event_loop()

        def _call_api():
            t0 = time.time()
            client = OpenAI(api_key=self.api_key)
            response = client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="pcm",  # raw 24kHz 16-bit mono PCM
            )
            pcm24k = response.content
            elapsed = time.time() - t0
            print(f"[OpenAI-TTS] synthesis={elapsed:.2f}s text_len={len(text)} model={self.model} voice={self.voice}")
            return pcm24k

        pcm24k = await loop.run_in_executor(None, _call_api)
        return _pcm24k_to_8k(pcm24k)
