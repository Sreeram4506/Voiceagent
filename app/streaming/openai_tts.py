"""OpenAI TTS adapter — fastest cloud TTS, supports all languages including Telugu.

Model: tts-1 (lowest latency)
Voices: alloy, echo, fable, onyx, nova, shimmer
Speed: 1.25 — natural, energetic, human phone pace
Streaming: yields 8kHz PCM chunks as they arrive from OpenAI (first audio in ~200ms)
"""

import os
from math import gcd

import numpy as np

OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "nova")
OPENAI_TTS_SPEED = float(os.environ.get("OPENAI_TTS_SPEED", "1.0"))
TELEPHONY_SAMPLE_RATE = 8000
OPENAI_TTS_SAMPLE_RATE = 24000  # OpenAI PCM output is 24kHz

# Accumulate this many 24kHz bytes before downsampling and yielding a chunk.
# 4800 bytes = 100ms at 24kHz 16-bit mono — small enough for low latency.
_STREAM_CHUNK_BYTES = 4800


def _pcm24k_to_8k(pcm24k: bytes) -> bytes:
    """Downsample 24kHz 16-bit mono PCM to 8kHz."""
    from scipy.signal import resample_poly

    audio = np.frombuffer(pcm24k, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return b""

    g = gcd(OPENAI_TTS_SAMPLE_RATE, TELEPHONY_SAMPLE_RATE)
    audio = resample_poly(audio, TELEPHONY_SAMPLE_RATE // g, OPENAI_TTS_SAMPLE_RATE // g)

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


class OpenAITTS:
    """Async TTS using OpenAI TTS API. 8kHz 16-bit mono PCM output with streaming."""

    def __init__(self, api_key: str, model: str = OPENAI_TTS_MODEL, voice: str = OPENAI_TTS_VOICE,
                 speed: float = OPENAI_TTS_SPEED):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.speed = speed

    async def stream_synthesize(self, text: str, language: str, voice_tier: str):
        """Stream 8kHz PCM chunks as they arrive from OpenAI — first audio in ~200ms."""
        import asyncio
        import time
        from openai import AsyncOpenAI

        t0 = time.time()
        client = AsyncOpenAI(api_key=self.api_key)
        buf = b""
        first = True

        async with client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            speed=self.speed,
            response_format="pcm",
        ) as response:
            async for raw_chunk in response.iter_bytes(_STREAM_CHUNK_BYTES):
                buf += raw_chunk
                # Align to 2-byte PCM samples
                aligned = (len(buf) // 2) * 2
                if aligned >= _STREAM_CHUNK_BYTES:
                    pcm = _pcm24k_to_8k(buf[:aligned])
                    buf = buf[aligned:]
                    if pcm:
                        if first:
                            print(f"[OpenAI-TTS] first chunk at {time.time()-t0:.2f}s voice={self.voice}")
                            first = False
                        yield pcm

        # Flush remaining bytes
        if buf:
            aligned = (len(buf) // 2) * 2
            if aligned:
                pcm = _pcm24k_to_8k(buf[:aligned])
                if pcm:
                    yield pcm

        print(f"[OpenAI-TTS] stream done in {time.time()-t0:.2f}s text_len={len(text)}")

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        """Collect full audio — use stream_synthesize for lower latency."""
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
                speed=self.speed,
                response_format="pcm",
            )
            elapsed = time.time() - t0
            print(f"[OpenAI-TTS] synthesis={elapsed:.2f}s text_len={len(text)} voice={self.voice}")
            return response.content

        pcm24k = await loop.run_in_executor(None, _call_api)
        return _pcm24k_to_8k(pcm24k)
