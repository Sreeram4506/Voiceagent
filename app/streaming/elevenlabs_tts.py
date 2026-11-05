"""ElevenLabs TTS — returns 8 kHz 16-bit mono PCM for the telephony stream.

Uses the streaming endpoint so first audio arrives in ~300ms instead of waiting
for the full response (~1.5-2s).
"""

import array
import os
from collections.abc import AsyncIterator

import httpx


class ElevenLabsTTS:
    """Calls ElevenLabs v1 text-to-speech/stream and downsamples to 8 kHz linear16."""

    VOICE_IDS = {
        "hindi_female": "pFZP5JQG7iQjIQuC4Bku",
        "hindi_male": "onwK4e9ZLuTAKqWW03F9",
        "telugu_male": "onwK4e9ZLuTAKqWW03F9",
        "telugu_female": "pFZP5JQG7iQjIQuC4Bku",
        "default": "pFZP5JQG7iQjIQuC4Bku",
    }

    def __init__(self, api_key: str | None = None, voice_id: str | None = None):
        self._api_key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self._voice_id = voice_id or self.VOICE_IDS["default"]
        self._client = httpx.AsyncClient(timeout=30.0)

    def _payload(self, text: str) -> dict:
        return {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

    async def stream_synthesize(
        self, text: str, language: str, voice_tier: str
    ) -> AsyncIterator[bytes]:
        """Yields 8kHz 16-bit mono PCM chunks as they arrive from ElevenLabs streaming."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream?output_format=pcm_16000"
        async with self._client.stream(
            "POST",
            url,
            headers={"xi-api-key": self._api_key, "Content-Type": "application/json"},
            json=self._payload(text),
        ) as resp:
            resp.raise_for_status()
            buf = b""
            async for chunk in resp.aiter_bytes(chunk_size=3200):  # 0.1s of 16kHz
                buf += chunk
                # align to 2-byte boundary before downsampling
                usable = len(buf) - (len(buf) % 2)
                if usable >= 3200:
                    yield _downsample_16k_to_8k(buf[:usable])
                    buf = buf[usable:]
            if len(buf) >= 2:
                yield _downsample_16k_to_8k(buf[: len(buf) - (len(buf) % 2)])

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        """Collect all streaming chunks into one buffer (for protocol compatibility)."""
        chunks = []
        async for chunk in self.stream_synthesize(text, language, voice_tier):
            chunks.append(chunk)
        return b"".join(chunks)


class DeepgramTTS:
    """Deepgram Aura TTS — outputs linear16 at 16 kHz, downsampled to 8 kHz."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self._client = httpx.AsyncClient(timeout=15.0)

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=16000"
        resp = await self._client.post(
            url,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
        resp.raise_for_status()
        return _downsample_16k_to_8k(resp.content)


def _downsample_16k_to_8k(pcm_16k: bytes) -> bytes:
    """Simple 2:1 decimation from 16 kHz to 8 kHz 16-bit mono PCM."""
    samples = array.array("h")
    samples.frombytes(pcm_16k[: len(pcm_16k) - (len(pcm_16k) % 2)])
    out = array.array("h", samples[::2])
    return out.tobytes()
