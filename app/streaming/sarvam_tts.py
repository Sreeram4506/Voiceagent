"""Sarvam Bulbul text-to-speech: Telugu voices made for Indian languages, streamed as 8 kHz phone audio.

Used when an agent's voice_engine setting is "sarvam" and SARVAM_API_KEY holds a Sarvam Model API key
(from the Sarvam dashboard's key management page, sent as the api-subscription-key header). A Samvaad
(Voice Agents) key does not work here.
"""

import os
import struct
import time
from collections.abc import AsyncIterator

import httpx

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech/stream"
SARVAM_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v4-flash")
SAMPLE_RATE = 8000
MIN_PACE, MAX_PACE = 0.5, 2.0
REQUEST_TIMEOUT_SECONDS = 20.0

# Sarvam's conversational Telugu voices on bulbul:v4-flash. SARVAM_SPEAKER overrides the choice.
SPEAKERS = {"te": {"female": "kavitha_te_conversation", "male": "tarun_te_narration"}}
LANGUAGE_CODES = {"te": "te-IN"}
SARVAM_LANGUAGES = frozenset(SPEAKERS)


class SarvamTTSError(RuntimeError):
    """Sarvam refused or failed a request; the message says why."""


def _strip_wav_header(audio: bytes) -> bytes:
    """Drop a RIFF/WAV header if the stream starts with one, keeping only the samples."""
    if not audio.startswith(b"RIFF"):
        return audio
    offset = 12
    while offset + 8 <= len(audio):
        chunk_id, size = audio[offset:offset + 4], struct.unpack("<I", audio[offset + 4:offset + 8])[0]
        if chunk_id == b"data":
            return audio[offset + 8:]
        offset += 8 + size
    return b""


class SarvamTTS:
    """Async TTS returning 8 kHz 16-bit mono PCM, streamed as Sarvam produces it."""

    def __init__(self, api_key: str, gender: str = "female", pace: float = 1.0, speaker: str | None = None,
                 model: str = SARVAM_TTS_MODEL, transport: httpx.AsyncBaseTransport | None = None):
        self.api_key = api_key
        self.gender = gender if gender in ("female", "male") else "female"
        self.pace = round(min(MAX_PACE, max(MIN_PACE, float(pace))), 2)
        self.speaker = speaker or os.environ.get("SARVAM_SPEAKER") or ""
        self.model = model
        self._transport = transport

    def _body(self, text: str, language: str) -> dict:
        speaker = self.speaker or SPEAKERS.get(language, SPEAKERS["te"])[self.gender]
        return {
            "text": text,
            "language_code": LANGUAGE_CODES.get(language, "te-IN"),
            "model": self.model,
            "speaker": speaker,
            "pace": self.pace,
            "speech_sample_rate": SAMPLE_RATE,
            "output_audio_codec": "linear16",
        }

    async def stream_synthesize(self, text: str, language: str, voice_tier: str) -> AsyncIterator[bytes]:
        """Yield PCM chunks (whole 16-bit samples) as they arrive from Sarvam."""
        t0 = time.monotonic()
        headers = {"api-subscription-key": self.api_key}
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, transport=self._transport) as client:
            async with client.stream("POST", SARVAM_TTS_URL, headers=headers, json=self._body(text, language)) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode(errors="replace")[:300]
                    raise SarvamTTSError(f"Sarvam TTS returned {response.status_code}: {detail}")
                head, carry, first = b"", b"", True
                async for raw in response.aiter_bytes():
                    if first:
                        head += raw
                        if len(head) < 44 and head.startswith(b"RIFF"):
                            continue  # wait for the whole header before deciding where the samples start
                        raw, first = _strip_wav_header(head), False
                        print(f"[Sarvam-TTS] first audio at {time.monotonic() - t0:.2f}s speaker={self._body('', language)['speaker']}")
                    data = carry + raw
                    whole = len(data) - len(data) % 2
                    carry = data[whole:]
                    if whole:
                        yield data[:whole]
        print(f"[Sarvam-TTS] done in {time.monotonic() - t0:.2f}s text_len={len(text)}")

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        return b"".join([chunk async for chunk in self.stream_synthesize(text, language, voice_tier)])
