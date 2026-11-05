"""OpenAI Whisper STT — buffers audio during the utterance, transcribes on end."""

import io
import os
import struct

import openai


class OpenAISTT:
    """Collects 8 kHz 16-bit mono PCM frames and sends them to Whisper on end_utterance."""

    def __init__(self, api_key: str | None = None):
        self._client = openai.AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._buffer = bytearray()
        self._language = "hi"

    async def start_utterance(self, language: str) -> None:
        self._buffer = bytearray()
        self._language = language

    async def push(self, pcm_frame: bytes) -> None:
        self._buffer.extend(pcm_frame)

    # Languages officially supported by the Whisper API (ISO-639-1 codes).
    # Telugu ("te") is NOT in this list — the API rejects it even though the
    # model handles Telugu audio fine when language is omitted (auto-detect).
    WHISPER_LANGUAGES = {
        "af","ar","hy","az","be","bs","bg","ca","zh","hr","cs","da","nl","en",
        "et","fi","fr","gl","de","el","he","hi","hu","is","id","it","ja","kn",
        "kk","ko","lv","lt","mk","ms","mr","mi","ne","no","fa","pl","pt","ro",
        "ru","sr","sk","sl","es","sw","sv","tl","ta","th","tr","uk","ur","vi","cy",
    }

    async def end_utterance(self) -> str:
        buf_ms = len(self._buffer) * 1000 // (8000 * 2)
        print(f"[STT] end_utterance: buffer={len(self._buffer)}bytes ({buf_ms}ms)")
        if len(self._buffer) < 640:
            print(f"[STT] buffer too short, skipping")
            return ""
        wav = _pcm_to_wav(bytes(self._buffer), sample_rate=8000)
        lang = self._language[:2]
        kwargs = {"model": "whisper-1", "file": ("audio.wav", wav, "audio/wav")}
        if lang in self.WHISPER_LANGUAGES:
            kwargs["language"] = lang
        transcript = await self._client.audio.transcriptions.create(**kwargs)
        return transcript.text.strip()


def _pcm_to_wav(pcm: bytes, sample_rate: int = 8000, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM in a WAV header."""
    data_size = len(pcm)
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()
