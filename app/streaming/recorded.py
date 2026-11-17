"""Recorded-audio session: plays a pre-recorded audio file over the call and hangs up."""

import asyncio
import base64
import io
import subprocess
import tempfile
from contextlib import suppress

from app.streaming.protocol import media_message


def _to_pcm_8k(audio_bytes: bytes, content_type: str) -> bytes:
    """Convert any supported audio format to 8 kHz 16-bit LE mono PCM via ffmpeg."""
    ext_map = {
        "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav",
        "audio/ogg": ".ogg", "audio/webm": ".webm", "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a", "audio/aac": ".aac",
    }
    ext = ext_map.get(content_type, ".mp3")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp.name,
             "-ar", "8000", "-ac", "1", "-f", "s16le", "-acodec", "pcm_s16le", "-"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:200].decode(errors='replace')}")
        return result.stdout


class RecordedAudioSession:
    """Plays a stored audio greeting over the telephony stream, then ends the call."""

    def __init__(
        self,
        audio_b64: str,
        content_type: str,
        filename: str,
        stream_sid: str,
        send,
        on_end,
    ):
        self._audio_b64 = audio_b64
        self._content_type = content_type
        self._filename = filename
        self.stream_sid = stream_sid
        self._send = send
        self._on_end = on_end

        self.turns: list[tuple[str, str]] = [("agent", f"(recorded audio: {filename})")]
        self.actions: list[str] = []
        self.ended = False
        self.frames_received = 0
        self.persisted = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._play())

    async def _play(self) -> None:
        try:
            raw_audio = base64.b64decode(self._audio_b64)
            pcm = await asyncio.to_thread(_to_pcm_8k, raw_audio, self._content_type)
            print(f"[RECORDED] playing {self._filename}: {len(pcm)} bytes PCM")
            chunk_bytes = 8000 * 2  # 1 second of 8 kHz 16-bit mono
            for i in range(0, len(pcm), chunk_bytes):
                chunk = pcm[i:i + chunk_bytes]
                await self._safe_send(media_message(self.stream_sid, chunk))
                await asyncio.sleep(len(chunk) / (8000 * 2))
        except Exception as e:
            print(f"[RECORDED] playback error: {e}")
        self.ended = True
        if self._on_end:
            with suppress(RuntimeError, OSError):
                await self._on_end()

    async def on_audio(self, frame: bytes) -> None:
        self.frames_received += 1

    async def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._task

    def outcome(self) -> str:
        return "recorded_audio_played" if self.ended else "no_resolution"

    async def wait_ready(self, timeout: float = 2.0) -> bool:
        return True

    async def attach(self, stream_sid: str, send, end) -> None:
        self.stream_sid = stream_sid
        self._send = send
        self._on_end = end
        await self.start()

    async def _safe_send(self, message: str) -> None:
        with suppress(RuntimeError, OSError):
            await self._send(message)
