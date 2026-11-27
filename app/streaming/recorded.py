"""Recorded-audio session: plays a pre-recorded audio file, takes the caller's yes/no answer, thanks them, then
hangs up."""

import asyncio
import base64
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

from app.streaming.call_recorder import recorder_from_env
from app.streaming.protocol import clear_message, media_message
from app.streaming.thanks import thanks_audio
from app.streaming.yes_no import (
    AnswerListener, answer_from_digit, classify_reply, transcribe_reply, wav_bytes,
)

# After the message, how long to wait for the caller to start answering before hanging up.
ANSWER_WAIT_SECONDS = 6.0


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
    """Plays a stored audio message over the telephony stream, then listens for a yes/no answer and ends the call.

    The caller can press 1 (yes) or 2 (no) at any point, or say their answer once the message has finished.
    """

    def __init__(
        self,
        *,
        pcm: bytes | None = None,
        audio_b64: str = "",
        content_type: str = "audio/mpeg",
        filename: str = "greeting.mp3",
        stream_sid: str,
        send,
        on_end,
        transcribe: Callable[[bytes], Awaitable[str]] = transcribe_reply,
        answer_wait: float = ANSWER_WAIT_SECONDS,
        language: str = "te",
        thanks: Callable[[str], bytes] | None = None,
        record: bool = True,
    ):
        self._pcm = pcm
        self._audio_b64 = audio_b64
        self._content_type = content_type
        self._filename = filename
        self.stream_sid = stream_sid
        self._send = send
        self._on_end = on_end
        self._transcribe = transcribe
        self._answer_wait = answer_wait
        # The thank-you for an answer: "yes", "no" or "unclear" -> phone-ready PCM.
        self._thanks = thanks or (lambda answer: thanks_audio(language, answer))

        self.turns: list[tuple[str, str]] = [("agent", f"(recorded audio: {filename})")]
        self.actions: list[str] = []
        self.ended = False
        self.frames_received = 0
        self.persisted = False
        self._task: asyncio.Task | None = None
        self._listener: AnswerListener | None = None  # set once the message has played
        self._reply_heard = asyncio.Event()
        self._key_pressed = asyncio.Event()
        self._answer = ""  # "yes" / "no" once known
        self._result = ""  # said_yes, said_no, unclear_reply, no_reply
        self._record = record
        self.recorder = recorder_from_env(self._record)  # the message and answer, when on and kept for this call

    async def start(self) -> None:
        if self.recorder:
            self.recorder.start(time.monotonic())
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            await self._play()
            if not self._answer:
                await self._listen()
        except Exception as e:
            print(f"[RECORDED] error: {e}")
        self._result = self._result or (f"said_{self._answer}" if self._answer else "no_reply")
        print(f"[RECORDED] answer: {self._result}")
        await self._thank()
        self.ended = True
        if self._on_end:
            try:
                await self._on_end()
            except Exception as e:
                print(f"[RECORDED] on_end error: {e}")

    async def _play(self) -> None:
        if self._pcm is not None:
            pcm = self._pcm
        else:
            raw_audio = base64.b64decode(self._audio_b64)
            pcm = await asyncio.to_thread(_to_pcm_8k, raw_audio, self._content_type)
        print(f"[RECORDED] playing {self._filename}: {len(pcm)} bytes PCM")
        chunk_bytes = 8000 * 2  # 1 second of 8 kHz 16-bit mono
        for i in range(0, len(pcm), chunk_bytes):
            chunk = pcm[i:i + chunk_bytes]
            if self.recorder:
                self.recorder.agent(chunk, start=time.monotonic())
            await self._safe_send(media_message(self.stream_sid, chunk))
            if await self._pressed_within(len(chunk) / (8000 * 2)):
                await self._safe_send(clear_message(self.stream_sid))  # they answered: stop the message
                if self.recorder:
                    self.recorder.cut_agent(time.monotonic())
                return
        await self._pressed_within(0.5)

    async def _pressed_within(self, seconds: float) -> bool:
        """Wait up to `seconds`; True as soon as the caller presses an answer key."""
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._key_pressed.wait(), seconds)
        return self._key_pressed.is_set()

    async def _thank(self) -> None:
        """Thank them once they have answered. Silence gets no thank-you: nobody may be listening."""
        answer = self._answer or ("unclear" if self._result == "unclear_reply" else "")
        if not answer:
            return
        try:
            await self._say(self._thanks(answer))
        except Exception as e:
            print(f"[RECORDED] could not play the thank-you: {e}")

    async def _say(self, pcm: bytes) -> None:
        """Play audio to the end, then a short pause so the last word is not cut off by the hang-up."""
        if not pcm:
            return
        chunk_bytes = 8000 * 2
        for i in range(0, len(pcm), chunk_bytes):
            chunk = pcm[i:i + chunk_bytes]
            if self.recorder:
                self.recorder.agent(chunk, start=time.monotonic())
            await self._safe_send(media_message(self.stream_sid, chunk))
            await asyncio.sleep(len(chunk) / (8000 * 2))
        await asyncio.sleep(0.3)

    async def _listen(self) -> None:
        """After the message: wait for a keypress or a spoken reply."""
        self._listener = AnswerListener()
        deadline = time.monotonic() + self._answer_wait
        while not self._answer and not self._reply_heard.is_set():
            if self._listener.spoke:
                deadline = max(deadline, time.monotonic() + 0.1)  # they are mid-reply: let them finish
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.05)
        if self._answer or not self._listener.spoke:
            return
        try:
            text = await self._transcribe(wav_bytes(self._listener.audio()))
        except Exception as e:
            print(f"[RECORDED] could not transcribe the reply: {e}")
            text = ""
        self.turns.append(("caller", text or "(reply could not be transcribed)"))
        answer = classify_reply(text)
        if answer:
            self._answer = answer
        else:
            self._result = "unclear_reply"

    async def on_audio(self, frame: bytes) -> None:
        self.frames_received += 1
        if self.recorder:
            self.recorder.caller(frame)
        if self._listener is not None and not self._reply_heard.is_set() and self._listener.feed(frame):
            self._reply_heard.set()

    async def on_dtmf(self, digit: str) -> None:
        answer = answer_from_digit(digit)
        if answer and not self._answer:
            self._answer = answer
            self.turns.append(("caller", f"(pressed {digit}: {answer})"))
            self._key_pressed.set()

    async def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._task

    def outcome(self) -> str:
        if self._answer:
            return f"said_{self._answer}"
        return self._result if self.ended and self._result else "no_resolution"

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
