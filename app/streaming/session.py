"""One live call: audio in, transcript, reply, speech out, barge-in, end of call."""

import asyncio
import logging
import re
from collections import deque
from pathlib import Path
from collections.abc import Awaitable, Callable
from contextlib import suppress

log = logging.getLogger(__name__)

from app.providers.base import ToolRequest
from app.streaming.audio import FRAME_MS, split_frames
from app.streaming.protocol import clear_message, media_message
from app.streaming.providers import AsyncLLM, AsyncSTT, AsyncTTS
from app.streaming.vad import EnergyVAD

PREROLL_FRAMES = 15  # 300 ms of audio kept from before speech_start so first words aren't clipped
_SENTENCE = re.compile(r"(.+?[.?!।])(?:\s+|$)", re.S)

Send = Callable[[str], Awaitable[None]]
OnEnd = Callable[[], Awaitable[None]]


def pop_sentences(buffer: str) -> tuple[list[str], str]:
    """Split complete sentences off the front of a streaming text buffer."""
    sentences: list[str] = []
    pos = 0
    for match in _SENTENCE.finditer(buffer):
        sentences.append(match.group(1).strip())
        pos = match.end()
    return sentences, buffer[pos:]


class StreamSession:
    def __init__(
        self,
        stt: AsyncSTT,
        llm: AsyncLLM,
        tts: AsyncTTS,
        script: str,
        language: str,
        voice_tier: str,
        stream_sid: str,
        send: Send,
        on_end: OnEnd | None = None,
        vad: EnergyVAD | None = None,
        mode: str = "campaign",
    ):
        self.stt, self.llm, self.tts = stt, llm, tts
        self.script, self.language, self.voice_tier = script, language, voice_tier
        self.stream_sid = stream_sid
        self._send = send
        self._on_end = on_end
        self.vad = vad or EnergyVAD()
        self.mode = mode

        self.history: list[tuple[str, str]] = []
        self.turns: list[tuple[str, str]] = []   # (role, text) in call order, for persistence
        self.actions: list[str] = []
        self.ended = False
        self.frames_received = 0
        self.persisted = False

        self._preroll: deque[bytes] = deque(maxlen=PREROLL_FRAMES)
        self._in_utterance = False
        self._response: asyncio.Task | None = None
        self._speaking_opening = False
        self._echo_until: float = 0.0  # suppress VAD until this monotonic time
        self._filler_cache: list[bytes] = []
        self._filler_idx: int = 0
        self._filler_task: asyncio.Task | None = None

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Opening line: speak the script directly, then wait for caller."""
        self._response = asyncio.create_task(self._speak_opening())

    async def cancel(self) -> None:
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._filler_task
        if self._response and not self._response.done():
            self._response.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._response

    def outcome(self) -> str:
        if "book_appointment" in self.actions:
            return "appointment_booked"
        if self.ended:
            return "completed"
        return "no_resolution"

    async def _speak_opening(self) -> None:
        """Play pre-recorded audio or TTS greeting depending on mode."""
        self._speaking_opening = True
        try:
            if self.mode == "ai_agent":
                print(f"[AI-AGENT] starting conversational mode")
                await self._respond(None)
            else:
                audio_path = Path(__file__).parent / "campaign_audio.raw"
                if audio_path.exists():
                    pcm = audio_path.read_bytes()
                    print(f"[AUDIO] playing pre-recorded file: {len(pcm)} bytes")
                    self.turns.append(("agent", self.script or "(pre-recorded audio)"))
                    chunk_bytes = 8000 * 2
                    chunks = [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
                    for chunk in chunks:
                        await self._safe_send(media_message(self.stream_sid, chunk))
                        await asyncio.sleep(len(chunk) / (8000 * 2))
                    print(f"[AUDIO] all {len(chunks)} chunks sent")
                elif self.script:
                    self.turns.append(("agent", self.script))
                    self.history.append(("agent", self.script))
                    await self._speak(self.script)
                self.ended = True
                if self._on_end:
                    await self._safe_end()
        finally:
            self._speaking_opening = False

    # ---- filler audio (instant response while STT/LLM run) ---------------

    _FILLER_TEXTS = ["అవున...", "సరే...", "హా..."]

    async def _warm_fillers(self) -> None:
        """Pre-synthesize short filler phrases at session start."""
        for text in self._FILLER_TEXTS:
            try:
                pcm = await self.tts.synthesize(text, self.language, self.voice_tier)
                self._filler_cache.append(pcm)
            except Exception as e:
                print(f"[Filler] warm failed for '{text}': {e}")
        print(f"[Filler] {len(self._filler_cache)} phrases cached")

    async def _play_filler(self) -> None:
        """Play a short acknowledgment sound immediately so caller hears a response."""
        if not self._filler_cache or self.ended:
            return
        import time as _time
        pcm = self._filler_cache[self._filler_idx % len(self._filler_cache)]
        self._filler_idx += 1
        frame_bytes = 8000 * 2 // 5  # 200ms frames
        for i in range(0, len(pcm), frame_bytes):
            if self.ended:
                return
            chunk = pcm[i:i + frame_bytes]
            await self._safe_send(media_message(self.stream_sid, chunk))
            await asyncio.sleep(len(chunk) / (8000 * 2))
        self._echo_until = _time.monotonic() + 0.5

    # ---- inbound audio ---------------------------------------------------

    async def on_audio(self, frame: bytes) -> None:
        if self._speaking_opening or self.ended:
            return
        self.frames_received += 1
        import time as _time
        if _time.monotonic() < self._echo_until:
            self._preroll.append(frame)
            return
        try:
            event = self.vad.feed(frame)

            if event == "speech_start":
                print(f"[VAD] speech_start (frames={self.frames_received})")
                if self._response and not self._response.done():
                    await self._barge_in()
                await self.stt.start_utterance(self.language)
                for buffered in self._preroll:
                    await self.stt.push(buffered)
                self._in_utterance = True

            if self._in_utterance:
                await self.stt.push(frame)
            else:
                self._preroll.append(frame)

            if event == "speech_end" and self._in_utterance:
                self._in_utterance = False
                print(f"[VAD] speech_end, transcribing...")
                text = (await self.stt.end_utterance()).strip()
                print(f"[STT] transcript: {repr(text)}")
                self._preroll.clear()
                if text:
                    self.turns.append(("caller", text))
                    self._response = asyncio.create_task(self._respond(text))
        except Exception as e:
            print(f"[on_audio ERROR] {type(e).__name__}: {e}")

    # ---- outbound reply --------------------------------------------------

    async def _barge_in(self) -> None:
        """Caller spoke over the agent: stop the reply and clear the provider's audio buffer."""
        await self.cancel()
        self._filler_task = None
        await self._safe_send(clear_message(self.stream_sid))

    async def _respond(self, caller_text: str | None) -> None:
        if caller_text is not None:
            self.history.append(("caller", caller_text))

        end_requested = False
        is_opening = caller_text is None  # never end call on the opening greeting
        spoken: list[str] = []
        buffer = ""

        async for item in self.llm.stream_reply(self.script, self.history, caller_text or ""):
            if isinstance(item, ToolRequest):
                self.actions.append(item.name)
                if item.name == "end_call" and not is_opening:
                    end_requested = True
                continue
            buffer += item
            sentences, buffer = pop_sentences(buffer)
            for sentence in sentences:
                spoken.append(sentence)
                await self._speak(sentence)

        if buffer.strip():
            spoken.append(buffer.strip())
            await self._speak(buffer.strip())

        reply = " ".join(spoken)
        if reply:
            self.history.append(("agent", reply))
            self.turns.append(("agent", reply))

        if end_requested:
            self.ended = True
            if self._on_end:
                await self._safe_end()

    async def _speak(self, text: str) -> None:
        import time
        t0 = time.time()
        print(f"[TTS] synthesize start: {len(text)} chars")
        # 200ms frames: smaller chunks mean at most 200ms of unplayed audio
        # when _speak returns — critical so end_call doesn't cut off audio
        frame_bytes = 8000 * 2 // 5  # 3200 bytes = 200ms at 8kHz 16-bit
        frame_dur = frame_bytes / (8000 * 2)
        first_sent = False
        total_bytes = 0

        try:
            # Use streaming TTS if available — first audio arrives sooner
            if hasattr(self.tts, "stream_synthesize"):
                buf = b""
                async for pcm_chunk in self.tts.stream_synthesize(text, self.language, self.voice_tier):
                    buf += pcm_chunk
                    while len(buf) >= frame_bytes:
                        frame = buf[:frame_bytes]
                        buf = buf[frame_bytes:]
                        await self._safe_send(media_message(self.stream_sid, frame))
                        if not first_sent:
                            print(f"[TTS] first chunk sent at {time.time()-t0:.2f}s")
                            first_sent = True
                        total_bytes += len(frame)
                        await asyncio.sleep(frame_dur)
                if buf:
                    await self._safe_send(media_message(self.stream_sid, buf))
                    total_bytes += len(buf)
                    # sleep so Plivo finishes playing this last partial frame
                    await asyncio.sleep(len(buf) / (8000 * 2))
            else:
                audio = await self.tts.synthesize(text, self.language, self.voice_tier)
                chunks = [audio[i:i + frame_bytes] for i in range(0, len(audio), frame_bytes)]
                for chunk in chunks:
                    await self._safe_send(media_message(self.stream_sid, chunk))
                    if not first_sent:
                        print(f"[TTS] first chunk sent at {time.time()-t0:.2f}s")
                        first_sent = True
                    # sleep after EVERY chunk including last so Plivo finishes
                    # playing before we return (prevents end_call cutting off audio)
                    await asyncio.sleep(len(chunk) / (8000 * 2))
                total_bytes = len(audio)
        except Exception as e:
            print(f"[TTS] FAILED: {e}")
            return

        print(f"[TTS] done in {time.time()-t0:.2f}s: {total_bytes} bytes")
        # suppress echo/sidetone for 500ms after each agent utterance
        import time as _time
        self._echo_until = _time.monotonic() + 0.5

    async def _safe_send(self, message: str) -> None:
        with suppress(RuntimeError, OSError):
            await self._send(message)

    async def _safe_end(self) -> None:
        with suppress(RuntimeError, OSError):
            await self._on_end()
