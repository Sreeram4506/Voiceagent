"""One live call: audio in, transcript, reply, speech out, barge-in, end of call."""

import asyncio
import logging
import re
import time
from collections import deque
from pathlib import Path
from collections.abc import Awaitable, Callable
from contextlib import suppress

log = logging.getLogger(__name__)

from app.providers.base import ToolRequest
from app.streaming.echo_gate import ECHO_TAIL_SECONDS, EchoGate
from app.streaming.emotion import detect_emotion
from app.streaming.protocol import clear_message, media_message
from app.streaming.providers import AsyncLLM, AsyncSTT, AsyncTTS
from app.streaming.vad import EnergyVAD

PREROLL_FRAMES = 15  # 300 ms of audio kept from before speech_start so first words aren't clipped
# A sentence is complete once whitespace follows its end mark, so "18." is not cut off before ".5" arrives.
_SENTENCE = re.compile(r"(.+?[.?!।])\s+", re.S)
# Most replies start within this; only a slower one gets a short "హా..." so the line is not dead.
FILLER_DELAY_SECONDS = 1.5

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
        mode: str = "ai_agent",
        greeting: str = "",
        opening: tuple[str, bytes] | None = None,
    ):
        self.stt, self.llm, self.tts = stt, llm, tts
        self.script, self.language, self.voice_tier = script, language, voice_tier
        self.stream_sid = stream_sid
        self._send = send
        self._on_end = on_end
        self.vad = vad or EnergyVAD()
        self.mode = mode
        self.greeting = greeting
        self.opening = opening  # (text, 8 kHz PCM) written and voiced while the phone rang

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
        self._echo_gate = EchoGate()  # the agent's own voice leaking back must not count as the caller
        self._play_until = 0.0  # monotonic time at which the phone finishes playing what we sent
        self._caller_done_at: float | None = None  # when the caller last stopped talking
        self._filler_cache: list[bytes] = []
        self._filler_idx: int = 0
        self._filler_task: asyncio.Task | None = None

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Opening line: speak the script directly, then wait for caller."""
        asyncio.create_task(self._warm_fillers())
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
        """Play pre-recorded audio or speak greeting depending on mode."""
        self._speaking_opening = True
        try:
            if self.mode == "broadcast":
                audio_path = Path(__file__).parent / "campaign_audio.raw"
                if audio_path.exists():
                    pcm = audio_path.read_bytes()
                    print(f"[AUDIO] playing broadcast file: {len(pcm)} bytes")
                    self.turns.append(("agent", self.script or "(pre-recorded audio)"))
                    chunk_bytes = 8000 * 2
                    chunks = [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
                    for chunk in chunks:
                        await self._send_agent_audio(chunk)
                        await asyncio.sleep(len(chunk) / (8000 * 2))
                self.ended = True
                if self._on_end:
                    await self._safe_end()
            elif self.opening and self.opening[1]:
                text, pcm = self.opening
                print(f"[AI-AGENT] playing opening prepared while ringing: {text}")
                self.turns.append(("agent", text))
                self.history.append(("agent", text))
                await self._play_pcm(pcm)
            elif self.greeting and self.greeting.strip():
                # Speak the predefined opening greeting/question, then listen for caller
                greet_text = self.greeting.strip()
                print(f"[AI-AGENT] speaking predefined greeting/ask: {greet_text}")
                self.turns.append(("agent", greet_text))
                self.history.append(("agent", greet_text))
                await self._speak(greet_text)
            else:
                print(f"[AI-AGENT] starting conversational mode (mode={self.mode})")
                await self._respond(None)
        finally:
            self._speaking_opening = False

    # ---- filler audio (instant response while STT/LLM run) ---------------

    _FILLER_TEXTS = {
        "te": ["అవునా...", "సరే...", "హా..."],
        "hi": ["अच्छा...", "हाँ...", "जी..."],
        "en": ["Hmm...", "Right...", "Okay..."],
        "ta": ["ஆமா...", "சரி..."],
        "kn": ["ಹೌದಾ...", "ಸರಿ..."],
    }

    async def _warm_fillers(self) -> None:
        """Pre-synthesize short filler phrases at session start."""
        for text in self._FILLER_TEXTS.get(self.language[:2], []):
            try:
                pcm = await self.tts.synthesize(text, self.language, self.voice_tier)
                self._filler_cache.append(pcm)
            except Exception as e:
                print(f"[Filler] warm failed for '{text}': {e}")
        print(f"[Filler] {len(self._filler_cache)} phrases cached")

    async def _play_filler(self) -> None:
        """If the reply is slow to start, play one short acknowledgement; cancelled when real speech begins."""
        if not self._filler_cache:
            return
        try:
            await asyncio.sleep(FILLER_DELAY_SECONDS)
            pcm = self._filler_cache[self._filler_idx % len(self._filler_cache)]
            self._filler_idx += 1
            await self._send_agent_audio(pcm)
            await asyncio.sleep(len(pcm) / (8000 * 2))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Filler] play error: {e}")

    # ---- inbound audio ---------------------------------------------------

    async def on_audio(self, frame: bytes) -> None:
        if self._speaking_opening or self.ended:
            return
        self.frames_received += 1
        import time as _time
        if _time.monotonic() < self._echo_until:
            self._preroll.append(frame)
            return
        frame = self._without_echo(frame)
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
                self._caller_done_at = time.monotonic()
                if self._filler_cache and (self._filler_task is None or self._filler_task.done()):
                    self._filler_task = asyncio.create_task(self._play_filler())
                text = (await self.stt.end_utterance()).strip()
                print(f"[STT] transcript: {repr(text)}")
                self._preroll.clear()
                if text:
                    self.turns.append(("caller", text))
                    self._response = asyncio.create_task(self._respond(text))
        except Exception as e:
            print(f"[on_audio ERROR] {type(e).__name__}: {e}")

    # ---- outbound reply --------------------------------------------------

    def _without_echo(self, frame: bytes) -> bytes:
        """The caller's audio, with the agent's own voice leaking back down the line turned into silence."""
        now = time.monotonic()
        return self._echo_gate.filter(frame, agent_playing=now < self._play_until + ECHO_TAIL_SECONDS, now=now)

    async def _send_agent_audio(self, pcm: bytes) -> None:
        """Send agent audio to the phone, keeping track of what is playing so its echo can be told apart."""
        now = time.monotonic()
        self._echo_gate.agent_audio(pcm)
        self._play_until = max(now, self._play_until) + len(pcm) / (8000 * 2)
        await self._safe_send(media_message(self.stream_sid, pcm))

    async def _barge_in(self) -> None:
        """Caller spoke over the agent: stop the reply and clear the provider's audio buffer."""
        self._play_until = time.monotonic()
        await self.cancel()
        self._filler_task = None  # reset so next speech_end can restart filler loop
        await self._safe_send(clear_message(self.stream_sid))

    async def _respond(self, caller_text: str | None) -> None:
        import time as _t
        t0 = _t.time()
        if caller_text is not None:
            self.history.append(("caller", caller_text))

        # Detect emotion and weave a hint into the script for this turn
        emotion_hint = ""
        if caller_text:
            emo = detect_emotion(caller_text)
            if emo.prompt_hint:
                emotion_hint = emo.prompt_hint
                print(f"[EMOTION] {emo.primary} (conf={emo.confidence:.1f})")

        script_for_turn = self.script
        if emotion_hint:
            script_for_turn = f"{self.script}\n\n{emotion_hint}"

        end_requested = False
        is_opening = caller_text is None
        max_sentences = 1 if is_opening else 2
        spoken: list[str] = []
        lines: asyncio.Queue = asyncio.Queue()
        speaker = asyncio.create_task(self._speak_lines(lines))
        buffer = ""

        def queue(sentence: str) -> None:
            if not spoken:
                print(f"[RESPOND] first sentence ready at {_t.time()-t0:.2f}s")
                lines.put_nowait(sentence)  # voiced as a stream straight away
            else:
                # voiced in parallel, so it plays right after the first sentence with no dead air
                lines.put_nowait(asyncio.create_task(self.tts.synthesize(sentence, self.language, self.voice_tier)))
            spoken.append(sentence)

        try:
            async for item in self.llm.stream_reply(script_for_turn, self.history, caller_text or ""):
                if isinstance(item, ToolRequest):
                    self.actions.append(item.name)
                    if item.name == "end_call" and not is_opening:
                        end_requested = True
                    continue
                if len(spoken) >= max_sentences:
                    continue  # keep reading: a booking or hang-up can follow the words
                buffer += item
                done, buffer = pop_sentences(buffer)
                for sentence in done[: max_sentences - len(spoken)]:
                    queue(sentence)
            if buffer.strip() and len(spoken) < max_sentences:
                queue(buffer.strip())
            lines.put_nowait(None)
            await speaker
        except BaseException:
            speaker.cancel()
            while not lines.empty():
                pending = lines.get_nowait()
                if isinstance(pending, asyncio.Task):
                    pending.cancel()
            raise

        reply = " ".join(spoken)
        if reply:
            self.history.append(("agent", reply))
            self.turns.append(("agent", reply))

        if end_requested:
            self.ended = True
            if self._on_end:
                await self._safe_end()

    async def _speak_lines(self, lines: asyncio.Queue) -> None:
        """The first sentence streams as it is voiced; later ones were voiced in parallel and follow at once."""
        while (item := await lines.get()) is not None:
            if isinstance(item, str):
                await self._speak(item)
                continue
            try:
                pcm = await item
            except Exception as e:
                print(f"[TTS] FAILED: {e}")
                continue
            await self._play_pcm(pcm)

    def _log_reply_gap(self) -> None:
        if self._caller_done_at is not None:
            print(f"[LATENCY] caller stopped -> reply audio: {time.monotonic() - self._caller_done_at:.2f}s")
            self._caller_done_at = None

    async def _speak(self, text: str) -> None:
        """Voice text as one stream and play it as it arrives."""
        import time
        t0 = time.time()
        print(f"[TTS] synthesize start: {len(text)} chars")
        frame_bytes = 8000 * 2 // 5  # 3200 bytes = 200ms at 8kHz 16-bit
        frame_dur = frame_bytes / (8000 * 2)
        first_sent = False
        total_bytes = 0

        try:
            if hasattr(self.tts, "stream_synthesize"):
                buf = b""
                async for pcm_chunk in self.tts.stream_synthesize(text, self.language, self.voice_tier):
                    buf += pcm_chunk
                    while len(buf) >= frame_bytes:
                        frame = buf[:frame_bytes]
                        buf = buf[frame_bytes:]
                        if not first_sent:
                            if self._filler_task and not self._filler_task.done():
                                self._filler_task.cancel()
                                with suppress(asyncio.CancelledError, Exception):
                                    await self._filler_task
                            self._filler_task = None
                        await self._send_agent_audio(frame)
                        if not first_sent:
                            print(f"[TTS] first chunk sent at {time.time()-t0:.2f}s")
                            self._log_reply_gap()
                            first_sent = True
                        total_bytes += len(frame)
                        await asyncio.sleep(frame_dur)
                if buf:
                    await self._send_agent_audio(buf)
                    total_bytes += len(buf)
                    await asyncio.sleep(len(buf) / (8000 * 2))
            else:
                audio = await self.tts.synthesize(text, self.language, self.voice_tier)
                chunks = [audio[i:i + frame_bytes] for i in range(0, len(audio), frame_bytes)]
                for chunk in chunks:
                    await self._send_agent_audio(chunk)
                    if not first_sent:
                        print(f"[TTS] first chunk sent at {time.time()-t0:.2f}s")
                        self._log_reply_gap()
                        first_sent = True
                    await asyncio.sleep(len(chunk) / (8000 * 2))
                total_bytes += len(audio)
        except Exception as e:
            print(f"[TTS] FAILED: {e}")
            return

        print(f"[TTS] done in {time.time()-t0:.2f}s: {total_bytes} bytes")
        import time as _time
        self._echo_until = _time.monotonic() + 0.2

    async def _play_pcm(self, pcm: bytes) -> None:
        """Send ready-made audio at the pace the phone plays it."""
        frame_bytes = 8000 * 2 // 5
        for i in range(0, len(pcm), frame_bytes):
            chunk = pcm[i:i + frame_bytes]
            await self._send_agent_audio(chunk)
            await asyncio.sleep(len(chunk) / (8000 * 2))
        import time as _time
        self._echo_until = _time.monotonic() + 0.2

    async def _safe_send(self, message: str) -> None:
        with suppress(RuntimeError, OSError):
            await self._send(message)

    async def _safe_end(self) -> None:
        with suppress(RuntimeError, OSError):
            await self._on_end()
