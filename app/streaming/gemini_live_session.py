"""One live call bridged to Google's Gemini Live API instead of OpenAI Realtime.

Behaves like RealtimeSession (same prompt, greeting made while the phone rings, silence check-ins,
booking, goodbye before hang-up, cost log); only the wire protocol and audio rates differ.
Phone audio goes in at 8 kHz as-is; Gemini speaks at 24 kHz, which is brought down to 8 kHz.
"""

import asyncio
import base64
import json
import os
import time
from contextlib import suppress

import numpy as np
import websockets
from scipy.signal import firwin, lfilter

from app.streaming.echo_gate import frame_levels
from app.streaming.protocol import clear_message
from app.streaming.realtime_cost import GeminiUsage
from app.streaming.realtime_session import _HEADERS_KW, END_OF_TURN_SILENCE_MS, GOODBYE_ASK, TOOLS, RealtimeSession

GEMINI_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
GEMINI_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
# Aoede: picked by the owner on 2026-11-26 from all 30 Gemini voices. Gacrux, the earlier pick, measures
# about 127 Hz and sounded like a man on the phone; Aoede is about 178 Hz.
GEMINI_VOICES = {"female": "Aoede", "male": "Charon"}
LANGUAGE_CODES = {"te": "te-IN", "hi": "hi-IN", "en": "en-IN", "ta": "ta-IN", "kn": "kn-IN"}
FUNCTIONS = [{"name": tool["name"], "description": tool["description"]} for tool in TOOLS]
CALLER_VOICE_LEVEL = 500.0  # frame RMS above line hiss (~50-200): the caller is talking
HANG_UP_WAIT_SECONDS = 5.0  # how long to wait for the model to finish its goodbye after end_call


class Downsampler:
    """24 kHz -> 8 kHz PCM with a filter that carries over between chunks, so chunk edges don't click."""

    def __init__(self) -> None:
        self._taps = firwin(63, 3400, fs=24000)
        self._state = np.zeros(len(self._taps) - 1)
        self._offset = 0  # samples seen so far, to keep every third sample across chunk boundaries

    def __call__(self, pcm: bytes) -> bytes:
        samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float64)
        filtered, self._state = lfilter(self._taps, 1.0, samples, zi=self._state)
        kept = filtered[(-self._offset) % 3::3]
        self._offset += len(samples)
        return np.clip(kept, -32768, 32767).astype("<i2").tobytes()


class GeminiLiveSession(RealtimeSession):
    def __init__(self, *args, voice_gender: str = "female", **kwargs):
        super().__init__(*args, voice_gender=voice_gender, **kwargs)
        self._voice = GEMINI_VOICES.get(voice_gender, GEMINI_VOICES["female"])
        self.usage = GeminiUsage()
        self._downsample = Downsampler()
        self._log: list[list[str]] = []  # [role, text] in call order
        self._turn = 0
        self._turn_done = asyncio.Event()

    @property
    def turns(self) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        for role, text in self._log:
            text = text.strip()
            if not text:
                continue
            if turns and turns[-1][0] == role:
                turns[-1] = (role, f"{turns[-1][1]} {text}")
            else:
                turns.append((role, text))
        return turns

    async def on_audio(self, frame: bytes) -> None:
        self.frames_received += 1
        if self.ended or self._ws is None or time.monotonic() < self._guard_until:
            return
        blocked_before = self._echo_gate.blocked_frames
        heard = self._without_echo(frame)
        if self._echo_gate.blocked_frames > blocked_before:
            # Only the agent's own voice coming back. Gemini bills every second of audio it is sent and
            # re-bills it as context on each later reply, so silence here is not worth sending.
            return
        if heard and frame_levels(heard).max() >= CALLER_VOICE_LEVEL:
            # Gemini sends the caller's words only after they stop, so without this the "are you still
            # there?" check fires while they are mid-sentence.
            self._quiet_since = time.monotonic()
        audio = {"data": base64.b64encode(heard).decode(), "mimeType": "audio/pcm;rate=8000"}
        await self._send_json({"realtimeInput": {"audio": audio}})

    # ---- Gemini socket ---------------------------------------------------

    async def _run(self) -> None:
        try:
            headers = {"x-goog-api-key": self._api_key}
            async with websockets.connect(GEMINI_URL, max_size=None, **{_HEADERS_KW: headers}) as ws:
                self._ws = ws
                print(f"[Gemini] connected at {time.monotonic() - self._t0:.2f}s model={self._model}")
                await self._send_json(self._setup_message())
                async for raw in ws:
                    await self._on_message(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Gemini] socket error: {type(e).__name__}: {e}")
        finally:
            self._ws = None
            if not self._ready.is_set():
                self.failed = True
                self._ready.set()
            elif self._on_end and not self.ended and self._end_task is None:
                self._end_task = asyncio.create_task(self._finish())

    def _setup_message(self) -> dict:
        code = LANGUAGE_CODES.get(self._language)
        speech = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}}
        transcription = {}
        if code:
            speech["languageCode"] = code
            transcription = {"languageCodes": [code]}
        return {"setup": {
            "model": f"models/{self._model}",
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": speech},
            "systemInstruction": {"parts": [{"text": self._instructions}]},
            "tools": [{"functionDeclarations": FUNCTIONS}],
            "realtimeInputConfig": {"automaticActivityDetection": {
                "silenceDurationMs": END_OF_TURN_SILENCE_MS, "prefixPaddingMs": 300}},
            "inputAudioTranscription": transcription,
            "outputAudioTranscription": transcription,
        }}

    async def _send_json(self, message: dict) -> None:
        ws = self._ws
        if ws is not None:
            with suppress(websockets.ConnectionClosed):
                await ws.send(json.dumps(message))

    async def _say(self, instruction: str) -> None:
        """Ask the model to speak now, e.g. the greeting or a check-in."""
        turn = {"role": "user", "parts": [{"text": f"[From the call system, not the caller] {instruction}"}]}
        await self._send_json({"clientContent": {"turns": [turn], "turnComplete": True}})

    async def _on_message(self, message: dict) -> None:
        if "setupComplete" in message and not self._ready.is_set():
            self._ready.set()
            print(f"[Gemini] session ready at {time.monotonic() - self._t0:.2f}s voice={self._voice}")
            await self._say(self._opening_instruction())
        if "usageMetadata" in message:
            self.usage.add(message["usageMetadata"])
        if "toolCall" in message:
            await self._handle_calls(message["toolCall"].get("functionCalls") or [])
        if "serverContent" in message:
            await self._on_content(message["serverContent"])

    async def _on_content(self, content: dict) -> None:
        heard = (content.get("inputTranscription") or {}).get("text")
        if heard:
            self._add_text("caller", heard)
            self._quiet_since = time.monotonic()
            self._silence_checks = 0
            self._giving_up = False
            self._spoke_since_caller = False
        said = (content.get("outputTranscription") or {}).get("text")
        if said:
            self._add_text("agent", said)
        if content.get("interrupted"):
            await self._barge_in()
        for part in (content.get("modelTurn") or {}).get("parts") or []:
            data = (part.get("inlineData") or {}).get("data")
            if data:
                self._responding = True
                await self._play(f"turn-{self._turn}", self._downsample(base64.b64decode(data)))
        if content.get("turnComplete"):
            self._turn += 1
            self._responding = False
            self._turn_done.set()
            if self._giving_up and self._spoke_since_caller and self._end_task is None:
                self._end_task = asyncio.create_task(self._finish())

    def _add_text(self, role: str, text: str) -> None:
        if not self._log or self._log[-1][0] != role:
            self._log.append([role, ""])
        self._log[-1][1] += text

    async def _barge_in(self) -> None:
        """Gemini stopped talking because the caller spoke: drop what the phone still has queued."""
        if self._send is None or self._play_until <= time.monotonic():
            return
        self._play_until = time.monotonic()
        with suppress(RuntimeError, OSError):
            await self._send(clear_message(self.stream_sid))

    async def _check_in(self) -> None:
        await self._say(self._next_check_in())

    async def _handle_calls(self, calls: list[dict]) -> None:
        names = [call.get("name", "") for call in calls]
        self.actions.extend(names)
        print(f"[Gemini] tools: {names}")
        responses = [{"id": call.get("id"), "name": call.get("name"), "response": {"result": "ok"}} for call in calls]
        await self._send_json({"toolResponse": {"functionResponses": responses}})
        if "end_call" not in names or self._end_task is not None:
            return
        if self._spoke_since_caller:
            self._turn_done.clear()
            self._end_task = asyncio.create_task(self._hang_up_after_turn())
        elif not self._giving_up:
            # The model hung up without a word: have it say goodbye, then hang up once that has played.
            self._giving_up = True
            await self._say(GOODBYE_ASK)

    async def _hang_up_after_turn(self) -> None:
        """end_call can arrive mid-goodbye; let the model finish the turn before hanging up."""
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._turn_done.wait(), HANG_UP_WAIT_SECONDS)
        await self._finish()
