"""A recording of each live call: the caller and the agent mixed onto one 8 kHz track, saved as a WAV.

Recording is on only when VA_RECORDINGS_DIR names a folder, so a server with a throwaway disk (Render wipes
its disk on every deploy) does not hand out links to files that will vanish.
"""

import io
import os
import wave
from pathlib import Path

import numpy as np

RATE = 8000  # phone audio: 8 kHz, 16-bit mono, as Plivo streams it
_BYTES_PER_SECOND = RATE * 2
MAX_SECONDS = 60 * 60  # a stuck call stops growing its recording after an hour


class CallRecorder:
    """Caller frames arrive every 20 ms in order; agent audio is placed at the moment the phone plays it."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._caller = bytearray()
        self._agent = bytearray()

    def start(self, now: float) -> None:
        """The phone is connected: time zero of the recording."""
        if self._start is None:
            self._start = now

    def caller(self, frame: bytes) -> None:
        if self._start is not None and len(self._caller) < MAX_SECONDS * _BYTES_PER_SECOND:
            self._caller += frame

    def agent(self, pcm: bytes, start: float) -> None:
        """Agent audio that the phone starts playing at `start` (a time.monotonic() value)."""
        if self._start is None:
            return
        offset = self._offset(start)
        if offset > MAX_SECONDS * _BYTES_PER_SECOND:
            return
        if len(self._agent) < offset:
            self._agent += bytes(offset - len(self._agent))
        self._agent += pcm

    def cut_agent(self, at: float) -> None:
        """The caller cut in at `at`: the agent audio queued after that never played, so drop it."""
        if self._start is not None:
            del self._agent[self._offset(at):]

    def has_audio(self) -> bool:
        return bool(self._caller or self._agent)

    def wav(self) -> bytes:
        caller = np.frombuffer(bytes(self._caller[: len(self._caller) // 2 * 2]), dtype="<i2").astype(np.int32)
        agent = np.frombuffer(bytes(self._agent[: len(self._agent) // 2 * 2]), dtype="<i2").astype(np.int32)
        mixed = np.zeros(max(len(caller), len(agent)), dtype=np.int32)
        mixed[: len(caller)] += caller
        mixed[: len(agent)] += agent
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(np.clip(mixed, -32768, 32767).astype("<i2").tobytes())
        return buf.getvalue()

    def _offset(self, moment: float) -> int:
        """Byte position in a 16-bit track of a time.monotonic() moment."""
        return max(0, int((moment - self._start) * RATE)) * 2


def recordings_dir() -> Path | None:
    folder = os.environ.get("VA_RECORDINGS_DIR", "").strip()
    return Path(folder) if folder else None


def recorder_from_env() -> CallRecorder | None:
    """A recorder for a new call, or None when recording is switched off."""
    return CallRecorder() if recordings_dir() else None


def save_recording(call_id: int, recorder: CallRecorder | None) -> str | None:
    """Write the call's WAV and return its file name, or None when there is nothing to save."""
    folder = recordings_dir()
    if folder is None or recorder is None or not recorder.has_audio():
        return None
    folder.mkdir(parents=True, exist_ok=True)
    name = f"call_{int(call_id)}.wav"
    (folder / name).write_bytes(recorder.wav())
    return name


def recording_path(name: str | None) -> Path | None:
    """The saved file for a call, never outside the recordings folder."""
    folder = recordings_dir()
    if folder is None or not name or Path(name).name != name:
        return None
    path = folder / name
    return path if path.is_file() else None


def recording_url(call) -> str | None:
    """Where the dashboard plays a call's recording, or None when the call has none on this server."""
    return f"/dashboard/api/calls/{call.id}/recording" if recording_path(call.recording_file) else None
