"""Keep the voice agent from hearing its own voice as the caller cutting in.

Phones leak some of the agent's voice back down the line (earpiece echo, speakerphone). The speech
detector hears even a 5% leak as the caller talking and stops the agent mid-sentence every couple of
seconds, which callers hear as the audio breaking up. While the agent's audio is playing, inbound audio
much quieter than the agent's own voice is replaced with silence; audio loud enough to be the caller
talking over the agent opens the gate for the rest of what they say.
"""

import numpy as np

FRAME_BYTES = 320  # 20 ms of 8 kHz 16-bit mono
BARGE_IN_RATIO = 0.45  # the caller must be at least this loud, relative to the agent's voice, to cut in
OPEN_FRAMES = 2  # consecutive loud frames that open the gate: a click or a pop is not the caller
HOLD_SECONDS = 1.0  # stays open this long after the last loud frame, so the whole interruption is heard
DEFAULT_AGENT_LEVEL = 3000.0  # typical frame RMS of the agent's voice, until this call's is measured
SPEECH_FLOOR = 300.0  # agent frames quieter than this are pauses and do not count towards its level
LEVEL_SMOOTHING = 0.05  # how fast the measured agent level follows new audio


def frame_levels(pcm: bytes) -> np.ndarray:
    """RMS of each 20 ms frame."""
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32)
    usable = len(samples) - len(samples) % (FRAME_BYTES // 2)
    if usable == 0:
        return np.sqrt(np.mean(samples**2, keepdims=True)) if len(samples) else np.zeros(0)
    return np.sqrt(np.mean(samples[:usable].reshape(-1, FRAME_BYTES // 2) ** 2, axis=1))


class EchoGate:
    """Decides, frame by frame, whether inbound audio is the caller or the agent's own echo."""

    def __init__(self) -> None:
        self.agent_level = DEFAULT_AGENT_LEVEL
        self.blocked_frames = 0
        self._loud_run = 0
        self._open_until = 0.0

    def agent_audio(self, pcm: bytes) -> None:
        """Learn how loud the agent's voice is on this call from the audio sent to the phone."""
        for level in frame_levels(pcm):
            if level >= SPEECH_FLOOR:
                self.agent_level += (float(level) - self.agent_level) * LEVEL_SMOOTHING

    def filter(self, frame: bytes, agent_playing: bool, now: float) -> bytes:
        """The frame to pass on: unchanged, or silence when it can only be the agent's echo."""
        if not agent_playing:
            self._loud_run = 0
            return frame
        levels = frame_levels(frame)
        loud = levels.size > 0 and float(levels.max()) >= self.agent_level * BARGE_IN_RATIO
        self._loud_run = self._loud_run + 1 if loud else 0
        if self._loud_run >= OPEN_FRAMES:
            self._open_until = now + HOLD_SECONDS
        if now < self._open_until:
            return frame
        self.blocked_frames += 1
        return bytes(len(frame))
