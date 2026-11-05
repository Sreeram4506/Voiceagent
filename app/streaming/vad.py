"""Energy-based voice activity detection to find where the caller starts and stops speaking."""

from dataclasses import dataclass

from app.streaming.audio import FRAME_MS, frame_rms


@dataclass
class EnergyVAD:
    threshold: float = 500.0      # RMS level that counts as speech
    min_speech_ms: int = 200      # ignore clicks and short noises
    end_silence_ms: int = 200     # silence that closes an utterance

    def __post_init__(self) -> None:
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, frame: bytes) -> str | None:
        """Returns "speech_start", "speech_end", or None for each 20 ms frame."""
        loud = frame_rms(frame) >= self.threshold
        if loud:
            self._silence_ms = 0
            self._speech_ms += FRAME_MS
            if not self._in_speech and self._speech_ms >= self.min_speech_ms:
                self._in_speech = True
                return "speech_start"
            return None

        if self._in_speech:
            self._silence_ms += FRAME_MS
            if self._silence_ms >= self.end_silence_ms:
                self._reset()
                return "speech_end"
        else:
            self._speech_ms = 0
        return None

    def _reset(self) -> None:
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
