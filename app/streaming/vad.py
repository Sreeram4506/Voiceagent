"""Voice activity detection with adaptive noise floor for telephony audio."""

from dataclasses import dataclass, field

from app.streaming.audio import FRAME_MS, frame_rms


@dataclass
class EnergyVAD:
    threshold: float = 500.0
    min_speech_ms: int = 150
    end_silence_ms: int = 500      # callers pause 300-450 ms at a comma; shorter answers half a sentence
    adaptive: bool = True          # track background noise and adjust threshold

    def __post_init__(self) -> None:
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._noise_floor: float = 0.0
        self._frame_count: int = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, frame: bytes) -> str | None:
        """Returns "speech_start", "speech_end", or None for each 20 ms frame."""
        rms = frame_rms(frame)

        # Adaptive noise floor from quiet frames only: if the caller talks straight away, their voice
        # must not be learned as background noise, or the detector goes deaf to them.
        if self.adaptive and not self._in_speech and rms < self.threshold:
            self._frame_count += 1
            if self._frame_count <= 25:
                # First 500 ms of quiet: bootstrap the noise estimate
                self._noise_floor = (self._noise_floor * (self._frame_count - 1) + rms) / self._frame_count
            else:
                self._noise_floor = self._noise_floor * 0.97 + rms * 0.03

        effective_threshold = self.threshold
        if self.adaptive and self._noise_floor > 0:
            # Threshold = max(fixed floor, 2.5x the noise floor)
            effective_threshold = max(self.threshold, self._noise_floor * 2.5)

        loud = rms >= effective_threshold
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
