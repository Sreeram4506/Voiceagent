"""Telephony audio helpers. Telephony audio is 16-bit little-endian mono PCM."""

import array
import math
import sys

SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2  # 320 bytes per 20 ms frame


def silence(duration_ms: int) -> bytes:
    return bytes(SAMPLE_RATE * duration_ms // 1000 * 2)


def frame_rms(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def split_frames(pcm: bytes) -> list[bytes]:
    """Split into fixed 20 ms frames, zero-padding the last one."""
    frames = []
    for offset in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[offset : offset + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + bytes(FRAME_BYTES - len(chunk))
        frames.append(chunk)
    return frames
