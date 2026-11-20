"""Day 3-4: cut raw recordings into 2-15 s training clips at natural pauses.

    python tts_training/scripts/prepare_clips.py tts_training/data/raw tts_training/data/wavs

Each clip is mono, 22050 Hz, 16-bit, rumble filtered, loudness matched, with
short silence kept at both ends. Pieces that stay longer than 15 s are listed,
not exported, so you can cut them by hand.
"""

import argparse
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import high_pass_filter
from pydub.silence import split_on_silence

SAMPLE_RATE = 22050
MIN_CLIP_MS = 2_000
MAX_CLIP_MS = 15_000
PAUSE_MS = 350            # a breath between phrases
SILENCE_BELOW_AVG_DB = 16  # quieter than the recording's average by this much counts as a pause
KEEP_SILENCE_MS = 150
RUMBLE_CUTOFF_HZ = 70
TARGET_DBFS = -20.0
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def clean(audio: AudioSegment) -> AudioSegment:
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    return high_pass_filter(audio, RUMBLE_CUTOFF_HZ)


def group_pieces(pieces: list[AudioSegment]) -> list[AudioSegment]:
    """Join neighbouring phrases until each clip is at least MIN_CLIP_MS, never past MAX_CLIP_MS."""
    clips: list[AudioSegment] = []
    current: AudioSegment | None = None
    for piece in pieces:
        if current is None:
            current = piece
        elif len(current) < MIN_CLIP_MS and len(current) + len(piece) <= MAX_CLIP_MS:
            current += piece
        else:
            clips.append(current)
            current = piece
    if current is not None:
        clips.append(current)
    return clips


def level(clip: AudioSegment) -> AudioSegment:
    clip = clip.apply_gain(TARGET_DBFS - clip.dBFS)
    return clip.apply_gain(min(0.0, -1.0 - clip.max_dBFS))  # keep peaks below -1 dBFS


def split_recording(path: Path, out_dir: Path) -> tuple[int, list[str]]:
    audio = clean(AudioSegment.from_file(path))
    pieces = split_on_silence(audio, min_silence_len=PAUSE_MS, silence_thresh=audio.dBFS - SILENCE_BELOW_AVG_DB,
                              keep_silence=KEEP_SILENCE_MS)
    saved, too_long = 0, []
    for n, clip in enumerate(group_pieces(pieces)):
        name = f"{path.stem}_{n:04d}.wav"
        if len(clip) > MAX_CLIP_MS:
            too_long.append(f"{name} ({len(clip) / 1000:.1f}s)")
            continue
        if len(clip) < MIN_CLIP_MS:
            continue
        level(clip).export(out_dir / name, format="wav")
        saved += 1
    return saved, too_long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in sorted(p for p in args.raw_dir.iterdir() if p.suffix.lower() in AUDIO_TYPES):
        saved, too_long = split_recording(path, args.out_dir)
        total += saved
        print(f"{path.name}: {saved} clips" + (f"; too long, cut by hand: {', '.join(too_long)}" if too_long else ""))
    print(f"{total} clips in {args.out_dir}")


if __name__ == "__main__":
    main()
