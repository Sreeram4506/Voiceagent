"""Day 3-4: turn the corrected transcripts into Piper's metadata.csv, and check the dataset.

    python tts_training/scripts/build_metadata.py tts_training/data/wavs tts_training/data/transcripts.tsv \
        tts_training/data/metadata.csv

Text goes through the same prepare_text() the live calls use (English words in
Telugu letters, amounts, dates, phone numbers, final ం), so the voice is trained
on exactly the forms it will later be asked to say. Lines starting with # in
the .tsv are skipped: use that to drop a bad clip.
"""

import argparse
import re
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root, for app.*

from app.streaming.piper_tts import prepare_text  # noqa: E402  pylint: disable=wrong-import-position

MIN_SECONDS, MAX_SECONDS = 1.0, 15.0
MIN_CHARS_PER_SECOND, MAX_CHARS_PER_SECOND = 4, 30  # outside this the text and audio likely don't match
START_HOURS, BEST_HOURS = 2.0, 6.0
_LATIN = re.compile(r"[A-Za-z]")


def check_line(wavs_dir: Path, name: str, raw_text: str) -> tuple[str | None, float, str | None]:
    """(spoken text, seconds, problem). Problem is None when the clip is fit for training."""
    path = wavs_dir / name
    if not path.is_file():
        return None, 0.0, "audio file missing"
    info = sf.info(str(path))
    seconds = info.frames / info.samplerate
    text = prepare_text(raw_text).strip()
    if info.samplerate != 22050 or info.channels != 1:
        return None, seconds, f"needs 22050 Hz mono, is {info.samplerate} Hz x{info.channels}"
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        return None, seconds, f"{seconds:.1f}s long, keep clips {MIN_SECONDS:.0f}-{MAX_SECONDS:.0f}s"
    if not text:
        return None, seconds, "empty transcript"
    if _LATIN.search(text):
        return None, seconds, f"English letters left: {text}"
    if not MIN_CHARS_PER_SECOND <= len(text) / seconds <= MAX_CHARS_PER_SECOND:
        return None, seconds, f"{len(text) / seconds:.0f} chars/s, text and audio may not match"
    return text, seconds, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wavs_dir", type=Path)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()

    rows, problems, seconds_total = [], [], 0.0
    for line in args.tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _, raw_text = line.partition("\t")
        text, seconds, problem = check_line(args.wavs_dir, name.strip(), raw_text)
        if problem:
            problems.append(f"{name}: {problem}")
            continue
        rows.append(f"{name.strip()}|{text}")
        seconds_total += seconds

    args.metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
    hours = seconds_total / 3600
    print(f"{len(rows)} clips, {hours:.2f} h written to {args.metadata}")
    print(f"left out {len(problems)}:" if problems else "no clips left out")
    for problem in problems:
        print("  " + problem)
    if hours < START_HOURS:
        print(f"need at least {START_HOURS:.0f} h to start fine-tuning ({BEST_HOURS:.0f} h+ for the most natural voice)")


if __name__ == "__main__":
    main()
