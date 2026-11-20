"""Day 3-4: first-draft transcripts for the clips, for you to correct by hand.

    OPENAI_API_KEY=... python tts_training/scripts/transcribe_clips.py tts_training/data/wavs tts_training/data/transcripts.tsv

Writes "clip.wav<TAB>text" lines. Clips already in the file are skipped, so a
stopped run picks up where it left off. About $0.006 per audio minute
(7 hours ≈ $2.50). Every line still needs a human check: the voice learns
exactly what the text says.
"""

import argparse
import os
from pathlib import Path

from openai import OpenAI

# The convention the voice is trained on and spoken with: all Telugu script, English said the Indian way.
STYLE_PROMPT = ("తెలుగు లిపిలో మాత్రమే రాయండి. ఇంగ్లీష్ పదాలు పలికినట్టే తెలుగు అక్షరాల్లో రాయండి: "
                "ఓకే, లోన్, ఫోన్, మీటింగ్, సార్. అంకెలు పలికినట్టే పదాల్లో రాయండి.")


def done_clips(tsv: Path) -> set[str]:
    if not tsv.exists():
        return set()
    return {line.split("\t", 1)[0] for line in tsv.read_text(encoding="utf-8").splitlines() if "\t" in line}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wavs_dir", type=Path)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--model", default="gpt-4o-transcribe", help="gpt-4o-transcribe, gpt-4o-mini-transcribe or whisper-1")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    skip = done_clips(args.tsv)
    todo = sorted(p for p in args.wavs_dir.glob("*.wav") if p.name not in skip)
    print(f"{len(todo)} clips to transcribe ({len(skip)} already done)")
    with args.tsv.open("a", encoding="utf-8") as out:
        for n, clip in enumerate(todo, 1):
            with clip.open("rb") as audio:
                text = client.audio.transcriptions.create(model=args.model, file=audio, language="te",
                                                          prompt=STYLE_PROMPT).text
            out.write(f"{clip.name}\t{' '.join(text.split())}\n")
            out.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"done; now correct {args.tsv} by listening to each clip")


if __name__ == "__main__":
    main()
