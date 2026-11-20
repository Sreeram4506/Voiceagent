"""Day 8 and Day 10: voice the product test sentences, save them, and measure CPU speed.

    python tts_training/scripts/compare_voices.py --piper models/piper_voices/te_IN-custom-medium.onnx --edge

Writes one WAV per sentence per voice into tts_training/output/<voice>/, at the
voice's own rate (not phone quality, so you hear the model itself), and prints
time-to-audio and real-time factor (synthesis time / audio length; under 1 is
faster than real time).
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # the repo root, for app.*

from app.streaming.piper_tts import prepare_text  # noqa: E402  pylint: disable=wrong-import-position

SENTENCES = ROOT / "tts_training" / "test_sentences.txt"
OUTPUT = ROOT / "tts_training" / "output"


def piper_voice(model: Path, out_dir: Path, sentences: list[str], length_scale: float) -> list[tuple[float, float]]:
    from piper.config import SynthesisConfig
    from piper.voice import PiperVoice

    t0 = time.perf_counter()
    voice = PiperVoice.load(str(model))
    print(f"{model.name}: loaded in {time.perf_counter() - t0:.2f}s")
    config = SynthesisConfig(length_scale=length_scale)
    timings = []
    for n, sentence in enumerate(sentences, 1):
        t0 = time.perf_counter()
        audio = np.concatenate([c.audio_float_array for c in voice.synthesize(prepare_text(sentence), syn_config=config)])
        took = time.perf_counter() - t0
        sf.write(out_dir / f"{n:02d}.wav", audio, voice.config.sample_rate)
        timings.append((took, len(audio) / voice.config.sample_rate))
    return timings


async def edge_voice(out_dir: Path, sentences: list[str]) -> list[tuple[float, float]]:
    import edge_tts

    timings = []
    for n, sentence in enumerate(sentences, 1):
        path = out_dir / f"{n:02d}.mp3"
        t0 = time.perf_counter()
        await edge_tts.Communicate(sentence, "te-IN-ShrutiNeural").save(str(path))
        took = time.perf_counter() - t0
        timings.append((took, sf.info(str(path)).duration if path.stat().st_size else 0.0))
    return timings


def report(name: str, timings: list[tuple[float, float]]) -> None:
    took = np.array([t for t, _ in timings])
    rtf = np.array([t / a for t, a in timings if a])
    print(f"{name}: per sentence median {np.median(took) * 1000:.0f} ms, worst {took.max() * 1000:.0f} ms, "
          f"real-time factor {np.median(rtf):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--piper", type=Path, action="append", default=[], help="a voice .onnx; repeat to compare")
    parser.add_argument("--edge", action="store_true", help="also voice the sentences with Microsoft Edge TTS")
    parser.add_argument("--length-scale", type=float, default=0.9, help="Piper pace; below 1 is faster")
    args = parser.parse_args()

    sentences = [s for s in SENTENCES.read_text(encoding="utf-8").splitlines() if s.strip()]
    for model in args.piper:
        out_dir = OUTPUT / model.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        report(model.stem, piper_voice(model, out_dir, sentences, args.length_scale))
    if args.edge:
        out_dir = OUTPUT / "edge"
        out_dir.mkdir(parents=True, exist_ok=True)
        report("edge (network)", asyncio.run(edge_voice(out_dir, sentences)))
    print(f"listen in {OUTPUT}")


if __name__ == "__main__":
    main()
