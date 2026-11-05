"""Piper TTS adapter — local offline Telugu voice synthesis.

Uses the Piper TTS engine with a pre-trained Telugu voice model.
Inference is ~50x faster than real-time (~50ms for 2.5s of speech).

Model: te_IN-venkatesh-medium (male) or te_IN-padmavathi-medium (female)
Downloaded from: https://huggingface.co/rhasspy/piper-voices
"""

import os
import re
from math import gcd

import numpy as np

# Common English words used in Telugu speech → Telugu script transliterations.
# This lets the Telugu Piper model pronounce them naturally instead of awkwardly.
_EN_TO_TE: dict[str, str] = {
    # greetings / fillers
    "hello": "హలో",
    "hi": "హాయ్",
    "okay": "ఓకే",
    "ok": "ఓకే",
    "yes": "యస్",
    "no": "నో",
    "sure": "షూర్",
    "thanks": "థాంక్స్",
    "thank": "థాంక్",
    "sorry": "సారీ",
    "please": "ప్లీజ్",
    "good": "గుడ్",
    "great": "గ్రేట్",
    "nice": "నైస్",
    "fine": "ఫైన్",
    "right": "రైట్",
    # common nouns in Telugu conversation
    "support": "సపోర్ట్",
    "call": "కాల్",
    "calls": "కాల్స్",
    "team": "టీమ్",
    "campaign": "కాంపెయిన్",
    "meeting": "మీటింగ్",
    "appointment": "అపాయింట్మెంట్",
    "area": "ఏరియా",
    "work": "వర్క్",
    "problem": "ప్రాబ్లమ్",
    "issue": "ఇష్యూ",
    "simple": "సింపుల్",
    "number": "నంబర్",
    "mobile": "మొబైల్",
    "phone": "ఫోన్",
    "message": "మెసేజ్",
    "address": "అడ్రెస్",
    "vote": "వోటు",
    "votes": "వోట్లు",
    "time": "టైమ్",
    "minute": "మినిట్",
    "minutes": "మినిట్లు",
    "second": "సెకెండ్",
    "day": "డే",
    "date": "డేట్",
    "month": "మంత్",
    # convenience
    "free": "ఫ్రీ",
    "busy": "బిజీ",
    "available": "అవైలబుల్",
    "convenient": "కన్వీనియంట్",
    "information": "ఇన్ఫర్మేషన్",
    "details": "డీటెయిల్స్",
    "update": "అప్డేట్",
    "confirm": "కన్ఫర్మ్",
    "connect": "కనెక్ట్",
    "register": "రిజిస్టర్",
    "candidate": "కాండిడేట్",
    "party": "పార్టీ",
    "election": "ఎలెక్షన్",
    "local": "లోకల్",
    "government": "గవర్నమెంట్",
    "development": "డెవలప్మెంట్",
    "project": "ప్రాజెక్ట్",
    "road": "రోడ్",
    "water": "వాటర్",
    "school": "స్కూల్",
    "hospital": "హాస్పిటల్",
}

# Build a regex that matches whole English words (case-insensitive)
_EN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_EN_TO_TE, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _transliterate(text: str) -> str:
    """Replace English words in mixed Telugu+English text with Telugu script."""
    def _replace(m: re.Match) -> str:
        return _EN_TO_TE.get(m.group(0).lower(), m.group(0))
    return _EN_PATTERN.sub(_replace, text)

PIPER_VOICES_DIR = os.environ.get(
    "PIPER_VOICES_DIR",
    "/Users/koushikeslavath/Downloads/piper_voices",
)
PIPER_VOICE = os.environ.get("PIPER_VOICE", "te_IN-venkatesh-medium")

TELEPHONY_SAMPLE_RATE = 8000

_voice = None


def _get_voice():
    global _voice
    if _voice is None:
        from piper.voice import PiperVoice
        model_path = os.path.join(
            PIPER_VOICES_DIR, "te", "te_IN", "venkatesh", "medium",
            f"{PIPER_VOICE}.onnx",
        )
        print(f"[Piper] Loading Telugu voice from {model_path}")
        _voice = PiperVoice.load(model_path)
        print(f"[Piper] Voice loaded, sample_rate={_voice.config.sample_rate}")
    return _voice


def _synthesize_pcm8k(text: str) -> bytes:
    """Synthesize text and return 8kHz 16-bit mono PCM bytes."""
    from scipy.signal import resample_poly

    voice = _get_voice()
    sr = voice.config.sample_rate

    text = _transliterate(text)
    chunks = list(voice.synthesize(text))
    if not chunks:
        return b""

    audio = np.concatenate([c.audio_float_array for c in chunks])
    if audio.size == 0:
        return b""

    # Resample from 22050Hz to 8000Hz
    g = gcd(sr, TELEPHONY_SAMPLE_RATE)
    audio = resample_poly(audio, TELEPHONY_SAMPLE_RATE // g, sr // g).astype(np.float32)

    # Normalize to -20 dBFS, cap at +20dB
    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        target = 10 ** (-20.0 / 20)
        gain = min(target / peak, 10 ** (20 / 20))
        audio = np.clip(audio * gain, -1.0, 1.0)

    return (audio * 32767).astype(np.int16).tobytes()


class PiperTTS:
    """Async TTS adapter using local Piper Telugu voice. 8kHz 16-bit mono PCM output."""

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        import asyncio
        import time

        loop = asyncio.get_event_loop()
        t0 = time.time()
        pcm = await loop.run_in_executor(None, _synthesize_pcm8k, text)
        print(f"[Piper] synthesis={time.time()-t0:.3f}s text_len={len(text)} pcm={len(pcm)}bytes")
        return pcm
