"""The short thank-you a recorded-message call plays once the caller has answered.

The clips are 8 kHz 16-bit mono WAVs in audio/, made with gpt-4o-mini-tts (voice marin) and checked by
transcribing them back. "Yes" hears that the team will call; "no" and unclear replies hear a plain thanks.
"""

import wave
from functools import cache
from pathlib import Path

AUDIO = Path(__file__).parent / "audio"
LANGUAGES = ("te", "en")  # agents in other languages hear the English clip


@cache
def thanks_audio(language: str, answer: str) -> bytes:
    """Phone-ready PCM for the thank-you after a 'yes', 'no' or 'unclear' answer."""
    lang = language if language in LANGUAGES else "en"
    kind = "yes" if answer == "yes" else "no"
    with wave.open(str(AUDIO / f"thanks_{lang}_{kind}.wav")) as clip:
        return clip.readframes(clip.getnframes())
