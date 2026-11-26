"""The caller's yes/no answer after a recorded message: a keypress (1 = yes, 2 = no) or a short spoken reply."""

import io
import os
import re
import wave

import openai

from app.streaming.echo_gate import frame_levels

RATE = 8000
FRAME_BYTES = 320  # 20 ms of 8 kHz 16-bit audio, as the phone sends it
VOICE_LEVEL = 500.0  # frame RMS above line hiss (~50-200): the caller is talking
END_SILENCE_FRAMES = 40  # 0.8 s of quiet after speech: they have finished
MAX_SPEECH_FRAMES = 200  # 4 s: a yes/no needs no longer; stop listening even if they keep talking
TRANSCRIBE_MODEL = os.environ.get("VA_ANSWER_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

DIGITS = {"1": "yes", "2": "no"}

# Checked before the yes words, so "not interested" and "అవసరం లేదు" are a no.
_NO_PHRASES = ("not interested", "no thanks", "don't", "dont", "అవసరం లేదు", "ఇంట్రెస్ట్ లేదు", "వద్దు", "లేదు",
               "లేదండి", "నహీ", "नहीं", "नही", "मत")
_NO_WORDS = {"no", "nope", "nah", "ना", "na"}
_YES_PHRASES = ("interested", "కావాలి", "ఇంట్రెస్ట్ ఉంది", "అవును", "అవునండి", "సరే", "ఓకే", "हाँ", "हां", "ठीक")
_YES_WORDS = {"yes", "yeah", "yep", "ok", "okay", "sure", "haan", "ha", "హా", "హాఁ", "హాం", "हा", "जी"}


def classify_reply(text: str) -> str | None:
    """'yes', 'no', or None when the reply says neither (a question, a hello, "I'll tell you later")."""
    lowered = text.lower()
    words = set(re.findall(r"[\wऀ-ॿఀ-౿']+", lowered))
    if any(phrase in lowered for phrase in _NO_PHRASES) or words & _NO_WORDS:
        return "no"
    if any(phrase in lowered for phrase in _YES_PHRASES) or words & _YES_WORDS:
        return "yes"
    return None


def answer_from_digit(digit: str) -> str | None:
    return DIGITS.get(str(digit).strip())


class AnswerListener:
    """Collects the caller's reply frame by frame and says when they have finished speaking."""

    def __init__(self) -> None:
        self._frames: list[bytes] = []
        self._quiet = 0
        self.spoke = False

    def feed(self, frame: bytes) -> bool:
        """Take one inbound frame; True once a reply has been heard in full."""
        loud = len(frame) > 0 and float(frame_levels(frame).max()) >= VOICE_LEVEL
        if not self.spoke and not loud:
            return False
        self.spoke = True
        self._frames.append(frame)
        self._quiet = 0 if loud else self._quiet + 1
        return self._quiet >= END_SILENCE_FRAMES or len(self._frames) >= MAX_SPEECH_FRAMES

    def audio(self) -> bytes:
        return b"".join(self._frames)


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return buf.getvalue()


async def transcribe_reply(wav: bytes) -> str:
    """The caller's few words as text (about $0.003 a minute of audio)."""
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    result = await client.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL, file=("reply.wav", wav, "audio/wav"),
        prompt="A short answer to a yes or no question, in Telugu, Hindi or English.",
    )
    return (result.text or "").strip()
