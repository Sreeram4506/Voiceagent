"""Piper TTS adapter — local offline Telugu voice synthesis.

Uses the Piper TTS engine with a pre-trained Telugu voice model.
Inference is ~50x faster than real-time (~50ms for 2.5s of speech).

Model: te_IN-venkatesh-medium (male) or te_IN-padmavathi-medium (female)
Downloaded from: https://huggingface.co/rhasspy/piper-voices
"""

import os
import re
from functools import lru_cache
from pathlib import Path
from math import gcd

import numpy as np

# Comprehensive English to Telugu phonetic transliteration dictionary.
# This ensures common English words used in spoken Telugu/English calls
# are pronounced fluently and natively by the Telugu voice engine without stumbling.
_EN_TO_TE: dict[str, str] = {
    # Greetings, address & politeness
    "hello": "హలో",
    "hi": "హాయ్",
    "hey": "హే",
    "sir": "సార్",
    "madam": "మేడమ్",
    "boss": "బాస్",
    "bro": "బ్రో",
    "brother": "బ్రదర్",
    "friend": "ఫ్రెండ్",
    "mr": "మిస్టర్",
    "mrs": "మిసెస్",
    "dr": "డాక్టర్",
    "doctor": "డాక్టర్",
    "miss": "మిస్",
    "thanks": "థాంక్స్",
    "thank": "థాంక్",
    "thankyou": "థాంక్యూ",
    "welcome": "వెల్‌కమ్",
    "please": "ప్లీజ్",
    "sorry": "సారీ",
    "excuse": "ఎక్స్‌క్యూజ్",
    "pardon": "పార్డన్",

    # Conversation affirmative / negative & reactions
    "okay": "ఓకే",
    "ok": "ఓకే",
    "yes": "యస్",
    "yeah": "యా",
    "no": "నో",
    "sure": "షూర్",
    "fine": "ఫైన్",
    "good": "గుడ్",
    "great": "గ్రేట్",
    "super": "సూపర్",
    "nice": "నైస్",
    "right": "రైట్",
    "correct": "కరెక్ట్",
    "wrong": "రాంగ్",
    "exactly": "ఎగ్జాక్ట్లీ",
    "definitely": "డెఫినెట్లీ",
    "obviously": "ఆబ్వియస్లీ",
    "really": "రియల్లీ",
    "perfect": "పర్ఫెక్ట్",
    "done": "డన్",
    "ready": "రెడీ",
    "true": "ట్రూ",
    "false": "ఫాల్స్",

    # Common verbs & conversation phrases
    "call": "కాల్",
    "calls": "కాల్స్",
    "calling": "కాలింగ్",
    "called": "కాల్డ్",
    "recall": "రీకాల్",
    "speak": "స్పీక్",
    "speaking": "స్పీకింగ్",
    "talk": "టాక్",
    "talking": "టాకింగ్",
    "tell": "టెల్",
    "telling": "టెల్లింగ్",
    "say": "సే",
    "saying": "సేయింగ్",
    "ask": "ఆస్క్",
    "asking": "ఆస్కింగ్",
    "hear": "హియర్",
    "hearing": "హియరింగ్",
    "listen": "లిసన్",
    "listening": "లిసనింగ్",
    "know": "నో",
    "knowing": "నోయింగ్",
    "understand": "అండర్‌స్టాండ్",
    "share": "షేర్",
    "sharing": "షేరింగ్",
    "send": "సెండ్",
    "sending": "సెండింగ్",
    "sent": "సెంట్",
    "receive": "రిసీవ్",
    "check": "చెక్",
    "checking": "చెకింగ్",
    "verify": "వెరిఫై",
    "confirm": "కన్ఫర్మ్",
    "confirmed": "కన్ఫర్మ్డ్",
    "confirmation": "కన్ఫర్మేషన్",
    "book": "బుక్",
    "booking": "బుకింగ్",
    "booked": "బుక్డ్",
    "register": "రిజిస్టర్",
    "help": "హెల్ప్",
    "need": "నీడ్",
    "want": "వాంట్",
    "like": "లైక్",
    "connect": "కనెక్ట్",
    "connected": "కనెక్టెడ్",
    "connection": "కనెక్షన్",
    "disconnect": "డిస్కనెక్ట్",
    "wait": "వెయిట్",
    "hold": "హోల్డ్",
    "cancel": "క్యాన్సిల్",
    "cancelled": "క్యాన్సిల్డ్",
    "update": "అప్‌డేట్",
    "updated": "అప్‌డేటెడ్",
    "renew": "రెన్యూ",
    "renewal": "రెన్యూవల్",

    # Time, dates & scheduling
    "time": "టైమ్",
    "times": "టైమ్స్",
    "timing": "టైమింగ్",
    "timings": "టైమింగ్స్",
    "minute": "మినిట్",
    "minutes": "మినిట్స్",
    "second": "సెకండ్",
    "seconds": "సెకండ్స్",
    "hour": "అవర్",
    "hours": "అవర్స్",
    "day": "డే",
    "days": "డేస్",
    "today": "టుడే",
    "tomorrow": "టుమారో",
    "yesterday": "ఎస్టర్డే",
    "morning": "మార్నింగ్",
    "afternoon": "ఆఫ్టర్‌నూన్",
    "evening": "ఈవెనింగ్",
    "night": "నైట్",
    "now": "నౌ",
    "later": "లేటర్",
    "soon": "సూన్",
    "early": "ఎర్లీ",
    "late": "లేట్",
    "daily": "డైలీ",
    "weekly": "వీక్లీ",
    "monthly": "మంత్లీ",
    "week": "వీక్",
    "weeks": "వీక్స్",
    "month": "మంత్",
    "months": "మంత్స్",
    "year": "ఇయర్",
    "years": "ఇయర్స్",
    "weekend": "వీకెండ్",
    "date": "డేట్",
    "slot": "స్లాట్",
    "slots": "స్లాట్స్",
    "schedule": "షెడ్యూల్",
    "scheduled": "షెడ్యూల్డ్",
    "calendar": "క్యాలెండర్",
    "appointment": "అపాయింట్‌మెంట్",
    "appointments": "అపాయింట్‌మెంట్స్",
    "meeting": "మీటింగ్",
    "meetings": "మీటింగ్స్",

    # Banking, finance & credit
    "loan": "లోన్",
    "loans": "లోన్స్",
    "bank": "బ్యాంక్",
    "banks": "బ్యాంక్స్",
    "banking": "బ్యాంకింగ్",
    "branch": "బ్రాంచ్",
    "account": "అకౌంట్",
    "accounts": "అకౌంట్స్",
    "money": "మనీ",
    "cash": "క్యాష్",
    "payment": "పేమెంట్",
    "payments": "పేమెంట్స్",
    "pay": "పే",
    "interest": "ఇంట్రెస్ట్",
    "rate": "రేట్",
    "rates": "రేట్స్",
    "amount": "అమౌంట్",
    "credit": "క్రెడిట్",
    "debit": "డెబిట్",
    "card": "కార్డ్",
    "cards": "కార్డ్స్",
    "emi": "ఈఎంఐ",
    "finance": "ఫైనాన్స్",
    "financial": "ఫైనాన్షియల్",
    "offer": "ఆఫర్",
    "offers": "ఆఫర్స్",
    "scheme": "స్కీమ్",
    "schemes": "స్కీమ్స్",
    "discount": "డిస్కౌంట్",
    "limit": "లిమిట్",
    "balance": "బ్యాలెన్స్",
    "statement": "స్టేట్‌మెంట్",
    "transfer": "ట్రాన్స్‌ఫర్",
    "deposit": "డిపాజిట్",
    "otp": "ఓటీపీ",
    "pin": "పిన్",
    "kyc": "కేవైసీ",
    "document": "డాక్యుమెంట్",
    "documents": "డాక్యుమెంట్స్",
    "proof": "ప్రూఫ్",
    "id": "ఐడీ",
    "pan": "పాన్",
    "aadhar": "ఆధార్",
    "bill": "బిల్",
    "charges": "ఛార్జెస్",
    "fee": "ఫీ",
    "fees": "ఫీజ్",

    # Business, real estate & campaign
    "candidate": "కాండిడేట్",
    "campaign": "క్యాంపెయిన్",
    "party": "పార్టీ",
    "vote": "వోటు",
    "votes": "వోట్లు",
    "election": "ఎలక్షన్",
    "area": "ఏరియా",
    "location": "లొకేషన్",
    "site": "సైట్",
    "visit": "విజిట్",
    "visiting": "విజిటింగ్",
    "plot": "ప్లాట్",
    "plots": "ప్లాట్స్",
    "flat": "ఫ్లాట్",
    "flats": "ఫ్లాట్స్",
    "house": "హౌస్",
    "villa": "విల్లా",
    "property": "ప్రాపర్టీ",
    "office": "ఆఫీస్",
    "address": "అడ్రెస్",
    "road": "రోడ్",
    "street": "స్ట్రీట్",
    "city": "సిటీ",
    "team": "టీమ్",
    "customer": "కస్టమర్",
    "customers": "కస్టమర్స్",
    "client": "క్లయింట్",
    "clients": "క్లయింట్స్",
    "staff": "స్టాఫ్",
    "manager": "మేనేజర్",
    "agent": "ఏజెంట్",
    "service": "సర్వీస్",
    "support": "సపోర్ట్",
    "information": "ఇన్ఫర్మేషన్",
    "details": "డీటెయిల్స్",
    "issue": "ఇష్యూ",
    "problem": "ప్రాబ్లమ్",
    "free": "ఫ్రీ",
    "busy": "బిజీ",
    "available": "అవైలబుల్",
    "convenient": "కన్వీనియంట్",
    "simple": "సింపుల్",
    "easy": "ఈజీ",
    "fast": "ఫాస్ట్",
    "quick": "క్విక్",
    "urgent": "అర్జెంట్",
    "important": "ఇంపార్టెంట్",
    "phone": "ఫోన్",
    "mobile": "మొబైల్",
    "number": "నంబర్",
    "numbers": "నంబర్స్",
    "message": "మెసేజ్",
    "messages": "మెసేజెస్",
    "app": "యాప్",
    "link": "లింక్",
    "online": "ఆన్‌లైన్",
    "system": "సిస్టమ్",
    "work": "వర్క్",

    # Common short words
    "you": "యూ",
    "your": "యువర్",
    "me": "మీ",
    "my": "మై",
    "we": "వీ",
    "our": "అవర్",
    "is": "ఈజ్",
    "are": "ఆర్",
    "am": "యామ్",
    "was": "వాజ్",
    "were": "వర్",
    "have": "హావ్",
    "has": "హ్యాజ్",
    "had": "హ్యాడ్",
    "will": "విల్",
    "can": "క్యాన్",
    "for": "ఫర్",
    "to": "టు",
    "in": "ఇన్",
    "on": "ఆన్",
    "at": "ఎట్",
    "by": "బై",
    "from": "ఫ్రమ్",
    "with": "విత్",
    "and": "అండ్",
    "but": "బట్",
    "so": "సో",
    "if": "ఇఫ్",
    "all": "ఆల్",
    "any": "ఎనీ",
    "some": "సమ్",
    "now": "నౌ",
    "not": "నాట్",
    "just": "జస్ట్",
    "how": "హౌ",
    "what": "వాట్",
    "when": "వెన్",
    "where": "వేర్",
    "why": "వై",
}

# Phonetic syllable mapping for unmapped English words to ensure
# NO Latin letters reach Piper and cause inconsistent/garbled speech.
_CONSONANTS_MAP = {
    'b': 'బ', 'c': 'క', 'd': 'డ', 'f': 'ఫ', 'g': 'గ', 'h': 'హ',
    'j': 'జ', 'k': 'క', 'l': 'ల', 'm': 'మ', 'n': 'న', 'p': 'ప',
    'q': 'క', 'r': 'ర', 's': 'స', 't': 'ట', 'v': 'వ', 'w': 'వ',
    'x': 'క్స్', 'y': 'య', 'z': 'జ'
}
_VOWEL_SIGNS = {
    'a': 'ా', 'e': 'ె', 'i': 'ి', 'o': 'ొ', 'u': 'ు',
}
_MULTI_MAP = [
    ('tion', 'షన్'), ('sion', 'షన్'), ('sh', 'ష'), ('ch', 'చ'),
    ('th', 'త'), ('ph', 'ఫ'), ('wh', 'వ'), ('ck', 'క్'),
    ('ee', 'ీ'), ('oo', 'ూ'), ('ea', 'ీ'), ('ai', 'ై'),
    ('ay', 'ే'), ('ou', 'ౌ'), ('ow', 'ౌ'),
]


def _phonetic_fallback(word: str) -> str:
    """Phonetically map an English word into Telugu syllables so Piper pronounces it fluently."""
    w = word.lower()
    res = []
    i = 0
    n = len(w)
    while i < n:
        matched = False
        for eng, te in _MULTI_MAP:
            if w.startswith(eng, i):
                res.append(te)
                i += len(eng)
                matched = True
                break
        if matched:
            continue
        c = w[i]
        if c in _CONSONANTS_MAP:
            if i + 1 < n and w[i+1] in _VOWEL_SIGNS:
                v = w[i+1]
                res.append(_CONSONANTS_MAP[c] + _VOWEL_SIGNS[v])
                i += 2
            elif i + 1 < n and w[i+1] in _CONSONANTS_MAP:
                res.append(_CONSONANTS_MAP[c] + '్')
                i += 1
            else:
                res.append(_CONSONANTS_MAP[c])
                i += 1
        elif c in _VOWEL_SIGNS:
            v_init = {'a': 'ఆ', 'e': 'ఎ', 'i': 'ఇ', 'o': 'ఒ', 'u': 'ఉ'}
            res.append(v_init.get(c, 'అ'))
            i += 1
        else:
            res.append(c)
            i += 1
    return ''.join(res)


# Build a regex that matches whole English words (case-insensitive)
_EN_PATTERN = re.compile(r"\b[A-Za-z]+\b")


def _transliterate(text: str) -> str:
    """Replace English words in mixed Telugu+English text with natural Telugu script phonetics."""
    def _replace(m: re.Match) -> str:
        word = m.group(0).lower()
        if word in _EN_TO_TE:
            return _EN_TO_TE[word]
        return _phonetic_fallback(word)
    return _EN_PATTERN.sub(_replace, text)

_default_piper_dir = Path(__file__).resolve().parent.parent.parent / "models" / "piper_voices"
if not _default_piper_dir.exists() and Path("/Users/koushikeslavath/Downloads/piper_voices").exists():
    _default_piper_dir = Path("/Users/koushikeslavath/Downloads/piper_voices")

PIPER_VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", str(_default_piper_dir))
PIPER_VOICE = os.environ.get("PIPER_VOICE", "te_IN-venkatesh-medium")
# Full path to a voice's .onnx (its .onnx.json beside it). Set this for a custom-trained voice.
PIPER_MODEL = os.environ.get("PIPER_MODEL", "")
# length_scale < 1.0 speeds up speech. 0.65 is ~40% faster — natural, energetic, human pace.
PIPER_LENGTH_SCALE = float(os.environ.get("PIPER_LENGTH_SCALE", "0.65"))
# Expressiveness: noise_scale varies the voice, noise_w varies the rhythm. Piper's defaults are 0.667 / 0.8.
PIPER_NOISE_SCALE = float(os.environ.get("PIPER_NOISE_SCALE", "0.667"))
PIPER_NOISE_W = float(os.environ.get("PIPER_NOISE_W", "0.8"))
PHRASE_CACHE_SIZE = 512  # greetings and stock replies repeat on every call

TELEPHONY_SAMPLE_RATE = 8000

_voice = None


def model_path() -> Path | None:
    """The voice to load: PIPER_MODEL if set, else PIPER_VOICE found anywhere under PIPER_VOICES_DIR."""
    if PIPER_MODEL:
        path = Path(PIPER_MODEL)
        return path if path.is_file() else None
    root = Path(PIPER_VOICES_DIR)
    return next(root.rglob(f"{PIPER_VOICE}.onnx"), None) if root.is_dir() else None


def _get_voice():
    global _voice
    if _voice is None:
        from piper.voice import PiperVoice
        path = model_path()
        if path is None:
            raise FileNotFoundError(f"Piper voice not found (PIPER_MODEL={PIPER_MODEL!r}, "
                                    f"PIPER_VOICE={PIPER_VOICE!r} under {PIPER_VOICES_DIR})")
        print(f"[Piper] Loading Telugu voice from {path}")
        _voice = PiperVoice.load(str(path))
        print(f"[Piper] Voice loaded, sample_rate={_voice.config.sample_rate}")
    return _voice


def warmup() -> bool:
    """At server start, in the background: download the voice if it isn't on disk yet, then load it.

    Calls use Edge until the voice is ready (the factory checks model_path() per call), so a slow or
    failed download never delays or breaks a call.
    """
    if os.environ.get("VA_TTS_ENGINE", "edge").strip().lower() != "piper":
        return False
    import threading

    def _load() -> None:
        try:
            if model_path() is None:
                from app.streaming.piper_fetch import main as fetch_voice

                if fetch_voice() != 0 or model_path() is None:
                    print("[Startup] Piper voice not available; Telugu calls stay on Edge")
                    return
            _synthesize_pcm8k("నమస్కారమ్")
            print("[Startup] Piper voice warmed up")
        except Exception as e:  # a broken voice file must not take the server down; calls fall back to Edge
            print(f"[Startup] Piper warmup failed: {e}")

    threading.Thread(target=_load, name="piper-warmup", daemon=True).start()
    return True


def prepare_text(text: str) -> str:
    """Spoken-form Telugu script for Piper: English words in Telugu letters, then numbers/dates/anusvara fixed."""
    from app.streaming.telugu_text import normalize_telugu

    return normalize_telugu(_transliterate(text))


@lru_cache(maxsize=PHRASE_CACHE_SIZE)
def _synthesize_pcm8k(text: str, length_scale: float | None = None) -> bytes:
    """Synthesize text and return 8kHz 16-bit mono PCM bytes. Repeated phrases come from the cache."""
    from scipy.signal import resample_poly
    from piper.config import SynthesisConfig

    voice = _get_voice()
    sr = voice.config.sample_rate

    text = prepare_text(text)
    ls = length_scale if length_scale is not None else PIPER_LENGTH_SCALE
    cfg = SynthesisConfig(length_scale=ls, noise_scale=PIPER_NOISE_SCALE, noise_w_scale=PIPER_NOISE_W)
    chunks = list(voice.synthesize(text, syn_config=cfg))
    if not chunks:
        return b""

    audio = np.concatenate([c.audio_float_array for c in chunks])
    if audio.size == 0:
        return b""

    # Resample from 22050Hz to 8000Hz
    g = gcd(sr, TELEPHONY_SAMPLE_RATE)
    audio = resample_poly(audio, TELEPHONY_SAMPLE_RATE // g, sr // g).astype(np.float32)

    # Normalize to -14 dBFS — loud, present, human call volume; cap at +20dB
    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        target = 10 ** (-14.0 / 20)
        gain = min(target / peak, 10 ** (20 / 20))
        audio = np.clip(audio * gain, -1.0, 1.0)

    return (audio * 32767).astype(np.int16).tobytes()


def _split_phrases(text: str) -> list[str]:
    """Split at natural clause boundaries so each phrase can be synthesized and played in turn."""
    parts = [p.strip() for p in re.split(r'(?<=[,।;.!?])\s+', text) if p.strip()]
    return parts if parts else [text]


class PiperTTS:
    """Async TTS adapter using local Piper Telugu voice. 8kHz 16-bit mono PCM output."""

    def __init__(self, length_scale: float | None = None):
        self.length_scale = length_scale

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        import asyncio
        import time

        loop = asyncio.get_event_loop()
        t0 = time.time()
        pcm = await loop.run_in_executor(None, _synthesize_pcm8k, text, self.length_scale)
        print(f"[Piper] synthesis={time.time()-t0:.3f}s text_len={len(text)} speed={PIPER_LENGTH_SCALE} pcm={len(pcm)}bytes")
        return pcm

    async def stream_synthesize(self, text: str, language: str, voice_tier: str):
        """Yield PCM chunks phrase-by-phrase so the first audio plays while the rest is generated."""
        import asyncio
        import time

        loop = asyncio.get_event_loop()
        phrases = _split_phrases(text)
        t0 = time.time()
        for phrase in phrases:
            pcm = await loop.run_in_executor(None, _synthesize_pcm8k, phrase, self.length_scale)
            if pcm:
                print(f"[Piper] stream phrase={time.time()-t0:.3f}s '{phrase[:30]}'")
                yield pcm
