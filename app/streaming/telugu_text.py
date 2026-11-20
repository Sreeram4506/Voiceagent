"""Turns written Telugu into the words a person would say, before Piper voices it.

Piper reads text through espeak-ng, which already speaks plain numbers, lakhs and
percentages in Telugu. It gets these wrong, so they are rewritten here:
  * ₹ / Rs amounts lose the currency word      → "5000 రూపాయలు"
  * dates read the slashes aloud               → "15 ఆగస్టు 2026"
  * phone numbers are read as crores           → digit by digit
  * a word-final ం (anusvara) comes out as "n" → written మ్ so it is said "m"
    (also before a vowel letter, as in ఈఎంఐ, which then reads letter by letter)

The same function runs on training transcripts and on live text, so the voice
is trained on exactly the forms it will be asked to say.
"""

import re

DIGITS = ("సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది")
MONTHS = ("జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్",
          "జులై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్")
RUPEES = "రూపాయలు"
MIN_PHONE_DIGITS = 7  # shorter digit runs are amounts or counts, read as numbers

_CURRENCY = re.compile(r"(?:₹|\bRs\.?|\bINR)\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
# Not part of a bigger number like 1,50,000 or 2.5, but punctuation after it is fine.
_PHONE = re.compile(r"(?<!\d)(?<!\d[,.])(\+?\d[\d\s-]{%d,}\d)(?!\d|[,.]\d)" % (MIN_PHONE_DIGITS - 2))
_FINAL_ANUSVARA = re.compile(r"ం(?=[\s.,!?;:।'\"]|$)")
# Before a vowel letter ం only occurs in spelled-out letters like ఈఎంఐ (EMI): said "m", letter by letter.
_ANUSVARA_BEFORE_VOWEL = re.compile(r"ం(?=[అ-ఔ])")


def _say_digits(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(1))
    if len(digits) < MIN_PHONE_DIGITS:
        return match.group(0)
    return " ".join(DIGITS[int(d)] for d in digits)


def _say_date(match: re.Match) -> str:
    day, month, year = (int(g) for g in match.groups())
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return match.group(0)
    return f"{day} {MONTHS[month - 1]} {year}"


def normalize_telugu(text: str) -> str:
    """Rewrite currency, dates, phone numbers and final anusvara into spoken Telugu forms."""
    text = _CURRENCY.sub(lambda m: f"{m.group(1)} {RUPEES}", text)
    text = _DATE.sub(_say_date, text)
    text = _PHONE.sub(_say_digits, text)
    text = _ANUSVARA_BEFORE_VOWEL.sub("మ్ ", text)
    return _FINAL_ANUSVARA.sub("మ్", text)
