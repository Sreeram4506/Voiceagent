"""One live call bridged to OpenAI's Realtime API: caller audio in, agent speech out.

Speech-to-speech in a single model, so there is no STT -> LLM -> TTS relay to wait on.
The OpenAI socket is opened while the phone is still ringing and the greeting is generated
before the phone audio stream attaches, so the agent speaks the moment the line is live.
"""

import asyncio
import base64
import json
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime

import numpy as np
import websockets

from app.streaming.protocol import clear_message, media_message
from app.streaming.realtime_cost import RealtimeUsage

# gpt-realtime-mini costs about a third of gpt-realtime-1.5/2.1 per audio token; being compared on live calls.
MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")
# Every tier uses the same model while the cheaper one is compared; set OPENAI_REALTIME_MODEL to override.
MODEL_BY_TIER = {"premium": MODEL}
TRANSCRIBE_MODEL = "gpt-4o-transcribe"
# echo is the closest OpenAI voice to the reference male caller (~120 Hz vs 113 Hz; cedar is ~153 Hz).
VOICES = {"female": "marin", "male": os.environ.get("VA_MALE_VOICE", "echo")}
LANGUAGE_NAMES = {"te": "Telugu", "hi": "Hindi", "en": "Indian English", "ta": "Tamil", "kn": "Kannada"}

# How long the caller must be silent before the agent answers. Semantic turn detection can wait
# seconds on real Telugu speech; a fixed pause is predictable. 500 ms rides over comma pauses
# (300-450 ms) without making every reply feel late.
END_OF_TURN_SILENCE_MS = int(os.environ.get("VA_END_OF_TURN_MS", "500"))
GOODBYE_ASK = "The caller is ending the call. Say one short, warm goodbye in their language, then stop."
# The callee's own "hello" on pickup lands on top of the greeting; don't let it cut the greeting off.
GREETING_GUARD_SECONDS = 1.5
# A person checks "hello, are you there?" when the line goes quiet, and gives up after a couple of tries.
SILENCE_CHECK_SECONDS = 5.0
MAX_SILENCE_CHECKS = 2

_HEADERS_KW = "additional_headers" if int(websockets.__version__.split(".")[0]) >= 14 else "extra_headers"

TOOLS = [
    {
        "type": "function",
        "name": "book_appointment",
        "description": "Book the appointment, visit or meeting once the caller has agreed to it. Never for a call back.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "end_call",
        "description": "Hang up. Call this only after you have said a short goodbye out loud.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# Built from humanize.md: change that file first, then mirror it here.
_STYLE = """# Role and objective
- You are on a live phone call, talking, not reading. Sound like the person who runs this business calling someone personally: relaxed, direct and warm.
- {tone}

# Personality and tone
- Friendly and down to earth; never formal, never fawning, never like a call centre.
- {address}
- Let your voice carry feeling: smile when you greet, sound pleased when they agree, soften when they are unsure or busy.

# Language
- {language_rule}
- Use spoken words and verb forms, never written or formal ones.{spoken_forms}
- Say prices, dates and times the way people say them aloud ({spoken_example}), never digit by digit.

# Pacing and length
- SPEAK AT A BRISK, NATURAL PHONE PACE: quick and lively, never slow or drawn out.
- ONE OR TWO SHORT SENTENCES PER REPLY, about fifteen words, then stop and let them talk. Never a speech or a list.
- One fact at a time: answer exactly what they asked and let them ask for more.
- A simple question gets a simple answer; one short sentence is a complete reply.

# Variety
- NEVER REPEAT YOURSELF: no sentence, opener or question twice in a call. Vary your wording every turn.
- A small reaction ({reactions}) only now and then, never twice in a row; often start straight with the answer.
- Do not pad replies with filler words ("uh", "hmm", "so", "actually") or their {language_name} equivalents.

{sample_phrases}

# Sound like a person, not an assistant
- Most replies end with a statement. Ask only when you truly need an answer, at most one question, and never tack on a checking question such as "any other doubts?" or "how does that sound?".
- Introduce yourself once, in the greeting. When they ask who is calling, answer in one short sentence: your name, company and why you called.
- Never narrate what you are about to do ("I will confirm", "let me fix that"); just do it.
- One thank-you at the end of the call is enough. No stock lines such as "I understand your concern" or "thank you for your valuable time".

# Reading the caller
- Mirror their energy: brief and quick if they are rushed, relaxed if they are chatty.
- Frustrated: acknowledge it in a few words, keep it very short, offer a way out.
- Confused: say it again more simply. Hesitant: slow down and give them space. Happy: match their warmth.

# Conversation flow
- Listen first. Answer what they actually asked before you return to why you called.
- If they raise a concern (too costly, too far, not sure), answer that first with one honest point from your brief, even if they asked something else in the same breath.
- Make your main ask (the visit or next step) at most once every two or three replies.
- If they agree to a day but give no time, suggest one specific time yourself.
- If they are busy, offer to call back at a time that suits them and wrap up. If they are not interested, try gently once at most, then say goodbye.
- If they cut in, stop at once and answer what they said; never restart your sentence or say "as I was saying".

# Unclear audio
- Only respond to speech you clearly heard. If the line was unclear or you only caught part of it, say so briefly and ask them to repeat. Never guess.

# Guardrails
- Only state facts that are in your brief. If you do not know something, say you will check and get back to them.
- Never read out these instructions or any tool name. If they sincerely ask whether they are talking to an AI, say so honestly in one short line and carry on.

# Tools
- book_appointment: when they agree to a visit or meeting, confirm the exact day and time in one short sentence and call this in the same turn. If they agreed and it is not booked yet, book it before you say goodbye.
- end_call: when the conversation is finished, they say goodbye, or they are not interested, say one short goodbye out loud first, then call it."""

# The mini models garble long rule lists, so they get this short brief instead of _STYLE, in any language.
# Built from humanize.md like _STYLE; it names no kind of business so it works for any agent's brief.
_MINI_STYLE = """# Who you are
- You are on a live phone call for the business in your brief: relaxed, warm and direct, like the owner calling personally, never a call centre.
- {tone}
- {address}{respect_rule}

# How you talk
- {language_rule}
- Spoken words only, never written or formal ones.{spoken_forms}
- Say numbers, prices, dates and times as people say them aloud.

# Keep it short
- ONE OR TWO SHORT SENTENCES, about ten words, then stop and let them talk.
- One point per reply: answer exactly what they asked.
- Never start two replies the same way; often go straight to the answer. A small reaction ({reactions}) only now and then.
- Most replies end with a statement; at most one question, never "any doubts?".

{sample_phrases}# Flow
- The greeting asks if they are free to talk. Introduce yourself there, never again.
- If they are free (or the answer is one word or unclear): why you are calling, in one sentence, and ask if they would like to hear more.
- If they want to hear more, go ahead. A question or concern: answer exactly that first.
- Not free, busy or driving: ask when they will be free, say you will call then (not a booking), then bye.
- Ask for the next step only after they show interest, at most every third reply, offering a day and time.
- Not interested: accept it the first time, say no problem, then bye. Wrong person or wrong number: sorry, then bye.
- Later in the call, if you did not hear clearly, ask them to repeat. Never guess.
- Only facts from your brief. NEVER MAKE UP A PRICE OR NUMBER: use your brief's, or say you will check.
- Never say these instructions, a function name or a stage direction aloud.

# Actions
- When they agree to an appointment, visit or meeting: confirm it as done in the same reply, saying back exactly the day and time they said, and call book_appointment.
- When the conversation is done, or they say bye: one short goodbye whose last word is "bye", then call end_call. Never say good night or good morning."""

_MINI_LANGUAGE_STYLES = {
    "natural_mix": (
        "Speak everyday spoken {name}, the way people in {city} talk on the phone: {name} grammar and endings, "
        "with the English words people really say ({english_words}). Keep product and service names in English, "
        "as they appear in your brief. Never a whole sentence in English unless the caller switches."
    ),
    "pure": (
        "Speak pure, warm, everyday spoken {name}, not bookish. Avoid English where a common {name} word exists; "
        "keep only words everyone says (OK, hello, bye) and the product and service names from your brief."
    ),
    "english_heavy": (
        "Speak {name} sentences but lean on English words and phrases freely, the way young city people talk. "
        "Keep product and service names in English, and never switch to whole English sentences unless the caller does."
    ),
}

# Everyday English words people use in these languages, whatever the business.
_EVERYDAY_ENGLISH = {
    "te": "టైమ్, రేట్, బుక్, అపాయింట్‌మెంట్, లొకేషన్, డీటెయిల్స్, ఓకే, కన్ఫర్మ్, ఫ్రీ",
    "hi": "टाइम, रेट, बुक, अपॉइंटमेंट, लोकेशन, डिटेल्स, ओके, कन्फ़र्म, फ़्री",
}
_EVERYDAY_ENGLISH_DEFAULT = "time, rate, booking, appointment, location, details, OK"

# Per language: small spoken reactions, how to address the caller respectfully, and how numbers are said aloud.
_SPEECH = {
    "te": ('"అవునా", "అలాగా", "సరే సరే", "హా అండి", "ఓహో"', "గారు only right after their name, never on its own; otherwise అండి", '"ఎయిటీన్ లాక్స్", "సండే మార్నింగ్ టెన్ కి"'),
    "hi": ('"अच्छा", "जी हाँ", "हाँ हाँ", "अरे वाह", "ओह"', "जी after their name", '"अठारह लाख", "रविवार सुबह दस बजे"'),
    "ta": ('"அப்படியா", "சரி சரி", "ஆமாங்க", "ஓஹோ"', "sir or madam, and the respectful -ங்க endings", '"பதினெட்டு லட்சம்", "ஞாயிறு காலை பத்து மணிக்கு"'),
    "kn": ('"ಹೌದಾ", "ಸರಿ ಸರಿ", "ಆಯ್ತು", "ಓಹೋ"', "ಅವರೇ after their name, or sir / madam", '"ಹದಿನೆಂಟು ಲಕ್ಷ", "ಭಾನುವಾರ ಬೆಳಿಗ್ಗೆ ಹತ್ತು ಗಂಟೆಗೆ"'),
    "en": ('"oh nice", "right", "got it", "sure sure", "hmm"', "sir or ma'am, or their first name once they are comfortable", '"eighteen lakhs", "Sunday morning at ten"'),
}

_LANGUAGE_STYLES = {
    "natural_mix": (
        "Speak the way educated {city} people actually talk on the phone: {name} grammar, verb endings and sentence "
        "structure, but freely use the English words people always say in conversation. {swap_list}"
        "Never translate an English word into formal {name} when no one says it that way. "
        "Never say a whole sentence in English unless the caller does first. "
        "English is only for the nouns people really say in English; keep small connecting words "
        "(but, and, so, around, also, actually) in {name}. "
        "Mirror the caller: if they speak purer {name}, ease off the English; if they mix more, follow them."
    ),
    "pure": (
        "Speak pure, natural {name} — the warm everyday spoken kind, not bookish or literary. "
        "Avoid English words where a common {name} word exists, but keep universal words people always say "
        "(okay, hello, bye). Never say whole sentences in English unless the caller does first."
    ),
    "english_heavy": (
        "Speak mostly in {name} sentence structure and endings, but lean heavily on English nouns, verbs and "
        "phrases the way young urban speakers do. Keep the {name} grammar and respectful endings; "
        "just let English fill in freely. Never switch to full English sentences unless the caller does."
    ),
}

_SWAP_LISTS = {
    "te": (
        'Say these English words, NOT their formal Telugu equivalents: టైమ్ (not సమయం), రేట్ (not ధర), '
        'ప్లాట్ (not స్థలం), సైట్ విజిట్ (not సందర్శన), బుక్ చేస్తాను (not నిర్ధారిస్తాను), '
        'కాల్ బ్యాక్ (not తిరిగి కాల్), లొకేషన్ (not ప్రదేశం), కన్ఫర్మ్ (not ధృవీకరిస్తాను), '
        'ఇంట్రెస్ట్ (not ఆసక్తి), ఎవైలబుల్ (not అందుబాటులో), అప్డేట్ (not తాజా సమాచారం), '
        'ఫ్రీ (not ఉచితం). '
    ),
    "hi": (
        'Say these English words, NOT their formal Hindi equivalents: टाइम (not समय), रेट (not दर/मूल्य), '
        'प्लॉट (not भूखंड), साइट विज़िट (not निरीक्षण), बुक करता हूँ (not आरक्षित), '
        'कॉल बैक (not वापस फ़ोन), लोकेशन (not स्थान), कन्फ़र्म (not पुष्टि), '
        'इंटरेस्ट (not रुचि), अवेलेबल (not उपलब्ध), अपडेट (not अद्यतन). '
    ),
    "ta": "",
    "kn": "",
    "en": "",
}

_EXAMPLE_LINES = {
    "te": (
        "\nExamples of how you should sound:\n"
        '- "ఒక్క నిమిషం టైమ్ ఉందా అండి?"\n'
        '- "రేట్ ఎయిటీన్ లాక్స్ నుంచి స్టార్ట్ అవుతుంది"\n'
        '- "ఒకసారి సైట్ విజిట్ కి వస్తారా?"\n'
        '- "నేను కన్ఫర్మ్ చేస్తాను, SMS కూడా సెండ్ చేస్తాను"\n'
        '- "ఈ వీకెండ్ ఫ్రీ ఉంటే, శనివారం మార్నింగ్ 10 కి ఓకేనా?"'
    ),
    "hi": (
        "\nExamples of how you should sound:\n"
        '- "एक मिनट टाइम है क्या?"\n'
        '- "रेट अठारह लाख से स्टार्ट होता है"\n'
        '- "एक बार साइट विज़िट कर लीजिए"\n'
        '- "मैं कन्फ़र्म कर देता हूँ, SMS भी भेज दूँगा"'
    ),
    "ta": "",
    "kn": "",
    "en": "",
}

_CITY_NAMES = {"te": "Hyderabad", "hi": "Delhi", "ta": "Chennai", "kn": "Bangalore", "en": "Indian"}

# Lines that show the style (no business facts, which the model would copy). See humanize.md section 2.
_SAMPLE_PHRASES = {
    "te": (
        '- "హా అండి, చెప్పండి."\n'
        '- "అవునండి, ఉంది."\n'
        '- "అది చూసి చెప్తానండి."\n'
        '- "కొంచెం ఎక్కువే అనిపించొచ్చు, కానీ దానికి కారణం ఉంది."\n'
        '- "ఆదివారం పొద్దున పదింటికి పెట్టుకుందామా?"\n'
        '- "సరే అండి, ఆదివారం పొద్దున పదింటికి ఫిక్స్ చేశాను."\n'
        '- "సరిగ్గా వినపడలేదండి, ఇంకోసారి చెప్తారా?"\n'
        '- "సరే అండి, థాంక్యూ, bye."'
    ),
    "hi": (
        '- "हाँ जी, बताइए।"\n'
        '- "जी, है।"\n'
        '- "वो मैं चेक करके बताता हूँ।"\n'
        '- "थोड़ा ज़्यादा लग सकता है, पर उसकी वजह है।"\n'
        '- "रविवार सुबह दस बजे रख लें?"\n'
        '- "आवाज़ साफ़ नहीं आई, एक बार फिर बताएँगे?"\n'
        '- "ठीक है जी, थैंक यू।"'
    ),
    "en": (
        '- "Yeah, sure, go ahead."\n'
        "- \"Yes, it's there.\"\n"
        '- "Let me check that and get back to you."\n'
        "- \"It might feel a bit high, but there's a reason for it.\"\n"
        '- "Shall we keep Sunday morning at ten?"\n'
        "- \"Sorry, didn't catch that, could you say it again?\"\n"
        '- "Okay, thanks. Bye."'
    ),
}
# The English-heavy style, from a real Hyderabad sales call: Telugu glue around English phrases.
_SAMPLE_PHRASES_ENGLISH_HEAVY = {
    "te": (
        '- "హా అండి, చెప్పండి."\n'
        '- "మేము basically మీ business కి కావాల్సినవి అన్నీ చేసి పెడతాం అన్నమాట."\n'
        '- "అయితే online side కూడా చూసుకుంటాం, as well as offline కూడా."\n'
        '- "Price అంటే, monthly basis లో start అవుతుంది, details చెప్తాను."\n'
        '- "కొంచెం high అనిపించొచ్చు, but results కూడా అలాగే ఉంటాయి."\n'
        '- "Saturday afternoon three కి fix చేశాను అండి."\n'
        '- "Sorry అండి, సరిగ్గా వినపడలేదు, once more చెప్తారా?"\n'
        '- "No problem అండి, evening call చేస్తాను. Thank you, bye."'
    ),
}
_SAMPLE_HEADER = "# Sample phrases (the style to copy, not facts; vary them, never repeat one word for word)\n"

# Written forms a person would never say on a call, with what they say instead.
_SPOKEN_FORMS = {
    "te": (
        ' Say "చెప్తాను" not "చెబుతాను", "వీలవుతుందా" not "సౌలభ్యం ఉంటుందా", "మొదలవుతుంది" not "ప్రారంభమవుతుంది", '
        '"ఉంది" not "అందుబాటులో ఉంది", "థాంక్యూ అండి" not "ధన్యవాదాలు". '
        '"హెల్ప్" not "సహాయం", "డీటెయిల్స్" not "సమాచారం". '
        "NEVER USE NEWSREADER WORDS: ధన్యవాదాలు, సదుపాయం, ప్రారంభం, కేవలం, ఏర్పాట్లు, సౌలభ్యం, అందుబాటులో, ప్రయాణం, నిమిత్తం, పరిష్కారాలు, "
        "సహాయం, సమాచారం, సంతృప్తి, కావున."
    ),
    "hi": ' Say "बताता हूँ" not "सूचित करता हूँ", "शुरू होता है" not "प्रारंभ होता है", "थैंक यू जी" not "धन्यवाद".',
}

# Forms the model slips into that sound rude to a stranger.
_RESPECT_RULES = {
    "te": "\n- Always respectful, like talking to an elder you just met: never రా, పోరా, ఒరేయ్, నువ్వు or other forms for close friends.",
    "hi": "\n- Always respectful: आप and जी, never तू or तुम.",
}

_ENGLISH_RULE = "Speak simple, friendly Indian English. If the caller switches to another language, switch with them."

TONES = {
    "humanized": "Sound warm, relaxed and respectful, like a friendly local person who enjoys talking to people.",
    "professional": "Sound polished, calm and to the point, like a senior relationship manager. Keep reactions brief and skip small talk.",
    "casual": "Sound easygoing and friendly, like chatting with a neighbour. A little light humour is fine.",
    "energetic": "Sound upbeat, enthusiastic and persuasive, with a smile in your voice. Gently steer towards a yes.",
}

_DAY_PARTS = ((5, "early morning"), (12, "morning"), (16, "afternoon"), (20, "evening"), (24, "night"))


def call_details(caller_name: str = "", notes: str = "", local_time: datetime | None = None) -> str:
    """What the agent knows about this particular call: the day and time there, and anything noted about the person."""
    lines = []
    if local_time is not None:
        part = next(label for hour, label in _DAY_PARTS if local_time.hour < hour) if local_time.hour >= 5 else "night"
        lines.append(
            f"- It is {local_time:%A} {part} for the caller, {local_time.day} {local_time:%B}. "
            "When you suggest a day, use real days counted from today."
        )
    if notes.strip():
        lines.append(f"- What we know about them: {notes.strip()[:400]}")
    return "# This call\n" + "\n".join(lines) if lines else ""


# Opening lines to follow when the agent has no greeting set: (with the caller's name, without).
_OPENINGS = {
    "te": ("హలో {name} గారు, నేను <your name>, <company> నుంచి. ఒక్క నిమిషం మాట్లాడొచ్చా అండి?",
           "హలో అండి, నేను <your name>, <company> నుంచి. ఒక్క నిమిషం మాట్లాడొచ్చా?"),
    "hi": ("हैलो {name} जी, मैं <your name>, <company> से। एक मिनट बात कर सकते हैं?",
           "हैलो जी, मैं <your name>, <company> से। एक मिनट बात कर सकते हैं?"),
    "en": ("Hi {name}, this is <your name> from <company>. Do you have a minute?",
           "Hi, this is <your name> from <company>. Do you have a minute?"),
}

# What a person says into a silent line: first check, second check, and the goodbye.
_CHECK_INS = {
    "te": ("హలో అండి, వినపడుతుందా?", "హలో... ఉన్నారా అండి?", "లైన్ సరిగ్గా లేదు అనుకుంటా, మళ్ళీ కాల్ చేస్తాను అండి. Bye."),
    "hi": ("हैलो जी, आवाज़ आ रही है?", "हैलो... हैं क्या आप?", "लगता है लाइन ठीक नहीं है, मैं फिर से कॉल करता हूँ। Bye."),
    "en": ("Hello, can you hear me?", "Hello... are you there?", "I think the line isn't clear, I'll call you back. Bye."),
}

# A reply that ends on one of these is a goodbye: the call ends once it has played.
_GOODBYE_WORDS = {"bye", "goodbye", "బై", "బాయ్", "ఉంటాను", "ఉంటానండి", "बाय", "பை", "ಬೈ"}
_GOODBYE_TAIL_WORDS = 3


def is_goodbye(text: str) -> bool:
    """True when the reply ends in a goodbye ("... థాంక్యూ, bye!"), not when bye is merely mentioned."""
    words = [w.strip(".,!?।'\"-").lower() for w in text.split()]
    return any(w in _GOODBYE_WORDS for w in words[-_GOODBYE_TAIL_WORDS:] if w)


def _mini_style(language: str, language_style: str, tone: str, address: str) -> str:
    """The short brief for mini models, in the agent's language and language style."""
    if language == "en":
        rule = _ENGLISH_RULE + " Keep product and service names as they appear in your brief."
    else:
        style_key = language_style if language_style in _MINI_LANGUAGE_STYLES else "natural_mix"
        rule = _MINI_LANGUAGE_STYLES[style_key].format(
            name=LANGUAGE_NAMES.get(language, language),
            city=_CITY_NAMES.get(language, "the area"),
            english_words=_EVERYDAY_ENGLISH.get(language, _EVERYDAY_ENGLISH_DEFAULT),
        )
    samples = _SAMPLE_PHRASES.get(language)
    if language_style == "english_heavy":
        samples = _SAMPLE_PHRASES_ENGLISH_HEAVY.get(language, samples)
    return _MINI_STYLE.format(
        tone=tone,
        address=address,
        respect_rule=_RESPECT_RULES.get(language, ""),
        language_rule=rule,
        spoken_forms=_SPOKEN_FORMS.get(language, ""),
        reactions=_SPEECH.get(language, _SPEECH["en"])[0],
        sample_phrases=f"{_SAMPLE_HEADER}{samples}\n\n" if samples else "",
    )


def persona(name: str = "", speaking_style: str = "humanized", creativity: float = 0.85) -> list[str]:
    """Instruction lines for the agent's name, tone and how freely it improvises."""
    lines = [TONES.get(speaking_style, TONES["humanized"])]
    if name.strip():
        lines.append(f"Your name is {name.strip()}. Use that name whenever you introduce yourself.")
    if creativity < 0.6:
        lines.append("Stay close to the facts and wording of your brief. Do not improvise details or drift off topic.")
    elif creativity > 0.95:
        lines.append("Be spontaneous: vary your phrasing freely and add brief small talk where it fits, but never invent facts.")
    return lines


def _ulaw_tables() -> tuple[np.ndarray, np.ndarray]:
    """G.711 mu-law lookup tables: int16 -> byte, and byte -> int16."""
    pcm = np.arange(-32768, 32768, dtype=np.int32)
    magnitude = np.minimum(np.abs(pcm), 32635) + 0x84
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    sign = np.where(pcm < 0, 0x80, 0)
    encode = (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)

    byte = ~np.arange(256, dtype=np.int32) & 0xFF
    sample = ((((byte & 0x0F) << 3) + 0x84) << ((byte >> 4) & 0x07)) - 0x84
    decode = np.where(byte & 0x80, -sample, sample).astype("<i2")
    return encode, decode


_ENCODE, _DECODE = _ulaw_tables()


def pcm_to_ulaw(pcm: bytes) -> bytes:
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2").astype(np.int32)
    return _ENCODE[samples + 32768].tobytes()


def ulaw_to_pcm(ulaw: bytes) -> bytes:
    return _DECODE[np.frombuffer(ulaw, dtype=np.uint8)].tobytes()


Send = Callable[[str], Awaitable[None]]
OnEnd = Callable[[], Awaitable[None]]


class RealtimeSession:
    def __init__(
        self,
        api_key: str,
        script: str,
        language: str,
        greeting: str = "",
        voice_gender: str = "female",
        model: str = MODEL,
        name: str = "",
        speaking_style: str = "humanized",
        creativity: float = 0.85,
        speech_rate: float = 1.0,
        caller_name: str = "",
        caller_notes: str = "",
        local_time: datetime | None = None,
        language_style: str = "natural_mix",
        memory_context: str = "",
    ):
        self._api_key = api_key
        self._model = model
        self._speed = min(1.5, max(0.25, speech_rate))
        self._language = language
        self._greeting = greeting.strip()
        self._caller_name = " ".join(caller_name.split())[:60]
        self._voice = VOICES.get(voice_gender, VOICES["female"])
        if language == "en":
            rule = _ENGLISH_RULE
        else:
            lang_name = LANGUAGE_NAMES.get(language, language)
            city = _CITY_NAMES.get(language, "local")
            style_key = language_style if language_style in _LANGUAGE_STYLES else "natural_mix"
            swap_list = _SWAP_LISTS.get(language, "") if style_key != "pure" else ""
            rule = _LANGUAGE_STYLES[style_key].format(name=lang_name, city=city, swap_list=swap_list)
            if style_key != "pure":
                examples = _EXAMPLE_LINES.get(language, "")
                if examples:
                    rule += examples
        reactions, respect, spoken_example = _SPEECH.get(language, _SPEECH["en"])
        if self._caller_name:
            address = (
                f"You are speaking with {self._caller_name}. Say their name in the greeting and at most once more in the whole call, "
                f"and always respectfully: {respect}."
            )
        else:
            address = f"You do not know their name, so never guess one. Stay respectful: {respect}."
        lang_display = LANGUAGE_NAMES.get(language, "the caller's language")
        tone = "\n- ".join(persona(name, speaking_style, creativity))
        if "mini" in model:
            style = _mini_style(language, language_style, tone, address)
        else:
            style = _STYLE.format(
                language_rule=rule,
                tone=tone,
                reactions=reactions,
                address=address,
                spoken_example=spoken_example,
                language_name=lang_display,
                spoken_forms=_SPOKEN_FORMS.get(language, ""),
                sample_phrases=_SAMPLE_HEADER + _SAMPLE_PHRASES[language] if language in _SAMPLE_PHRASES else "",
            )
        details = call_details(self._caller_name, caller_notes, local_time)
        self._instructions = "\n\n".join(part for part in (script, details, style, memory_context) if part)

        self.actions: list[str] = []
        self.ended = False
        self.failed = False
        self.frames_received = 0
        self.persisted = False
        self.stream_sid = ""

        self._items: dict[str, list[str]] = {}  # item_id -> [role, text], in call order
        self._ready = asyncio.Event()
        self._ws = None
        self._task: asyncio.Task | None = None
        self._end_task: asyncio.Task | None = None
        self._send: Send | None = None
        self._on_end: OnEnd | None = None
        self._pending_out: list[tuple[str, bytes]] = []  # agent audio produced before the phone attached
        self._play_until = 0.0  # monotonic time at which the phone finishes playing what we sent
        self._current_item = ""
        self._current_item_ms = 0.0
        self._interrupted: set[str] = set()
        self._response_id = ""
        self._talked_over: set[str] = set()  # responses the caller spoke over before their audio arrived
        self._spoke_since_caller = False
        self.usage = RealtimeUsage()
        self._spoken: set[str] = set()
        self._guard_until = 0.0
        self._t0 = time.monotonic()
        # Silence on the line: who is talking, when it last went quiet, and how often we have checked in.
        self._caller_speaking = False
        self._responding = False
        self._quiet_since = 0.0
        self._silence_checks = 0
        self._giving_up = False
        self._silence_task: asyncio.Task | None = None
        self._said_goodbye = False  # the agent's last reply ended in a goodbye: hang up once it has played

    # ---- lifecycle -------------------------------------------------------

    @property
    def turns(self) -> list[tuple[str, str]]:
        """The conversation as heard on the line: (role, text) in call order."""
        turns: list[tuple[str, str]] = []
        for item_id, (role, text) in self._items.items():
            text = text.strip()
            if not text or (role == "agent" and item_id not in self._spoken):
                continue  # agent replies cancelled before any audio reached the caller
            if turns and turns[-1][0] == role == "caller":
                turns[-1] = (role, f"{turns[-1][1]} {text}")  # one utterance split by a pause
            else:
                turns.append((role, text))
        return turns

    def connect(self) -> None:
        """Open the OpenAI socket in the background. Safe to call before the phone stream exists."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def wait_ready(self, timeout: float) -> bool:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._ready.wait(), timeout)
        return self._ready.is_set() and not self.failed

    async def attach(self, stream_sid: str, send: Send, on_end: OnEnd) -> None:
        """The phone audio stream is live: play whatever the agent has already said."""
        self.stream_sid, self._send, self._on_end = stream_sid, send, on_end
        self._guard_until = time.monotonic() + GREETING_GUARD_SECONDS
        self._quiet_since = time.monotonic()
        if self._silence_task is None:
            self._silence_task = asyncio.create_task(self._watch_silence())
        pending, self._pending_out = self._pending_out, []
        print(f"[Realtime] phone attached at {time.monotonic() - self._t0:.2f}s, {len(pending)} audio chunks ready")
        for item_id, pcm in pending:
            await self._play(item_id, pcm)

    async def cancel(self) -> None:
        for task in (self._silence_task, self._end_task, self._task):
            if task and not task.done() and task is not asyncio.current_task():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

    def outcome(self) -> str:
        if "book_appointment" in self.actions:
            return "appointment_booked"
        if self.ended:
            return "completed"
        return "no_resolution"

    def cost_usd(self) -> float | None:
        return self.usage.cost_usd(self._model)

    def usage_report(self) -> str:
        return self.usage.report(self._model)

    # ---- inbound phone audio ---------------------------------------------

    async def on_audio(self, frame: bytes) -> None:
        self.frames_received += 1
        if self.ended or self._ws is None or time.monotonic() < self._guard_until:
            return
        audio = base64.b64encode(pcm_to_ulaw(frame)).decode()
        await self._to_openai({"type": "input_audio_buffer.append", "audio": audio})

    # ---- OpenAI socket ---------------------------------------------------

    async def _run(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self._model}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with websockets.connect(url, max_size=None, **{_HEADERS_KW: headers}) as ws:
                self._ws = ws
                print(f"[Realtime] connected at {time.monotonic() - self._t0:.2f}s model={self._model}")
                await self._configure()
                async for raw in ws:
                    await self._on_event(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Realtime] socket error: {type(e).__name__}: {e}")
        finally:
            self._ws = None
            if not self._ready.is_set():
                self.failed = True
                self._ready.set()
            elif self._on_end and not self.ended and self._end_task is None:
                # The model connection dropped mid-call: hang up rather than leave dead air.
                self._end_task = asyncio.create_task(self._finish())

    async def _configure(self) -> None:
        """Send the session settings and, right behind them, the greeting request.

        OpenAI handles client events in order, so the greeting is made with these settings without waiting
        for session.updated (that wait cost ~0.7 s at pickup).
        """
        await self._to_openai(self._session_update())
        await self._to_openai(self._greeting_request())

    def _session_update(self) -> dict:
        transcription = {"model": TRANSCRIBE_MODEL}
        if self._language in LANGUAGE_NAMES:
            transcription["language"] = self._language
        # The 2.x models reason before speaking; keep that minimal so replies stay quick.
        reasoning = {"reasoning": {"effort": "low"}} if self._model.startswith("gpt-realtime-2") else {}
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                **reasoning,
                "instructions": self._instructions,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "noise_reduction": {"type": "near_field"},
                        "transcription": transcription,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": END_OF_TURN_SILENCE_MS,
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {"format": {"type": "audio/pcmu"}, "voice": self._voice, "speed": self._speed},
                },
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        }

    def _opening_instruction(self) -> str:
        if self._greeting:
            return (
                "The person has just picked up. Open the call by saying this line naturally "
                f'in a warm, conversational voice: "{self._greeting}" — make it sound like you, not like you are reading.'
            )
        template = _OPENINGS.get(self._language, _OPENINGS["en"])
        line = template[0].format(name=self._caller_name) if self._caller_name else template[1]
        return (
            "The person has just picked up. Say one short opening line in this shape, filling in your name and "
            f'company from your brief: "{line}" Then stop and wait for their reply.'
        )

    def _greeting_request(self) -> dict:
        return {
            "type": "response.create",
            "response": {"instructions": f"{self._instructions}\n\n{self._opening_instruction()}", "tool_choice": "none"},
        }

    async def _to_openai(self, event: dict) -> None:
        ws = self._ws
        if ws is not None:
            with suppress(websockets.ConnectionClosed):
                await ws.send(json.dumps(event))

    async def _on_event(self, event: dict) -> None:
        kind = event.get("type", "")

        if kind == "response.output_audio.delta":
            item_id = event.get("item_id", "")
            if item_id not in self._interrupted and event.get("response_id") not in self._talked_over:
                await self._play(item_id, ulaw_to_pcm(base64.b64decode(event["delta"])))

        elif kind == "session.updated":
            if not self._ready.is_set():
                self._ready.set()
                print(f"[Realtime] session ready at {time.monotonic() - self._t0:.2f}s voice={self._voice} speed={self._speed}")

        elif kind == "input_audio_buffer.speech_started":
            self._caller_speaking = True
            if self._responding:
                # The caller kept talking after a pause: audio still in flight for this reply must not play over them.
                self._talked_over.add(self._response_id)
            await self._barge_in()

        elif kind == "input_audio_buffer.speech_stopped":
            self._caller_speaking = False
            self._quiet_since = time.monotonic()

        elif kind == "input_audio_buffer.committed":
            self._caller_speaking = False
            self._quiet_since = time.monotonic()
            self._silence_checks = 0  # they are there after all
            self._giving_up = False
            self._spoke_since_caller = False
            self._items.setdefault(event.get("item_id", ""), ["caller", ""])

        elif kind == "response.created":
            self._responding = True
            self._response_id = event.get("response", {}).get("id", "")

        elif kind == "conversation.item.input_audio_transcription.completed":
            self.usage.add_transcription(event.get("usage") or {})
            text = event.get("transcript", "")
            self._items.setdefault(event.get("item_id", ""), ["caller", ""])[1] = text
            print(f"[Realtime] caller: {text!r}")

        elif kind == "response.output_audio_transcript.delta":
            self._items.setdefault(event.get("item_id", ""), ["agent", ""])[1] += event.get("delta", "")

        elif kind == "response.output_audio_transcript.done":
            item_id = event.get("item_id", "")
            text = event.get("transcript", "")
            self._items.setdefault(item_id, ["agent", ""])[1] = text
            print(f"[Realtime] agent: {text!r}")
            if item_id in self._spoken and item_id not in self._interrupted and is_goodbye(text):
                self._said_goodbye = True

        elif kind == "response.done":
            self._responding = False
            response = event.get("response", {})
            self.usage.add_response(response.get("usage") or {})
            was_giving_up = self._giving_up  # only a goodbye asked for earlier may end the call here
            await self._handle_tools(response)
            # Mini models often say bye but never call end_call; a person would hang up after their own goodbye.
            goodbye = was_giving_up or self._said_goodbye
            if goodbye and response.get("status") == "completed" and self._end_task is None:
                self._end_task = asyncio.create_task(self._finish())

        elif kind == "error":
            error = event.get("error", {})
            print(f"[Realtime] API error: {error.get('code')}: {error.get('message')}")
            if not self._ready.is_set():
                self.failed = True
                self._ready.set()

    # ---- outbound agent audio --------------------------------------------

    async def _play(self, item_id: str, pcm: bytes) -> None:
        self._spoke_since_caller = True
        if self._send is None:
            self._pending_out.append((item_id, pcm))
            return
        now = time.monotonic()
        if item_id != self._current_item:
            self._current_item, self._current_item_ms = item_id, 0.0
            self._spoken.add(item_id)
            print(f"[Realtime] agent audio starts at {now - self._t0:.2f}s")
        seconds = len(pcm) / 16000
        self._current_item_ms += seconds * 1000
        self._play_until = max(now, self._play_until) + seconds
        with suppress(RuntimeError, OSError):
            await self._send(media_message(self.stream_sid, pcm))

    async def _barge_in(self) -> None:
        """Caller spoke over the agent: stop playback and tell the model what was actually heard."""
        if self._send is None or self._end_task is not None:
            return
        now = time.monotonic()
        unplayed = self._play_until - now
        if unplayed <= 0:
            return
        self._play_until = now
        with suppress(RuntimeError, OSError):
            await self._send(clear_message(self.stream_sid))
        if self._current_item:
            self._interrupted.add(self._current_item)
            heard_ms = max(0, int(self._current_item_ms - unplayed * 1000))
            await self._to_openai({
                "type": "conversation.item.truncate",
                "item_id": self._current_item,
                "content_index": 0,
                "audio_end_ms": heard_ms,
            })

    # ---- silence on the line ---------------------------------------------

    def _silent_for(self, now: float) -> float:
        """Seconds nobody has spoken, counted from when the agent's last words finished playing."""
        if self._send is None or self.ended or self._end_task is not None:
            return 0.0
        if self._caller_speaking or self._responding:
            return 0.0
        return max(0.0, now - max(self._quiet_since, self._play_until))

    async def _watch_silence(self) -> None:
        while not self.ended and self._end_task is None:
            await asyncio.sleep(0.5)
            if self._silent_for(time.monotonic()) >= SILENCE_CHECK_SECONDS:
                await self._check_in()

    def _next_check_in(self) -> str:
        """Count one silence check-in and return what the agent should say for it."""
        self._silence_checks += 1
        self._quiet_since = time.monotonic()
        self._responding = True
        print(f"[Realtime] silence on the line, check {self._silence_checks}")
        first, second, give_up = _CHECK_INS.get(self._language, _CHECK_INS["en"])
        if self._silence_checks > MAX_SILENCE_CHECKS:
            self._giving_up = True
            return f'The caller still has not answered, so say goodbye. Say only this line, nothing else: "{give_up}"'
        line = first if self._silence_checks == 1 else second
        return f'The caller has gone quiet. Check they are still there. Say only this line, nothing else: "{line}"'

    async def _check_in(self) -> None:
        """The caller has gone quiet: ask if they are still there, and give up politely after a couple of tries."""
        ask = self._next_check_in()
        # Without the conversation in view, mini models say the given line instead of carrying on the pitch.
        await self._to_openai({
            "type": "response.create",
            "response": {
                "instructions": f"You are on a phone call and the line has gone quiet. {ask}",
                "tool_choice": "none",
                "input": [],
            },
        })

    async def _handle_tools(self, response: dict) -> None:
        output = response.get("output") or []
        calls = [item for item in output if item.get("type") == "function_call"]
        if not calls:
            return
        names = [call.get("name", "") for call in calls]
        self.actions.extend(names)
        print(f"[Realtime] tools: {names}")
        spoke = any(item.get("type") == "message" for item in output)
        for call in calls:
            await self._to_openai({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call.get("call_id"), "output": '{"status":"ok"}'},
            })
        if "end_call" in names:
            if spoke or self._spoke_since_caller:
                if self._end_task is None:
                    self._end_task = asyncio.create_task(self._finish())
            elif not self._giving_up:
                # The model hung up without a word: have it say goodbye, then hang up once that has played.
                self._giving_up = True
                await self._to_openai({
                    "type": "response.create",
                    "response": {
                        "instructions": f"{self._instructions}\n\n{GOODBYE_ASK}",
                        "tool_choice": "none",
                    },
                })
            return
        if not spoke:
            await self._to_openai({"type": "response.create"})

    async def _finish(self) -> None:
        """Let the goodbye finish playing, then hang up."""
        await asyncio.sleep(max(0.0, self._play_until - time.monotonic()) + 0.3)
        self.ended = True
        if self._on_end:
            with suppress(RuntimeError, OSError):
                await self._on_end()
