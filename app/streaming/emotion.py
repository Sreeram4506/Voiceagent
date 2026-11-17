"""Lightweight emotion detection from caller transcript text.

Detects the caller's likely emotional state from their words so the agent
can adapt its tone and pacing in real time.
"""

import re
from dataclasses import dataclass


@dataclass
class EmotionState:
    primary: str       # neutral, frustrated, interested, confused, impatient, happy
    confidence: float  # 0.0–1.0
    suggestion: str    # how the agent should adapt

    @property
    def prompt_hint(self) -> str:
        if self.primary == "neutral" or self.confidence < 0.3:
            return ""
        return f"[The caller sounds {self.primary}. {self.suggestion}]"


_PATTERNS: dict[str, re.Pattern] = {
    "frustrated": re.compile(
        r"\b(no|nahi|naa|vaddu|don'?t|stop|already told|again|waste|irritat|bore|enough|fed up|problem"
        r"|annoyed|angry|ridiculous|terrible|worst|bad|wrong|disappointed|frustrated)\b", re.I,
    ),
    "interested": re.compile(
        r"\b(yes|haa|avunu|tell me|interested|how much|when|where|details|more|show me|visit|book|want"
        r"|sounds good|okay|sure|definitely|please|cheppandi|cheppu)\b", re.I,
    ),
    "confused": re.compile(
        r"\b(what\??|sorry\??|huh|didn'?t understand|repeat|pardon|meaning|which one|confused|unclear"
        r"|what do you mean|ardam kaale|samajh nahi|entandi)\b", re.I,
    ),
    "impatient": re.compile(
        r"\b(busy|quick|hurry|short|fast|no time|later|call back|not now|minute only|brief|wrap up"
        r"|come to the point|jaldi|time ledu)\b", re.I,
    ),
    "happy": re.compile(
        r"\b(great|wonderful|perfect|excellent|super|nice|good|happy|thanks|thank you|awesome|amazing"
        r"|love it|fantastic|bagundi|bahut accha|dhanyavaad)\b", re.I,
    ),
}

_SUGGESTIONS = {
    "frustrated": "Be extra empathetic, acknowledge their feeling, keep your reply very short.",
    "interested": "Match their enthusiasm, give the details they want, suggest a next step.",
    "confused": "Slow down, rephrase simply, ask which part was unclear.",
    "impatient": "Be very brief and direct, offer to call back at a better time.",
    "happy": "Match their energy, reinforce the positive moment, good time to suggest next steps.",
}


def detect_emotion(text: str) -> EmotionState:
    if not text or not text.strip():
        return EmotionState("neutral", 0.0, "")

    scores = {name: len(pat.findall(text)) for name, pat in _PATTERNS.items()}

    words = len(text.split())
    if words <= 2 and scores["interested"] == 0 and scores["happy"] == 0 and any(scores[k] > 0 for k in scores):
        scores["impatient"] += 1

    top = max(scores, key=scores.__getitem__)
    score = scores[top]

    if score == 0:
        return EmotionState("neutral", 0.0, "")

    return EmotionState(top, min(1.0, score / 3), _SUGGESTIONS[top])
