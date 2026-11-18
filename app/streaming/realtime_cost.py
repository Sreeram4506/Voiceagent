"""What one Realtime call used, from the token counts OpenAI reports, and what that costs."""

import os
from dataclasses import dataclass

# USD per 1M tokens: text in, cached text in, text out, audio in, cached audio in, audio out.
# From https://developers.openai.com/api/docs/pricing (Nov 2026); update when OpenAI changes them.
REALTIME_PRICES = {
    "gpt-realtime": (4.00, 0.40, 16.00, 32.00, 0.40, 64.00),
    "gpt-realtime-1.5": (4.00, 0.40, 16.00, 32.00, 0.40, 64.00),
    "gpt-realtime-2": (4.00, 0.40, 24.00, 32.00, 0.40, 64.00),
    "gpt-realtime-2.1": (4.00, 0.40, 24.00, 32.00, 0.40, 64.00),
    "gpt-realtime-mini": (0.60, 0.06, 2.40, 10.00, 0.30, 20.00),
    "gpt-realtime-2.1-mini": (0.60, 0.06, 2.40, 10.00, 0.30, 20.00),
}
# Caller transcription (gpt-4o-transcribe), USD per 1M tokens: input, output.
TRANSCRIBE_PRICES = (2.50, 10.00)
USD_TO_INR = float(os.environ.get("VA_USD_INR", "88"))


@dataclass
class RealtimeUsage:
    text_in: int = 0
    cached_text_in: int = 0
    text_out: int = 0
    audio_in: int = 0
    cached_audio_in: int = 0
    audio_out: int = 0
    transcribe_in: int = 0
    transcribe_out: int = 0
    replies: int = 0

    def add_response(self, usage: dict) -> None:
        """Add the usage block of one response.done event."""
        given = usage.get("input_token_details") or {}
        cached = given.get("cached_tokens_details") or {}
        made = usage.get("output_token_details") or {}
        self.text_in += given.get("text_tokens", 0)
        self.audio_in += given.get("audio_tokens", 0)
        self.cached_text_in += cached.get("text_tokens", 0)
        self.cached_audio_in += cached.get("audio_tokens", 0)
        self.text_out += made.get("text_tokens", 0)
        self.audio_out += made.get("audio_tokens", 0)
        self.replies += 1

    def add_transcription(self, usage: dict) -> None:
        if usage.get("type") == "tokens":
            self.transcribe_in += usage.get("input_tokens", 0)
            self.transcribe_out += usage.get("output_tokens", 0)

    def cost_usd(self, model: str) -> float | None:
        """None when the model has no known price, rather than a wrong number."""
        prices = REALTIME_PRICES.get(model)
        if prices is None:
            return None
        counts = (
            self.text_in - self.cached_text_in, self.cached_text_in, self.text_out,
            self.audio_in - self.cached_audio_in, self.cached_audio_in, self.audio_out,
        )
        speech = sum(count * price for count, price in zip(counts, prices))
        transcription = self.transcribe_in * TRANSCRIBE_PRICES[0] + self.transcribe_out * TRANSCRIBE_PRICES[1]
        return (speech + transcription) / 1_000_000

    def report(self, model: str) -> str:
        cost = self.cost_usd(model)
        price = f"${cost:.4f} ≈ ₹{cost * USD_TO_INR:.2f}" if cost is not None else "no price known for this model"
        return (
            f"{model} {price} | {self.replies} replies, audio in {self.audio_in} (cached {self.cached_audio_in}), "
            f"audio out {self.audio_out}, text in {self.text_in} (cached {self.cached_text_in}), "
            f"text out {self.text_out}, transcription {self.transcribe_in}+{self.transcribe_out} tokens"
        )


# Gemini Live, USD per 1M tokens: text in, audio in, text out, audio out.
# From https://aistudio.google.com/docs/pricing (Nov 2026, paid tier); update when Google changes them.
GEMINI_PRICES = {
    "gemini-3.8-live": (0.75, 3.00, 4.50, 12.00),
    "gemini-3.8-live-extended-thinking": (0.75, 3.00, 4.50, 12.00),
    "gemini-3.1-flash-live-preview": (0.75, 3.00, 4.50, 12.00),
    "gemini-2.5-flash-native-audio-preview-12-2025": (0.50, 3.00, 2.00, 12.00),
    "gemini-2.5-flash-native-audio-latest": (0.50, 3.00, 2.00, 12.00),
}


@dataclass
class GeminiUsage:
    text_in: int = 0
    audio_in: int = 0
    text_out: int = 0
    audio_out: int = 0
    replies: int = 0

    def add(self, meta: dict) -> None:
        """Add one usageMetadata block; Gemini sends one per model turn."""
        for detail in meta.get("promptTokensDetails") or []:
            if detail.get("modality") == "AUDIO":
                self.audio_in += detail.get("tokenCount", 0)
            else:
                self.text_in += detail.get("tokenCount", 0)
        for detail in meta.get("responseTokensDetails") or []:
            if detail.get("modality") == "AUDIO":
                self.audio_out += detail.get("tokenCount", 0)
            else:
                self.text_out += detail.get("tokenCount", 0)
        self.text_out += meta.get("thoughtsTokenCount", 0)
        self.replies += 1

    def cost_usd(self, model: str) -> float | None:
        prices = GEMINI_PRICES.get(model)
        if prices is None:
            return None
        counts = (self.text_in, self.audio_in, self.text_out, self.audio_out)
        return sum(count * price for count, price in zip(counts, prices)) / 1_000_000

    def report(self, model: str) -> str:
        cost = self.cost_usd(model)
        price = f"${cost:.4f} ≈ ₹{cost * USD_TO_INR:.2f}" if cost is not None else "no price known for this model"
        return (
            f"{model} {price} | {self.replies} replies, audio in {self.audio_in}, audio out {self.audio_out}, "
            f"text in {self.text_in}, text out {self.text_out} tokens"
        )
