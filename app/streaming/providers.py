"""Async provider interfaces for the live pipeline. Vendor adapters implement these."""

from collections.abc import AsyncIterator
from typing import Protocol

from app.providers.base import ToolRequest


class AsyncSTT(Protocol):
    """Speech-to-text for one caller utterance at a time.

    Vendor streaming APIs (e.g. a websocket STT service) keep the connection open between
    push() calls, so audio is transcribed as it arrives rather than after the caller stops.
    """

    async def start_utterance(self, language: str) -> None: ...

    async def push(self, pcm_frame: bytes) -> None: ...

    async def end_utterance(self) -> str: ...


class AsyncTTS(Protocol):
    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        """Returns 8 kHz 16-bit mono PCM."""
        ...


class AsyncLLM(Protocol):
    def stream_reply(
        self, script: str, history: list[tuple[str, str]], caller_text: str
    ) -> AsyncIterator[str | ToolRequest]:
        """Yields text tokens, and ToolRequest items for actions. "end_call" ends the call."""
        ...
