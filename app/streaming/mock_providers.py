"""Scripted async providers for tests and local runs. No network, deterministic."""

from collections.abc import AsyncIterator

from app.providers.base import ToolRequest
from app.streaming.audio import silence


class MockAsyncSTT:
    def __init__(self, transcripts: list[str] | None = None):
        self._transcripts = list(transcripts or [])
        self.frames_pushed = 0

    async def start_utterance(self, language: str) -> None:
        self.frames_pushed = 0

    async def push(self, pcm_frame: bytes) -> None:
        self.frames_pushed += 1

    async def end_utterance(self) -> str:
        return self._transcripts.pop(0) if self._transcripts else ""


class MockAsyncTTS:
    """Produces silence whose length scales with the text, so pacing and barge-in can be tested."""

    async def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        return silence(min(max(len(text) * 15, 100), 800))


class MockAsyncLLM:
    """Each entry in `turns` is the scripted output for one reply, as a list of tokens and tool requests."""

    def __init__(self, turns: list[list[str | ToolRequest]] | None = None):
        self._turns = list(turns or [])

    async def stream_reply(self, script: str, history, caller_text: str) -> AsyncIterator[str | ToolRequest]:
        items = self._turns.pop(0) if self._turns else ["Ji, samajh gaya."]
        for item in items:
            yield item
