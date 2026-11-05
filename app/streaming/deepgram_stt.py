"""Deepgram real-time streaming STT via WebSocket."""

import asyncio
import json
import os

import websockets


class DeepgramSTT:
    """Streams 8 kHz 16-bit mono PCM to Deepgram and collects the transcript."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ["DEEPGRAM_API_KEY"]
        self._ws = None
        self._transcript_parts: list[str] = []
        self._receive_task: asyncio.Task | None = None

    async def start_utterance(self, language: str) -> None:
        lang = language if "-" in language else f"{language}-IN" if language == "hi" else f"{language}-US"
        url = (
            "wss://api.deepgram.com/v1/listen"
            f"?encoding=linear16&sample_rate=8000&channels=1"
            f"&language={lang}&model=nova-2&punctuate=true"
            f"&endpointing=false&interim_results=false"
        )
        self._transcript_parts = []
        self._ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        try:
            async for msg in self._ws:
                data = json.loads(msg)
                if data.get("type") == "Results":
                    alt = (data.get("channel", {}).get("alternatives") or [{}])[0]
                    text = alt.get("transcript", "").strip()
                    if text and data.get("is_final"):
                        self._transcript_parts.append(text)
        except websockets.ConnectionClosed:
            pass

    async def push(self, pcm_frame: bytes) -> None:
        if self._ws:
            try:
                await self._ws.send(pcm_frame)
            except websockets.ConnectionClosed:
                pass

    async def end_utterance(self) -> str:
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.wait_for(self._receive_task, timeout=3.0)
            except (websockets.ConnectionClosed, asyncio.TimeoutError, Exception):
                pass
            finally:
                if self._receive_task and not self._receive_task.done():
                    self._receive_task.cancel()
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None
                self._receive_task = None
        return " ".join(self._transcript_parts)
