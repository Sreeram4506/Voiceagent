"""Claude (Anthropic) streaming LLM for voice agent replies."""

import os
from collections.abc import AsyncIterator

import anthropic

from app.providers.base import ToolRequest

TOOLS = [
    {
        "name": "book_appointment",
        "description": "Book a site visit appointment for the caller.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "end_call",
        "description": "End the call politely after the conversation is complete.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


class ClaudeLLM:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    async def stream_reply(
        self, script: str, history: list[tuple[str, str]], caller_text: str
    ) -> AsyncIterator[str | ToolRequest]:
        system = (
            f"{script}\n\n"
            "You are an AI voice agent on a phone call. Keep responses short and natural — "
            "1-2 sentences max. Speak in the same language the caller uses. "
            "Use book_appointment tool when the caller agrees to a visit. "
            "Use end_call tool when the conversation is naturally over."
        )

        messages = []
        for role, text in history:
            messages.append({
                "role": "user" if role == "caller" else "assistant",
                "content": text,
            })
        if caller_text and (not history or history[-1] != ("caller", caller_text)):
            messages.append({"role": "user", "content": caller_text})

        if not messages:
            messages.append({"role": "user", "content": "[Call connected. Greet the caller and introduce yourself.]"})

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=200,
            system=system,
            messages=messages,
            tools=TOOLS,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield event.delta.text
                elif event.type == "content_block_stop":
                    pass
                elif event.type == "message_stop":
                    pass

            response = await stream.get_final_message()
            for block in response.content:
                if block.type == "tool_use":
                    yield ToolRequest(block.name, block.input or {})
