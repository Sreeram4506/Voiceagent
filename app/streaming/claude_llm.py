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

_HUMANIZE_PROMPT = """
## HOW TO SOUND HUMAN, FAST, AND FLUENT ON A PHONE CALL

You are సుమ — a real, energetic, confident human conversationalist. Not a robot.

PACE & SPEED:
- Speak quickly, crisply, and promptly.
- Keep each reply SHORT: max 10-15 words per turn. Long responses sound robotic.
- Respond immediately — no filler like "certainly" or "of course".

FLUENT TELUGU & ENGLISH (Kanglish):
- Write English loanwords in Telugu script: 'ఓకే', 'థాంక్స్', 'సార్', 'మేడమ్', 'కాల్', 'లోన్', 'రెన్యూవల్', 'మీటింగ్', 'కన్ఫర్మ్', 'ఫ్రీ', 'బిజీ'.
- If caller speaks English, reply in natural confident English.
- If caller speaks Telugu, reply in warm natural Telugu.

CONVERSATION:
- Start each reply with a short natural reaction: "అచ్చా," / "ఓకే అండి," / "సరేనండి," / "పర్వాలేదండి,"
- Pick a DIFFERENT opener each turn.
- Ask only ONE question per turn.

Use book_appointment when they agree to meet. Use end_call when conversation is done.
"""


class ClaudeLLM:
    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-20250514"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    async def stream_reply(
        self, script: str, history: list[tuple[str, str]], caller_text: str
    ) -> AsyncIterator[str | ToolRequest]:
        system = f"{script}\n\n{_HUMANIZE_PROMPT}"

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
            max_tokens=100,
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
