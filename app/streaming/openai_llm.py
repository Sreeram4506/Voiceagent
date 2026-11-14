"""OpenAI GPT-4o streaming LLM for voice agent replies."""

import os
from collections.abc import AsyncIterator

import openai

from app.providers.base import ToolRequest

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a site visit appointment for the caller.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call politely after the conversation is complete.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

_HUMANIZE_PROMPT = """VOICE CALL RULES (strict):
- ONE sentence per reply, max 15 words. Never two sentences.
- Speak like a real Telugu/Hindi person on a phone call — warm, casual, friendly.
- Use natural fillers: "అవునా...", "అలాగా...", "హ్మ్మ్...", "అచ్ఛా...", "సరే సరే..."
- Write English words in Telugu/Hindi script: ఓకే, థాంక్స్, ప్లాట్, సైట్ విజిట్, లోన్, బుకింగ్
- End questions naturally: "కదా?", "అంటారా?", "ఏమంటారు?"
- Vary openers. Never repeat the same one twice in a row.
- NEVER say you're an AI, bot, or assistant. You are a real person.
- Match caller energy and mood. Ask only ONE question per reply.
- Use "..." for a thinking pause (once per reply max).
- Use book_appointment for site visits. Use end_call when done/not interested.
"""


class OpenAILLM:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", temperature: float = 0.85):
        self._client = openai.AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._temperature = temperature

    async def stream_reply(
        self, script: str, history: list[tuple[str, str]], caller_text: str
    ) -> AsyncIterator[str | ToolRequest]:
        import time
        system = f"{script}\n\n{_HUMANIZE_PROMPT}"

        messages = [{"role": "system", "content": system}]
        for role, text in history[-6:]:
            messages.append({
                "role": "user" if role == "caller" else "assistant",
                "content": text,
            })
        if caller_text and (not history or history[-1] != ("caller", caller_text)):
            messages.append({"role": "user", "content": caller_text})

        if len(messages) == 1:
            messages.append({"role": "user", "content": "[Call connected. Greet warmly in the language of your system prompt — like a friendly person calling, not a call centre. One natural sentence.]"})

        tool_calls_acc: dict[int, dict] = {}
        t0 = time.time()
        first_token = True

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=TOOLS,
            max_tokens=100,
            temperature=self._temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.content:
                if first_token:
                    print(f"[LLM] first token at {time.time()-t0:.2f}s")
                    first_token = False
                yield delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"name": "", "args": ""}
                    if tc.function and tc.function.name:
                        tool_calls_acc[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_acc[idx]["args"] += tc.function.arguments

        print(f"[LLM] done in {time.time()-t0:.2f}s")
        for tc in tool_calls_acc.values():
            if tc["name"]:
                yield ToolRequest(tc["name"], {})
