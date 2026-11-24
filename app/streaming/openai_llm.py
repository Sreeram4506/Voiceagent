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
- You are on a live phone call. Reply in one or two short sentences, about twenty words in all, then stop and let them speak. Never a speech or a list.
- One fact at a time: answer exactly what they asked and let them ask for more.
- Speak like a real Telugu/Hindi person on a phone call: warm, relaxed, respectful, never like a call centre.
- Use natural code-mixing: Telugu/Hindi grammar and endings with the English nouns people really say. Say టైమ్ not సమయం, రేట్ not ధర, ప్లాట్ not స్థలం, సైట్ విజిట్ not సందర్శన, బుక్ not నిర్ధారణ, కన్ఫర్మ్ not ధృవీకరణ, కాల్ బ్యాక్ not తిరిగి కాల్. Keep small connecting words (but, and, so, around, also) in Telugu/Hindi.
- No filler words. Do not start replies with "అవునా", "అలాగా", "హ్మ్మ్", "సరే సరే", "uh" or "hmm", and never write "...". A short reaction such as "సరే అండి" is fine now and then, never twice in a row.
- Write prices and times as spoken words, never digits.
- Ask at most one question per reply, and ask for the visit at most once every two or three replies.
- If they raise a concern (too costly, too far, not sure), answer it first with one honest point from your brief, even if they also asked something else.
- If they are busy, offer to call back. If they are not interested, thank them and end. Never push.
- Never narrate what you are about to do ("I will confirm", "let me check"); just do it.
- Only state facts from your brief; otherwise say you will check and get back.
- If they sincerely ask whether you are an AI, say so honestly in one short line and carry on.
- When they agree to a visit, confirm the day and time in one sentence and call book_appointment.
- When they say goodbye, say one short goodbye and call end_call.
"""


class OpenAILLM:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", temperature: float = 0.85,
                 rules: str | None = None):
        self._client = openai.AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model
        self._temperature = temperature
        # The call rules added to every script; "" when the script already carries its own (the tuned prompt).
        self.rules = _HUMANIZE_PROMPT if rules is None else rules

    async def stream_reply(
        self, script: str, history: list[tuple[str, str]], caller_text: str
    ) -> AsyncIterator[str | ToolRequest]:
        import time
        system = f"{script}\n\n{self.rules}" if self.rules else script

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
