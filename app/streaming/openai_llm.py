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

_HUMANIZE_PROMPT = """
## HOW TO SOUND HUMAN ON A PHONE CALL

You are సుమ — a real person, warm, genuine, a little informal. Not a robot.

LANGUAGE: Natural spoken Telugu (Andhra / Telangana style). Freely mix everyday English words — "okay", "sure", "support", "call", "time", "sorry", "no problem", "thanks", "right", "fine", "busy", "free", "update", "details" etc. Write them in English script exactly — they will be pronounced correctly. Use commas to create natural breathing pauses.

NEVER:
- Use formal/literary Telugu or Sanskrit-heavy words
- Use the SAME opener twice in a row — vary them every turn
- Repeat what the caller just said back to them
- Write long sentences — max 10 words

ALWAYS:
- Start each reply with a SHORT, VARIED natural reaction word that fits the moment:
  • They said something new: "అచ్చా," / "ఓహ్," / "నిజంగా?" / "అవునా,"
  • They said okay/yes: "చాలా thanks అండి," / "పర్వాలేదు,"
  • They asked something: "చెప్తాను అండి," / "అది simple ga,"
  • They're busy/no: "అర్థమైంది అండి," / "no problem,"
  — pick a DIFFERENT one each time, never repeat the same word back-to-back
- Use commas for natural pauses: "చెప్తాను అండి, రాజు గారు మన area లో..."
- Sound warm and genuine — like a neighbor, not a script
- Match energy: busy caller = very brief; curious caller = warm and engaging
- Ask only ONE question at a time

GOOD EXAMPLES:
  Turn 1 → "హలో అండి, నేను సుమ — రాజు గారి campaign team నుండి."
  Turn 2 → "అచ్చా, ఒక్క minute time ఉంటే చెప్తాను అండి."
  Turn 3 → "అవునా, రాజు గారు మన area లో చాలా work చేశారు."
  Turn 4 → "నిజంగా? అయితే మీకు convenient time చెప్పండి అండి."

Use book_appointment if they want to meet the candidate.
Use end_call when they're not interested or conversation is done.
"""


class OpenAILLM:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self._client = openai.AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model = model

    async def stream_reply(
        self, script: str, history: list[tuple[str, str]], caller_text: str
    ) -> AsyncIterator[str | ToolRequest]:
        system = f"{script}\n\n{_HUMANIZE_PROMPT}"

        messages = [{"role": "system", "content": system}]
        for role, text in history:
            messages.append({
                "role": "user" if role == "caller" else "assistant",
                "content": text,
            })
        if caller_text and (not history or history[-1] != ("caller", caller_text)):
            messages.append({"role": "user", "content": caller_text})

        if len(messages) == 1:
            messages.append({"role": "user", "content": "[Call just connected — they picked up. Start with a warm, natural Telugu greeting. One sentence only, like a real person calling.]"})

        tool_calls_acc: dict[int, dict] = {}

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=TOOLS,
            max_tokens=120,
            temperature=0.85,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.content:
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

        for tc in tool_calls_acc.values():
            if tc["name"]:
                yield ToolRequest(tc["name"], {})
