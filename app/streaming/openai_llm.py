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
## HOW TO SOUND HUMAN, FAST, AND FLUENT ON A PHONE CALL

You are సుమ — a real, energetic, confident human conversationalist. Not a robot.

PACE & SPEED:
- Speak quickly, crisply, and promptly.
- Keep each reply short and punchy: max 10 to 15 words per turn. Long responses sound robotic and boring.
- Respond immediately to what the caller says without hesitation or repetition.

FLUENT TELUGU & ENGLISH:
- Language: Natural conversational Telugu (Andhra/Telangana style) mixed with common everyday English words (Kanglish/Telugish).
- Write everyday English loanwords directly in Telugu script (e.g. 'ఓకే', 'థాంక్స్', 'సార్', 'మేడమ్', 'కాల్', 'అపాయింట్‌మెంట్', 'లోన్', 'రెన్యూవల్', 'మీటింగ్', 'షెడ్యూల్', 'కన్ఫర్మ్', 'డిటైల్స్', 'టైమ్', 'ఫ్రీ', 'బిజీ', 'హలో', 'హాయ్', 'సపోర్ట్', 'బ్యాంక్', 'నెంబర్', 'షుర్', 'నో ప్రాబ్లమ్'). This guarantees 100% fluent, seamless pronunciation without any stuttering.
- If the caller speaks in pure English, reply in natural, fluent, confident English.
- If the caller speaks in Telugu, reply in warm, respectful, natural Telugu.

CONVERSATION FLOW:
- Start each reply with a short, natural reaction word:
  • They said yes/agreed: "ఖచ్చితంగా అండి," / "చాలా థాంక్స్ అండి," / "ఓకే అండి,"
  • They shared info: "అచ్చా," / "ఓహ్ సరేనండి," / "అర్థమైంది అండి,"
  • They asked something: "తప్పకుండా చెప్తాను అండి," / "సింపుల్ గా చెప్తాను,"
  • They're busy/hesitant: "పర్వాలేదండి," / "నో ప్రాబ్లమ్ అండి,"
  — pick a DIFFERENT opener each turn, never repeat the same one back-to-back.
- Ask only ONE clear question at a time.
- Match the caller's energy: brief if they are in a hurry, warm and helpful if they want details.

Use book_appointment if they want to meet, book an appointment, or confirm a visit.
Use end_call politely when they're not interested or the conversation is concluded.
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
