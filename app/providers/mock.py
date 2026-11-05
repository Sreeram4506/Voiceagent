"""Offline implementations of every provider. Used in tests and local development."""

from itertools import count

from app.providers.base import AgentReply, DialResult, ToolRequest


class MockTelephony:
    """Scripted outcomes: pass a function that maps a phone number to a result string."""

    def __init__(self, script=None):
        self._ids = count(1)
        self._script = script or (lambda number: "answered")
        self.dialed: list[dict] = []
        self.hung_up: list[str] = []

    def dial(self, to_e164: str, from_number: str, callback_url: str, custom_field: str = "") -> DialResult:
        self.dialed.append({"to": to_e164, "from": from_number, "custom_field": custom_field})
        outcome = self._script(to_e164)
        if outcome == "refused":
            return DialResult(provider_call_id="", accepted=False, reason="refused")
        return DialResult(provider_call_id=f"mock-{next(self._ids)}", accepted=True)

    def hangup(self, provider_call_id: str) -> None:
        self.hung_up.append(provider_call_id)


class MockSTT:
    def __init__(self, transcripts: list[str] | None = None):
        self._transcripts = list(transcripts or [])

    def transcribe(self, audio_chunk: bytes, language: str) -> str:
        return self._transcripts.pop(0) if self._transcripts else ""


class MockTTS:
    def synthesize(self, text: str, language: str, voice_tier: str) -> bytes:
        return f"[{voice_tier}:{language}] {text}".encode()


class MockLLM:
    """Replies with a fixed greeting, then answers yes/no questions with simple rules."""

    def respond(self, script: str, history: list[tuple[str, str]], caller_text: str) -> AgentReply:
        if not history:
            return AgentReply(text="Namaste, main aapki madad ke liye call kar raha hoon.")
        lowered = caller_text.lower()
        if "book" in lowered or "appointment" in lowered:
            return AgentReply(
                text="Theek hai, main slot book kar deta hoon.",
                tool_requests=[ToolRequest(name="book_appointment", arguments={"note": caller_text})],
            )
        if "bye" in lowered or "nahi" in lowered:
            return AgentReply(text="Dhanyavaad, aapka din shubh ho.", end_call=True)
        return AgentReply(text="Samajh gaya. Aur kuch jaanna chahenge?")
