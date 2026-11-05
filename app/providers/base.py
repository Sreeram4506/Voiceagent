"""Provider interfaces. The call engine depends only on these, never on a vendor SDK."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class DialResult:
    provider_call_id: str
    accepted: bool
    reason: str | None = None


@dataclass
class ToolRequest:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class AgentReply:
    text: str
    tool_requests: list[ToolRequest] = field(default_factory=list)
    end_call: bool = False


class TelephonyProvider(Protocol):
    def dial(
        self, to_e164: str, from_number: str, callback_url: str, custom_field: str = ""
    ) -> DialResult:
        """from_number is the ExoPhone shown to the customer. custom_field carries our call id."""
        ...

    def hangup(self, provider_call_id: str) -> None: ...


class STTProvider(Protocol):
    def transcribe(self, audio_chunk: bytes, language: str) -> str: ...


class TTSProvider(Protocol):
    def synthesize(self, text: str, language: str, voice_tier: str) -> bytes: ...


class LLMProvider(Protocol):
    def respond(self, script: str, history: list[tuple[str, str]], caller_text: str) -> AgentReply: ...
