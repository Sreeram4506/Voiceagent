"""Wire format for Plivo's bidirectional audio stream.

Plivo opens a WebSocket and sends JSON frames with an "event" field.
Inbound audio arrives as base64 in media.payload. Outbound audio uses the
"playAudio" event, and barge-in uses "clearAudio".
"""

import base64
import json
from dataclasses import dataclass, field


@dataclass
class InboundEvent:
    kind: str
    stream_sid: str | None = None
    audio: bytes | None = None
    raw: dict = field(default_factory=dict)


def parse_event(text: str) -> InboundEvent:
    data = json.loads(text)
    kind = data.get("event", "")
    stream_sid = (
        data.get("streamId")
        or (data.get("start") or {}).get("streamId")
        or data.get("stream_sid")
        or (data.get("start") or {}).get("stream_sid")
    )
    audio = None
    if kind == "media":
        audio = base64.b64decode(data["media"]["payload"])
    return InboundEvent(kind=kind, stream_sid=stream_sid, audio=audio, raw=data)


def media_message(stream_sid: str, pcm: bytes) -> str:
    return json.dumps(
        {
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-l16",
                "sampleRate": 8000,
                "payload": base64.b64encode(pcm).decode(),
            },
        }
    )


def clear_message(stream_sid: str) -> str:
    """Tells Plivo to drop buffered audio, used for barge-in."""
    return json.dumps({"event": "clearAudio", "streamId": stream_sid})
