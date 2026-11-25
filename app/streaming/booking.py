"""The appointment time the agent agreed with the caller, as the agent passes it to book_appointment."""

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Where the caller is when the call gives no time zone: the platform's default market.
DEFAULT_ZONE = ZoneInfo("Asia/Kolkata")

BOOKING_PARAMETERS = {
    "type": "object",
    "properties": {
        "date": {"type": "string", "description": "The agreed day as YYYY-MM-DD, counted from today's date."},
        "time": {"type": "string", "description": "The agreed time in the caller's local time, 24-hour HH:MM (4 PM is 16:00)."},
    },
    "required": ["date", "time"],
}


def booking_args(raw: object) -> dict:
    """The tool call's arguments as a dict: OpenAI sends a JSON string, Gemini an object."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def booking_time(args: dict, call_time: datetime) -> datetime | None:
    """The agreed date and time as a naive UTC datetime (how appointments are stored), or None when the
    agent's values are unreadable or before the call itself, so the caller never gets a wrong slot saved."""
    zone = call_time.tzinfo or DEFAULT_ZONE
    try:
        local = datetime.strptime(f"{args['date']} {args['time']}", "%Y-%m-%d %H:%M").replace(tzinfo=zone)
    except (KeyError, TypeError, ValueError):
        return None
    reference = call_time if call_time.tzinfo else call_time.replace(tzinfo=zone)
    if local < reference:
        return None
    return local.astimezone(timezone.utc).replace(tzinfo=None)
