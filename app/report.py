"""Per-agent client report: an agent's details and every call's yes/no outcome, for a shareable dashboard.

The same shape serves any agent — a recorded yes/no agent (said_yes / said_no) or an AI conversation agent
(appointment_booked / do_not_call) — so one dashboard page works for all of them.
"""

import hashlib
import hmac
import os
import re
from datetime import datetime, timezone

from app.config import get_settings
from app.models import AIEmployee, AppSetting, Appointment, Call, CallStatus, CallTurn, Lead

# How each saved outcome maps to what a client wants to see.
_INTERESTED = {"said_yes", "appointment_booked"}
_NOT_INTERESTED = {"said_no", "do_not_call"}
_NOT_REACHED_STATUS = {CallStatus.NO_ANSWER, CallStatus.BUSY, CallStatus.FAILED, CallStatus.DIALING}

CATEGORIES = ("interested", "not_interested", "maybe", "not_reached")
CATEGORY_LABEL = {
    "interested": "Interested (yes)",
    "not_interested": "Not interested (no)",
    "maybe": "Answered, no clear answer",
    "not_reached": "Not reached",
}


def report_token(employee_id: int) -> str:
    """An unguessable token that lets the agent's own report be opened without a dashboard login."""
    key = get_settings().stream_secret.encode()
    return hmac.new(key, f"report:{int(employee_id)}".encode(), hashlib.sha256).hexdigest()[:32]


def _agent_outcome(call: Call) -> str:
    """The agent's own outcome, read from the summary where the telephony hang-up cause has overwritten it."""
    match = re.search(r"outcome=([a-z_]+)", call.summary or "")
    return match.group(1) if match else (call.outcome or "")


def category(call: Call) -> str:
    answered = (call.billed_seconds or 0) > 0
    if not answered or call.status in _NOT_REACHED_STATUS:
        return "not_reached"
    outcome = _agent_outcome(call)
    if outcome in _INTERESTED:
        return "interested"
    if outcome in _NOT_INTERESTED:
        return "not_interested"
    return "maybe"


def _extras(db, employee: AIEmployee) -> dict:
    row = db.get(AppSetting, f"t{employee.tenant_id}:employee_extras")
    value = row.value if row and isinstance(row.value, dict) else {}
    extra = value.get(str(employee.id), {})
    return extra if isinstance(extra, dict) else {}


def _engine() -> str:
    return "Google Gemini" if os.getenv("GEMINI_API_KEY") else "OpenAI"


def _agent_block(db, employee: AIEmployee, extras: dict) -> dict:
    mode = str(extras.get("agent_mode") or "ai_agent")
    return {
        "id": employee.id,
        "name": employee.name,
        "language": employee.language,
        "mode": mode,
        "mode_label": "Recorded message" if mode == "recorded_audio" else "AI conversation",
        "voice": str(extras.get("voice_gender") or "female").capitalize(),
        "engine": _engine(),
        "style": str(extras.get("speaking_style") or "professional").capitalize(),
        "purpose": (employee.script or "").strip()[:280],
    }


def _caller_words(db, call_id: int) -> str:
    turns = db.query(CallTurn).filter(CallTurn.call_id == call_id, CallTurn.role == "caller").order_by(CallTurn.id).all()
    return " ".join(t.text.strip() for t in turns if t.text and t.text.strip())[:200]


def build_report(db, employee: AIEmployee) -> dict:
    """The agent's details, a yes/no summary, and every call — the whole payload the dashboard page renders."""
    extras = _extras(db, employee)
    calls = db.query(Call).filter(Call.employee_id == employee.id).order_by(Call.id.desc()).all()
    booked = {
        row.call_id
        for row in db.query(Appointment.call_id).filter(Appointment.employee_id == employee.id).all()
        if row.call_id
    }

    rows, counts = [], dict.fromkeys(CATEGORIES, 0)
    for call in calls:
        cat = category(call)
        counts[cat] += 1
        lead = db.get(Lead, call.lead_id) if call.lead_id else None
        captured = call.captured_fields or {}
        rows.append({
            "id": call.id,
            "name": (lead.name if lead else captured.get("name")) or "—",
            "phone": (lead.phone_e164 if lead else captured.get("phone")) or "—",
            "category": cat,
            "category_label": CATEGORY_LABEL[cat],
            "said": _caller_words(db, call.id),
            "talk_seconds": call.billed_seconds or 0,
            "booked": call.id in booked,
            "at": (call.started_at.replace(tzinfo=timezone.utc).isoformat() if call.started_at else None),
        })

    answered = sum(1 for r in rows if r["talk_seconds"] > 0)
    return {
        "agent": _agent_block(db, employee, extras),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(rows),
            "answered": answered,
            "booked": len(booked & {r["id"] for r in rows}),
            **counts,
        },
        "calls": rows,
    }
