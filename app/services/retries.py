"""What happens to a lead once its call ends: done, or queued to be tried again later.

Plivo's hangup callback reports CallStatus=completed even for calls nobody picked up, so a call
counts as answered only when our answer webhook fired or Plivo reports an answer time or billed
seconds. The hangup cause, when Plivo sends one, says why an unanswered call failed.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, Lead, LeadStatus

# Calls placed by hand from the dashboard are one-offs: nobody expects the phone to ring again later.
ONE_OFF_SOURCES = frozenset({"single_call"})

RETRYABLE = frozenset({CallStatus.NO_ANSWER, CallStatus.BUSY, CallStatus.FAILED, CallStatus.REFUSED})
NEVER_CONNECTED = frozenset({CallStatus.FAILED, CallStatus.REFUSED})

# Plivo HangupCauseName values (https://www.plivo.com/docs/voice/troubleshooting/hangup-causes), lower-cased,
# plus the raw SIP-style HangupCause values Plivo sends on XML callbacks.
_NO_ANSWER_CAUSES = frozenset({"no answer", "ring timeout reached", "no_answer", "no_user_response"})
_BUSY_CAUSES = frozenset({"busy line", "busy everywhere", "rejected", "declined", "user_busy", "call_rejected"})
_NORMAL_CAUSES = frozenset({"normal hangup", "end of xml instructions", "normal_clearing", ""})
_STATUS_BY_CALL_STATUS = {
    "no-answer": CallStatus.NO_ANSWER,
    "busy": CallStatus.BUSY,
    "failed": CallStatus.FAILED,
    "timeout": CallStatus.FAILED,
    "cancel": CallStatus.FAILED,
}
_EMPTY = frozenset({"", "none", "null"})
OUTCOME_MAX_LENGTH = 50  # Call.outcome column size


def utc_naive(moment: datetime) -> datetime:
    """Timestamps are stored as naive UTC."""
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def settle_lead(lead: Lead, employee: AIEmployee | None, call_status: CallStatus, now: datetime) -> None:
    """Finish the lead, or queue it for another try if the call reached nobody and attempts remain."""
    if call_status not in RETRYABLE:
        lead.status = LeadStatus.DONE
        return
    settings = get_settings()
    max_attempts = employee.max_attempts if employee else settings.max_dial_attempts
    delay_minutes = employee.retry_delay_minutes if employee else settings.retry_delay_minutes
    if lead.source not in ONE_OFF_SOURCES and lead.attempts < max_attempts:
        lead.status = LeadStatus.QUEUED
        lead.next_attempt_at = utc_naive(now) + timedelta(minutes=delay_minutes)
        return
    lead.status = LeadStatus.FAILED if call_status in NEVER_CONNECTED else LeadStatus.DONE


def _field(data: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = str(data.get(name) or "").strip()
        if value.lower() not in _EMPTY:
            return value
    return ""


def _seconds(data: Mapping[str, str]) -> int | None:
    raw = _field(data, "BillDuration", "Duration")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def was_answered(call: Call, data: Mapping[str, str]) -> bool:
    return call.connected_at is not None or bool(_field(data, "AnswerTime")) or (_seconds(data) or 0) > 0


def classify_hangup(data: Mapping[str, str], answered: bool) -> CallStatus:
    """The final status of a call from Plivo's hangup callback."""
    if answered:
        return CallStatus.COMPLETED
    cause = _field(data, "HangupCauseName", "HangupCause").lower()
    if cause in _NO_ANSWER_CAUSES:
        return CallStatus.NO_ANSWER
    if cause in _BUSY_CAUSES:
        return CallStatus.BUSY
    by_status = _STATUS_BY_CALL_STATUS.get(_field(data, "CallStatus", "Status").lower())
    if by_status is not None:
        return by_status
    if cause not in _NORMAL_CAUSES:
        return CallStatus.FAILED  # a carrier or network error
    return CallStatus.NO_ANSWER


def apply_hangup(session: Session, call_id: int, data: Mapping[str, str], now: datetime) -> None:
    """Record how a call ended and decide whether its lead is called again."""
    call = session.get(Call, call_id)
    if call is None:
        return
    status = classify_hangup(data, was_answered(call, data))
    call.status = status
    if not call.outcome:  # keep what the conversation saved, e.g. appointment_booked
        reason = _field(data, "HangupCauseName", "HangupCause", "CallStatus", "Status")
        call.outcome = reason[:OUTCOME_MAX_LENGTH] or None
    seconds = _seconds(data)
    if seconds is not None:
        call.billed_seconds = seconds
    call.ended_at = utc_naive(now)

    lead = session.get(Lead, call.lead_id) if call.lead_id else None
    if lead is not None and lead.status == LeadStatus.CALLING:
        settle_lead(lead, session.get(AIEmployee, call.employee_id), status, now)


def mark_answered(session: Session, call_id: int, now: datetime) -> None:
    """The callee picked up: the call is live, not ringing."""
    call = session.get(Call, call_id)
    if call is None:
        return
    if call.status == CallStatus.DIALING:
        call.status = CallStatus.CONNECTED
    if call.connected_at is None:
        call.connected_at = utc_naive(now)


def record_hangup(call_id: int, data: Mapping[str, str]) -> None:
    """Plivo's hangup callback, saved in its own transaction. Blocking: run it in a worker thread."""
    with SessionLocal() as db:
        apply_hangup(db, call_id, data, datetime.now(timezone.utc))
        db.commit()


def record_answer(call_id: int) -> None:
    """Plivo's answer callback, saved in its own transaction. Blocking: run it in a worker thread."""
    with SessionLocal() as db:
        mark_answered(db, call_id, datetime.now(timezone.utc))
        db.commit()
