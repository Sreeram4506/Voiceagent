"""A call that reaches nobody is tried again later, until the agent's attempts run out."""

from datetime import datetime, time, timedelta
from uuid import uuid4

import pytest

from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, Lead, LeadStatus, LedgerEntry, LedgerKind, Tenant
from app.providers.mock import MockTelephony
from app.services.compliance import InMemoryDNDRegistry
from app.services.retries import classify_hangup
from app.services.scheduler import dispatch_due

RETRY_MINUTES = 45


def seed(*, attempts: int = 1, max_attempts: int = 3, source: str = "api", outcome: str | None = None) -> tuple[int, int]:
    """A lead whose call is ringing right now. Returns (call_id, lead_id)."""
    with SessionLocal() as db:
        tenant = Tenant(name="T", api_key_hash=f"h-{uuid4().hex}", webhook_secret="s")
        db.add(tenant)
        db.flush()
        emp = AIEmployee(tenant_id=tenant.id, name="A", script="", max_attempts=max_attempts,
                         retry_delay_minutes=RETRY_MINUTES, window_start=time(0, 0), window_end=time(23, 59))
        db.add(emp)
        db.flush()
        lead = Lead(tenant_id=tenant.id, employee_id=emp.id, phone_e164="+919999911111", status=LeadStatus.CALLING,
                    attempts=attempts, source=source, next_attempt_at=datetime.utcnow())
        db.add(lead)
        db.flush()
        call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=emp.id, status=CallStatus.DIALING,
                    started_at=datetime.utcnow(), outcome=outcome)
        db.add(call)
        db.commit()
        return call.id, lead.id


def hangup(client, call_id: int, **form) -> None:
    data = {"CallStatus": "completed", **form}
    assert client.post(f"/v1/plivo/status/{call_id}", data=data).status_code == 200


def load(call_id: int, lead_id: int) -> tuple[Call, Lead]:
    with SessionLocal() as db:
        return db.get(Call, call_id), db.get(Lead, lead_id)


def assert_retry_in(lead: Lead, minutes: int) -> None:
    expected = datetime.utcnow() + timedelta(minutes=minutes)
    assert abs((lead.next_attempt_at - expected).total_seconds()) < 120


def test_unanswered_call_is_retried_after_the_agents_retry_delay(client):
    call_id, lead_id = seed(attempts=1)
    hangup(client, call_id, HangupCauseName="No Answer")
    call, lead = load(call_id, lead_id)
    assert call.status == CallStatus.NO_ANSWER
    assert lead.status == LeadStatus.QUEUED
    assert_retry_in(lead, RETRY_MINUTES)


def test_busy_line_is_retried(client):
    call_id, lead_id = seed(attempts=1)
    hangup(client, call_id, HangupCauseName="Busy Line")
    call, lead = load(call_id, lead_id)
    assert (call.status, lead.status) == (CallStatus.BUSY, LeadStatus.QUEUED)


def test_completed_callback_without_an_answer_counts_as_no_answer(client):
    """Plivo reports CallStatus=completed even for calls nobody picked up."""
    call_id, lead_id = seed(attempts=1)
    hangup(client, call_id)
    call, lead = load(call_id, lead_id)
    assert (call.status, lead.status) == (CallStatus.NO_ANSWER, LeadStatus.QUEUED)


def test_answered_call_is_marked_connected_and_not_retried(client, monkeypatch):
    monkeypatch.setattr("app.main.prepare_call", lambda call_id: None)  # no voice agent needed here
    call_id, lead_id = seed(attempts=1)
    assert client.post(f"/v1/plivo/answer/{call_id}").status_code == 200
    call, _ = load(call_id, lead_id)
    assert call.status == CallStatus.CONNECTED and call.connected_at is not None

    hangup(client, call_id, HangupCauseName="Normal Hangup")
    call, lead = load(call_id, lead_id)
    assert (call.status, lead.status) == (CallStatus.COMPLETED, LeadStatus.DONE)


def test_a_billed_duration_means_the_call_was_answered(client):
    call_id, lead_id = seed(attempts=1)
    hangup(client, call_id, BillDuration="35")
    call, lead = load(call_id, lead_id)
    assert (call.status, lead.status, call.billed_seconds) == (CallStatus.COMPLETED, LeadStatus.DONE, 35)


def test_last_attempt_unanswered_finishes_the_lead(client):
    call_id, lead_id = seed(attempts=3, max_attempts=3)
    hangup(client, call_id, HangupCauseName="No Answer")
    call, lead = load(call_id, lead_id)
    assert (call.status, lead.status) == (CallStatus.NO_ANSWER, LeadStatus.DONE)


def test_network_failure_is_retried_then_marked_failed(client):
    call_id, lead_id = seed(attempts=1)
    hangup(client, call_id, HangupCauseName="Network Error")
    call, lead = load(call_id, lead_id)
    assert (call.status, lead.status) == (CallStatus.FAILED, LeadStatus.QUEUED)

    call_id, lead_id = seed(attempts=3, max_attempts=3)
    hangup(client, call_id, HangupCauseName="Network Error")
    _, lead = load(call_id, lead_id)
    assert lead.status == LeadStatus.FAILED


def test_a_one_off_dashboard_call_is_not_redialed(client):
    call_id, lead_id = seed(attempts=1, source="single_call")
    hangup(client, call_id, HangupCauseName="No Answer")
    _, lead = load(call_id, lead_id)
    assert lead.status == LeadStatus.DONE


def test_hangup_keeps_the_outcome_the_call_already_saved(client):
    call_id, lead_id = seed(attempts=1, outcome="appointment_booked")
    hangup(client, call_id, BillDuration="60", HangupCauseName="Normal Hangup")
    call, _ = load(call_id, lead_id)
    assert call.outcome == "appointment_booked"


def test_hangup_without_a_duration_keeps_the_billed_seconds(client):
    call_id, lead_id = seed(attempts=1)
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        call.billed_seconds, call.connected_at = 42, datetime.utcnow()
        db.commit()
    hangup(client, call_id)
    call, _ = load(call_id, lead_id)
    assert call.billed_seconds == 42


@pytest.mark.parametrize(
    ("form", "answered", "expected"),
    [
        ({"HangupCauseName": "Ring Timeout Reached"}, False, CallStatus.NO_ANSWER),
        ({"HangupCauseName": "Rejected"}, False, CallStatus.BUSY),
        ({"HangupCauseName": "Busy everywhere"}, False, CallStatus.BUSY),
        ({"HangupCauseName": "Request timeout"}, False, CallStatus.FAILED),
        ({"CallStatus": "no-answer"}, False, CallStatus.NO_ANSWER),
        ({"CallStatus": "busy"}, False, CallStatus.BUSY),
        ({"CallStatus": "failed"}, False, CallStatus.FAILED),
        ({"CallStatus": "completed"}, False, CallStatus.NO_ANSWER),
        ({"CallStatus": "completed", "HangupCauseName": "Normal Hangup"}, True, CallStatus.COMPLETED),
    ],
)
def test_hangup_classification(form, answered, expected):
    assert classify_hangup(form, answered=answered) == expected


# ── refused dials go through the same retry rules ───────────────────────────


def _dialable_lead(session, *, attempts: int, max_attempts: int = 3) -> Lead:
    tenant = Tenant(name="T", api_key_hash=f"h-{uuid4().hex}", webhook_secret="s")
    session.add(tenant)
    session.flush()
    emp = AIEmployee(tenant_id=tenant.id, name="A", script="", phone_number="+914000000000", max_concurrency=10,
                     max_attempts=max_attempts, retry_delay_minutes=RETRY_MINUTES,
                     window_start=time(0, 0), window_end=time(23, 59))
    session.add_all([emp, LedgerEntry(tenant_id=tenant.id, kind=LedgerKind.TOPUP, amount_paise=10_000)])
    session.flush()
    lead = Lead(tenant_id=tenant.id, employee_id=emp.id, phone_e164="+917000000009", status=LeadStatus.NEW,
                attempts=attempts, next_attempt_at=datetime.utcnow() - timedelta(minutes=1))
    session.add(lead)
    session.commit()
    return lead


def test_refused_dial_is_retried_later(session):
    lead = _dialable_lead(session, attempts=0)
    dispatch_due(session, MockTelephony(script=lambda n: "refused"), InMemoryDNDRegistry(), datetime.utcnow(), "cb", 30)
    session.refresh(lead)
    assert (lead.status, lead.attempts) == (LeadStatus.QUEUED, 1)
    assert_retry_in(lead, RETRY_MINUTES)


def test_refused_dial_on_the_last_attempt_fails_the_lead(session):
    lead = _dialable_lead(session, attempts=2, max_attempts=3)
    dispatch_due(session, MockTelephony(script=lambda n: "refused"), InMemoryDNDRegistry(), datetime.utcnow(), "cb", 30)
    session.refresh(lead)
    assert (lead.status, lead.attempts) == (LeadStatus.FAILED, 3)


def test_a_queued_retry_is_dialed_once_it_is_due(session):
    lead = _dialable_lead(session, attempts=1)
    lead.status, lead.next_attempt_at = LeadStatus.QUEUED, datetime.utcnow() + timedelta(minutes=RETRY_MINUTES)
    lead.last_called_at = datetime.utcnow()
    session.commit()

    now = datetime.utcnow()
    assert dispatch_due(session, MockTelephony(), InMemoryDNDRegistry(), now, "cb", 30).dialed == []
    later = now + timedelta(minutes=RETRY_MINUTES + 1)
    assert dispatch_due(session, MockTelephony(), InMemoryDNDRegistry(), later, "cb", 30).dialed == [lead.id]
