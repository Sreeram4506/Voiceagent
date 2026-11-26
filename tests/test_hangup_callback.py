"""Plivo's hang-up callback must not erase what the agent decided on the call."""

from datetime import datetime
from datetime import time as dtime

from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, Lead, LeadStatus, Tenant, VoiceTier

HANGUP = {"CallStatus": "completed", "HangupCause": "NORMAL_CLEARING", "Duration": "88"}


def make_call(outcome: str | None, lead_status: LeadStatus = LeadStatus.CALLING) -> tuple[int, int]:
    with SessionLocal() as db:
        tenant = Tenant(name="Acme", api_key_hash="x", webhook_secret="s", webhook_url="https://hooks.example.com/in")
        db.add(tenant)
        db.flush()
        agent = AIEmployee(tenant_id=tenant.id, name="Suma", language="te", voice_tier=VoiceTier.STANDARD,
                           script="Book a demo.", window_start=dtime(9, 0), window_end=dtime(20, 0))
        db.add(agent)
        db.flush()
        lead = Lead(tenant_id=tenant.id, employee_id=agent.id, phone_e164="+919999900031", status=lead_status)
        db.add(lead)
        db.flush()
        call = Call(tenant_id=tenant.id, employee_id=agent.id, lead_id=lead.id, outcome=outcome,
                    status=CallStatus.COMPLETED if outcome else CallStatus.DIALING, started_at=datetime(2026, 11, 26, 11))
        db.add(call)
        db.commit()
        return call.id, lead.id


def reload(call_id: int) -> Call:
    with SessionLocal() as db:
        return db.get(Call, call_id)


def test_booked_call_keeps_its_outcome_after_the_hangup_callback(client):
    call_id, _ = make_call("appointment_booked")
    assert client.post(f"/v1/plivo/status/{call_id}", data=HANGUP).status_code == 200
    call = reload(call_id)
    assert call.outcome == "appointment_booked"
    assert call.billed_seconds == 88  # the callback still records the billed time


def test_opted_out_caller_stays_do_not_call(client):
    call_id, lead_id = make_call("do_not_call", lead_status=LeadStatus.DNC)
    client.post(f"/v1/plivo/status/{call_id}", data=HANGUP)
    assert reload(call_id).outcome == "do_not_call"
    with SessionLocal() as db:
        assert db.get(Lead, lead_id).status == LeadStatus.DNC


def test_unanswered_call_still_gets_the_hangup_cause(client):
    call_id, _ = make_call(None)
    client.post(f"/v1/plivo/status/{call_id}", data={"CallStatus": "no-answer", "HangupCause": "NO_ANSWER"})
    call = reload(call_id)
    assert call.status == CallStatus.NO_ANSWER
    assert call.outcome == "NO_ANSWER"
