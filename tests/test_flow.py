"""End-to-end flows using mock providers: scheduling, conversation, billing, delivery."""

import hashlib
import hmac
import json
from datetime import datetime, time

import httpx
import pytest

from app.models import AIEmployee, Call, CallDirection, CallStatus, Lead, LeadStatus, LedgerEntry, Tenant, VoiceTier
from app.providers.mock import MockLLM, MockSTT, MockTelephony, MockTTS
from app.services.billing import balance_paise, charge_call, top_up
from app.services.call_engine import finalize_call, run_turns
from app.services.compliance import InMemoryDNDRegistry
from app.services.delivery import deliver_outcome, sign
from app.services.scheduler import dispatch_due

NOON_IST_UTC = datetime(2026, 11, 1, 4, 30)


@pytest.fixture
def tenant(session):
    t = Tenant(name="Acme", api_key_hash="x", webhook_secret="whsec", webhook_url="https://hooks.example.com/in")
    session.add(t)
    session.flush()
    return t


@pytest.fixture
def employee(session, tenant):
    e = AIEmployee(
        tenant_id=tenant.id,
        name="Riya",
        language="hi",
        voice_tier=VoiceTier.STANDARD,
        script="Book a site visit for the caller.",
        phone_number="+918012345678",
        max_concurrency=2,
        window_start=time(9, 0),
        window_end=time(20, 0),
        timezone="Asia/Kolkata",
        max_attempts=3,
        retry_delay_minutes=30,
    )
    session.add(e)
    session.flush()
    return e


def add_lead(session, tenant, employee, phone):
    lead = Lead(
        tenant_id=tenant.id,
        employee_id=employee.id,
        phone_e164=phone,
        status=LeadStatus.NEW,
        next_attempt_at=NOON_IST_UTC,
    )
    session.add(lead)
    session.flush()
    return lead


def test_dispatch_skips_agent_without_number(session, tenant, employee):
    top_up(session, tenant.id, 100)
    employee.phone_number = None
    add_lead(session, tenant, employee, "+919999900001")
    session.flush()
    report = dispatch_due(session, MockTelephony(), InMemoryDNDRegistry(), NOON_IST_UTC, "cb", 30)
    assert report.dialed == []
    assert list(report.skipped.values()) == ["no_phone_number_assigned"]


def test_dispatch_uses_agent_exophone_and_tags_call_id(session, tenant, employee):
    top_up(session, tenant.id, 100)
    add_lead(session, tenant, employee, "+919999900001")
    session.flush()
    telephony = MockTelephony()
    dispatch_due(session, telephony, InMemoryDNDRegistry(), NOON_IST_UTC, "cb", 30)
    assert len(telephony.dialed) == 1
    dial = telephony.dialed[0]
    assert dial["from"] == "+918012345678"
    assert dial["to"] == "+919999900001"
    call = session.query(Call).one()
    assert dial["custom_field"] == str(call.id)


def test_dispatch_skips_without_credits(session, tenant, employee):
    add_lead(session, tenant, employee, "+919999900001")
    report = dispatch_due(session, MockTelephony(), InMemoryDNDRegistry(), NOON_IST_UTC, "cb", 30)
    assert report.dialed == []
    assert list(report.skipped.values()) == ["no_credits"]


def test_dispatch_dials_and_respects_concurrency(session, tenant, employee):
    top_up(session, tenant.id, 100)
    for i in range(4):
        add_lead(session, tenant, employee, f"+91999990000{i}")
    session.flush()
    report = dispatch_due(session, MockTelephony(), InMemoryDNDRegistry(), NOON_IST_UTC, "cb", 30)
    assert len(report.dialed) == 2  # max_concurrency = 2
    assert "concurrency_limit" in report.skipped.values()
    assert session.query(Call).count() == 2


def test_refused_call_pauses_campaign(session, tenant, employee):
    top_up(session, tenant.id, 100)
    add_lead(session, tenant, employee, "+919999900009")
    session.flush()
    telephony = MockTelephony(script=lambda n: "refused")
    dispatch_due(session, telephony, InMemoryDNDRegistry(), NOON_IST_UTC, "cb", 30)
    assert employee.paused is True


def test_dnd_number_is_never_dialed(session, tenant, employee):
    top_up(session, tenant.id, 100)
    add_lead(session, tenant, employee, "+919999900009")
    session.flush()
    report = dispatch_due(
        session, MockTelephony(), InMemoryDNDRegistry({"+919999900009"}), NOON_IST_UTC, "cb", 30
    )
    assert report.dialed == []
    assert report.skipped and "dnd_registry" in report.skipped.values()


def test_booked_call_bills_and_completes_lead(session, tenant, employee):
    top_up(session, tenant.id, 100)  # 10000 paise
    lead = add_lead(session, tenant, employee, "+919999900001")
    call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=employee.id, status=CallStatus.CONNECTED, started_at=NOON_IST_UTC)
    session.add(call)
    session.flush()

    actions: list[str] = []
    result = run_turns(
        session,
        call,
        employee,
        MockLLM(),
        MockTTS(),
        caller_utterances=["Haan, appointment book kar dijiye", "Dhanyavaad bye"],
        actions_log=actions,
    )
    assert result.outcome == "appointment_booked"
    assert "book_appointment" in actions

    charged = finalize_call(session, call, employee, lead, result, connected_seconds=90, now=NOON_IST_UTC)
    assert charged == 750  # 90s at 500 paise/min
    assert call.status == CallStatus.COMPLETED
    assert lead.status == LeadStatus.DONE
    assert balance_paise(session, tenant.id) == 10000 - 750
    assert [t.role for t in call.turns][:2] == ["agent", "caller"]


def test_no_charge_for_zero_seconds(session, tenant, employee):
    top_up(session, tenant.id, 1)
    assert charge_call(session, tenant.id, 1, VoiceTier.STANDARD, 0) == 0


def test_outcome_webhook_is_signed_and_retried(session, tenant, employee):
    lead = add_lead(session, tenant, employee, "+919999900001")
    call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=employee.id, status=CallStatus.COMPLETED, outcome="completed", summary="ok", started_at=NOON_IST_UTC, ended_at=NOON_IST_UTC, billed_seconds=30)
    session.add(call)
    session.flush()

    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        # First attempt fails, second succeeds.
        return httpx.Response(500 if len(received) == 1 else 200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ok = deliver_outcome(session, call, client=client, max_attempts=3)
    assert ok is True
    assert len(received) == 2

    body = received[-1].content
    expected = hmac.new(b"whsec", body, hashlib.sha256).hexdigest()
    assert received[-1].headers["X-Signature-SHA256"] == expected
    assert json.loads(body)["outcome"] == "completed"


def test_deliver_outcome_handles_missing_tenant_gracefully(session):
    call = Call(id=999, tenant_id=99999, status=CallStatus.COMPLETED)
    assert deliver_outcome(session, call) is False


def test_finalize_call_creates_appointment_record(session, tenant, employee):
    from app.models import Appointment
    from app.services.call_engine import CallResult
    lead = add_lead(session, tenant, employee, "+919999900001")
    call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=employee.id, status=CallStatus.CONNECTED, started_at=NOON_IST_UTC)
    session.add(call)
    session.flush()

    res = CallResult(outcome="appointment_booked", summary="Booked", actions=["book_appointment"])
    finalize_call(session, call, employee, lead, res, connected_seconds=60, now=NOON_IST_UTC)
    session.flush()

    appts = session.query(Appointment).filter(Appointment.call_id == call.id).all()
    assert len(appts) == 1
    assert appts[0].phone == "+919999900001"


def test_refused_call_not_in_dialed_report(session, tenant, employee):
    top_up(session, tenant.id, 100)
    add_lead(session, tenant, employee, "+919999900009")
    session.flush()
    telephony = MockTelephony(script=lambda n: "refused")
    report = dispatch_due(session, telephony, InMemoryDNDRegistry(), NOON_IST_UTC, "cb", 30)
    assert report.dialed == []
    assert "+919999900009" not in report.dialed
