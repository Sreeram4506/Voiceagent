"""A lead is claimed before it is dialed, so two dialers never ring the same person; one Plivo client is shared."""

from datetime import datetime, time, timedelta
from uuid import uuid4

import httpx

from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, Lead, LeadStatus, LedgerEntry, LedgerKind, Tenant
from app.providers.mock import MockTelephony
from app.services.compliance import InMemoryDNDRegistry
from app.services.scheduler import dispatch_due


def dialable_leads(session, phones: list[str]) -> list[Lead]:
    tenant = Tenant(name="T", api_key_hash=f"h-{uuid4().hex}", webhook_secret="s")
    session.add(tenant)
    session.flush()
    emp = AIEmployee(tenant_id=tenant.id, name="A", script="", phone_number="+914000000000", max_concurrency=10,
                     window_start=time(0, 0), window_end=time(23, 59))
    session.add_all([emp, LedgerEntry(tenant_id=tenant.id, kind=LedgerKind.TOPUP, amount_paise=10_000)])
    session.flush()
    due = datetime.utcnow() - timedelta(minutes=1)
    leads = [Lead(tenant_id=tenant.id, employee_id=emp.id, phone_e164=p, status=LeadStatus.NEW, next_attempt_at=due)
             for p in phones]
    session.add_all(leads)
    session.commit()
    return leads


def dispatch(session, telephony) -> object:
    return dispatch_due(session, telephony, InMemoryDNDRegistry(), datetime.utcnow(), "cb", 30)


class RacingTelephony(MockTelephony):
    """While the first call rings, a second worker runs its own dispatch pass over the same leads."""

    def __init__(self):
        super().__init__()
        self.raced = False

    def dial(self, to_e164, from_number, callback_url, custom_field=""):
        if not self.raced:
            self.raced = True
            with SessionLocal() as other_worker:
                dispatch(other_worker, self)
        return super().dial(to_e164, from_number, callback_url, custom_field)


def test_two_dispatchers_never_dial_the_same_lead(session):
    dialable_leads(session, ["+917000000001", "+917000000002"])
    telephony = RacingTelephony()

    report = dispatch(session, telephony)

    numbers = [d["to"] for d in telephony.dialed]
    assert sorted(numbers) == ["+917000000001", "+917000000002"]  # each person rang exactly once
    assert "already_dialing" in report.skipped.values()
    assert session.query(Call).count() == 2


def test_the_call_is_saved_before_the_phone_rings(session):
    """Plivo's ring and answer webhooks look the call up by id, often before the dial request returns."""
    dialable_leads(session, ["+917000000003"])
    seen: list[Call | None] = []

    class Peeking(MockTelephony):
        def dial(self, to_e164, from_number, callback_url, custom_field=""):
            with SessionLocal() as webhook:
                seen.append(webhook.get(Call, int(custom_field)))
            return super().dial(to_e164, from_number, callback_url, custom_field)

    dispatch(session, Peeking())
    assert seen and seen[0] is not None


def test_a_network_error_while_dialing_releases_the_lead(session):
    first, second = dialable_leads(session, ["+917000000004", "+917000000005"])

    class Flaky(MockTelephony):
        def dial(self, to_e164, from_number, callback_url, custom_field=""):
            if to_e164 == first.phone_e164:
                raise httpx.ConnectError("plivo unreachable")
            return super().dial(to_e164, from_number, callback_url, custom_field)

    report = dispatch(session, Flaky())

    session.refresh(first)
    assert first.status == LeadStatus.QUEUED  # free to be tried again, not stuck as "calling"
    assert report.skipped[first.id] == "dial_error"
    assert report.dialed == [second.id]
    failed_call = session.query(Call).filter(Call.lead_id == first.id).one()
    assert failed_call.status == CallStatus.REFUSED


def test_one_plivo_client_is_shared_across_dials(monkeypatch):
    from app.providers import factory
    from app.providers.plivo import PlivoTelephony

    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setenv("PLIVO_AUTH_ID", "AUTH_ID")
    monkeypatch.setenv("PLIVO_AUTH_TOKEN", "AUTH_TOKEN")
    try:
        first, second = factory.get_telephony(), factory.get_telephony()
        assert isinstance(first, PlivoTelephony)
        assert first is second
    finally:
        factory.reset_telephony()


def test_without_plivo_credentials_calls_go_to_the_mock(monkeypatch):
    from app.providers import factory

    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("PLIVO_AUTH_ID", raising=False)
    assert isinstance(factory.get_telephony(), MockTelephony)
