"""The auto-dialer's on/off switch and progress live in the database, and a lease keeps it on one worker."""

from datetime import datetime, time, timedelta
from uuid import uuid4

from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, Lead, LeadStatus, LedgerEntry, LedgerKind, Tenant
from app.services import autodialer


def ready_tenant(contacts: int = 2) -> int:
    """A tenant whose agent can dial right now, with `contacts` due contacts."""
    with SessionLocal() as db:
        tenant = Tenant(name="T", api_key_hash=f"h-{uuid4().hex}", webhook_secret="s")
        db.add(tenant)
        db.flush()
        emp = AIEmployee(tenant_id=tenant.id, name="A", script="", phone_number="+914000000000", max_concurrency=40,
                         window_start=time(0, 0), window_end=time(23, 59))
        db.add_all([emp, LedgerEntry(tenant_id=tenant.id, kind=LedgerKind.TOPUP, amount_paise=10_000)])
        db.flush()
        due = datetime.utcnow() - timedelta(minutes=1)
        db.add_all([Lead(tenant_id=tenant.id, employee_id=emp.id, phone_e164=f"+91700000{n:04d}",
                         status=LeadStatus.NEW, next_attempt_at=due) for n in range(contacts)])
        db.commit()
        return tenant.id


def test_start_and_stop_are_saved_in_the_database(no_dialer_threads):
    tid = ready_tenant()

    assert autodialer.start(tid)["status"] == "started"
    assert no_dialer_threads == [tid]
    assert autodialer.status(tid)["active"] is True
    assert autodialer.start(tid)["status"] == "already_running"

    autodialer.stop(tid)
    assert autodialer.status(tid)["active"] is False


def test_status_of_a_tenant_that_never_dialed():
    assert autodialer.status(ready_tenant()) == {"active": False, "total_dialed": 0, "remaining": 0, "waiting": ""}


def test_a_pass_dials_due_contacts_and_saves_progress():
    tid = ready_tenant(contacts=2)
    autodialer.start(tid)

    result = autodialer.run_pass(tid)

    assert result.dialed == 2 and not result.finished
    state = autodialer.status(tid)
    assert (state["active"], state["total_dialed"], state["remaining"]) == (True, 2, 0)


def test_the_dialer_keeps_running_while_calls_are_in_flight():
    """A call still ringing may come back unanswered and need a retry."""
    tid = ready_tenant(contacts=1)
    autodialer.start(tid)
    autodialer.run_pass(tid)

    assert not autodialer.run_pass(tid).finished
    assert autodialer.status(tid)["active"] is True


def test_the_dialer_stops_itself_when_nothing_is_left():
    tid = ready_tenant(contacts=0)
    autodialer.start(tid)

    assert autodialer.run_pass(tid).finished
    assert autodialer.status(tid)["active"] is False


def test_a_call_that_never_reported_back_is_requeued_for_retry():
    tid = ready_tenant(contacts=0)
    with SessionLocal() as db:
        emp = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).one()
        lead = Lead(tenant_id=tid, employee_id=emp.id, phone_e164="+917000009999", status=LeadStatus.CALLING,
                    attempts=1, next_attempt_at=datetime.utcnow())
        db.add(lead)
        db.flush()
        ringing_too_long = datetime.utcnow() - timedelta(minutes=5)
        db.add(Call(tenant_id=tid, lead_id=lead.id, employee_id=emp.id, status=CallStatus.DIALING,
                    started_at=ringing_too_long))
        db.commit()
        lead_id = lead.id
    autodialer.start(tid)

    autodialer.run_pass(tid)

    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        call = db.query(Call).filter(Call.lead_id == lead_id).one()
        assert call.status == CallStatus.NO_ANSWER
        assert lead.status == LeadStatus.QUEUED and lead.next_attempt_at > datetime.utcnow()


def test_only_one_worker_holds_a_tenants_lease():
    tid = ready_tenant()
    autodialer.start(tid)
    now = datetime.utcnow()

    assert autodialer.acquire_lease(tid, "worker-1", now)
    assert autodialer.acquire_lease(tid, "worker-1", now)  # renewing its own lease
    assert not autodialer.acquire_lease(tid, "worker-2", now)
    assert autodialer.acquire_lease(tid, "worker-2", now + autodialer.LEASE + timedelta(seconds=1))


def test_a_stopped_dialer_cannot_be_leased():
    tid = ready_tenant()
    autodialer.start(tid)
    autodialer.stop(tid)
    assert not autodialer.acquire_lease(tid, "worker-1", datetime.utcnow())


def test_the_worker_stops_and_hands_back_its_lease_when_done():
    from app.models import AutoDialer

    tid = ready_tenant(contacts=0)
    autodialer.start(tid)

    autodialer._work(tid)  # returns once nothing is left to call

    with SessionLocal() as db:
        state = db.get(AutoDialer, tid)
        assert (state.active, state.lease_owner) == (False, None)


def test_a_worker_that_was_stopped_does_not_dial():
    tid = ready_tenant(contacts=2)
    autodialer.start(tid)
    autodialer.stop(tid)

    autodialer._work(tid)

    assert autodialer.status(tid)["total_dialed"] == 0


def test_a_restart_resumes_dialers_that_were_running(no_dialer_threads):
    running, stopped = ready_tenant(), ready_tenant()
    autodialer.start(running)
    autodialer.start(stopped)
    autodialer.stop(stopped)
    no_dialer_threads.clear()

    autodialer.resume_all()

    assert no_dialer_threads == [running]


def login(client) -> None:
    assert client.post("/dashboard/api/auth/login", json={"email": "admin@suma.ai", "password": "suma2026"}).status_code == 200


def test_emergency_pause_is_saved_in_the_database(client):
    login(client)
    client.post("/dashboard/api/employees", json={"name": "Suma"})
    tid = client.get("/dashboard/api/auth/me").json()["tenant_id"]

    client.post("/dashboard/api/pause-all")
    assert autodialer.status(tid)["active"] is False
    assert autodialer.is_paused(tid) is True
    assert client.get("/dashboard/api/live-calls").json()["paused"] is True

    client.post("/dashboard/api/resume-all")
    assert autodialer.is_paused(tid) is False
    assert client.get("/dashboard/api/autocall/status").json()["active"] is True
