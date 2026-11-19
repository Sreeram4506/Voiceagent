"""Campaign launch: it must say why calls can't start, and re-launches must really call again."""

from datetime import datetime, time, timedelta, timezone

from app.db import SessionLocal
from app.models import AIEmployee, ContactList, Lead, LeadStatus, LedgerEntry, LedgerKind
from app.services.scheduler import dial_blocker, explain_wait


def login(client):
    assert client.post("/dashboard/api/auth/login", json={"email": "admin@suma.ai", "password": "suma2026"}).status_code == 200


def stop_dialer(tid: int) -> None:
    from app.dashboard import _auto_call_active

    _auto_call_active[tid] = False


def agent_with_list(client, *, window: tuple[time, time], attempts: int) -> tuple[int, int, int]:
    """An agent ready to dial (number + credits) with a two-contact list that has used its attempts."""
    eid = client.post("/dashboard/api/employees", json={"name": "Suma"}).json()["id"]
    with SessionLocal() as db:
        emp = db.get(AIEmployee, eid)
        emp.phone_number, emp.window_start, emp.window_end = "+914000000000", *window
        db.add(LedgerEntry(tenant_id=emp.tenant_id, kind=LedgerKind.TOPUP, amount_paise=10_000))
        cl = ContactList(tenant_id=emp.tenant_id, name="List", filename="list.csv", contact_count=2)
        db.add(cl)
        db.flush()
        for n in range(2):
            db.add(Lead(tenant_id=emp.tenant_id, employee_id=eid, phone_e164=f"+9199999000{n}", status=LeadStatus.DONE,
                        source=f"list:{cl.id}", attempts=attempts, next_attempt_at=datetime.utcnow()))
        db.commit()
        return eid, cl.id, emp.tenant_id


def test_launch_outside_calling_hours_says_so_and_resets_attempts(client):
    login(client)
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    closed = (time((now_ist.hour + 2) % 24, 0), time((now_ist.hour + 3) % 24, 0))  # a window that is not now
    eid, list_id, tid = agent_with_list(client, window=closed, attempts=3)
    cid = client.post("/dashboard/api/campaigns", json={"name": "Night", "employee_id": eid, "contact_list_id": list_id}).json()["id"]
    try:
        r = client.post(f"/dashboard/api/campaigns/{cid}/launch").json()
    finally:
        stop_dialer(tid)
    assert r["ok"] and "outside calling hours" in r["warning"]
    with SessionLocal() as db:
        leads = db.query(Lead).filter(Lead.source == f"list:{list_id}").all()
        assert {(lead.status, lead.attempts) for lead in leads} == {(LeadStatus.NEW, 0)}


def test_launch_with_nobody_to_call_says_so(client):
    login(client)
    cid = client.post("/dashboard/api/campaigns", json={"name": "Empty"}).json()["id"]
    r = client.post(f"/dashboard/api/campaigns/{cid}/launch").json()
    assert r["warning"] == explain_wait("no_contacts")


def test_excel_upload_requeues_a_number_that_was_called_before(client):
    login(client)
    eid, _, tid = agent_with_list(client, window=(time(0, 0), time(23, 59)), attempts=3)
    csv = b"name,phone\nRavi,+919999900000\n"
    try:
        r = client.post(f"/dashboard/api/employees/{eid}/excel-campaign",
                        files={"file": ("list.csv", csv, "text/csv")}, data={"campaign_name": "Again"}).json()
    finally:
        stop_dialer(tid)
    assert r["added"] == 1 and r["warning"] is None
    with SessionLocal() as db:
        lead = db.query(Lead).filter(Lead.phone_e164 == "+919999900000").one()
        assert (lead.status, lead.attempts, lead.campaign_id) == (LeadStatus.NEW, 0, r["campaign_id"])


def test_dial_blocker_names_the_first_thing_in_the_way(session):
    emp = AIEmployee(tenant_id=1, name="A", script="", window_start=time(0, 0), window_end=time(23, 59),
                     timezone="Asia/Kolkata", paused=False, phone_number=None)
    assert dial_blocker(session, emp, datetime.now(timezone.utc)) == explain_wait("no_phone_number_assigned")
    emp.phone_number = "+914000000000"
    assert dial_blocker(session, emp, datetime.now(timezone.utc)) == explain_wait("no_credits")
