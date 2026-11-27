"""Database, Plivo and file work stays off the event loop that carries every live call's audio."""

import asyncio
import inspect
import json
from datetime import datetime, time
from uuid import uuid4

from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, CallTurn, Lead, LeadStatus, LedgerEntry, Tenant
from app.streaming import factory, ws
from app.streaming.mock_providers import MockAsyncLLM, MockAsyncSTT, MockAsyncTTS


def loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def spy_off_loop(monkeypatch, module, name: str) -> list[bool]:
    """Wrap module.name so each call records whether it ran on the event loop thread."""
    original = getattr(module, name)
    on_loop: list[bool] = []

    def wrapper(*args, **kwargs):
        on_loop.append(loop_running())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return on_loop


BLOCKING_ROUTES = {
    ("POST", "/dashboard/api/employees/{employee_id}/single-call"),
    ("POST", "/dashboard/api/employees/{employee_id}/excel-campaign"),
    ("POST", "/dashboard/api/employees/{employee_id}/audio"),
    ("POST", "/dashboard/api/upload"),
    ("POST", "/dashboard/api/voice/profile"),
    ("POST", "/dashboard/api/voice/test"),
    ("GET", "/dashboard/"),
    ("GET", "/dashboard/single"),
    ("GET", "/dashboard/campaign"),
    ("GET", "/dashboard/history"),
    ("GET", "/dashboard/ai-agent"),
    ("GET", "/dashboard/api/auth/me"),
    ("GET", "/dashboard/api/tenants"),
}


def test_routes_that_touch_the_database_or_files_run_in_the_threadpool():
    from app.main import app

    found = set()
    for route in app.routes:
        for method in getattr(route, "methods", ()) or ():
            if (method, route.path) in BLOCKING_ROUTES:
                found.add((method, route.path))
                assert not inspect.iscoroutinefunction(route.endpoint), f"{method} {route.path} blocks the event loop"
    assert found == BLOCKING_ROUTES


def test_login_hashes_the_password_off_the_event_loop(client, monkeypatch):
    import app.main as main

    on_loop = spy_off_loop(monkeypatch, main, "authenticate_user")
    r = client.post("/dashboard/api/auth/login", json={"email": "admin@suma.ai", "password": "suma2026"})
    assert r.status_code == 200
    assert on_loop == [False]


def test_logout_runs_off_the_event_loop(client, monkeypatch):
    import app.main as main

    on_loop = spy_off_loop(monkeypatch, main, "destroy_session")
    client.post("/dashboard/api/auth/logout")
    assert on_loop == [False]


def test_plivo_webhooks_write_the_database_off_the_event_loop(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "prepare_call", lambda call_id: None)
    answered = spy_off_loop(monkeypatch, main, "record_answer")
    hung_up = spy_off_loop(monkeypatch, main, "record_hangup")
    client.post("/v1/plivo/answer/1")
    client.post("/v1/plivo/status/1", data={"CallStatus": "completed"})
    assert answered == [False] and hung_up == [False]


def seed_call() -> int:
    with SessionLocal() as db:
        tenant = Tenant(name="T", api_key_hash=f"h-{uuid4().hex}", webhook_secret="s")
        db.add(tenant)
        db.flush()
        emp = AIEmployee(tenant_id=tenant.id, name="A", script="Hi.", window_start=time(0, 0), window_end=time(23, 59))
        db.add(emp)
        db.flush()
        lead = Lead(tenant_id=tenant.id, employee_id=emp.id, phone_e164="+919999900001", status=LeadStatus.CALLING,
                    next_attempt_at=datetime.utcnow())
        db.add(lead)
        db.flush()
        call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=emp.id, status=CallStatus.CONNECTED,
                    started_at=datetime.utcnow())
        db.add(call)
        db.commit()
        return call.id


def call_status(call_id: int) -> CallStatus:
    with SessionLocal() as db:
        return db.get(Call, call_id).status


def test_live_call_loads_and_saves_off_the_event_loop(client, monkeypatch, eventually):
    monkeypatch.setattr(factory, "build_providers",
                        lambda employee, extras=None: (MockAsyncSTT([]), MockAsyncLLM([[]]), MockAsyncTTS()))
    loads = spy_off_loop(monkeypatch, ws, "_load_context")
    saves = spy_off_loop(monkeypatch, ws, "_persist")
    call_id = seed_call()

    with client.websocket_connect(f"/v1/stream/{call_id}?token={ws.stream_token(call_id)}") as sock:
        sock.send_text(json.dumps({"event": "start", "stream_sid": "S1"}))
        sock.send_text(json.dumps({"event": "stop"}))
        # Wait while the socket is open: leaving the block makes the test client cancel the handler.
        assert eventually(lambda: call_status(call_id) == CallStatus.COMPLETED)

    assert loads and not any(loads)
    assert saves and not any(saves)


class FinishedCall:
    """The parts of a live session that saving a call reads."""

    def __init__(self):
        self.turns = [("agent", "Namaskaram"), ("caller", "Cheppandi")]
        self.actions: list[str] = []
        self.frames_received = 3000  # 60 s of audio
        self.persisted = False

    def outcome(self) -> str:
        return "completed"


def test_a_call_ending_twice_at_once_is_saved_and_billed_once():
    """The end-of-call callback and the socket closing can both save the call at the same moment."""
    call_id = seed_call()
    with SessionLocal() as db:
        tenant_id = db.get(Call, call_id).tenant_id
    session = FinishedCall()

    async def end_twice() -> None:
        await asyncio.gather(ws._persist_once(call_id, session), ws._persist_once(call_id, session))

    asyncio.run(end_twice())

    with SessionLocal() as db:
        assert db.query(CallTurn).filter(CallTurn.call_id == call_id).count() == 2
        assert db.query(LedgerEntry).filter(LedgerEntry.tenant_id == tenant_id).count() == 1
