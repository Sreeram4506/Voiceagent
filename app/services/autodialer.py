"""A tenant's auto-dialer: keeps placing calls for the tenant's due contacts until none are left.

Its on/off switch, emergency pause and progress live in the auto_dialers table, so any worker can start or
stop it and a restart picks it back up. A lease in the same row makes sure only one worker dials for a
tenant at a time; if that worker dies, another takes over once the lease runs out.
"""

import logging
import os
import socket
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.models import AIEmployee, AutoDialer, Call, CallStatus, Lead, LeadStatus
from app.providers.factory import get_telephony
from app.services.compliance import InMemoryDNDRegistry
from app.services.retries import settle_lead, utc_naive
from app.services.scheduler import DIALABLE, DispatchReport, dispatch_due, explain_wait

log = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
LEASE = timedelta(seconds=120)
MAX_LINES = 40  # calls at once per tenant
DIALING_TIMEOUT = timedelta(seconds=120)  # rings for 30 s at most; still "dialing" after this means no callback came
STUCK_CALL = timedelta(minutes=30)  # Plivo cuts calls at 10 minutes; one "connected" this long lost its callback
BUSY_PAUSE_SECONDS = 15  # after dialing, or while every line is busy or another worker holds the lease
IDLE_PAUSE_SECONDS = 10
IN_FLIGHT = (CallStatus.DIALING, CallStatus.CONNECTED)

_threads: dict[int, threading.Thread] = {}  # this process's dialer threads; the state itself is in the database
_threads_lock = threading.Lock()


@dataclass(frozen=True)
class PassResult:
    dialed: int = 0
    finished: bool = False
    lines_full: bool = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── switch and status ──────────────────────────────────────────────────────


def start(tenant_id: int) -> dict:
    """Turn the dialer on and make sure a thread in this process works it."""
    with SessionLocal() as db:
        state = _state(db, tenant_id)
        if state.active:
            return {"status": "already_running", **_progress(state)}
        state.active = True
        state.total_dialed, state.remaining, state.waiting = 0, _pending(db, tenant_id), ""
        db.commit()
        progress = _progress(state)
    _spawn_worker(tenant_id)
    return {"status": "started", **progress}


def stop(tenant_id: int) -> dict:
    """Turn the dialer off; whichever worker runs it stops before its next pass."""
    with SessionLocal() as db:
        state = _state(db, tenant_id)
        state.active = False
        db.commit()
        return {"status": "stopped", **_progress(state)}


def status(tenant_id: int) -> dict:
    with SessionLocal() as db:
        state = db.get(AutoDialer, tenant_id)
        if state is None:
            return {"active": False, "total_dialed": 0, "remaining": 0, "waiting": ""}
        return {"active": state.active, **_progress(state), "waiting": state.waiting or ""}


def is_paused(tenant_id: int) -> bool:
    with SessionLocal() as db:
        state = db.get(AutoDialer, tenant_id)
        return bool(state and state.paused)


def set_paused(tenant_id: int, paused: bool) -> None:
    """The dashboard's emergency pause. Pausing also turns the dialer off."""
    with SessionLocal() as db:
        state = _state(db, tenant_id)
        state.paused = paused
        if paused:
            state.active = False
        db.commit()


def resume_all() -> None:
    """At startup: work every dialer that was on when the server last stopped."""
    with SessionLocal() as db:
        tenant_ids = db.scalars(select(AutoDialer.tenant_id).where(AutoDialer.active.is_(True))).all()
    for tenant_id in tenant_ids:
        _spawn_worker(tenant_id)


def _progress(state: AutoDialer) -> dict:
    return {"total_dialed": state.total_dialed or 0, "remaining": state.remaining or 0}


def _state(db: Session, tenant_id: int) -> AutoDialer:
    state = db.get(AutoDialer, tenant_id)
    if state is None:
        _create_state(tenant_id)
        state = db.get(AutoDialer, tenant_id)
    return state


def _create_state(tenant_id: int) -> None:
    with SessionLocal() as db:
        db.add(AutoDialer(tenant_id=tenant_id, active=False, paused=False, total_dialed=0, remaining=0, waiting=""))
        try:
            db.commit()
        except IntegrityError:  # another worker created it a moment ago
            db.rollback()


# ── lease ──────────────────────────────────────────────────────────────────


def acquire_lease(tenant_id: int, owner: str = WORKER_ID, now: datetime | None = None) -> bool:
    """Take or renew the right to dial for this tenant. False while another live worker holds it."""
    now = utc_naive(now or _utcnow())
    with SessionLocal() as db:
        taken = db.execute(
            update(AutoDialer)
            .where(
                AutoDialer.tenant_id == tenant_id,
                AutoDialer.active.is_(True),
                or_(
                    AutoDialer.lease_owner.is_(None),
                    AutoDialer.lease_owner == owner,
                    AutoDialer.lease_expires_at < now,
                ),
            )
            .values(lease_owner=owner, lease_expires_at=now + LEASE)
            .execution_options(synchronize_session=False)
        ).rowcount == 1
        db.commit()
        return taken


def release_lease(tenant_id: int, owner: str = WORKER_ID) -> None:
    with SessionLocal() as db:
        db.execute(
            update(AutoDialer)
            .where(AutoDialer.tenant_id == tenant_id, AutoDialer.lease_owner == owner)
            .values(lease_owner=None, lease_expires_at=None)
            .execution_options(synchronize_session=False)
        )
        db.commit()


# ── one pass ───────────────────────────────────────────────────────────────


def run_pass(tenant_id: int, now: datetime | None = None) -> PassResult:
    """Free lines whose calls never reported back, then dial as many due contacts as there are free lines."""
    now = utc_naive(now or _utcnow())
    with SessionLocal() as db:
        _free_stuck_lines(db, tenant_id, now)
        pending, in_flight = _pending(db, tenant_id), _in_flight(db, tenant_id)
        state = _state(db, tenant_id)
        if pending == 0 and in_flight == 0:
            state.active, state.remaining, state.waiting = False, 0, ""
            db.commit()
            return PassResult(finished=True)
        if in_flight >= MAX_LINES:
            state.remaining = pending
            db.commit()
            return PassResult(lines_full=True)

        employee = db.scalar(
            select(AIEmployee).where(AIEmployee.tenant_id == tenant_id).order_by(AIEmployee.id).limit(1)
        )
        if employee and employee.max_concurrency < MAX_LINES:
            employee.max_concurrency = MAX_LINES
        db.commit()

        settings = get_settings()
        report = dispatch_due(
            db,
            telephony=get_telephony(),
            dnd=InMemoryDNDRegistry(),
            now=now,
            callback_url=f"{settings.public_url}/v1/plivo/status",
            retry_delay_minutes=settings.retry_delay_minutes,
            batch_size=MAX_LINES - in_flight,
        )
        state.total_dialed = (state.total_dialed or 0) + len(report.dialed)
        state.remaining = _pending(db, tenant_id)
        state.waiting = _waiting(tenant_id, report, employee, state.waiting or "")
        db.commit()
        return PassResult(dialed=len(report.dialed))


def _pending(db: Session, tenant_id: int) -> int:
    """Contacts still to be called, including retries waiting for their time."""
    return db.scalar(
        select(func.count(Lead.id)).where(Lead.tenant_id == tenant_id, Lead.status.in_(DIALABLE))
    ) or 0


def _in_flight(db: Session, tenant_id: int) -> int:
    return db.scalar(
        select(func.count(Call.id)).where(Call.tenant_id == tenant_id, Call.status.in_(IN_FLIGHT))
    ) or 0


def _free_stuck_lines(db: Session, tenant_id: int, now: datetime) -> None:
    """Close calls whose hang-up callback never came, so they stop holding lines and their leads move on."""
    unanswered = db.scalars(select(Call).where(
        Call.tenant_id == tenant_id, Call.status == CallStatus.DIALING, Call.started_at < now - DIALING_TIMEOUT,
    )).all()
    for call in unanswered:
        _close(db, call, CallStatus.NO_ANSWER, lead_result=CallStatus.NO_ANSWER, now=now)
    stuck = db.scalars(select(Call).where(
        Call.tenant_id == tenant_id, Call.status == CallStatus.CONNECTED, Call.started_at < now - STUCK_CALL,
    )).all()
    for call in stuck:
        # The person was reached, so the lead is done even though the call's end was never reported.
        _close(db, call, CallStatus.FAILED, lead_result=CallStatus.COMPLETED, now=now)
    if stuck:
        log.warning("[AUTO-CALL][tid=%s] freed %d lines held by calls stuck as connected", tenant_id, len(stuck))
    if unanswered or stuck:
        db.commit()


def _close(db: Session, call: Call, status: CallStatus, lead_result: CallStatus, now: datetime) -> None:
    call.status = status
    call.ended_at = call.ended_at or now
    lead = db.get(Lead, call.lead_id) if call.lead_id else None
    if lead is not None and lead.status == LeadStatus.CALLING:
        settle_lead(lead, db.get(AIEmployee, call.employee_id), lead_result, now)


def _waiting(tenant_id: int, report: DispatchReport, employee: AIEmployee | None, previous: str) -> str:
    """Why contacts are waiting when nothing could be dialed, in plain words; logged when it changes."""
    if report.dialed or not report.skipped:
        return ""
    reason = Counter(report.skipped.values()).most_common(1)[0][0]
    waiting = explain_wait(reason, employee)
    if waiting != previous:
        log.info("[AUTO-CALL][tid=%s] %d contacts waiting: %s", tenant_id, len(report.skipped), waiting)
    return waiting


# ── worker thread ──────────────────────────────────────────────────────────


def _spawn_worker(tenant_id: int) -> None:
    with _threads_lock:
        running = _threads.get(tenant_id)
        if running is not None and running.is_alive():
            return
        thread = threading.Thread(target=_work, args=(tenant_id,), name=f"autodialer-{tenant_id}", daemon=True)
        _threads[tenant_id] = thread
        thread.start()


def _work(tenant_id: int) -> None:
    """Run passes while the dialer is on. Errors are logged and retried, never allowed to end the dialer."""
    try:
        while True:
            try:
                pause = _work_once(tenant_id)
            except Exception:
                log.exception("[AUTO-CALL][tid=%s] pass failed; trying again shortly", tenant_id)
                pause = IDLE_PAUSE_SECONDS
            if pause is None:
                return
            time.sleep(pause)
    finally:
        try:
            release_lease(tenant_id)
        except Exception:
            log.exception("[AUTO-CALL][tid=%s] could not release the lease; it expires on its own", tenant_id)
        with _threads_lock:
            if _threads.get(tenant_id) is threading.current_thread():
                del _threads[tenant_id]


def _work_once(tenant_id: int) -> float | None:
    """One step of the worker loop: seconds to wait before the next, or None when the dialer is done."""
    if not status(tenant_id)["active"]:
        return None
    if not acquire_lease(tenant_id):
        return BUSY_PAUSE_SECONDS  # another worker is dialing for this tenant
    result = run_pass(tenant_id)
    if result.finished:
        return None
    return BUSY_PAUSE_SECONDS if result.dialed or result.lines_full else IDLE_PAUSE_SECONDS
