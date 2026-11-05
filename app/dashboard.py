"""Dashboard API endpoints for Excel upload, call management, and data viewing."""

import io
import os
import re
import threading
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse

from app.db import SessionLocal
from app.models import AIEmployee, Call, CallDirection, CallStatus, CallTurn, ContactList, Lead, LeadStatus

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Per-tenant auto-call state  (tenant_id -> value)
_auto_call_active: dict[int, bool] = {}
_auto_call_stats: dict[int, dict] = {}

# Per-tenant AI agent config
_ai_agent_config: dict[int, dict] = {}


def _to_e164(raw: str) -> str:
    digits = re.sub(r"\D", "", str(raw))
    if digits.startswith("0"):
        digits = "91" + digits[1:]
    if not digits.startswith("91"):
        digits = "91" + digits
    return "+" + digits


def _suma_html() -> HTMLResponse:
    from pathlib import Path
    html = (Path(__file__).parent / "suma.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


def _require_tid(request: Request) -> int:
    from app.auth import require_tenant_id
    return require_tenant_id(request)


def _default_agent_config(tid: int) -> dict:
    return {
        "system_prompt": (
            "Nuvvu oka political campaign assistant vi. Candidate gari tarapuna phone chestunnav. "
            "Telugu lo natural ga maatlaadu — like a real person, not a robot. "
            "Short ga cheppu, friendly ga undu. Vaallaki interest lekapothe politely end cheyyi. "
            "Vaallaki candidate gurinchi brief ga cheppu, support adugu. "
            "Ekkuva formality vaddu — normal phone conversation la undu."
        ),
        "greeting": "",
        "language": "te",
        "voice": "default",
    }


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    from app.auth import get_session_data
    if not get_session_data(request):
        return RedirectResponse("/login?next=/dashboard/", status_code=302)
    return _suma_html()


@router.get("/single", response_class=HTMLResponse)
async def single_call_page(request: Request):
    from app.auth import get_session_data
    if not get_session_data(request):
        return RedirectResponse("/login?next=/dashboard/single", status_code=302)
    return _suma_html()


@router.get("/campaign", response_class=HTMLResponse)
async def campaign_page(request: Request):
    from app.auth import get_session_data
    if not get_session_data(request):
        return RedirectResponse("/login?next=/dashboard/campaign", status_code=302)
    return _suma_html()


@router.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    from app.auth import get_session_data
    if not get_session_data(request):
        return RedirectResponse("/login?next=/dashboard/history", status_code=302)
    return _suma_html()


@router.get("/api/stats")
def get_stats(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        total_leads = db.query(Lead).filter(Lead.tenant_id == tid).count()
        new_leads = db.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.NEW).count()
        calling = db.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.CALLING).count()
        done = db.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.DONE).count()
        failed = db.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.FAILED).count()
        total_calls = db.query(Call).filter(Call.tenant_id == tid).count()
        completed_calls = db.query(Call).filter(Call.tenant_id == tid, Call.status == CallStatus.COMPLETED).count()
        unanswered_busy = db.query(Call).filter(
            Call.tenant_id == tid, Call.status.in_([CallStatus.NO_ANSWER, CallStatus.BUSY])
        ).count()
        return {
            "total_leads": total_leads,
            "new_leads": new_leads,
            "calling": calling,
            "done": done,
            "failed": failed,
            "total_calls": total_calls,
            "completed_calls": completed_calls,
            "unanswered_busy": unanswered_busy,
        }


@router.get("/api/leads")
def list_leads(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        leads = db.query(Lead).filter(Lead.tenant_id == tid).order_by(Lead.id.desc()).limit(1000).all()
        return [
            {
                "id": l.id,
                "name": l.name or "",
                "phone": l.phone_e164,
                "status": l.status.value,
                "attempts": l.attempts,
                "last_called": l.last_called_at.isoformat() if l.last_called_at else None,
                "created": l.created_at.isoformat() if l.created_at else None,
            }
            for l in leads
        ]


@router.get("/api/calls")
def list_calls(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        calls = db.query(Call).filter(Call.tenant_id == tid).order_by(Call.id.desc()).limit(1000).all()
        result = []
        for c in calls:
            lead = db.get(Lead, c.lead_id) if c.lead_id else None
            turns = db.query(CallTurn).filter_by(call_id=c.id).order_by(CallTurn.id).all()
            cf = c.captured_fields or {}
            result.append({
                "id": c.id,
                "phone": lead.phone_e164 if lead else cf.get("phone", "—"),
                "name": lead.name if lead else cf.get("name", "—"),
                "status": c.status.value,
                "outcome": c.outcome or "—",
                "duration": c.billed_seconds,
                "started": c.started_at.isoformat() if c.started_at else None,
                "turns": [{"role": t.role, "text": t.text} for t in turns],
            })
        return result


@router.post("/api/add_lead")
def add_lead_manually(request: Request, payload: dict):
    tid = _require_tid(request)
    phone_raw = str(payload.get("phone", "")).strip()
    name = str(payload.get("name", "")).strip() or None
    if not phone_raw:
        return {"error": "Phone number is required"}
    phone = _to_e164(phone_raw)
    if len(phone) < 12:
        return {"error": "Invalid phone number"}
    with SessionLocal() as db:
        existing = db.query(Lead).filter(Lead.tenant_id == tid, Lead.phone_e164 == phone).first()
        if existing:
            return {"error": "Lead already exists"}
        emp = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).first()
        lead = Lead(
            tenant_id=tid,
            employee_id=emp.id if emp else 1,
            phone_e164=phone,
            name=name,
            status=LeadStatus.NEW,
            source="manual",
            next_attempt_at=datetime.now(timezone.utc),
        )
        db.add(lead)
        db.commit()
        db.refresh(lead)
        return {"added": phone, "name": name, "id": lead.id}


@router.post("/api/recall/{lead_id}")
def recall_lead(lead_id: int):
    import os
    from app.config import get_settings
    from app.providers.mock import MockTelephony
    from app.services.scheduler import dispatch_due

    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        if not lead:
            return {"error": "Lead not found"}
        lead.status = LeadStatus.NEW
        lead.next_attempt_at = datetime.now(timezone.utc)
        db.commit()

        settings = get_settings()
        auth_id = os.getenv("PLIVO_AUTH_ID")
        if auth_id:
            from app.providers.plivo import PlivoTelephony
            telephony = PlivoTelephony(auth_id, os.environ["PLIVO_AUTH_TOKEN"], public_url=settings.public_url)
        else:
            telephony = MockTelephony()

        from app.services.compliance import InMemoryDNDRegistry
        report = dispatch_due(
            db,
            telephony=telephony,
            dnd=InMemoryDNDRegistry(),
            now=datetime.now(timezone.utc),
            callback_url=f"{settings.public_url}/v1/plivo/status",
            retry_delay_minutes=settings.retry_delay_minutes,
        )
        db.commit()
        return {"dialed": report.dialed}


@router.post("/api/upload")
async def upload_excel(request: Request, file: UploadFile = File(...)):
    import openpyxl
    tid = _require_tid(request)

    content = await file.read()
    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
    ws = wb.active

    headers = [str(c.value or "").strip().lower() for c in next(ws.iter_rows(min_row=1, max_row=1))]
    phone_col = None
    name_col = None
    for i, h in enumerate(headers):
        if "phone" in h or "mobile" in h or "number" in h or "contact" in h:
            phone_col = i
        if "name" in h:
            name_col = i

    if phone_col is None:
        for i, h in enumerate(headers):
            if h:
                phone_col = i
                break

    if phone_col is None:
        return {"error": "No phone column found", "added": 0}

    added = 0
    skipped = 0
    with SessionLocal() as db:
        emp = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).first()
        employee_id = emp.id if emp else 1

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or not row[phone_col]:
                continue
            raw_phone = str(row[phone_col]).strip()
            if not raw_phone or raw_phone == "None":
                continue
            phone = _to_e164(raw_phone)
            if len(phone) < 12:
                skipped += 1
                continue

            existing = db.query(Lead).filter(Lead.tenant_id == tid, Lead.phone_e164 == phone).first()
            if existing:
                skipped += 1
                continue

            name = str(row[name_col]).strip() if name_col is not None and row[name_col] else None
            db.add(Lead(
                tenant_id=tid,
                employee_id=employee_id,
                phone_e164=phone,
                name=name,
                status=LeadStatus.NEW,
                source="excel",
                next_attempt_at=datetime.now(timezone.utc),
            ))
            added += 1
        cl = ContactList(
            tenant_id=tid,
            name=file.filename or "Uploaded list",
            filename=file.filename,
            contact_count=added,
        )
        db.add(cl)
        db.flush()
        if added:
            db.query(Lead).filter(Lead.source == "excel", Lead.campaign_id.is_(None)).update(
                {"source": f"list:{cl.id}"}, synchronize_session=False
            )
        db.commit()
    wb.close()
    return {"added": added, "skipped": skipped, "list_id": cl.id}


@router.post("/api/dispatch")
def dispatch_all(request: Request):
    from app.config import get_settings
    from app.providers.mock import MockTelephony
    from app.services.compliance import InMemoryDNDRegistry
    from app.services.scheduler import dispatch_due
    import os
    tid = _require_tid(request)

    with SessionLocal() as db:
        emp = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).first()
        if emp and emp.max_concurrency < 40:
            emp.max_concurrency = 40
            db.flush()

        settings = get_settings()
        auth_id = os.getenv("PLIVO_AUTH_ID")
        if auth_id:
            from app.providers.plivo import PlivoTelephony
            telephony = PlivoTelephony(auth_id, os.environ["PLIVO_AUTH_TOKEN"], public_url=settings.public_url)
        else:
            telephony = MockTelephony()

        report = dispatch_due(
            db,
            telephony=telephony,
            dnd=InMemoryDNDRegistry(),
            now=datetime.now(timezone.utc),
            callback_url=f"{settings.public_url}/v1/plivo/status",
            retry_delay_minutes=settings.retry_delay_minutes,
            batch_size=40,
        )
        db.commit()
        return {"dialed": report.dialed, "skipped": report.skipped}


def _auto_call_loop(tenant_id: int):
    import os
    import time
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as SASession
    from app.config import get_settings
    from app.services.compliance import InMemoryDNDRegistry
    from app.services.scheduler import dispatch_due

    engine = create_engine(get_settings().database_url, connect_args={"check_same_thread": False})
    _auto_call_stats[tenant_id] = {"total_dialed": 0, "remaining": 0}

    while _auto_call_active.get(tenant_id):
        should_wait = False
        did_dial = False

        try:
            with SASession(engine) as db:
                remaining = db.query(Lead).filter(Lead.tenant_id == tenant_id, Lead.status == LeadStatus.NEW).count()
                _auto_call_stats[tenant_id]["remaining"] = remaining
                if remaining == 0:
                    _auto_call_active[tenant_id] = False
                    break

                stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=45)
                stale = db.query(Call).filter(
                    Call.tenant_id == tenant_id,
                    Call.status.in_([CallStatus.DIALING, CallStatus.CONNECTED]),
                    Call.started_at < stale_cutoff,
                ).all()
                for c in stale:
                    c.status = CallStatus.COMPLETED
                if stale:
                    db.commit()

                in_flight = db.query(Call).filter(
                    Call.tenant_id == tenant_id,
                    Call.status.in_([CallStatus.DIALING, CallStatus.CONNECTED])
                ).count()
                if in_flight >= 40:
                    should_wait = True
                else:
                    batch = 40 - in_flight
                    emp = db.query(AIEmployee).filter(AIEmployee.tenant_id == tenant_id).first()
                    if emp and emp.max_concurrency < 40:
                        emp.max_concurrency = 40
                        db.flush()

                    settings = get_settings()
                    auth_id = os.getenv("PLIVO_AUTH_ID")
                    if auth_id:
                        from app.providers.plivo import PlivoTelephony
                        telephony = PlivoTelephony(auth_id, os.environ["PLIVO_AUTH_TOKEN"], public_url=settings.public_url)
                    else:
                        from app.providers.mock import MockTelephony
                        telephony = MockTelephony()

                    report = dispatch_due(
                        db,
                        telephony=telephony,
                        dnd=InMemoryDNDRegistry(),
                        now=datetime.now(timezone.utc),
                        callback_url=f"{settings.public_url}/v1/plivo/status",
                        retry_delay_minutes=settings.retry_delay_minutes,
                        batch_size=batch,
                    )
                    db.commit()
                    _auto_call_stats[tenant_id]["total_dialed"] += len(report.dialed)
                    did_dial = len(report.dialed) > 0
        except Exception as e:
            print(f"[AUTO-CALL][tid={tenant_id}] error: {e}")

        if should_wait:
            time.sleep(15)
        elif did_dial:
            time.sleep(15)
        else:
            time.sleep(10)

    engine.dispose()


def _get_tid_from_request_or_first(request=None) -> int:
    if request is not None:
        from app.auth import get_session_data
        sess = get_session_data(request)
        if sess:
            return sess["tenant_id"]
    # fallback for internal calls (e.g. from campaign launch)
    with SessionLocal() as db:
        from app.models import Tenant
        t = db.query(Tenant).first()
        return t.id if t else 1


@router.post("/api/autocall/start")
def start_auto_call(request: Request):
    tid = _require_tid(request)
    if _auto_call_active.get(tid):
        stats = _auto_call_stats.get(tid, {"total_dialed": 0, "remaining": 0})
        return {"status": "already_running", **stats}
    _auto_call_active[tid] = True
    t = threading.Thread(target=_auto_call_loop, args=(tid,), daemon=True)
    t.start()
    with SessionLocal() as db:
        remaining = db.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.NEW).count()
    return {"status": "started", "remaining": remaining}


@router.post("/api/autocall/stop")
def stop_auto_call(request: Request):
    tid = _require_tid(request)
    _auto_call_active[tid] = False
    stats = _auto_call_stats.get(tid, {"total_dialed": 0, "remaining": 0})
    return {"status": "stopped", **stats}


@router.get("/api/autocall/status")
def auto_call_status(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        remaining = db.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.NEW).count()
        in_flight = db.query(Call).filter(
            Call.tenant_id == tid,
            Call.status.in_([CallStatus.DIALING, CallStatus.CONNECTED])
        ).count()
    stats = _auto_call_stats.get(tid, {"total_dialed": 0})
    return {
        "active": _auto_call_active.get(tid, False),
        "remaining": remaining,
        "in_flight": in_flight,
        "total_dialed": stats.get("total_dialed", 0),
    }


@router.post("/api/reset")
def reset_all_leads(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        leads = db.query(Lead).filter(Lead.tenant_id == tid).all()
        for lead in leads:
            lead.status = LeadStatus.NEW
            lead.attempts = 0
            lead.next_attempt_at = datetime.now(timezone.utc)
            lead.last_called_at = None
        db.commit()
        return {"reset": len(leads)}


@router.put("/api/lead/{lead_id}")
def update_lead(lead_id: int, payload: dict):
    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        if not lead:
            return {"error": "Lead not found"}
        if "name" in payload:
            lead.name = payload["name"].strip() or None
        if "phone" in payload:
            phone = _to_e164(payload["phone"])
            if len(phone) < 12:
                return {"error": "Invalid phone number"}
            existing = db.query(Lead).filter(Lead.phone_e164 == phone, Lead.id != lead_id).first()
            if existing:
                return {"error": "Phone number already exists"}
            lead.phone_e164 = phone
        if "status" in payload:
            try:
                lead.status = LeadStatus(payload["status"])
            except ValueError:
                return {"error": "Invalid status"}
            if lead.status == LeadStatus.NEW:
                lead.next_attempt_at = datetime.now(timezone.utc)
        db.commit()
        return {"ok": True}


@router.delete("/api/lead/{lead_id}")
def delete_lead(lead_id: int):
    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        if not lead:
            return {"error": "Lead not found"}
        db.query(CallTurn).filter(
            CallTurn.call_id.in_(db.query(Call.id).filter_by(lead_id=lead_id))
        ).delete(synchronize_session=False)
        db.query(Call).filter_by(lead_id=lead_id).delete()
        db.delete(lead)
        db.commit()
        return {"deleted": lead_id}


@router.post("/api/lead/{lead_id}/move")
def move_lead(lead_id: int, payload: dict):
    direction = payload.get("direction", "up")
    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        if not lead:
            return {"error": "Lead not found"}
        if direction == "up":
            swap = db.query(Lead).filter(Lead.id > lead_id).order_by(Lead.id.asc()).first()
        else:
            swap = db.query(Lead).filter(Lead.id < lead_id).order_by(Lead.id.desc()).first()
        if not swap:
            return {"error": "Cannot move further"}
        lead.phone_e164, swap.phone_e164 = swap.phone_e164, lead.phone_e164
        lead.name, swap.name = swap.name, lead.name
        lead.status, swap.status = swap.status, lead.status
        lead.attempts, swap.attempts = swap.attempts, lead.attempts
        lead.last_called_at, swap.last_called_at = swap.last_called_at, lead.last_called_at
        lead.source, swap.source = swap.source, lead.source
        db.commit()
        return {"ok": True}


@router.delete("/api/leads/bulk")
def bulk_delete_leads(payload: dict):
    ids = payload.get("ids", [])
    if not ids:
        return {"error": "No IDs provided"}
    with SessionLocal() as db:
        db.query(CallTurn).filter(
            CallTurn.call_id.in_(db.query(Call.id).filter(Call.lead_id.in_(ids)))
        ).delete(synchronize_session=False)
        db.query(Call).filter(Call.lead_id.in_(ids)).delete(synchronize_session=False)
        deleted = db.query(Lead).filter(Lead.id.in_(ids)).delete(synchronize_session=False)
        db.commit()
        return {"deleted": deleted}


@router.put("/api/call/{call_id}")
def update_call(call_id: int, payload: dict):
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        if not call:
            return {"error": "Call not found"}
        if "status" in payload:
            try:
                call.status = CallStatus(payload["status"])
            except ValueError:
                return {"error": "Invalid status"}
        if "outcome" in payload:
            call.outcome = payload["outcome"].strip() or None
        db.commit()
        return {"ok": True}


@router.delete("/api/call/{call_id}")
def delete_call(call_id: int):
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        if not call:
            return {"error": "Call not found"}
        db.query(CallTurn).filter_by(call_id=call_id).delete()
        db.delete(call)
        db.commit()
        return {"deleted": call_id}


# ── AI Agent Section ─────────────────────────────────────────────────────


@router.get("/ai-agent", response_class=HTMLResponse)
async def ai_agent_page(request: Request):
    from app.auth import get_session_data
    if not get_session_data(request):
        return RedirectResponse("/login?next=/dashboard/ai-agent", status_code=302)
    return _suma_html()


@router.get("/api/ai-agent/config")
def get_ai_agent_config(request: Request):
    tid = _require_tid(request)
    return _ai_agent_config.get(tid, _default_agent_config(tid))


@router.post("/api/ai-agent/config")
def save_ai_agent_config(request: Request, payload: dict):
    tid = _require_tid(request)
    cfg = _ai_agent_config.setdefault(tid, _default_agent_config(tid))
    for k in ("system_prompt", "greeting", "language", "voice"):
        if k in payload:
            cfg[k] = payload[k]
    with SessionLocal() as db:
        emp = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).first()
        if emp:
            emp.script = cfg["system_prompt"]
            emp.language = cfg["language"]
            db.commit()
    return {"ok": True}


@router.post("/api/ai-agent/call")
def ai_agent_call(request: Request, payload: dict):
    """Make a single AI agent call to a phone number."""
    import os
    from app.config import get_settings
    from app.streaming.ws import set_call_mode
    tid = _require_tid(request)

    phone = payload.get("phone", "")
    name = payload.get("name", "")
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 10:
        return {"error": "Invalid phone number"}
    e164 = _to_e164(phone)

    cfg = _ai_agent_config.get(tid, _default_agent_config(tid))
    with SessionLocal() as db:
        emp = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).first()
        if not emp:
            return {"error": "No AI employee configured"}
        emp.script = cfg["system_prompt"]
        emp.language = cfg["language"]
        db.flush()

        call = Call(
            tenant_id=emp.tenant_id,
            employee_id=emp.id,
            direction=CallDirection.OUTBOUND,
            status=CallStatus.DIALING,
            started_at=datetime.now(timezone.utc),
            captured_fields={"phone": e164, "name": name},
        )
        db.add(call)
        db.flush()

        set_call_mode(call.id, "ai_agent")

        settings = get_settings()
        auth_id = os.getenv("PLIVO_AUTH_ID")
        if not auth_id:
            return {"error": "Plivo not configured"}

        from app.providers.plivo import PlivoTelephony
        telephony = PlivoTelephony(auth_id, os.environ["PLIVO_AUTH_TOKEN"], public_url=settings.public_url)
        result = telephony.dial(e164, emp.phone_number, f"{settings.public_url}/v1/plivo/status", custom_field=str(call.id))
        call.provider_call_id = result.provider_call_id or None
        if not result.accepted:
            call.status = CallStatus.REFUSED
        db.commit()
        return {"call_id": call.id, "accepted": result.accepted, "phone": e164, "name": name}


@router.get("/api/ai-agent/calls")
def get_ai_agent_calls(request: Request):
    """Get recent AI agent calls with their conversation transcripts."""
    tid = _require_tid(request)
    with SessionLocal() as db:
        calls = db.query(Call).filter(Call.tenant_id == tid).order_by(Call.id.desc()).limit(50).all()
        result = []
        for c in calls:
            turns = db.query(CallTurn).filter_by(call_id=c.id).order_by(CallTurn.id).all()
            if not turns and c.status not in (CallStatus.DIALING, CallStatus.CONNECTED):
                continue
            lead = db.get(Lead, c.lead_id) if c.lead_id else None
            cf = c.captured_fields or {}
            result.append({
                "id": c.id,
                "phone": lead.phone_e164 if lead else cf.get("phone", "—"),
                "name": lead.name if lead else cf.get("name", "—"),
                "status": c.status.value,
                "outcome": c.outcome or "—",
                "duration": c.billed_seconds or 0,
                "started_at": c.started_at.isoformat() if c.started_at else None,
                "turns": [{"role": t.role, "text": t.text} for t in turns],
            })
        return result


# ── Voice Tuning Section ────────────────────────────────────────────────

@router.get("/voice-tuning", response_class=HTMLResponse)
async def voice_tuning_page():
    return _suma_html()


@router.get("/api/voice/status")
def voice_model_status():
    from app.streaming.indicf5_tts import get_model_status, get_voice_profiles, get_active_profile
    status = get_model_status()
    profiles = get_voice_profiles()
    active = get_active_profile()
    hf_token = os.environ.get("HF_TOKEN", "")
    return {
        "model": status,
        "profiles": profiles,
        "active_profile": active["id"] if active else None,
        "hf_token_set": bool(hf_token),
    }


@router.post("/api/voice/hf-token")
def set_hf_token(payload: dict):
    token = payload.get("token", "").strip()
    if not token:
        return {"error": "Token is required"}
    os.environ["HF_TOKEN"] = token
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    lines.append(f"HF_TOKEN={token}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"HF_TOKEN={token}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    from app.streaming.indicf5_tts import reset_model
    reset_model()
    return {"ok": True}


@router.post("/api/voice/load-model")
def load_voice_model():
    from app.streaming.indicf5_tts import _load_model, get_model_status
    _load_model()
    return get_model_status()


@router.post("/api/voice/profile")
async def upload_voice_profile(
    audio: UploadFile = File(...),
    transcript: str = Form(""),
    name: str = Form(""),
):
    from app.streaming.indicf5_tts import save_voice_profile
    import re as _re

    audio_bytes = await audio.read()
    if not name.strip():
        name = audio.filename.rsplit(".", 1)[0] if audio.filename else "voice"
    profile_id = _re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    if not profile_id:
        profile_id = "voice_profile"

    save_voice_profile(profile_id, audio_bytes, transcript)
    return {"ok": True, "id": profile_id}


@router.post("/api/voice/profile/{profile_id}/activate")
def activate_voice_profile(profile_id: str):
    from app.streaming.indicf5_tts import set_active_profile
    set_active_profile(profile_id)
    return {"ok": True}


@router.delete("/api/voice/profile/{profile_id}")
def delete_voice_profile_endpoint(profile_id: str):
    from app.streaming.indicf5_tts import delete_voice_profile
    delete_voice_profile(profile_id)
    return {"ok": True}


@router.post("/api/voice/test")
async def test_voice(payload: dict):
    """Test TTS synthesis — returns a WAV file."""
    import tempfile
    from app.streaming.indicf5_tts import synthesize_to_file, get_active_profile

    text = payload.get("text", "").strip()
    if not text:
        return {"error": "Text is required"}

    profile = get_active_profile()
    if not profile:
        return {"error": "No active voice profile selected"}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    success = synthesize_to_file(text, tmp_path, profile)
    if not success:
        return {"error": "Synthesis failed. Check model status."}

    return FileResponse(tmp_path, media_type="audio/wav", filename="test_output.wav")
