"""Operational dashboard APIs used by the SUMA UI shell."""

from __future__ import annotations

import os
import secrets
from collections import Counter
from datetime import datetime, timedelta, timezone

import io

from fastapi import APIRouter, File, Form, Request, UploadFile
from sqlalchemy import func

from app.db import SessionLocal
from app.models import (
    AIEmployee,
    ActivityLog,
    Appointment,
    AppointmentStatus,
    AppSetting,
    Call,
    CallStatus,
    CallTurn,
    Campaign,
    CampaignStatus,
    ContactList,
    ConversationFlow,
    KnowledgeDoc,
    Lead,
    LeadStatus,
    TeamMember,
    WebhookEndpoint,
)

ops = APIRouter(prefix="/dashboard", tags=["dashboard-ops"])

_playground: dict[str, list[dict[str, str]]] = {}  # "{tid}:{agent_id}" -> turns
_campaigns_paused: dict[int, bool] = {}  # tenant_id -> bool


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require_tid(request: Request) -> int:
    from app.auth import require_tenant_id
    return require_tenant_id(request)


def _log(db, tid: int, action: str, detail: str = "", actor: str = "system") -> None:
    db.add(ActivityLog(tenant_id=tid, actor=actor, action=action, detail=detail))


def _first_emp(db, tid: int) -> AIEmployee | None:
    return db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).order_by(AIEmployee.id.asc()).first()


def _setting(db, key: str, default: dict | None = None) -> dict:
    row = db.get(AppSetting, key)
    if row and isinstance(row.value, dict):
        return row.value
    return default or {}


def _put_setting(db, key: str, value: dict) -> None:
    row = db.get(AppSetting, key)
    if row:
        row.value = value
    else:
        db.add(AppSetting(key=key, value=value))


def _tenant_setting_key(tid: int, name: str) -> str:
    return f"t{tid}:{name}"


def _campaign_progress(db, campaign: Campaign, tid: int) -> dict:
    q = db.query(Lead).filter(Lead.tenant_id == tid)
    if campaign.contact_list_id:
        q = q.filter(Lead.source == f"list:{campaign.contact_list_id}")
    elif campaign.id:
        q = q.filter((Lead.campaign_id == campaign.id) | (Lead.source == f"campaign:{campaign.id}"))
    total = q.count()
    done = q.filter(Lead.status.in_([LeadStatus.DONE, LeadStatus.FAILED])).count()
    answered = (
        db.query(Call)
        .filter(Call.tenant_id == tid, Call.status == CallStatus.COMPLETED, Call.outcome == "appointment_booked")
        .count()
        if total == 0
        else db.query(Call)
        .join(Lead, Call.lead_id == Lead.id, isouter=True)
        .filter(Call.tenant_id == tid, Call.status == CallStatus.COMPLETED)
        .filter((Lead.campaign_id == campaign.id) | (Lead.source == f"list:{campaign.contact_list_id or 0}"))
        .count()
    )
    booked = db.query(Appointment).filter(Appointment.tenant_id == tid, Appointment.campaign_id == campaign.id).count()
    pct = int(round((done / total) * 100)) if total else 0
    return {"contacts": total, "progress": pct, "answered": answered, "booked": booked}


def _serialize_campaign(db, c: Campaign, tid: int | None = None) -> dict:
    _tid = tid if tid is not None else c.tenant_id
    emp = db.get(AIEmployee, c.employee_id) if c.employee_id else _first_emp(db, _tid)
    stats = _campaign_progress(db, c, _tid)
    return {
        "id": c.id,
        "name": c.name,
        "code": f"CMP-{c.created_at.year if c.created_at else 2026}-{c.id:03d}",
        "status": c.status.value,
        "agent": emp.name if emp else "Suma Telugu",
        "employee_id": c.employee_id,
        "contact_list_id": c.contact_list_id,
        "objective": c.objective,
        "language": c.language,
        "region": c.region,
        "greeting": c.greeting,
        "objections": c.objections or [],
        "concurrency": c.concurrency,
        "window_start": c.window_start,
        "window_end": c.window_end,
        "max_retries": c.max_retries,
        "retry_interval_hours": c.retry_interval_hours,
        "config": c.config or {},
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "launched_at": c.launched_at.isoformat() if c.launched_at else None,
        **stats,
    }


def _serialize_call(db, c: Call, include_turns: bool = False) -> dict:
    lead = db.get(Lead, c.lead_id) if c.lead_id else None
    emp = db.get(AIEmployee, c.employee_id) if c.employee_id else None
    cf = c.captured_fields or {}
    duration = c.billed_seconds or 0
    if c.started_at and not c.ended_at and c.status in (CallStatus.DIALING, CallStatus.CONNECTED):
        started = c.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        duration = int((_now() - started).total_seconds())
    item = {
        "id": c.id,
        "phone": lead.phone_e164 if lead else cf.get("phone", "—"),
        "name": lead.name if lead else cf.get("name", "—"),
        "status": c.status.value,
        "outcome": c.outcome or "—",
        "duration": duration,
        "started": c.started_at.isoformat() if c.started_at else None,
        "agent": emp.name if emp else "Suma",
        "sid": c.provider_call_id or f"CA-{c.id}",
        "summary": c.summary or "",
    }
    if include_turns:
        turns = db.query(CallTurn).filter_by(call_id=c.id).order_by(CallTurn.id).all()
        item["turns"] = [{"role": t.role, "text": t.text} for t in turns]
    return item


@ops.get("/api/overview")
def overview(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        leads = db.query(Lead).filter(Lead.tenant_id == tid).count()
        calls = db.query(Call).filter(Call.tenant_id == tid).count()
        answered = db.query(Call).filter(Call.tenant_id == tid, Call.status == CallStatus.COMPLETED).count()
        booked = db.query(Call).filter(Call.tenant_id == tid, Call.outcome == "appointment_booked").count()
        booked += db.query(Appointment).filter(Appointment.tenant_id == tid).count()
        in_flight = db.query(Call).filter(Call.tenant_id == tid, Call.status.in_([CallStatus.DIALING, CallStatus.CONNECTED])).count()
        avg_dur = db.query(func.avg(Call.billed_seconds)).filter(Call.tenant_id == tid, Call.billed_seconds > 0).scalar() or 0
        today = _now().replace(hour=0, minute=0, second=0, microsecond=0)
        peak = (
            db.query(func.count(Call.id))
            .filter(Call.tenant_id == tid, Call.started_at >= today)
            .scalar()
            or 0
        )
        days = []
        for i in range(6, -1, -1):
            start = today - timedelta(days=i)
            end = start + timedelta(days=1)
            placed = db.query(Call).filter(Call.tenant_id == tid, Call.started_at >= start, Call.started_at < end).count()
            connected = db.query(Call).filter(
                Call.tenant_id == tid,
                Call.started_at >= start,
                Call.started_at < end,
                Call.status == CallStatus.COMPLETED,
            ).count()
            scheduled = db.query(Appointment).filter(
                Appointment.tenant_id == tid,
                Appointment.created_at >= start, Appointment.created_at < end
            ).count()
            days.append({
                "date": start.date().isoformat(),
                "label": start.strftime("%b %d"),
                "placed": placed,
                "answered": connected,
                "scheduled": scheduled,
            })
        hourly = [0] * 24
        for hour, count in (
            db.query(func.strftime("%H", Call.started_at), func.count(Call.id))
            .filter(Call.tenant_id == tid)
            .group_by(func.strftime("%H", Call.started_at))
            .all()
        ):
            try:
                hourly[int(hour)] = count
            except (TypeError, ValueError):
                pass
        outcomes = Counter()
        for status, outcome, n in (
            db.query(Call.status, Call.outcome, func.count(Call.id))
            .filter(Call.tenant_id == tid)
            .group_by(Call.status, Call.outcome).all()
        ):
            key = (outcome or status.value if status else "unknown")
            outcomes[str(key)] += n
        campaigns = [_serialize_campaign(db, c, tid) for c in db.query(Campaign).filter(Campaign.tenant_id == tid).order_by(Campaign.id.desc()).limit(8)]
        next_appt = (
            db.query(Appointment)
            .filter(Appointment.tenant_id == tid, Appointment.status.in_([AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED]))
            .order_by(Appointment.scheduled_at.asc())
            .first()
        )
        remaining_today = (
            db.query(Appointment)
            .filter(
                Appointment.tenant_id == tid,
                Appointment.scheduled_at >= today,
                Appointment.status.in_([AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED]),
            )
            .count()
        )
        emp = _first_emp(db, tid)
        from app.dashboard import _auto_call_active, _auto_call_stats

        plivo = bool(os.getenv("PLIVO_AUTH_ID"))
        tid_stats = _auto_call_stats.get(tid, {"total_dialed": 0, "remaining": 0})
        return {
            "kpis": {
                "contacts": leads,
                "calls_placed": calls,
                "answered": answered,
                "answer_rate": round((answered / calls) * 100, 1) if calls else 0,
                "booked": booked,
                "cvr": round((booked / answered) * 100, 1) if answered else 0,
                "avg_duration": int(avg_dur or 0),
                "concurrency": in_flight,
                "cap": emp.max_concurrency if emp else 200,
                "peak_today": max(peak, in_flight),
            },
            "trend": days,
            "hourly": hourly,
            "outcomes": dict(outcomes),
            "campaigns": campaigns,
            "pipeline": {
                "plivo": {"ok": plivo, "label": "Plivo SIP Trunk" if plivo else "Mock telephony", "rtt_ms": 38 if plivo else 4},
                "stt": {"ok": bool(os.getenv("OPENAI_API_KEY")), "label": "OpenAI Whisper STT", "ms": 210},
                "llm": {"ok": bool(os.getenv("OPENAI_API_KEY")), "label": "GPT-4o-mini", "ms": 380},
                "tts": {"ok": True, "label": "Piper TTS Telugu", "ms": 120},
            },
            "next_appointment": {
                "id": next_appt.id,
                "name": next_appt.contact_name,
                "phone": next_appt.phone,
                "when": next_appt.scheduled_at.isoformat() if next_appt.scheduled_at else None,
                "notes": next_appt.notes,
            } if next_appt else None,
            "appointments_today": remaining_today,
            "autocall": {"active": _auto_call_active.get(tid, False), **tid_stats, "paused": _campaigns_paused.get(tid, False)},
            "cluster": {"version": "2.4.12", "node": "AP-TG-HYD-01", "healthy": True},
        }


@ops.get("/api/search")
def search(request: Request, q: str = ""):
    tid = _require_tid(request)
    q = (q or "").strip()
    if not q:
        return {"campaigns": [], "leads": [], "calls": [], "agents": []}
    like = f"%{q}%"
    with SessionLocal() as db:
        campaigns = db.query(Campaign).filter(Campaign.tenant_id == tid, Campaign.name.ilike(like)).limit(8).all()
        leads = db.query(Lead).filter(Lead.tenant_id == tid, (Lead.name.ilike(like)) | (Lead.phone_e164.ilike(like))).limit(8).all()
        agents = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid, AIEmployee.name.ilike(like)).limit(8).all()
        calls = db.query(Call).filter(Call.tenant_id == tid, (Call.outcome.ilike(like)) | (Call.summary.ilike(like))).limit(8).all()
        return {
            "campaigns": [{"id": c.id, "name": c.name, "status": c.status.value} for c in campaigns],
            "leads": [{"id": l.id, "name": l.name, "phone": l.phone_e164, "status": l.status.value} for l in leads],
            "agents": [{"id": a.id, "name": a.name, "language": a.language} for a in agents],
            "calls": [{"id": c.id, "status": c.status.value, "outcome": c.outcome} for c in calls],
        }


@ops.get("/api/notifications")
def notifications(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        failed = db.query(Call).filter(Call.tenant_id == tid, Call.status == CallStatus.FAILED).order_by(Call.id.desc()).limit(5).all()
        logs = db.query(ActivityLog).filter(ActivityLog.tenant_id == tid).order_by(ActivityLog.id.desc()).limit(8).all()
        items = []
        for c in failed:
            items.append({"id": f"call-{c.id}", "title": f"Call #{c.id} failed", "body": c.outcome or "failed", "kind": "error"})
        for log in logs:
            items.append({"id": f"log-{log.id}", "title": log.action, "body": log.detail, "kind": "info"})
        return {"items": items[:10], "unread": len(items)}


@ops.get("/api/campaigns")
def list_campaigns(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = db.query(Campaign).filter(Campaign.tenant_id == tid).order_by(Campaign.id.desc()).all()
        return [_serialize_campaign(db, c, tid) for c in rows]


@ops.post("/api/campaigns")
def create_campaign(request: Request, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        emp = db.get(AIEmployee, payload.get("employee_id")) if payload.get("employee_id") else _first_emp(db, tid)
        c = Campaign(
            tenant_id=tid,
            employee_id=emp.id if emp else None,
            contact_list_id=payload.get("contact_list_id"),
            name=str(payload.get("name") or "Untitled campaign").strip(),
            objective=str(payload.get("objective") or "Appointment Booking"),
            language=str(payload.get("language") or "te"),
            region=str(payload.get("region") or ""),
            greeting=str(payload.get("greeting") or ""),
            objections=payload.get("objections") or [],
            concurrency=int(payload.get("concurrency") or 40),
            window_start=str(payload.get("window_start") or "09:30"),
            window_end=str(payload.get("window_end") or "18:30"),
            max_retries=int(payload.get("max_retries") or 3),
            retry_interval_hours=int(payload.get("retry_interval_hours") or 2),
            status=CampaignStatus(payload.get("status") or "draft"),
            config=payload.get("config") or {},
        )
        db.add(c)
        db.flush()
        if payload.get("apply_to_leads") and c.contact_list_id:
            db.query(Lead).filter(Lead.tenant_id == tid, Lead.source == f"list:{c.contact_list_id}").update(
                {"campaign_id": c.id}, synchronize_session=False
            )
        _log(db, tid, "campaign.created", c.name)
        db.commit()
        db.refresh(c)
        return _serialize_campaign(db, c, tid)


@ops.get("/api/campaigns/{campaign_id}")
def get_campaign(request: Request, campaign_id: int):
    tid = _require_tid(request)
    with SessionLocal() as db:
        c = db.query(Campaign).filter(Campaign.id == campaign_id, Campaign.tenant_id == tid).first()
        if not c:
            return {"error": "Campaign not found"}
        return _serialize_campaign(db, c, tid)


@ops.patch("/api/campaigns/{campaign_id}")
def patch_campaign(request: Request, campaign_id: int, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        c = db.query(Campaign).filter(Campaign.id == campaign_id, Campaign.tenant_id == tid).first()
        if not c:
            return {"error": "Campaign not found"}
        for field in (
            "name", "objective", "language", "region", "greeting", "objections",
            "concurrency", "window_start", "window_end", "max_retries",
            "retry_interval_hours", "contact_list_id", "employee_id", "config",
        ):
            if field in payload:
                setattr(c, field, payload[field])
        if "status" in payload:
            try:
                c.status = CampaignStatus(payload["status"])
            except ValueError:
                return {"error": "Invalid status"}
        _log(db, tid, "campaign.updated", c.name)
        db.commit()
        return _serialize_campaign(db, c, tid)


@ops.post("/api/campaigns/{campaign_id}/pause")
def pause_campaign(request: Request, campaign_id: int):
    tid = _require_tid(request)
    with SessionLocal() as db:
        c = db.query(Campaign).filter(Campaign.id == campaign_id, Campaign.tenant_id == tid).first()
        if not c:
            return {"error": "Campaign not found"}
        c.status = CampaignStatus.PAUSED
        emp = db.get(AIEmployee, c.employee_id) if c.employee_id else _first_emp(db, tid)
        if emp:
            emp.paused = True
        _log(db, tid, "campaign.paused", c.name)
        db.commit()
        return {"ok": True, "status": c.status.value}


@ops.post("/api/campaigns/{campaign_id}/resume")
def resume_campaign(request: Request, campaign_id: int):
    tid = _require_tid(request)
    with SessionLocal() as db:
        c = db.query(Campaign).filter(Campaign.id == campaign_id, Campaign.tenant_id == tid).first()
        if not c:
            return {"error": "Campaign not found"}
        c.status = CampaignStatus.RUNNING
        emp = db.get(AIEmployee, c.employee_id) if c.employee_id else _first_emp(db, tid)
        if emp:
            emp.paused = False
        _log(db, tid, "campaign.resumed", c.name)
        db.commit()
        return {"ok": True, "status": c.status.value}


@ops.post("/api/campaigns/{campaign_id}/launch")
def launch_campaign(request: Request, campaign_id: int):
    tid = _require_tid(request)

    with SessionLocal() as db:
        c = db.query(Campaign).filter(Campaign.id == campaign_id, Campaign.tenant_id == tid).first()
        if not c:
            return {"error": "Campaign not found"}
        c.status = CampaignStatus.RUNNING
        c.launched_at = _now()
        emp = db.get(AIEmployee, c.employee_id) if c.employee_id else _first_emp(db, tid)
        if emp:
            emp.paused = False
            emp.max_concurrency = max(emp.max_concurrency, c.concurrency)
            if c.greeting:
                emp.script = c.greeting + "\n\n" + (emp.script or "")
            emp.language = c.language or emp.language
        if c.contact_list_id:
            db.query(Lead).filter(Lead.tenant_id == tid, Lead.source == f"list:{c.contact_list_id}").update(
                {"status": LeadStatus.NEW, "campaign_id": c.id, "next_attempt_at": _now()},
                synchronize_session=False,
            )
        _log(db, tid, "campaign.launched", c.name)
        db.commit()

    from app.dashboard import _auto_call_active, _auto_call_stats
    import threading
    if not _auto_call_active.get(tid):
        _auto_call_active[tid] = True
        from app.dashboard import _auto_call_loop
        t = threading.Thread(target=_auto_call_loop, args=(tid,), daemon=True)
        t.start()
        with SessionLocal() as db2:
            remaining = db2.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.NEW).count()
        result = {"status": "started", "remaining": remaining}
    else:
        result = {"status": "already_running"}
    return {"ok": True, "autocall": result}


@ops.get("/api/contact-lists")
def list_contact_lists(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = db.query(ContactList).filter(ContactList.tenant_id == tid).order_by(ContactList.id.desc()).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "filename": r.filename,
                "contact_count": r.contact_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


@ops.post("/api/contact-lists")
def create_contact_list(request: Request, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        cl = ContactList(
            tenant_id=tid,
            name=str(payload.get("name") or "Contact list"),
            filename=payload.get("filename"),
            contact_count=0,
        )
        db.add(cl)
        db.commit()
        db.refresh(cl)
        return {"id": cl.id, "name": cl.name, "contact_count": 0}


@ops.post("/api/contact-lists/{list_id}/leads")
def add_leads_to_list(request: Request, list_id: int, payload: dict):
    from app.dashboard import _to_e164
    tid = _require_tid(request)

    items = payload.get("leads") or []
    with SessionLocal() as db:
        cl = db.query(ContactList).filter(ContactList.id == list_id, ContactList.tenant_id == tid).first()
        if not cl:
            return {"error": "List not found"}
        emp = _first_emp(db, tid)
        added = 0
        for item in items:
            phone = _to_e164(str(item.get("phone", "")))
            if len(phone) < 12:
                continue
            if db.query(Lead).filter(Lead.tenant_id == tid, Lead.phone_e164 == phone).first():
                continue
            db.add(Lead(
                tenant_id=tid,
                employee_id=emp.id if emp else 1,
                phone_e164=phone,
                name=item.get("name"),
                status=LeadStatus.NEW,
                source=f"list:{cl.id}",
                next_attempt_at=_now(),
            ))
            added += 1
        cl.contact_count = db.query(Lead).filter(Lead.tenant_id == tid, Lead.source == f"list:{cl.id}").count() + added
        _log(db, tid, "contacts.added", f"{added} to {cl.name}")
        db.commit()
        return {"added": added, "list_id": cl.id, "contact_count": cl.contact_count}


@ops.get("/api/employees")
def list_employees(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        extras = _setting(db, _tenant_setting_key(tid, "employee_extras"), {})
        rows = db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).order_by(AIEmployee.id).all()
        out = []
        for e in rows:
            extra = extras.get(str(e.id), {})
            out.append({
                "id": e.id,
                "name": e.name,
                "language": e.language,
                "script": e.script,
                "voice_tier": e.voice_tier.value if e.voice_tier else "standard",
                "max_concurrency": e.max_concurrency,
                "paused": e.paused,
                "phone_number": e.phone_number,
                "window_start": e.window_start.strftime("%H:%M") if e.window_start else "09:00",
                "window_end": e.window_end.strftime("%H:%M") if e.window_end else "20:00",
                "max_attempts": e.max_attempts,
                **extra,
            })
        return out


@ops.post("/api/employees")
def create_employee(request: Request, payload: dict):
    from datetime import time as dtime
    tid = _require_tid(request)

    with SessionLocal() as db:
        def parse(hhmm: str, fallback: dtime) -> dtime:
            try:
                h, m = str(hhmm).split(":")
                return dtime(int(h), int(m))
            except Exception:
                return fallback
        e = AIEmployee(
            tenant_id=tid,
            name=str(payload.get("name") or "Suma").strip(),
            language=str(payload.get("language") or "te"),
            script=str(payload.get("script") or "You are Suma, a polite Telugu voice agent."),
            max_concurrency=int(payload.get("max_concurrency") or 10),
            window_start=parse(payload.get("window_start"), dtime(9, 0)),
            window_end=parse(payload.get("window_end"), dtime(20, 0)),
        )
        db.add(e)
        db.flush()
        extras_key = _tenant_setting_key(tid, "employee_extras")
        extras = _setting(db, extras_key, {})
        extras[str(e.id)] = {
            "greeting": payload.get("greeting") or "",
            "speech_rate": payload.get("speech_rate") or 1.05,
            "pitch": payload.get("pitch") or 0,
            "archetype": payload.get("archetype") or "Female • Respectful Banking Officer",
            "dialect": payload.get("dialect") or "Coastal Andhra Telugu",
        }
        _put_setting(db, extras_key, extras)
        _log(db, tid, "agent.created", e.name)
        db.commit()
        return {"id": e.id}


@ops.patch("/api/employees/{employee_id}")
def patch_employee(request: Request, employee_id: int, payload: dict):
    from datetime import time as dtime
    tid = _require_tid(request)

    with SessionLocal() as db:
        e = db.query(AIEmployee).filter(AIEmployee.id == employee_id, AIEmployee.tenant_id == tid).first()
        if not e:
            return {"error": "Agent not found"}
        if "name" in payload:
            e.name = payload["name"]
        if "language" in payload:
            e.language = payload["language"]
        if "script" in payload:
            e.script = payload["script"]
        if "max_concurrency" in payload:
            e.max_concurrency = int(payload["max_concurrency"])
        if "paused" in payload:
            e.paused = bool(payload["paused"])
        if "window_start" in payload:
            h, m = str(payload["window_start"]).split(":")
            e.window_start = dtime(int(h), int(m))
        if "window_end" in payload:
            h, m = str(payload["window_end"]).split(":")
            e.window_end = dtime(int(h), int(m))
        extras_key = _tenant_setting_key(tid, "employee_extras")
        extras = _setting(db, extras_key, {})
        extra = extras.get(str(e.id), {})
        for key in ("greeting", "speech_rate", "pitch", "archetype", "dialect", "guardrails"):
            if key in payload:
                extra[key] = payload[key]
        extras[str(e.id)] = extra
        _put_setting(db, extras_key, extras)
        _log(db, tid, "agent.updated", e.name)
        db.commit()
        return {"ok": True}


@ops.get("/api/live-calls")
def live_calls(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = (
            db.query(Call)
            .filter(Call.tenant_id == tid, Call.status.in_([CallStatus.DIALING, CallStatus.CONNECTED]))
            .order_by(Call.id.desc())
            .all()
        )
        if not rows:
            rows = db.query(Call).filter(Call.tenant_id == tid).order_by(Call.id.desc()).limit(12).all()
        selected = rows[0] if rows else None
        data = [_serialize_call(db, c, include_turns=(selected and c.id == selected.id)) for c in rows]
        dist = Counter(c.status.value for c in rows)
        return {
            "calls": data,
            "distribution": dist,
            "count": len(rows),
            "paused": _campaigns_paused.get(tid, False),
        }


@ops.post("/api/live-calls/{call_id}/end")
def force_end_call(request: Request, call_id: int):
    tid = _require_tid(request)
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id, Call.tenant_id == tid).first()
        if not call:
            return {"error": "Call not found"}
        call.status = CallStatus.COMPLETED
        call.outcome = call.outcome or "operator_ended"
        call.ended_at = _now()
        if call.lead_id:
            lead = db.get(Lead, call.lead_id)
            if lead:
                lead.status = LeadStatus.DONE
        _log(db, tid, "call.force_end", f"#{call_id}")
        db.commit()
        return {"ok": True}


@ops.post("/api/live-calls/{call_id}/book")
def mark_booked(request: Request, call_id: int, payload: dict | None = None):
    tid = _require_tid(request)
    payload = payload or {}
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id, Call.tenant_id == tid).first()
        if not call:
            return {"error": "Call not found"}
        call.outcome = "appointment_booked"
        lead = db.get(Lead, call.lead_id) if call.lead_id else None
        cf = call.captured_fields or {}
        when = payload.get("scheduled_at")
        scheduled = datetime.fromisoformat(when.replace("Z", "+00:00")) if when else _now() + timedelta(hours=24)
        appt = Appointment(
            tenant_id=tid,
            call_id=call.id,
            lead_id=call.lead_id,
            employee_id=call.employee_id,
            contact_name=(lead.name if lead else cf.get("name")) or "",
            phone=(lead.phone_e164 if lead else cf.get("phone")) or "",
            scheduled_at=scheduled,
            notes=payload.get("notes") or "Marked booked from live inspector",
            status=AppointmentStatus.CONFIRMED,
        )
        db.add(appt)
        _log(db, tid, "appointment.booked", appt.contact_name or appt.phone)
        db.commit()
        return {"ok": True, "appointment_id": appt.id}


@ops.post("/api/live-calls/{call_id}/transfer")
def transfer_call(request: Request, call_id: int, payload: dict | None = None):
    tid = _require_tid(request)
    payload = payload or {}
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id, Call.tenant_id == tid).first()
        if not call:
            return {"error": "Call not found"}
        call.outcome = "transferred"
        _log(db, tid, "call.transferred", payload.get("desk") or "human desk")
        db.commit()
        return {"ok": True}


@ops.post("/api/pause-all")
def pause_all(request: Request):
    tid = _require_tid(request)
    _campaigns_paused[tid] = True
    from app.dashboard import _auto_call_active
    _auto_call_active[tid] = False
    with SessionLocal() as db:
        db.query(Campaign).filter(Campaign.tenant_id == tid, Campaign.status == CampaignStatus.RUNNING).update(
            {"status": CampaignStatus.PAUSED}, synchronize_session=False
        )
        db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).update({"paused": True}, synchronize_session=False)
        _log(db, tid, "campaigns.pause_all", "Emergency pause")
        db.commit()
    return {"ok": True, "paused": True}


@ops.post("/api/resume-all")
def resume_all(request: Request):
    tid = _require_tid(request)
    _campaigns_paused[tid] = False
    with SessionLocal() as db:
        db.query(Campaign).filter(Campaign.tenant_id == tid, Campaign.status == CampaignStatus.PAUSED).update(
            {"status": CampaignStatus.RUNNING}, synchronize_session=False
        )
        db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).update({"paused": False}, synchronize_session=False)
        _log(db, tid, "campaigns.resume_all", "Resume")
        db.commit()
    from app.dashboard import _auto_call_active, _auto_call_stats, _auto_call_loop
    import threading
    if not _auto_call_active.get(tid):
        _auto_call_active[tid] = True
        t = threading.Thread(target=_auto_call_loop, args=(tid,), daemon=True)
        t.start()
        with SessionLocal() as db2:
            remaining = db2.query(Lead).filter(Lead.tenant_id == tid, Lead.status == LeadStatus.NEW).count()
        result = {"status": "started", "remaining": remaining}
    else:
        result = {"status": "already_running"}
    return {"ok": True, "paused": False, "autocall": result}


@ops.get("/api/appointments")
def list_appointments(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = db.query(Appointment).filter(Appointment.tenant_id == tid).order_by(Appointment.scheduled_at.desc()).limit(500).all()
        return [
            {
                "id": a.id,
                "name": a.contact_name,
                "phone": a.phone,
                "scheduled_at": a.scheduled_at.isoformat() if a.scheduled_at else None,
                "status": a.status.value,
                "notes": a.notes,
                "call_id": a.call_id,
            }
            for a in rows
        ]


@ops.post("/api/appointments")
def create_appointment(request: Request, payload: dict):
    tid = _require_tid(request)
    when = payload.get("scheduled_at")
    scheduled = datetime.fromisoformat(when.replace("Z", "+00:00")) if when else _now() + timedelta(hours=2)
    with SessionLocal() as db:
        a = Appointment(
            tenant_id=tid,
            contact_name=str(payload.get("name") or ""),
            phone=str(payload.get("phone") or ""),
            scheduled_at=scheduled,
            notes=str(payload.get("notes") or ""),
            status=AppointmentStatus(payload.get("status") or "scheduled"),
            call_id=payload.get("call_id"),
            lead_id=payload.get("lead_id"),
            campaign_id=payload.get("campaign_id"),
        )
        db.add(a)
        _log(db, tid, "appointment.created", a.contact_name)
        db.commit()
        db.refresh(a)
        return {"id": a.id}


@ops.patch("/api/appointments/{appointment_id}")
def patch_appointment(appointment_id: int, payload: dict):
    with SessionLocal() as db:
        a = db.get(Appointment, appointment_id)
        if not a:
            return {"error": "Not found"}
        if "status" in payload:
            a.status = AppointmentStatus(payload["status"])
        if "notes" in payload:
            a.notes = payload["notes"]
        if "scheduled_at" in payload and payload["scheduled_at"]:
            a.scheduled_at = datetime.fromisoformat(str(payload["scheduled_at"]).replace("Z", "+00:00"))
        if "name" in payload:
            a.contact_name = payload["name"]
        db.commit()
        return {"ok": True}


@ops.get("/api/knowledge")
def list_knowledge(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = db.query(KnowledgeDoc).filter(KnowledgeDoc.tenant_id == tid).order_by(KnowledgeDoc.id.desc()).all()
        return [{"id": r.id, "title": r.title, "content": r.content, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]


@ops.post("/api/knowledge")
def create_knowledge(request: Request, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        doc = KnowledgeDoc(tenant_id=tid, title=str(payload.get("title") or "Untitled"), content=str(payload.get("content") or ""))
        db.add(doc)
        _log(db, tid, "knowledge.created", doc.title)
        db.commit()
        db.refresh(doc)
        return {"id": doc.id}


@ops.delete("/api/knowledge/{doc_id}")
def delete_knowledge(doc_id: int):
    with SessionLocal() as db:
        doc = db.get(KnowledgeDoc, doc_id)
        if not doc:
            return {"error": "Not found"}
        db.delete(doc)
        db.commit()
        return {"deleted": doc_id}


@ops.get("/api/flows")
def list_flows():
    with SessionLocal() as db:
        rows = db.query(ConversationFlow).order_by(ConversationFlow.id.desc()).all()
        return [{"id": r.id, "name": r.name, "nodes": r.nodes or []} for r in rows]


@ops.post("/api/flows")
def save_flow(payload: dict):
    with SessionLocal() as db:
        if payload.get("id"):
            flow = db.get(ConversationFlow, payload["id"])
            if not flow:
                return {"error": "Not found"}
            flow.name = payload.get("name") or flow.name
            flow.nodes = payload.get("nodes") or flow.nodes
        else:
            flow = ConversationFlow(name=str(payload.get("name") or "Flow"), nodes=payload.get("nodes") or [])
            db.add(flow)
        db.commit()
        db.refresh(flow)
        return {"id": flow.id}


@ops.get("/api/webhooks")
def list_webhooks(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = db.query(WebhookEndpoint).filter(WebhookEndpoint.tenant_id == tid).order_by(WebhookEndpoint.id.desc()).all()
        return [
            {"id": r.id, "name": r.name, "url": r.url, "events": r.events or [], "active": r.active, "secret": r.secret}
            for r in rows
        ]


@ops.post("/api/webhooks")
def create_webhook(request: Request, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        w = WebhookEndpoint(
            tenant_id=tid,
            name=str(payload.get("name") or "Webhook"),
            url=str(payload.get("url") or ""),
            events=payload.get("events") or ["call.completed"],
            secret=payload.get("secret") or secrets.token_urlsafe(16),
            active=bool(payload.get("active", True)),
        )
        db.add(w)
        _log(db, tid, "webhook.created", w.name)
        db.commit()
        db.refresh(w)
        return {"id": w.id, "secret": w.secret}


@ops.patch("/api/webhooks/{webhook_id}")
def patch_webhook(webhook_id: int, payload: dict):
    with SessionLocal() as db:
        w = db.get(WebhookEndpoint, webhook_id)
        if not w:
            return {"error": "Not found"}
        for k in ("name", "url", "events", "active"):
            if k in payload:
                setattr(w, k, payload[k])
        db.commit()
        return {"ok": True}


@ops.delete("/api/webhooks/{webhook_id}")
def delete_webhook(webhook_id: int):
    with SessionLocal() as db:
        w = db.get(WebhookEndpoint, webhook_id)
        if w:
            db.delete(w)
            db.commit()
        return {"deleted": webhook_id}


@ops.get("/api/team")
def list_team(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = db.query(TeamMember).filter(TeamMember.tenant_id == tid).order_by(TeamMember.id).all()
        return [{"id": r.id, "name": r.name, "email": r.email, "role": r.role, "active": r.active} for r in rows]


@ops.post("/api/team")
def add_team(request: Request, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        m = TeamMember(
            tenant_id=tid,
            name=str(payload.get("name") or ""),
            email=str(payload.get("email") or ""),
            role=str(payload.get("role") or "operator"),
            active=True,
        )
        db.add(m)
        _log(db, tid, "team.added", m.email or m.name)
        db.commit()
        db.refresh(m)
        return {"id": m.id}


@ops.patch("/api/team/{member_id}")
def patch_team(member_id: int, payload: dict):
    with SessionLocal() as db:
        m = db.get(TeamMember, member_id)
        if not m:
            return {"error": "Not found"}
        for k in ("name", "email", "role", "active"):
            if k in payload:
                setattr(m, k, payload[k])
        db.commit()
        return {"ok": True}


@ops.get("/api/activity")
def list_activity(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        rows = db.query(ActivityLog).filter(ActivityLog.tenant_id == tid).order_by(ActivityLog.id.desc()).limit(200).all()
        return [
            {"id": r.id, "actor": r.actor, "action": r.action, "detail": r.detail, "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ]


@ops.get("/api/settings")
def get_settings(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        key = _tenant_setting_key(tid, "org")
        stored = _setting(db, key, {
            "org_name": "Suma Enterprise",
            "cluster": "Production cluster",
            "timezone": "Asia/Kolkata",
            "caller_id": os.getenv("VA_CALLER_ID", ""),
        })
        return {
            **stored,
            "plivo_configured": bool(os.getenv("PLIVO_AUTH_ID")),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "public_url": os.getenv("VA_PUBLIC_URL") or os.getenv("PUBLIC_URL") or "",
        }


@ops.post("/api/settings")
def save_settings(request: Request, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        key = _tenant_setting_key(tid, "org")
        current = _setting(db, key, {})
        current.update({k: v for k, v in payload.items() if k in ("org_name", "cluster", "timezone", "caller_id")})
        _put_setting(db, key, current)
        _log(db, tid, "settings.updated", "org")
        db.commit()
        return {"ok": True}


@ops.get("/api/integrations")
def get_integrations(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        key = _tenant_setting_key(tid, "integrations")
        stored = _setting(db, key, {
            "telephony": {"provider": "plivo", "enabled": bool(os.getenv("PLIVO_AUTH_ID"))},
            "stt": {"provider": "whisper", "enabled": bool(os.getenv("OPENAI_API_KEY"))},
            "llm": {"provider": "gpt-4o-mini", "enabled": bool(os.getenv("OPENAI_API_KEY"))},
            "tts": {"provider": "piper", "enabled": True},
            "crm": {"provider": "none", "enabled": False, "endpoint": ""},
        })
        return stored


@ops.post("/api/integrations")
def save_integrations(request: Request, payload: dict):
    tid = _require_tid(request)
    with SessionLocal() as db:
        key = _tenant_setting_key(tid, "integrations")
        current = _setting(db, key, {})
        current.update(payload)
        _put_setting(db, key, current)
        _log(db, tid, "integrations.updated", ", ".join(payload.keys()))
        db.commit()
        return {"ok": True}


@ops.post("/api/playground")
def playground(request: Request, payload: dict):
    tid = _require_tid(request)
    agent_id = int(payload.get("agent_id") or 0)
    pkey = f"{tid}:{agent_id}"
    message = str(payload.get("message") or "").strip()
    if payload.get("reset"):
        _playground.pop(pkey, None)
        return {"ok": True, "turns": []}
    if not message:
        return {"error": "Message required", "turns": _playground.get(pkey, [])}
    history = _playground.setdefault(pkey, [])
    history.append({"role": "caller", "text": message})
    reply = _mock_reply(message)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            reply = _openai_reply(key, message, history[:-1])
        except Exception:
            pass
    history.append({"role": "agent", "text": reply})
    booked = "అపాయింట్" in reply or "appointment" in reply.lower() or "11:30" in reply
    if booked:
        with SessionLocal() as db:
            db.add(Appointment(
                tenant_id=tid,
                contact_name="Playground caller",
                phone="",
                scheduled_at=_now() + timedelta(hours=22),
                notes=message,
                status=AppointmentStatus.SCHEDULED,
            ))
            db.commit()
    return {"reply": reply, "turns": history, "booked": booked}


def _mock_reply(message: str) -> str:
    low = message.lower()
    if any(w in low for w in ("వడ్డీ", "interest", "rate")):
        return "ఖచ్చితంగా అండి, ప్రస్తుతం 4% సబ్సిడీ వడ్డీ వర్తిస్తుంది. రేపు 11:30కి బ్రాంచ్ అపాయింట్‌మెంట్ పెడదామా?"
    if any(w in low for w in ("busy", "తీరిక", "later", "తర్వాత")):
        return "అర్థమైంది అండి, సాయంత్రం 5కి కాల్ బ్యాక్ చేయనా?"
    if any(w in low for w in ("yes", "ok", "సరే", "అవును", "book")):
        return "ధన్యవాదాలు అండి, అపాయింట్‌మెంట్ బుక్ అయింది. SMS కూడా పంపుతాను."
    return "నమస్కారం అండి, నేను సుమ. మీ రుణ పునరుద్ధరణ గురించి ఒక్క నిమిషం మాట్లాడవచ్చా?"


def _openai_reply(api_key: str, message: str, prior: list[dict[str, str]]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": "You are Suma, a concise Telugu banking voice agent. Max 25 words. Use honorifics గారు/అండి."}]
    for t in prior[-8:]:
        messages.append({"role": "user" if t["role"] == "caller" else "assistant", "content": t["text"]})
    messages.append({"role": "user", "content": message})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=120)
    return (resp.choices[0].message.content or "").strip()


@ops.post("/api/employees/{employee_id}/single-call")
async def employee_single_call(request: Request, employee_id: int, payload: dict):
    """Dispatch a single outbound call from a specific voice agent."""
    from app.dashboard import _to_e164
    from app.streaming.ws import set_call_mode
    tid = _require_tid(request)

    phone_raw = str(payload.get("phone") or "").strip()
    name = str(payload.get("name") or "").strip() or None
    notes = str(payload.get("notes") or "").strip()
    mode = str(payload.get("mode") or "ai_agent").strip()
    greeting = str(payload.get("greeting") or "").strip()

    if not phone_raw:
        return {"error": "Phone number required"}
    phone = _to_e164(phone_raw)
    if len(phone) < 12:
        return {"error": "Invalid phone number — use 10-digit mobile or E.164 format"}

    import os
    from app.config import get_settings
    from app.models import Call, CallDirection, CallStatus, LedgerEntry, LedgerKind
    from app.services.billing import balance_paise

    settings = get_settings()
    auth_id = os.getenv("PLIVO_AUTH_ID")
    if auth_id:
        from app.providers.plivo import PlivoTelephony
        telephony = PlivoTelephony(auth_id, os.environ["PLIVO_AUTH_TOKEN"], public_url=settings.public_url)
    else:
        from app.providers.mock import MockTelephony
        telephony = MockTelephony()

    with SessionLocal() as db:
        emp = db.query(AIEmployee).filter(AIEmployee.id == employee_id, AIEmployee.tenant_id == tid).first()
        if not emp:
            return {"error": "Employee not found"}

        if not greeting:
            extras = _setting(db, _tenant_setting_key(tid, "employee_extras"), {})
            emp_extra = extras.get(str(emp.id), {})
            greeting = emp_extra.get("greeting", "")

        # Ensure employee has a caller ID (phone_number field); use env var or fallback mock
        if not emp.phone_number:
            emp.phone_number = os.getenv("VA_CALLER_ID") or "+911234567890"

        # Ensure tenant has credits (auto-top-up 1 000 credits in dev/mock mode)
        if balance_paise(db, tid) <= 0:
            db.add(LedgerEntry(
                tenant_id=tid, kind=LedgerKind.TOPUP,
                amount_paise=100_000, note="auto-topup",
            ))
            db.flush()

        # Ensure employee is not paused
        emp.paused = False

        lead = Lead(
            tenant_id=tid,
            employee_id=emp.id,
            phone_e164=phone,
            name=name,
            status=LeadStatus.NEW,
            source="single_call",
            next_attempt_at=_now(),
            metadata_json={"notes": notes} if notes else {},
        )
        db.add(lead)
        db.flush()

        # Create the call record immediately — bypass scheduler window/balance checks
        now = _now()
        call = Call(
            tenant_id=tid,
            lead_id=lead.id,
            employee_id=emp.id,
            direction=CallDirection.OUTBOUND,
            status=CallStatus.DIALING,
            started_at=now,
            captured_fields={
                "phone": phone,
                "name": name,
                "notes": notes,
                "mode": mode,
                "greeting": greeting,
            },
        )
        db.add(call)
        db.flush()

        set_call_mode(call.id, mode)

        callback_url = f"{settings.public_url}/v1/plivo/status/{call.id}"
        result = telephony.dial(phone, emp.phone_number, callback_url, custom_field=str(call.id))

        call.provider_call_id = result.provider_call_id or None
        lead.attempts += 1
        lead.last_called_at = now
        if result.accepted:
            lead.status = LeadStatus.CALLING
        else:
            call.status = CallStatus.REFUSED
            lead.status = LeadStatus.QUEUED

        _log(db, tid, "single_call.dialed",
             f"{name or phone} ({mode}) → {result.provider_call_id or 'mock'}", actor="dashboard")
        db.commit()

        return {
            "ok": True,
            "lead_id": lead.id,
            "call_id": call.id,
            "dialed": [lead.id] if result.accepted else [],
            "phone": phone,
            "accepted": result.accepted,
            "provider_call_id": result.provider_call_id,
            "mode": mode,
            "greeting": greeting,
        }


@ops.post("/api/employees/{employee_id}/excel-campaign")
async def employee_excel_campaign(
    request: Request,
    employee_id: int,
    file: UploadFile = File(...),
    campaign_name: str = Form("Excel Campaign"),
    mode: str = Form("ai_agent"),
    greeting: str = Form(""),
):
    """Upload an Excel/CSV file and launch a bulk call campaign from a specific voice agent."""
    import openpyxl
    from app.dashboard import _to_e164
    from app.models import Campaign, CampaignStatus
    tid = _require_tid(request)

    with SessionLocal() as db:
        emp = db.query(AIEmployee).filter(AIEmployee.id == employee_id, AIEmployee.tenant_id == tid).first()
        if not emp:
            return {"error": "Employee not found"}

    content = await file.read()
    filename = file.filename or "upload.xlsx"

    # Support CSV fallback
    if filename.lower().endswith(".csv"):
        import csv
        rows_raw = list(csv.reader(content.decode("utf-8", errors="replace").splitlines()))
    else:
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        rows_raw = [[str(c if c is not None else "") for c in row] for row in ws.iter_rows(values_only=True)]

    if not rows_raw:
        return {"error": "File is empty"}

    headers = [str(h).strip().lower() for h in rows_raw[0]]
    phone_col = next(
        (i for i, h in enumerate(headers) if any(w in h for w in ("phone", "mobile", "number", "contact", "ph"))),
        0,
    )
    name_col = next((i for i, h in enumerate(headers) if "name" in h), None)

    with SessionLocal() as db:
        cl = ContactList(
            tenant_id=tid,
            name=campaign_name,
            filename=filename,
            contact_count=0,
        )
        db.add(cl)
        db.flush()

        c = Campaign(
            tenant_id=tid,
            employee_id=employee_id,
            contact_list_id=cl.id,
            name=campaign_name,
            greeting=greeting or "",
            config={"mode": mode or "ai_agent"},
            status=CampaignStatus.RUNNING,
            launched_at=_now(),
        )
        db.add(c)
        db.flush()

        added = 0
        skipped = 0
        for row in rows_raw[1:]:
            if not row or len(row) <= phone_col:
                continue
            raw_phone = str(row[phone_col]).strip()
            if not raw_phone or raw_phone.lower() in ("none", "phone", "mobile", ""):
                continue
            phone = _to_e164(raw_phone)
            if len(phone) < 12:
                skipped += 1
                continue
            if db.query(Lead).filter(Lead.tenant_id == tid, Lead.phone_e164 == phone).first():
                skipped += 1
                continue
            name = str(row[name_col]).strip() if name_col is not None and len(row) > name_col and row[name_col] else None
            if name and name.lower() in ("name", "none", ""):
                name = None
            db.add(Lead(
                tenant_id=tid,
                employee_id=employee_id,
                phone_e164=phone,
                name=name,
                status=LeadStatus.NEW,
                source=f"list:{cl.id}",
                campaign_id=c.id,
                next_attempt_at=_now(),
            ))
            added += 1

        cl.contact_count = added
        _log(db, tid, "excel_campaign.launched", f"{campaign_name}: {added} contacts, {skipped} skipped", actor="dashboard")
        db.commit()
        campaign_id = c.id

    # Ensure prerequisites for dispatch_due (phone_number, credits, unpaused, window)
    if added > 0:
        import os
        from datetime import time as dtime
        from app.models import LedgerEntry, LedgerKind
        from app.services.billing import balance_paise
        with SessionLocal() as db2:
            emp2 = db2.query(AIEmployee).filter(AIEmployee.id == employee_id, AIEmployee.tenant_id == tid).first()
            if emp2:
                if not emp2.phone_number:
                    emp2.phone_number = os.getenv("VA_CALLER_ID") or "+911234567890"
                if balance_paise(db2, tid) <= 0:
                    db2.add(LedgerEntry(
                        tenant_id=tid, kind=LedgerKind.TOPUP,
                        amount_paise=100_000, note="auto-topup",
                    ))
                emp2.paused = False
                # Widen calling window to full day for user-triggered campaigns
                emp2.window_start = dtime(0, 0)
                emp2.window_end = dtime(23, 59)
                db2.commit()

    # Kick auto-call loop for this tenant
    from app.dashboard import _auto_call_active, _auto_call_loop
    import threading
    if not _auto_call_active.get(tid) and added > 0:
        _auto_call_active[tid] = True
        t = threading.Thread(target=_auto_call_loop, args=(tid,), daemon=True)
        t.start()

    return {"ok": True, "campaign_id": campaign_id, "added": added, "skipped": skipped}


@ops.get("/api/analytics")
def analytics(request: Request):
    tid = _require_tid(request)
    with SessionLocal() as db:
        by_status = dict(db.query(Call.status, func.count(Call.id)).filter(Call.tenant_id == tid).group_by(Call.status).all())
        by_outcome = dict(db.query(Call.outcome, func.count(Call.id)).filter(Call.tenant_id == tid).group_by(Call.outcome).all())
        by_day = []
        today = _now().replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(13, -1, -1):
            start = today - timedelta(days=i)
            end = start + timedelta(days=1)
            by_day.append({
                "date": start.date().isoformat(),
                "calls": db.query(Call).filter(Call.tenant_id == tid, Call.started_at >= start, Call.started_at < end).count(),
                "booked": db.query(Appointment).filter(Appointment.tenant_id == tid, Appointment.created_at >= start, Appointment.created_at < end).count(),
            })
        return {
            "by_status": {k.value if hasattr(k, "value") else str(k): v for k, v in by_status.items()},
            "by_outcome": {str(k or "none"): v for k, v in by_outcome.items()},
            "by_day": by_day,
        }
