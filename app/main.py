import os
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from app.api.deps import current_tenant
from app.api.routes import router
from app.config import get_settings
from app.db import get_session, init_db
from app.models import Tenant
from app.providers.base import TelephonyProvider
from app.providers.mock import MockTelephony
from app.services.compliance import InMemoryDNDRegistry
from app.services.scheduler import dispatch_due
from app.dashboard import router as dashboard_router
from app.dashboard_ops import ops as dashboard_ops_router
from app.streaming.ws import router as stream_router, stream_token
from app.auth import (
    SESSION_COOKIE,
    authenticate_user,
    create_session,
    destroy_session,
    ensure_default_tenant,
    get_session_data,
    require_session,
)

app = FastAPI(title="Voice Agent Platform", version="0.1.0")
app.include_router(router, prefix="/v1")
app.include_router(stream_router)
app.include_router(dashboard_router)
app.include_router(dashboard_ops_router)


# ── Auth page & API ───────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    html = (Path(__file__).parent / "login.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/dashboard/api/auth/login")
async def api_login(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid request body"}, status_code=400)
    email = str(body.get("email") or "").strip().lower()
    password = str(body.get("password") or "")
    if not email or not password:
        return JSONResponse({"detail": "Email and password required"}, status_code=400)
    payload = authenticate_user(email, password)
    if not payload:
        return JSONResponse({"detail": "Invalid email or password"}, status_code=401)
    token = create_session(**payload)
    resp = JSONResponse({"ok": True, "user": payload})
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=86400 * 30)
    return resp


@app.post("/dashboard/api/auth/logout")
async def api_logout(request: Request):
    destroy_session(request)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/dashboard/api/auth/me")
async def api_me(request: Request):
    sess = get_session_data(request)
    if not sess:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return {
        "tenant_id": sess["tenant_id"],
        "user_id": sess["user_id"],
        "email": sess["email"],
        "role": sess["role"],
        "name": sess["name"],
        "tenant_name": sess.get("tenant_name", ""),
    }


# ── Tenant management (admin) ─────────────────────────────────────────────

@app.get("/dashboard/api/tenants")
async def list_tenants(request: Request):
    sess = require_session(request)
    if sess.get("role") != "superadmin":
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    from app.db import SessionLocal
    from app.models import Tenant as TenantModel
    with SessionLocal() as db:
        rows = db.query(TenantModel).order_by(TenantModel.id).all()
        return [{"id": r.id, "name": r.name, "created_at": r.created_at.isoformat()} for r in rows]


@app.post("/dashboard/api/tenants")
async def create_tenant(request: Request):
    sess = require_session(request)
    if sess.get("role") not in ("admin", "superadmin"):
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid body"}, status_code=400)
    tenant_name = str(body.get("name") or "").strip()
    admin_email = str(body.get("email") or "").strip().lower()
    admin_password = str(body.get("password") or "suma2026")
    if not tenant_name or not admin_email:
        return JSONResponse({"detail": "name and email required"}, status_code=400)
    import secrets as _s
    from app.api.deps import hash_key
    from app.db import SessionLocal
    from app.models import Tenant as TenantModel, TenantUser
    from app.auth import hash_new_password
    with SessionLocal() as db:
        raw_key = _s.token_urlsafe(32)
        t = TenantModel(name=tenant_name, api_key_hash=hash_key(raw_key), webhook_secret=_s.token_urlsafe(32))
        db.add(t)
        db.flush()
        user = TenantUser(
            tenant_id=t.id, name="Admin", email=admin_email,
            password_hash=hash_new_password(admin_password), role="admin", active=True,
        )
        db.add(user)
        db.commit()
        return {"tenant_id": t.id, "api_key": raw_key, "admin_email": admin_email}


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/dashboard/")


def get_telephony() -> TelephonyProvider:
    """Plivo when credentials are configured; otherwise the offline mock."""
    auth_id = os.getenv("PLIVO_AUTH_ID")
    if auth_id:
        from app.providers.plivo import PlivoTelephony

        return PlivoTelephony(
            auth_id,
            os.environ["PLIVO_AUTH_TOKEN"],
            public_url=get_settings().public_url,
        )
    return MockTelephony()


@app.on_event("startup")
def _startup() -> None:
    init_db()
    ensure_default_tenant()


@app.post("/v1/plivo/answer/{call_id}")
async def plivo_answer(call_id: int, request: Request):
    """Plivo hits this when the callee answers. Returns XML that opens a bidirectional audio stream."""
    settings = get_settings()
    token = stream_token(call_id)
    ws_url = settings.public_url.replace("http://", "wss://").replace("https://", "wss://")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-l16;rate=8000" audioTrack="inbound">'
        f"{ws_url}/v1/stream/{call_id}?token={token}"
        "</Stream>"
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.post("/v1/plivo/status/{call_id}")
async def plivo_status(call_id: int, request: Request):
    """Plivo hangup/status callback. Updates call record with outcome and duration."""
    try:
        form = await request.form()
        data = dict(form)
    except Exception:
        data = {}

    from app.db import SessionLocal as SL
    from app.models import Call, CallStatus, Lead, LeadStatus

    status_map = {
        "completed": CallStatus.COMPLETED,
        "busy": CallStatus.BUSY,
        "failed": CallStatus.FAILED,
        "no-answer": CallStatus.NO_ANSWER,
        "cancel": CallStatus.FAILED,
        "hangup": CallStatus.COMPLETED,
    }

    with SL() as db:
        call = db.get(Call, call_id)
        if call:
            plivo_status_val = str(data.get("CallStatus", data.get("Status", ""))).lower()
            hangup_cause = str(data.get("HangupCause", data.get("HangupCauseName", "")))
            duration = data.get("Duration", data.get("BillDuration", "0"))

            new_status = status_map.get(plivo_status_val)
            if new_status:
                call.status = new_status
            elif plivo_status_val not in ("ringing", "in-progress", "early-media"):
                call.status = CallStatus.COMPLETED

            call.outcome = hangup_cause or plivo_status_val or None
            try:
                call.billed_seconds = int(duration)
            except (ValueError, TypeError):
                pass
            call.ended_at = datetime.utcnow()

            if call.lead_id:
                lead = db.get(Lead, call.lead_id)
                if lead and lead.status == LeadStatus.CALLING:
                    if call.status in (CallStatus.COMPLETED, CallStatus.NO_ANSWER, CallStatus.BUSY):
                        lead.status = LeadStatus.DONE
                    elif call.status == CallStatus.FAILED:
                        lead.status = LeadStatus.FAILED

            db.commit()

    return Response(content="OK", media_type="text/plain")


@app.post("/v1/dispatch")
def dispatch(tenant: Tenant = Depends(current_tenant), session: Session = Depends(get_session)):
    """Run one scheduling pass for this tenant's due leads."""
    settings = get_settings()
    report = dispatch_due(
        session,
        telephony=get_telephony(),
        dnd=InMemoryDNDRegistry(),
        now=datetime.utcnow(),
        callback_url=f"{settings.public_url}/v1/plivo/status",
        retry_delay_minutes=settings.retry_delay_minutes,
    )
    session.commit()
    return {"dialed": report.dialed, "skipped": report.skipped}
