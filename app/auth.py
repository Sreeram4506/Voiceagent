"""Session-based authentication for the SUMA dashboard.

Each browser session carries a random cookie token (suma_session). The session itself is a row in
user_sessions, keyed by a SHA-256 hash of the token, so logins survive restarts and work on every worker.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

from app.db import SessionLocal
from app.models import UserSession

SESSION_COOKIE = "suma_session"
SESSION_MAX_AGE_SECONDS = 86400 * 30  # matches the cookie's lifetime


# ── Password helpers ──────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return dk.hex()


def hash_new_password(password: str) -> str:
    """Return salt:hash string ready to store."""
    salt = secrets.token_hex(16)
    return f"{salt}:{_hash_password(password, salt)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, expected = stored.split(":", 1)
        return hmac_compare(_hash_password(password, salt), expected)
    except Exception:
        return False


def hmac_compare(a: str, b: str) -> bool:
    return secrets.compare_digest(a.encode(), b.encode())


# ── Session helpers ───────────────────────────────────────────────────────
# These hit the database: call them from sync routes (FastAPI runs those in its threadpool)
# or through run_in_threadpool, never directly on the event loop.

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_session(tenant_id: int, user_id: int, email: str, role: str, name: str, tenant_name: str = "") -> str:
    token = secrets.token_urlsafe(32)
    now = _utcnow()
    data = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "email": email,
        "role": role,
        "name": name,
        "tenant_name": tenant_name,
        "created_at": now.replace(tzinfo=timezone.utc).isoformat(),
    }
    with SessionLocal() as db:
        db.query(UserSession).filter(UserSession.expires_at < now).delete(synchronize_session=False)
        db.add(UserSession(
            token_hash=hash_token(token),
            data=data,
            created_at=now,
            expires_at=now + timedelta(seconds=SESSION_MAX_AGE_SECONDS),
        ))
        db.commit()
    return token


def get_session_data(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with SessionLocal() as db:
        row = db.get(UserSession, hash_token(token))
        if row is None or row.expires_at <= _utcnow():
            return None
        return dict(row.data or {})


def require_session(request: Request) -> dict:
    sess = get_session_data(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sess


def require_tenant_id(request: Request) -> int:
    return require_session(request)["tenant_id"]


def destroy_session(request: Request) -> None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return
    with SessionLocal() as db:
        db.query(UserSession).filter(UserSession.token_hash == hash_token(token)).delete(synchronize_session=False)
        db.commit()


def redirect_to_login(next_path: str = "/dashboard/") -> RedirectResponse:
    return RedirectResponse(url=f"/login?next={next_path}", status_code=302)


# ── Default tenant seed ───────────────────────────────────────────────────

def ensure_default_tenant() -> None:
    """On first startup create a default tenant + admin user if none exist."""
    from app.models import Tenant, TenantUser

    with SessionLocal() as db:
        existing = db.query(TenantUser).first()
        if existing:
            return

        # Create default tenant
        tenant = db.query(Tenant).first()
        if not tenant:
            from app.api.deps import hash_key, new_key
            import secrets as _s
            raw_key = new_key()
            tenant = Tenant(
                name="SUMA Enterprise",
                api_key_hash=hash_key(raw_key),
                webhook_secret=_s.token_urlsafe(32),
            )
            db.add(tenant)
            db.flush()
            print(f"\n╔══════════════════════════════════════╗")
            print(f"║  SUMA AI — Default Tenant Created    ║")
            print(f"║  Tenant: {tenant.name:<27}║")
            print(f"║  API Key: {raw_key[:30]}…║")
            print(f"╚══════════════════════════════════════╝\n")

        # Create default admin user
        password = "suma2026"
        phash = hash_new_password(password)
        user = TenantUser(
            tenant_id=tenant.id,
            name="Admin",
            email="admin@suma.ai",
            password_hash=phash,
            role="admin",
            active=True,
        )
        db.add(user)
        db.commit()
        print(f"[SUMA] Default login: admin@suma.ai / {password}")


# ── Login endpoint helpers ────────────────────────────────────────────────

def authenticate_user(email: str, password: str) -> dict | None:
    """Return session payload dict on success, None on failure."""
    from app.models import Tenant, TenantUser

    with SessionLocal() as db:
        user = (
            db.query(TenantUser)
            .filter(TenantUser.email == email.lower().strip(), TenantUser.active == True)
            .first()
        )
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        tenant = db.get(Tenant, user.tenant_id)
        return {
            "tenant_id": user.tenant_id,
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
            "name": user.name,
            "tenant_name": tenant.name if tenant else "Unknown",
        }
