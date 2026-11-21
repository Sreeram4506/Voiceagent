"""Dashboard logins live in the database, so they survive restarts and work on every worker."""

from datetime import datetime, timedelta

from app.auth import SESSION_COOKIE, hash_token
from app.db import SessionLocal
from app.models import Tenant, TenantUser, UserSession


def login(client) -> str:
    r = client.post("/dashboard/api/auth/login", json={"email": "admin@suma.ai", "password": "suma2026"})
    assert r.status_code == 200
    return client.cookies[SESSION_COOKIE]


def session_rows() -> list[UserSession]:
    with SessionLocal() as db:
        return db.query(UserSession).all()


def write_session(token: str, expires_at: datetime) -> None:
    """A session row as another worker, or this one before a restart, would have written it."""
    with SessionLocal() as db:
        user = db.query(TenantUser).first()
        tenant = db.get(Tenant, user.tenant_id)
        db.add(UserSession(
            token_hash=hash_token(token),
            data={"tenant_id": user.tenant_id, "user_id": user.id, "email": user.email, "role": user.role,
                  "name": user.name, "tenant_name": tenant.name},
            created_at=datetime.utcnow(),
            expires_at=expires_at,
        ))
        db.commit()


def test_login_is_stored_in_the_database_with_only_a_hash_of_the_token(client):
    token = login(client)
    rows = session_rows()
    assert len(rows) == 1
    assert rows[0].token_hash == hash_token(token) != token


def test_a_session_written_by_another_worker_is_accepted(client):
    write_session("token-from-another-worker", datetime.utcnow() + timedelta(days=1))
    client.cookies.set(SESSION_COOKIE, "token-from-another-worker")
    r = client.get("/dashboard/api/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == "admin@suma.ai"


def test_logout_deletes_the_session(client):
    login(client)
    assert client.post("/dashboard/api/auth/logout").status_code == 200
    assert session_rows() == []
    assert client.get("/dashboard/api/auth/me").status_code == 401


def test_an_expired_session_is_rejected(client):
    write_session("old-token", datetime.utcnow() - timedelta(minutes=1))
    client.cookies.set(SESSION_COOKIE, "old-token")
    assert client.get("/dashboard/api/auth/me").status_code == 401


def test_a_new_login_clears_out_expired_sessions(client):
    write_session("old-token", datetime.utcnow() - timedelta(minutes=1))
    login(client)
    assert [r.token_hash for r in session_rows()] != [hash_token("old-token")]
    assert len(session_rows()) == 1
