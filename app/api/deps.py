import hashlib
import secrets

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Tenant


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_key() -> str:
    return "va_" + secrets.token_urlsafe(24)


def current_tenant(
    x_api_key: str = Header(..., alias="X-API-Key"),
    session: Session = Depends(get_session),
) -> Tenant:
    tenant = session.scalar(select(Tenant).where(Tenant.api_key_hash == hash_key(x_api_key)))
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    return tenant
