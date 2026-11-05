"""Signed outcome webhooks to the tenant, with an attempt log."""

import hashlib
import hmac
import json

import httpx
from sqlalchemy.orm import Session

from app.models import Call, DeliveryAttempt, Tenant


def sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def build_payload(call: Call) -> dict:
    return {
        "event": "call.completed",
        "call_id": call.id,
        "lead_id": call.lead_id,
        "employee_id": call.employee_id,
        "outcome": call.outcome,
        "summary": call.summary,
        "captured_fields": call.captured_fields or {},
        "billed_seconds": call.billed_seconds,
        "ended_at": call.ended_at.isoformat() if call.ended_at else None,
    }


def deliver_outcome(session: Session, call: Call, client: httpx.Client | None = None, max_attempts: int = 3) -> bool:
    tenant = session.get(Tenant, call.tenant_id)
    if not tenant.webhook_url:
        return False

    body = json.dumps(build_payload(call), separators=(",", ":"), sort_keys=True).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Signature-SHA256": sign(tenant.webhook_secret, body),
    }
    http = client or httpx.Client(timeout=5.0)

    succeeded = False
    for _ in range(max_attempts):
        try:
            response = http.post(tenant.webhook_url, content=body, headers=headers)
            status = response.status_code
        except httpx.HTTPError:
            status = None
        session.add(DeliveryAttempt(call_id=call.id, url=tenant.webhook_url, status_code=status, succeeded=status is not None and 200 <= status < 300))
        if status is not None and 200 <= status < 300:
            succeeded = True
            break
    return succeeded
