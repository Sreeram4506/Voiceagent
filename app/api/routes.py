from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import current_tenant, hash_key, new_key
from app.db import get_session
from app.models import AIEmployee, Call, CallStatus, Lead, LeadStatus, Tenant, VoiceTier
from app.services.billing import balance_paise, top_up

router = APIRouter()


class TenantCreate(BaseModel):
    name: str
    webhook_url: str | None = None


class TenantCreated(BaseModel):
    tenant_id: int
    api_key: str
    webhook_secret: str


class EmployeeCreate(BaseModel):
    name: str
    language: str = "hi"
    voice_tier: VoiceTier = VoiceTier.STANDARD
    script: str
    capture_fields: dict = Field(default_factory=dict)
    max_concurrency: int = 10
    window_start: str = "09:00"
    window_end: str = "20:00"
    timezone: str = "Asia/Kolkata"
    max_attempts: int = 3
    retry_delay_minutes: int = 30


class LeadIn(BaseModel):
    employee_id: int
    phone_e164: str = Field(pattern=r"^\+\d{8,15}$")
    name: str | None = None
    source: str = "api"
    metadata: dict = Field(default_factory=dict)


class TopUp(BaseModel):
    credits: int = Field(gt=0)


@router.post("/tenants", response_model=TenantCreated)
def create_tenant(body: TenantCreate, session: Session = Depends(get_session)):
    import secrets

    raw_key = new_key()
    secret = secrets.token_urlsafe(32)
    tenant = Tenant(name=body.name, api_key_hash=hash_key(raw_key), webhook_url=body.webhook_url, webhook_secret=secret)
    session.add(tenant)
    session.commit()
    return TenantCreated(tenant_id=tenant.id, api_key=raw_key, webhook_secret=secret)


@router.get("/me/balance")
def my_balance(tenant: Tenant = Depends(current_tenant), session: Session = Depends(get_session)):
    paise = balance_paise(session, tenant.id)
    return {"credits": paise // 100, "paise": paise}


@router.post("/me/topup")
def my_topup(body: TopUp, tenant: Tenant = Depends(current_tenant), session: Session = Depends(get_session)):
    top_up(session, tenant.id, body.credits)
    session.commit()
    return {"credits": balance_paise(session, tenant.id) // 100}


@router.post("/employees")
def create_employee(body: EmployeeCreate, tenant: Tenant = Depends(current_tenant), session: Session = Depends(get_session)):
    from datetime import time

    def parse(hhmm: str) -> time:
        h, m = hhmm.split(":")
        return time(int(h), int(m))

    employee = AIEmployee(
        tenant_id=tenant.id,
        name=body.name,
        language=body.language,
        voice_tier=body.voice_tier,
        script=body.script,
        capture_fields=body.capture_fields,
        max_concurrency=body.max_concurrency,
        window_start=parse(body.window_start),
        window_end=parse(body.window_end),
        timezone=body.timezone,
        max_attempts=body.max_attempts,
        retry_delay_minutes=body.retry_delay_minutes,
    )
    session.add(employee)
    session.commit()
    return {"employee_id": employee.id}


class NumberAssign(BaseModel):
    phone_number: str = Field(pattern=r"^\+\d{10,15}$", description="ExoPhone in E.164, e.g. +918012345678")


@router.put("/employees/{employee_id}/number")
def assign_number(
    employee_id: int,
    body: NumberAssign,
    tenant: Tenant = Depends(current_tenant),
    session: Session = Depends(get_session),
):
    """Attach an ExoPhone to an agent. It becomes the caller ID for outbound calls and the number that routes inbound calls to this agent."""
    employee = _own_employee(session, tenant, employee_id)
    taken = session.scalar(
        select(AIEmployee.id).where(AIEmployee.phone_number == body.phone_number, AIEmployee.id != employee.id)
    )
    if taken is not None:
        raise HTTPException(status_code=409, detail="this number is already assigned to another agent")
    employee.phone_number = body.phone_number
    session.commit()
    return {"employee_id": employee.id, "phone_number": employee.phone_number}


def _own_employee(session: Session, tenant: Tenant, employee_id: int) -> AIEmployee:
    employee = session.get(AIEmployee, employee_id)
    if employee is None or employee.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="employee not found")
    return employee


@router.post("/leads")
def create_lead(body: LeadIn, tenant: Tenant = Depends(current_tenant), session: Session = Depends(get_session)):
    _own_employee(session, tenant, body.employee_id)
    lead = Lead(
        tenant_id=tenant.id,
        employee_id=body.employee_id,
        phone_e164=body.phone_e164,
        name=body.name,
        source=body.source,
        metadata_json=body.metadata,
        status=LeadStatus.NEW,
        next_attempt_at=datetime.utcnow(),
    )
    session.add(lead)
    session.commit()
    return {"lead_id": lead.id}


@router.post("/leads/bulk")
def create_leads_bulk(body: list[LeadIn], tenant: Tenant = Depends(current_tenant), session: Session = Depends(get_session)):
    now = datetime.utcnow()
    ids = []
    for item in body:
        _own_employee(session, tenant, item.employee_id)
        lead = Lead(
            tenant_id=tenant.id,
            employee_id=item.employee_id,
            phone_e164=item.phone_e164,
            name=item.name,
            source=item.source,
            metadata_json=item.metadata,
            status=LeadStatus.NEW,
            next_attempt_at=now,
        )
        session.add(lead)
        session.flush()
        ids.append(lead.id)
    session.commit()
    return {"created": len(ids), "lead_ids": ids}


@router.get("/calls/{call_id}")
def get_call(call_id: int, tenant: Tenant = Depends(current_tenant), session: Session = Depends(get_session)):
    call = session.get(Call, call_id)
    if call is None or call.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="call not found")
    return {
        "call_id": call.id,
        "status": call.status.value,
        "outcome": call.outcome,
        "summary": call.summary,
        "billed_seconds": call.billed_seconds,
        "captured_fields": call.captured_fields,
        "turns": [{"role": t.role, "text": t.text} for t in call.turns],
    }
