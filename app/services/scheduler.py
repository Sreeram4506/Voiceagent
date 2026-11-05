"""Picks due leads and places calls, respecting compliance, concurrency, and each agent's ExoPhone."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import AIEmployee, Call, CallDirection, CallStatus, Lead, LeadStatus, Tenant
from app.providers.base import TelephonyProvider
from app.services.billing import balance_paise
from app.services.compliance import DNDRegistry, check_lead


@dataclass
class DispatchReport:
    dialed: list[int]
    skipped: dict[int, str]


def dispatch_due(
    session: Session,
    telephony: TelephonyProvider,
    dnd: DNDRegistry,
    now: datetime,
    callback_url: str,
    retry_delay_minutes: int,
    batch_size: int = 50,
) -> DispatchReport:
    leads = session.scalars(
        select(Lead)
        .where(Lead.status.in_([LeadStatus.NEW, LeadStatus.QUEUED]))
        .where(Lead.next_attempt_at <= now)
        .order_by(Lead.next_attempt_at)
        .limit(batch_size)
    ).all()

    report = DispatchReport(dialed=[], skipped={})
    for lead in leads:
        employee = session.get(AIEmployee, lead.employee_id)
        tenant = session.get(Tenant, lead.tenant_id)

        if not employee or not tenant:
            report.skipped[lead.id] = "missing_employee_or_tenant"
            continue

        if not employee.phone_number:
            report.skipped[lead.id] = "no_phone_number_assigned"
            continue

        if balance_paise(session, tenant.id) <= 0:
            report.skipped[lead.id] = "no_credits"
            continue

        in_flight = session.scalar(
            select(func.count(Call.id)).where(
                Call.employee_id == employee.id,
                Call.status.in_([CallStatus.DIALING, CallStatus.CONNECTED]),
            )
        )
        if in_flight >= employee.max_concurrency:
            report.skipped[lead.id] = "concurrency_limit"
            continue

        decision = check_lead(lead, employee, now, dnd, retry_delay_minutes)
        if not decision.allowed:
            report.skipped[lead.id] = decision.reason
            continue

        # Create the call row first so its id can travel with the dial request.
        call = Call(
            tenant_id=tenant.id,
            lead_id=lead.id,
            employee_id=employee.id,
            direction=CallDirection.OUTBOUND,
            status=CallStatus.DIALING,
            started_at=now,
        )
        session.add(call)
        session.flush()

        result = telephony.dial(
            lead.phone_e164,
            employee.phone_number,
            callback_url,
            custom_field=str(call.id),
        )
        call.provider_call_id = result.provider_call_id or None
        if not result.accepted:
            call.status = CallStatus.REFUSED

        lead.attempts += 1
        lead.last_called_at = now
        lead.status = LeadStatus.CALLING if result.accepted else LeadStatus.QUEUED
        if not result.accepted and result.reason == "refused":
            # Carrier refusal: pause the campaign to protect the sender reputation.
            employee.paused = True
        if result.accepted:
            report.dialed.append(lead.id)
        else:
            report.skipped[lead.id] = result.reason or "dial_failed"

    session.flush()
    return report


def employee_for_number(session: Session, phone_number: str) -> AIEmployee | None:
    """Which agent answers calls to this ExoPhone. Used by inbound routing."""
    return session.scalar(select(AIEmployee).where(AIEmployee.phone_number == phone_number))
