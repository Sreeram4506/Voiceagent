"""Picks due leads and places calls, respecting compliance, concurrency, and each agent's ExoPhone."""

from dataclasses import dataclass
from datetime import datetime, time

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import AIEmployee, Call, CallDirection, CallStatus, Campaign, CampaignStatus, Lead, LeadStatus, Tenant
from app.providers.base import TelephonyProvider
from app.services.billing import balance_paise
from app.services.compliance import DNDRegistry, check_lead, within_calling_window
from app.streaming.ws import set_call_mode


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
        .outerjoin(Campaign, Lead.campaign_id == Campaign.id)
        .where(Lead.status.in_([LeadStatus.NEW, LeadStatus.QUEUED]))
        .where(Lead.next_attempt_at <= now)
        # A campaign's contacts are dialed only while it runs; contacts outside any campaign as before.
        .where(or_(Lead.campaign_id.is_(None), Campaign.status == CampaignStatus.RUNNING))
        .order_by(Lead.next_attempt_at)
        # Look past contacts that can't be dialed now, so they don't starve newer campaigns.
        .limit(max(batch_size * LOOKAHEAD_FACTOR, MIN_LOOKAHEAD))
    ).all()

    report = DispatchReport(dialed=[], skipped={})
    for lead in leads:
        if len(report.dialed) >= batch_size:
            break
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
        call_mode = "ai_agent"
        call_greeting = ""
        if lead.campaign_id:
            camp = session.get(Campaign, lead.campaign_id)
            if camp:
                call_greeting = camp.greeting or ""
                if camp.config and isinstance(camp.config, dict):
                    call_mode = camp.config.get("mode", "ai_agent")

        call = Call(
            tenant_id=tenant.id,
            lead_id=lead.id,
            employee_id=employee.id,
            direction=CallDirection.OUTBOUND,
            status=CallStatus.DIALING,
            started_at=now,
            captured_fields={
                "phone": lead.phone_e164,
                "name": lead.name,
                "mode": call_mode,
                "greeting": call_greeting,
            },
        )
        session.add(call)
        session.flush()
        set_call_mode(call.id, call_mode)

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


LOOKAHEAD_FACTOR = 10
MIN_LOOKAHEAD = 200

# Calling window for campaigns: the owner wants calls to go out at any hour.
FULL_DAY = (time(0, 0), time(23, 59))

_WAIT_REASONS = {
    "outside_calling_window": "outside calling hours ({window}); calls start automatically when the window opens",
    "no_credits": "no credits left; add credits to start dialing",
    "no_phone_number_assigned": "the agent has no phone number assigned",
    "concurrency_limit": "all lines are busy; waiting for calls to finish",
    "max_attempts_reached": "these contacts have used all their call attempts",
    "too_soon_since_last_call": "waiting for the retry gap since the last call",
    "campaign_paused": "the agent is paused",
    "dnd_registry": "the numbers are on the Do-Not-Call list",
    "missing_employee_or_tenant": "the contacts have no agent assigned",
    "no_contacts": "there are no contacts to call",
}


def explain_wait(reason: str, employee: AIEmployee | None = None) -> str:
    """A dispatch skip reason in plain words, for the dashboard and logs."""
    window = "9:00-20:00"
    if employee is not None:
        window = f"{employee.window_start:%H:%M}-{employee.window_end:%H:%M} {employee.timezone}"
    return _WAIT_REASONS.get(reason, reason.replace("_", " ")).format(window=window)


def dial_blocker(session: Session, employee: AIEmployee | None, now: datetime) -> str | None:
    """Why this agent's calls cannot start right now, in plain words, or None when they can."""
    if employee is None:
        return explain_wait("missing_employee_or_tenant")
    if employee.paused:
        return explain_wait("campaign_paused")
    if not employee.phone_number:
        return explain_wait("no_phone_number_assigned")
    if balance_paise(session, employee.tenant_id) <= 0:
        return explain_wait("no_credits")
    if not within_calling_window(employee, now):
        return explain_wait("outside_calling_window", employee)
    return None


def employee_for_number(session: Session, phone_number: str) -> AIEmployee | None:
    """Which agent answers calls to this ExoPhone. Used by inbound routing."""
    return session.scalar(select(AIEmployee).where(AIEmployee.phone_number == phone_number))
