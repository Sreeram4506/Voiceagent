"""Picks due leads and places calls, respecting compliance, concurrency, and each agent's ExoPhone."""

import logging
from dataclasses import dataclass
from datetime import datetime, time

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from app.models import AIEmployee, Call, CallDirection, CallStatus, Campaign, CampaignStatus, Lead, LeadStatus, Tenant
from app.providers.base import DialResult, TelephonyProvider
from app.services.billing import balance_paise
from app.services.compliance import DNDRegistry, check_lead, within_calling_window
from app.services.retries import settle_lead, utc_naive
from app.streaming.ws import set_call_mode

log = logging.getLogger(__name__)

DIALABLE = (LeadStatus.NEW, LeadStatus.QUEUED)


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
    """Dial due leads. Commits as it goes: each lead is claimed, and its call saved, before its phone rings."""
    leads = session.scalars(
        select(Lead)
        .outerjoin(Campaign, Lead.campaign_id == Campaign.id)
        .where(Lead.status.in_(DIALABLE))
        .where(Lead.next_attempt_at <= utc_naive(now))
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
        reason = _why_not_dial(session, lead, employee, now, dnd, retry_delay_minutes)
        if reason is None and not claim_lead(session, lead, now):
            reason = "already_dialing"  # another worker claimed it after we read it
        if reason is not None:
            report.skipped[lead.id] = reason
            continue

        call = _new_call(session, lead, employee, now)
        session.commit()  # the claim and the call are saved before the phone rings

        result = _dial(telephony, lead, employee, callback_url, call.id)
        call.provider_call_id = result.provider_call_id or None
        if result.accepted:
            report.dialed.append(lead.id)
        else:
            call.status = CallStatus.REFUSED
            settle_lead(lead, employee, CallStatus.REFUSED, now)
            report.skipped[lead.id] = result.reason or "dial_failed"
            if result.reason == "refused":
                # Carrier refusal: pause the campaign to protect the sender reputation.
                employee.paused = True
        session.commit()
    return report


def _why_not_dial(
    session: Session,
    lead: Lead,
    employee: AIEmployee | None,
    now: datetime,
    dnd: DNDRegistry,
    retry_delay_minutes: int,
) -> str | None:
    """The reason this lead can't be dialed right now, or None when it can."""
    tenant = session.get(Tenant, lead.tenant_id)
    if not employee or not tenant:
        return "missing_employee_or_tenant"
    if not employee.phone_number:
        return "no_phone_number_assigned"
    if balance_paise(session, tenant.id) <= 0:
        return "no_credits"
    in_flight = session.scalar(
        select(func.count(Call.id)).where(
            Call.employee_id == employee.id,
            Call.status.in_([CallStatus.DIALING, CallStatus.CONNECTED]),
        )
    )
    if in_flight >= employee.max_concurrency:
        return "concurrency_limit"
    decision = check_lead(lead, employee, now, dnd, retry_delay_minutes)
    return None if decision.allowed else decision.reason


def claim_lead(session: Session, lead: Lead, now: datetime) -> bool:
    """Mark the lead as being dialed, unless another dispatcher already did.

    One conditional UPDATE is the lock: whoever flips the status from new/queued to calling owns the
    lead, on SQLite and Postgres alike.
    """
    claimed = session.execute(
        update(Lead)
        .where(Lead.id == lead.id, Lead.status.in_(DIALABLE))
        .values(status=LeadStatus.CALLING, attempts=Lead.attempts + 1, last_called_at=utc_naive(now))
        .execution_options(synchronize_session=False)
    ).rowcount == 1
    if claimed:
        session.refresh(lead, ["status", "attempts", "last_called_at"])
    return claimed


def _new_call(session: Session, lead: Lead, employee: AIEmployee, now: datetime) -> Call:
    call_mode = "ai_agent"
    call_greeting = ""
    if lead.campaign_id:
        camp = session.get(Campaign, lead.campaign_id)
        if camp:
            call_greeting = camp.greeting or ""
            if camp.config and isinstance(camp.config, dict):
                call_mode = camp.config.get("mode", "ai_agent")

    call = Call(
        tenant_id=lead.tenant_id,
        lead_id=lead.id,
        employee_id=employee.id,
        direction=CallDirection.OUTBOUND,
        status=CallStatus.DIALING,
        started_at=utc_naive(now),
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
    return call


def _dial(telephony: TelephonyProvider, lead: Lead, employee: AIEmployee, callback_url: str, call_id: int) -> DialResult:
    """Place the call. A network error counts as a failed dial, so the lead is released, not stuck as calling."""
    try:
        return telephony.dial(lead.phone_e164, employee.phone_number, callback_url, custom_field=str(call_id))
    except Exception:
        log.exception("dialing lead %s (call %s) failed", lead.id, call_id)
        return DialResult(provider_call_id="", accepted=False, reason="dial_error")


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
