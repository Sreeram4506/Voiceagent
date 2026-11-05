"""Rules that decide whether a lead may be dialed right now."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

from app.models import AIEmployee, Lead, LeadStatus


class DNDRegistry(Protocol):
    def is_blocked(self, phone_e164: str) -> bool: ...


class InMemoryDNDRegistry:
    """Stand-in for the national DND/NDNC lookup. Replace with the real registry client before launch."""

    def __init__(self, blocked: set[str] | None = None):
        self._blocked = set(blocked or ())

    def is_blocked(self, phone_e164: str) -> bool:
        return phone_e164 in self._blocked


@dataclass
class Decision:
    allowed: bool
    reason: str = "ok"


def within_calling_window(employee: AIEmployee, now_utc: datetime) -> bool:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local = now_utc.astimezone(ZoneInfo(employee.timezone)).time()
    return employee.window_start <= local < employee.window_end


def check_lead(
    lead: Lead,
    employee: AIEmployee,
    now_utc: datetime,
    dnd: DNDRegistry,
    retry_delay_minutes: int,
) -> Decision:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    if lead.status == LeadStatus.DNC:
        return Decision(False, "do_not_call")
    if employee.paused:
        return Decision(False, "campaign_paused")
    if lead.status in (LeadStatus.PAUSED, LeadStatus.DONE, LeadStatus.FAILED):
        return Decision(False, f"status_{lead.status.value}")
    if lead.attempts >= employee.max_attempts:
        return Decision(False, "max_attempts_reached")
    last_called = lead.last_called_at
    if last_called and last_called.tzinfo is None:
        last_called = last_called.replace(tzinfo=timezone.utc)
    if last_called and now_utc - last_called < timedelta(minutes=retry_delay_minutes):
        return Decision(False, "too_soon_since_last_call")
    if not within_calling_window(employee, now_utc):
        return Decision(False, "outside_calling_window")
    if dnd.is_blocked(lead.phone_e164):
        return Decision(False, "dnd_registry")
    return Decision(True)
