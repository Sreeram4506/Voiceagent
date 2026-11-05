from datetime import datetime, time, timedelta

from app.models import AIEmployee, Lead, LeadStatus, VoiceTier
from app.services.billing import usage_charge_paise
from app.services.compliance import InMemoryDNDRegistry, check_lead, within_calling_window


def make_employee(**overrides) -> AIEmployee:
    base = dict(
        tenant_id=1,
        name="Riya",
        language="hi",
        voice_tier=VoiceTier.STANDARD,
        script="test",
        max_concurrency=5,
        window_start=time(9, 0),
        window_end=time(20, 0),
        timezone="Asia/Kolkata",
        max_attempts=3,
        retry_delay_minutes=30,
        paused=False,
    )
    base.update(overrides)
    return AIEmployee(**base)


def make_lead(**overrides) -> Lead:
    base = dict(
        tenant_id=1,
        employee_id=1,
        phone_e164="+919999900001",
        status=LeadStatus.NEW,
        attempts=0,
        next_attempt_at=datetime.utcnow(),
        last_called_at=None,
    )
    base.update(overrides)
    return Lead(**base)


# 10:00 IST == 04:30 UTC
NOON_IST_UTC = datetime(2026, 11, 1, 4, 30)
# 02:00 IST == 20:30 UTC previous day
NIGHT_IST_UTC = datetime(2026, 10, 31, 20, 30)


def test_calling_window_uses_employee_timezone():
    emp = make_employee()
    assert within_calling_window(emp, NOON_IST_UTC)
    assert not within_calling_window(emp, NIGHT_IST_UTC)


def test_allowed_lead_inside_window():
    decision = check_lead(make_lead(), make_employee(), NOON_IST_UTC, InMemoryDNDRegistry(), 30)
    assert decision.allowed


def test_blocks_outside_window():
    decision = check_lead(make_lead(), make_employee(), NIGHT_IST_UTC, InMemoryDNDRegistry(), 30)
    assert decision.reason == "outside_calling_window"


def test_blocks_dnd_numbers():
    dnd = InMemoryDNDRegistry({"+919999900001"})
    decision = check_lead(make_lead(), make_employee(), NOON_IST_UTC, dnd, 30)
    assert decision.reason == "dnd_registry"


def test_blocks_do_not_call_status():
    decision = check_lead(make_lead(status=LeadStatus.DNC), make_employee(), NOON_IST_UTC, InMemoryDNDRegistry(), 30)
    assert decision.reason == "do_not_call"


def test_respects_retry_delay():
    recent = NOON_IST_UTC - timedelta(minutes=5)
    decision = check_lead(make_lead(last_called_at=recent, attempts=1), make_employee(), NOON_IST_UTC, InMemoryDNDRegistry(), 30)
    assert decision.reason == "too_soon_since_last_call"


def test_caps_attempts():
    decision = check_lead(make_lead(attempts=3), make_employee(), NOON_IST_UTC, InMemoryDNDRegistry(), 30)
    assert decision.reason == "max_attempts_reached"


def test_paused_campaign_blocks_everything():
    decision = check_lead(make_lead(), make_employee(paused=True), NOON_IST_UTC, InMemoryDNDRegistry(), 30)
    assert decision.reason == "campaign_paused"


def test_billing_is_per_second_rounded_up_in_paise():
    # Standard is 500 paise/min. 90 seconds = 750 paise.
    assert usage_charge_paise(VoiceTier.STANDARD, 90) == 750
    # 1 second = ceil(500/60) = 9 paise.
    assert usage_charge_paise(VoiceTier.STANDARD, 1) == 9
    assert usage_charge_paise(VoiceTier.PREMIUM, 60) == 680
    assert usage_charge_paise(VoiceTier.CUSTOM, 0) == 0
