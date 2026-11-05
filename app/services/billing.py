"""Credit ledger. Money is integer paise; 1 credit = 100 paise = ₹1."""

import math

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import LedgerEntry, LedgerKind, VoiceTier

# Paise per connected minute, from the published price list.
RATE_PAISE_PER_MIN = {
    VoiceTier.STANDARD: 500,
    VoiceTier.PREMIUM: 680,
    VoiceTier.CUSTOM: 350,
}
NUMBER_FEE_PAISE_PER_MONTH = 64900


class InsufficientCredits(Exception):
    pass


def usage_charge_paise(voice_tier: VoiceTier, billed_seconds: int) -> int:
    """Per-second billing, rounded up to the next paisa."""
    rate = RATE_PAISE_PER_MIN[voice_tier]
    return math.ceil(billed_seconds * rate / 60)


def balance_paise(session: Session, tenant_id: int) -> int:
    session.flush()  # include pending ledger rows in the sum
    total = session.scalar(
        select(func.coalesce(func.sum(LedgerEntry.amount_paise), 0)).where(LedgerEntry.tenant_id == tenant_id)
    )
    return int(total)


def top_up(session: Session, tenant_id: int, credits: int, note: str = "top-up") -> None:
    session.add(
        LedgerEntry(tenant_id=tenant_id, kind=LedgerKind.TOPUP, amount_paise=credits * 100, note=note)
    )


def charge_call(session: Session, tenant_id: int, call_id: int, voice_tier: VoiceTier, billed_seconds: int) -> int:
    amount = usage_charge_paise(voice_tier, billed_seconds)
    if amount == 0:
        return 0
    session.add(
        LedgerEntry(
            tenant_id=tenant_id,
            kind=LedgerKind.USAGE,
            amount_paise=-amount,
            call_id=call_id,
            note=f"{billed_seconds}s on {voice_tier.value}",
        )
    )
    return amount
