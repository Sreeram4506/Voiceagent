"""Runs one call turn by turn and finalizes its outcome.

Audio streaming is provider-specific and sits behind the STT/TTS interfaces. This module
owns the conversation logic: turn loop, tool execution, end-of-call classification, and billing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import AIEmployee, Appointment, AppointmentStatus, Call, CallStatus, CallTurn, Lead, LeadStatus
from app.providers.base import AgentReply, LLMProvider, TTSProvider
from app.services.billing import charge_call

MAX_TURNS = 40


@dataclass
class CallResult:
    outcome: str
    summary: str
    actions: list[str] = field(default_factory=list)


def run_turns(
    session: Session,
    call: Call,
    employee: AIEmployee,
    llm: LLMProvider,
    tts: TTSProvider,
    caller_utterances: list[str],
    actions_log: list[str],
) -> CallResult:
    """Drive the conversation. caller_utterances stands in for STT output in this offline loop."""
    history: list[tuple[str, str]] = []
    booked = False
    ended_by_agent = False

    reply: AgentReply = llm.respond(employee.script, history, "")
    _record(session, call, "agent", reply.text, tts, employee)
    history.append(("agent", reply.text))

    for utterance in caller_utterances[:MAX_TURNS]:
        _record(session, call, "caller", utterance, None, employee)
        history.append(("caller", utterance))
        reply = llm.respond(employee.script, history, utterance)
        for tool in reply.tool_requests:
            actions_log.append(tool.name)
            if tool.name == "book_appointment":
                booked = True
        _record(session, call, "agent", reply.text, tts, employee)
        history.append(("agent", reply.text))
        if reply.end_call:
            ended_by_agent = True
            break

    lead = session.get(Lead, call.lead_id) if call.lead_id else None
    if booked:
        outcome = "appointment_booked"
        _persist_appointment(session, call, employee, lead)
    elif ended_by_agent:
        outcome = "completed"
    else:
        outcome = "no_resolution"
    summary = f"{len(history)} turns; outcome={outcome}"
    return CallResult(outcome=outcome, summary=summary, actions=actions_log)


def finalize_call(
    session: Session,
    call: Call,
    employee: AIEmployee,
    lead: Lead | None,
    result: CallResult,
    connected_seconds: int,
    now: datetime,
) -> int:
    """Close the call, bill connected seconds, and update the lead. Returns the amount charged in paise."""
    call.status = CallStatus.COMPLETED
    call.outcome = result.outcome
    call.summary = result.summary
    call.connected_at = now - timedelta(seconds=connected_seconds)
    call.ended_at = now
    call.billed_seconds = max(connected_seconds, 0)
    charged = charge_call(session, call.tenant_id, call.id, employee.voice_tier, call.billed_seconds)
    if lead is not None:
        lead.status = LeadStatus.DONE
    if result.outcome == "appointment_booked" or "book_appointment" in (result.actions or []):
        existing = session.query(Appointment).filter(Appointment.call_id == call.id).first()
        if not existing:
            _persist_appointment(session, call, employee, lead)
    return charged


def _persist_appointment(session: Session, call: Call, employee: AIEmployee, lead: Lead | None) -> None:
    name = ""
    phone = ""
    if lead is not None:
        name = lead.name or ""
        phone = lead.phone_e164
    else:
        cf = call.captured_fields or {}
        name = str(cf.get("name") or "")
        phone = str(cf.get("phone") or "")
    when = datetime.utcnow() + timedelta(hours=24)
    session.add(
        Appointment(
            tenant_id=call.tenant_id,
            call_id=call.id,
            lead_id=call.lead_id,
            employee_id=employee.id,
            campaign_id=getattr(lead, "campaign_id", None) if lead else None,
            contact_name=name,
            phone=phone,
            scheduled_at=when,
            notes="Booked by voice agent",
            status=AppointmentStatus.CONFIRMED,
        )
    )


def _record(session: Session, call: Call, role: str, text: str, tts: TTSProvider | None, employee: AIEmployee) -> None:
    if tts is not None:
        tts.synthesize(text, employee.language, employee.voice_tier.value)
    session.add(CallTurn(call_id=call.id, role=role, text=text))
