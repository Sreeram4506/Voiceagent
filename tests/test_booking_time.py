"""A booked appointment must be saved for the day and time the caller agreed, not "24 hours after the call"."""

import asyncio
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

from app.db import SessionLocal
from app.models import AIEmployee, Appointment, Call, CallStatus, Tenant, VoiceTier
from app.services.call_engine import CallResult, finalize_call
from app.streaming.gemini_live_session import FUNCTIONS, GeminiLiveSession
from app.streaming.realtime_session import TOOLS, RealtimeSession, call_details

IST = ZoneInfo("Asia/Kolkata")
CALL_TIME = datetime(2026, 11, 26, 16, 10, tzinfo=IST)  # Thursday afternoon in Hyderabad


def gemini_session() -> GeminiLiveSession:
    session = GeminiLiveSession("key", "Book a demo.", "te", model="gemini-3.8-live", local_time=CALL_TIME)

    async def capture(message: dict) -> None:
        pass

    session._send_json = capture
    return session


def test_booking_tool_asks_for_the_agreed_date_and_time():
    book = next(tool for tool in TOOLS if tool["name"] == "book_appointment")
    assert set(book["parameters"]["required"]) == {"date", "time"}
    gemini_book = next(f for f in FUNCTIONS if f["name"] == "book_appointment")
    assert set(gemini_book["parameters"]["required"]) == {"date", "time"}


def test_agent_is_told_the_year_so_it_can_write_the_date():
    details = call_details(local_time=CALL_TIME)
    assert "26 November 2026" in details
    assert "today Thursday 2026-11-26, tomorrow Friday 2026-11-27, day after tomorrow Saturday 2026-11-28" in details


def test_gemini_booking_is_saved_in_utc_for_the_callers_local_time():
    session = gemini_session()
    call = {"id": "1", "name": "book_appointment", "args": {"date": "2026-11-27", "time": "12:00"}}
    asyncio.run(session._handle_calls([call]))
    assert session.outcome() == "appointment_booked"
    assert session.appointment_at() == datetime(2026, 11, 27, 6, 30)  # 12:00 IST


def test_openai_booking_reads_the_json_arguments():
    session = RealtimeSession("key", "Book a demo.", "te", local_time=CALL_TIME)

    async def capture(event: dict) -> None:
        pass

    session._to_openai = capture
    book = {"type": "function_call", "name": "book_appointment", "call_id": "c1",
            "arguments": json.dumps({"date": "2026-11-28", "time": "16:30"})}
    asyncio.run(session._handle_tools({"output": [book]}))
    assert session.appointment_at() == datetime(2026, 11, 28, 11, 0)


def test_unreadable_or_past_booking_time_is_not_trusted():
    session = gemini_session()
    for args in ({"date": "tomorrow", "time": "noon"}, {"date": "2026-11-20", "time": "10:00"}, {}):
        session._booking = args
        assert session.appointment_at() is None


def test_later_booking_in_the_same_call_wins():
    session = gemini_session()
    for time in ("11:00", "15:00"):
        call = {"id": time, "name": "book_appointment", "args": {"date": "2026-11-27", "time": time}}
        asyncio.run(session._handle_calls([call]))
    assert session.appointment_at() == datetime(2026, 11, 27, 9, 30)


def _finalize(session, result: CallResult) -> Appointment:
    tenant = Tenant(name="Acme", api_key_hash="x", webhook_secret="s", webhook_url="https://hooks.example.com/in")
    session.add(tenant)
    session.flush()
    employee = AIEmployee(tenant_id=tenant.id, name="Suma", language="te", voice_tier=VoiceTier.STANDARD,
                          script="Book a demo.", window_start=time(9, 0), window_end=time(20, 0))
    session.add(employee)
    session.flush()
    call = Call(tenant_id=tenant.id, employee_id=employee.id, status=CallStatus.CONNECTED,
                started_at=datetime(2026, 11, 26, 10, 40))
    session.add(call)
    session.flush()
    finalize_call(session, call, employee, None, result, connected_seconds=60, now=datetime(2026, 11, 26, 10, 41))
    session.flush()
    return session.query(Appointment).filter(Appointment.call_id == call.id).one()


def test_finalized_call_saves_the_agreed_time(session):
    agreed = datetime(2026, 11, 27, 6, 30)
    result = CallResult(outcome="appointment_booked", summary="", actions=["book_appointment"], appointment_at=agreed)
    assert _finalize(session, result).scheduled_at == agreed


def test_booking_without_a_time_still_falls_back_to_a_day_later(session):
    result = CallResult(outcome="appointment_booked", summary="", actions=["book_appointment"])
    assert _finalize(session, result).scheduled_at == datetime(2026, 11, 27, 10, 41)


def test_dashboard_marks_appointment_times_as_utc(client):
    r = client.post("/dashboard/api/auth/login", json={"email": "admin@suma.ai", "password": "suma2026"})
    assert r.status_code == 200
    created = client.post("/dashboard/api/appointments", json={
        "name": "Ravi", "phone": "+919999900021", "scheduled_at": "2026-11-27T06:30:00.000Z", "status": "scheduled"})
    assert created.status_code == 200
    rows = client.get("/dashboard/api/appointments").json()
    rows = rows if isinstance(rows, list) else rows.get("appointments", [])
    mine = [row for row in rows if row.get("phone") == "+919999900021"]
    assert mine and mine[0]["scheduled_at"] == "2026-11-27T06:30:00+00:00"
