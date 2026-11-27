"""The per-agent client report: outcome mapping, payload shape, and the shareable API endpoint."""

from datetime import datetime
from datetime import time as dtime

from app.db import SessionLocal
from app.models import AIEmployee, AppSetting, Call, CallStatus, CallTurn, Lead, LeadStatus, Tenant, VoiceTier
from app.report import build_report, category, report_token


def _call(outcome="", summary="", billed=30, status=CallStatus.COMPLETED):
    return Call(outcome=outcome, summary=summary, billed_seconds=billed, status=status)


def test_outcome_maps_to_the_client_facing_category():
    assert category(_call("said_yes")) == "interested"
    assert category(_call("appointment_booked")) == "interested"
    assert category(_call("said_no")) == "not_interested"
    assert category(_call("do_not_call")) == "not_interested"
    assert category(_call("unclear_reply")) == "maybe"
    assert category(_call("completed")) == "maybe"
    assert category(_call("", billed=0, status=CallStatus.NO_ANSWER)) == "not_reached"
    assert category(_call("said_yes", billed=0, status=CallStatus.BUSY)) == "not_reached"  # never answered


def test_outcome_is_read_from_the_summary_when_the_hangup_cause_overwrote_it():
    call = _call(outcome="NORMAL_CLEARING", summary="2 turns; outcome=said_yes")
    assert category(call) == "interested"


def test_report_token_is_stable_and_specific_to_the_agent():
    assert report_token(135) == report_token(135)
    assert report_token(135) != report_token(136)
    assert len(report_token(135)) == 32


def _setup(db) -> int:
    tenant = Tenant(name="Acme", api_key_hash="x", webhook_secret="s", webhook_url="https://hooks.example.com/in")
    db.add(tenant)
    db.flush()
    agent = AIEmployee(tenant_id=tenant.id, name="PreKids Preschool", language="te", voice_tier=VoiceTier.PREMIUM,
                       script="Recorded preschool franchise message.", window_start=dtime(9, 0), window_end=dtime(20, 0))
    db.add(agent)
    db.flush()
    db.add(AppSetting(key=f"t{tenant.id}:employee_extras",
                      value={str(agent.id): {"agent_mode": "recorded_audio", "voice_gender": "female", "speaking_style": "professional"}}))
    for outcome, billed in (("said_yes", 40), ("said_no", 25), ("", 0)):
        lead = Lead(tenant_id=tenant.id, employee_id=agent.id, phone_e164="+91999990000" + str(billed), name="Lead", status=LeadStatus.DONE)
        db.add(lead)
        db.flush()
        status = CallStatus.NO_ANSWER if billed == 0 else CallStatus.COMPLETED
        call = Call(tenant_id=tenant.id, employee_id=agent.id, lead_id=lead.id, status=status, outcome=outcome,
                    summary=f"2 turns; outcome={outcome}", billed_seconds=billed, started_at=datetime(2026, 11, 26, 11))
        db.add(call)
        db.flush()
        if outcome == "said_yes":
            db.add(CallTurn(call_id=call.id, role="caller", text="అవును"))
    db.commit()
    return agent.id


def test_build_report_shape_and_counts():
    with SessionLocal() as db:
        agent_id = _setup(db)
        report = build_report(db, db.get(AIEmployee, agent_id))
    assert report["agent"]["mode_label"] == "Recorded message"
    assert report["summary"]["total"] == 3
    assert report["summary"]["answered"] == 2
    assert report["summary"]["interested"] == 1
    assert report["summary"]["not_interested"] == 1
    assert report["summary"]["not_reached"] == 1
    yes = next(c for c in report["calls"] if c["category"] == "interested")
    assert yes["said"] == "అవును" and yes["phone"].startswith("+91")


def test_report_api_opens_with_a_valid_token_and_refuses_a_bad_one(client):
    with SessionLocal() as db:
        agent_id = _setup(db)
    token = report_token(agent_id)
    ok = client.get(f"/dashboard/api/agents/{agent_id}/report", params={"token": token})
    assert ok.status_code == 200 and ok.json()["summary"]["interested"] == 1
    assert client.get(f"/dashboard/api/agents/{agent_id}/report", params={"token": "wrong"}).status_code == 403
    assert client.get(f"/dashboard/api/agents/{agent_id}/report").status_code in (401, 403)


def test_report_page_is_served(client):
    r = client.get("/dashboard/report")
    assert r.status_code == 200 and "Agent Call Report" in r.text


def test_report_includes_the_booked_demo_time():
    from app.models import Appointment, AppointmentStatus
    with SessionLocal() as db:
        agent_id = _setup(db)
        yes_call = next(c for c in db.query(Call).filter(Call.employee_id == agent_id) if "said_yes" in (c.summary or ""))
        db.add(Appointment(tenant_id=yes_call.tenant_id, call_id=yes_call.id, employee_id=agent_id,
                           contact_name="Lead", phone="+919999", scheduled_at=datetime(2026, 11, 28, 6, 30),
                           status=AppointmentStatus.CONFIRMED))
        db.commit()
        report = build_report(db, db.get(AIEmployee, agent_id))
    row = next(c for c in report["calls"] if c["booked"])
    assert row["demo_at"] == "2026-11-28T06:30:00+00:00"
    assert report["summary"]["booked"] == 1


def test_report_and_endpoint_expose_the_recording_with_the_share_token(client, monkeypatch, tmp_path):
    import wave
    from app.report import report_token
    monkeypatch.setenv("VA_RECORDINGS_DIR", str(tmp_path))
    with SessionLocal() as db:
        agent_id = _setup(db)
        call = next(c for c in db.query(Call).filter(Call.employee_id == agent_id) if (c.billed_seconds or 0) > 0)
        name = f"call_{call.id}.wav"
        with wave.open(str(tmp_path / name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\x00\x00" * 8000)
        call.recording_file = name
        db.commit()
        call_id = call.id
    token = report_token(agent_id)
    report = client.get(f"/dashboard/api/agents/{agent_id}/report", params={"token": token}).json()
    row = next(c for c in report["calls"] if c["id"] == call_id)
    assert row["recording"] == f"/dashboard/api/calls/{call_id}/recording?token={token}"
    audio = client.get(f"/dashboard/api/calls/{call_id}/recording", params={"token": token})
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert client.get(f"/dashboard/api/calls/{call_id}/recording", params={"token": "bad"}).status_code == 403
