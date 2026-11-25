"""Settings saved in the employee editor must reach the live call."""

from datetime import datetime

from app.db import SessionLocal
from app.models import Call, CallStatus
from app.streaming import factory
from app.streaming.realtime_session import MODEL, RealtimeSession, persona
from app.streaming.ws import _load_context, _persona_settings


def login(client):
    r = client.post("/dashboard/api/auth/login", json={"email": "admin@suma.ai", "password": "suma2026"})
    assert r.status_code == 200


def employee(client, employee_id):
    return next(e for e in client.get("/dashboard/api/employees").json() if e["id"] == employee_id)


def test_every_editor_setting_is_saved_and_returned(client):
    login(client)
    eid = client.post("/dashboard/api/employees", json={"name": "Suma"}).json()["id"]
    saved = {
        "name": "Priya", "language": "hi", "script": "Sell plots.", "voice_tier": "premium",
        "voice_gender": "male", "speaking_style": "energetic", "temperature": 0.4, "speech_rate": 1.25,
        "greeting": "नमस्ते", "max_concurrency": 25, "max_attempts": 5,
        "window_start": "10:30", "window_end": "18:00", "paused": True,
    }
    assert client.patch(f"/dashboard/api/employees/{eid}", json=saved).json() == {"ok": True}
    got = employee(client, eid)
    assert {k: got[k] for k in saved} == saved


def test_invalid_calling_window_is_rejected_and_nothing_changes(client):
    login(client)
    eid = client.post("/dashboard/api/employees", json={"name": "Suma"}).json()["id"]
    r = client.patch(f"/dashboard/api/employees/{eid}", json={"window_start": "20:00", "window_end": "09:00"})
    assert "error" in r.json()
    assert employee(client, eid)["window_start"] == "09:00"


def test_saved_voice_settings_reach_the_call(client):
    login(client)
    eid = client.post("/dashboard/api/employees", json={
        "name": "Priya", "voice_gender": "male", "speaking_style": "professional",
        "temperature": 0.4, "speech_rate": 1.2, "greeting": "నమస్కారం",
    }).json()["id"]
    with SessionLocal() as db:
        from app.models import AIEmployee
        emp = db.get(AIEmployee, eid)
        call = Call(tenant_id=emp.tenant_id, employee_id=eid, status=CallStatus.CONNECTED, started_at=datetime.utcnow())
        db.add(call)
        db.commit()
        call_id = call.id

    ctx = _load_context(call_id)
    assert ctx.extras["voice_gender"] == "male"
    assert ctx.extras["speech_rate"] == 1.2
    assert ctx.greeting == "నమస్కారం"
    assert _persona_settings(ctx) == {"name": "Priya", "speaking_style": "professional", "creativity": 0.4, "language_style": "natural_mix"}


def test_settings_shape_the_realtime_session():
    session = RealtimeSession(
        "key", "Sell plots.", "te", voice_gender="male", model="gpt-realtime-2.1",
        name="Priya", speaking_style="professional", creativity=0.4, speech_rate=1.2,
    )
    config = session._session_update()["session"]
    assert config["audio"]["output"] == {"format": {"type": "audio/pcmu"}, "voice": "echo", "speed": 1.2}
    assert config["reasoning"] == {"effort": "low"}
    assert "Your name is Priya" in config["instructions"]
    assert "senior relationship manager" in config["instructions"]
    assert "Stay close to the facts" in config["instructions"]

    default = RealtimeSession("key", "Sell plots.", "te", model=MODEL)._session_update()["session"]
    assert "reasoning" not in default
    assert persona() == [persona()[0]]  # balanced creativity adds no extra rule


def test_fallback_voice_accepts_the_numeric_speech_rate():
    class Agent:
        language = "te"
        voice_tier = "standard"

    tts = factory._build_tts(Agent(), openai_key=None, extras={"speech_rate": 1.05, "voice_gender": "male"})
    assert tts._rate == "+5%" and tts._gender == "male"


def test_lead_name_and_notes_reach_the_call(client):
    login(client)
    eid = client.post("/dashboard/api/employees", json={"name": "Suma"}).json()["id"]
    with SessionLocal() as db:
        from app.models import AIEmployee, Lead

        emp = db.get(AIEmployee, eid)
        lead = Lead(tenant_id=emp.tenant_id, employee_id=eid, phone_e164="+919999900001", name="Ravi Kumar",
                    metadata_json={"notes": "Prefers evening calls"})
        db.add(lead)
        db.flush()
        call = Call(tenant_id=emp.tenant_id, employee_id=eid, lead_id=lead.id, status=CallStatus.CONNECTED,
                    started_at=datetime.utcnow())
        db.add(call)
        db.commit()
        call_id = call.id

    ctx = _load_context(call_id)
    assert ctx.caller_name == "Ravi Kumar"
    assert ctx.caller_notes == "Prefers evening calls"
    assert ctx.timezone == "Asia/Kolkata"


def test_language_style_saved_and_reaches_call(client):
    login(client)
    eid = client.post("/dashboard/api/employees", json={
        "name": "Suma", "language_style": "english_heavy",
    }).json()["id"]
    got = employee(client, eid)
    assert got["language_style"] == "english_heavy"

    client.patch(f"/dashboard/api/employees/{eid}", json={"language_style": "pure"})
    got = employee(client, eid)
    assert got["language_style"] == "pure"

    with SessionLocal() as db:
        from app.models import AIEmployee
        emp = db.get(AIEmployee, eid)
        call = Call(tenant_id=emp.tenant_id, employee_id=eid, status=CallStatus.CONNECTED, started_at=datetime.utcnow())
        db.add(call)
        db.commit()
        call_id = call.id

    ctx = _load_context(call_id)
    assert _persona_settings(ctx)["language_style"] == "pure"


def _instructions(style: str, model: str = "gpt-realtime-2.1") -> str:
    return RealtimeSession("key", "Sell plots.", "te", model=model, speaking_style=style)._instructions


def test_professional_style_is_not_overruled_by_the_relaxed_base_prompt():
    text = _instructions("professional")
    assert "senior relationship manager" in text
    assert "composed, confident and courteous" in text
    assert "steady voice for the whole call" in text
    assert "apologise at most once" in text
    assert "no slang and no exclamations" in text
    assert "never make excuses or blame the line" in text
    for relaxed in ("never formal", "BRISK", "quick and lively", "smile when you greet", "relaxed, direct and warm"):
        assert relaxed not in text, relaxed


def test_professional_style_reaches_the_short_mini_brief_too():
    text = _instructions("professional", model=MODEL if "mini" in MODEL else "gpt-realtime-mini")
    assert "composed, courteous and confident" in text
    assert "relaxed, warm and direct" not in text


def test_humanized_style_keeps_its_relaxed_warm_voice():
    text = _instructions("humanized")
    assert "relaxed, direct and warm" in text
    assert "never formal" in text
    assert "BRISK, NATURAL PHONE PACE" in text

