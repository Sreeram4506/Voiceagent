"""Every live call is recorded (caller and agent on one track) when VA_RECORDINGS_DIR is set."""

import asyncio
import io
import wave

import numpy as np

from app.streaming.call_recorder import RATE, CallRecorder, recorder_from_env

SECOND = RATE * 2  # bytes of 16-bit audio per second


def tone(seconds: float, level: int) -> bytes:
    return (np.full(int(RATE * seconds), level, dtype="<i2")).tobytes()


def samples(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def test_agent_audio_lands_at_the_moment_it_played_and_mixes_with_the_caller():
    rec = CallRecorder()
    rec.start(now=100.0)
    for _ in range(50):  # one second of the caller, 20 ms frames
        rec.caller(tone(0.02, 1000))
    rec.agent(tone(0.5, 2000), start=100.5)  # the agent answers half a second in
    out = samples(rec.wav())
    assert len(out) == RATE
    assert out[: RATE // 2].max() == 1000  # caller only
    assert out[RATE // 2 + 10] == 3000  # caller and agent together


def test_barge_in_drops_the_agent_audio_that_never_played():
    rec = CallRecorder()
    rec.start(now=0.0)
    rec.agent(tone(2.0, 2000), start=0.0)
    rec.cut_agent(at=0.5)  # the caller cut in after half a second
    out = samples(rec.wav())
    assert len(out) == RATE // 2


def test_back_to_back_agent_chunks_stay_contiguous():
    rec = CallRecorder()
    rec.start(now=0.0)
    rec.agent(tone(0.1, 500), start=0.0)
    rec.agent(tone(0.1, 500), start=0.1)
    assert len(samples(rec.wav())) == int(RATE * 0.2)


def test_nothing_is_recorded_before_the_phone_is_attached():
    rec = CallRecorder()
    rec.caller(tone(0.02, 1000))
    rec.agent(tone(0.1, 1000), start=0.0)
    assert not rec.has_audio()


def test_recording_is_off_unless_a_folder_is_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("VA_RECORDINGS_DIR", raising=False)
    assert recorder_from_env() is None
    monkeypatch.setenv("VA_RECORDINGS_DIR", str(tmp_path))
    assert isinstance(recorder_from_env(), CallRecorder)


def test_gemini_call_records_both_sides(monkeypatch, tmp_path):
    from test_gemini_live import make_session, speech

    monkeypatch.setenv("VA_RECORDINGS_DIR", str(tmp_path))

    async def scenario():
        session, _, _, send, end, _ = make_session()
        session._ws = object()
        await session.attach("sid", send, end)
        await session.on_audio(tone(0.02, 1200))
        await session._on_message(speech(0.2))
        return session

    session = asyncio.run(scenario())
    out = samples(session.recorder.wav())
    assert out.max() > 0 and len(out) >= int(RATE * 0.2)


def test_saved_recording_is_written_and_served_only_to_its_tenant(monkeypatch, tmp_path, client):
    from app.db import SessionLocal
    from datetime import time

    from app.models import AIEmployee, Call, CallStatus, TenantUser, VoiceTier
    from app.streaming.call_recorder import save_recording

    monkeypatch.setenv("VA_RECORDINGS_DIR", str(tmp_path))
    rec = CallRecorder()
    rec.start(now=0.0)
    rec.caller(tone(0.5, 800))

    assert client.post("/dashboard/api/auth/login", json={"email": "admin@suma.ai", "password": "suma2026"}).status_code == 200
    with SessionLocal() as db:
        tenant_id = db.query(TenantUser).filter(TenantUser.email == "admin@suma.ai").one().tenant_id
        employee = AIEmployee(tenant_id=tenant_id, name="Suma", language="te", voice_tier=VoiceTier.STANDARD,
                              script="Book a demo.", window_start=time(9, 0), window_end=time(20, 0))
        db.add(employee)
        db.flush()
        call = Call(tenant_id=tenant_id, employee_id=employee.id, status=CallStatus.COMPLETED)
        db.add(call)
        db.flush()
        call.recording_file = save_recording(call.id, rec)
        db.commit()
        call_id = call.id
    assert (tmp_path / f"call_{call_id}.wav").exists()

    r = client.get(f"/dashboard/api/calls/{call_id}/recording")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert len(samples(r.content)) == RATE // 2
    listed = next(c for c in client.get("/dashboard/api/calls").json() if c["id"] == call_id)
    assert listed["recording"] == f"/dashboard/api/calls/{call_id}/recording"

    assert client.get("/dashboard/api/calls/999999/recording").status_code == 404


def test_recording_endpoint_needs_a_signed_in_user(client):
    assert client.get("/dashboard/api/calls/1/recording").status_code in (401, 403)
