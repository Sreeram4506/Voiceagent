"""Live streaming over the WebSocket, with mock speech providers."""

import base64
import json
import time as wallclock
from datetime import datetime, time

import pytest
from starlette.websockets import WebSocketDisconnect

from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, CallTurn, Lead, LeadStatus, Tenant, VoiceTier
from app.providers.base import ToolRequest
from app.streaming import factory
from app.streaming.audio import FRAME_BYTES
from app.streaming.mock_providers import MockAsyncLLM, MockAsyncSTT, MockAsyncTTS
from app.streaming.ws import stream_token

LOUD = (3000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
QUIET = bytes(FRAME_BYTES)


def media(frame: bytes) -> str:
    return json.dumps({"event": "media", "media": {"payload": base64.b64encode(frame).decode()}})


def speech(ms: int) -> list[str]:
    return [media(LOUD)] * (ms // 20)


def quiet(ms: int) -> list[str]:
    return [media(QUIET)] * (ms // 20)


def seed_call() -> int:
    now = datetime.utcnow()
    with SessionLocal() as db:
        tenant = Tenant(name="Acme", api_key_hash="h-test", webhook_secret="s")
        db.add(tenant)
        db.flush()
        employee = AIEmployee(
            tenant_id=tenant.id,
            name="Riya",
            language="hi",
            voice_tier=VoiceTier.STANDARD,
            script="Book a site visit.",
            window_start=time(0, 0),
            window_end=time(23, 59),
        )
        db.add(employee)
        db.flush()
        lead = Lead(tenant_id=tenant.id, employee_id=employee.id, phone_e164="+919999900001", status=LeadStatus.CALLING, next_attempt_at=now)
        db.add(lead)
        db.flush()
        call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=employee.id, status=CallStatus.CONNECTED, started_at=now)
        db.add(call)
        db.commit()
        return call.id


def use_providers(monkeypatch, stt, llm, tts=None):
    monkeypatch.setattr(factory, "build_providers", lambda employee, extras=None: (stt, llm, tts or MockAsyncTTS()))


def connect(client, call_id):
    return client.websocket_connect(f"/v1/stream/{call_id}?token={stream_token(call_id)}")


OPENING_SETTLE_SECONDS = 0.3


def start_and_wait_for_opening(ws, stream_sid: str) -> None:
    """Start the stream and give the opening line its turn, as on a real call where the caller
    speaks after answering. Audio that arrives during the opening is ignored by design."""
    ws.send_text(json.dumps({"event": "start", "stream_sid": stream_sid}))
    wallclock.sleep(OPENING_SETTLE_SECONDS)


def test_rejects_bad_token(client):
    call_id = seed_call()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/v1/stream/{call_id}?token=wrong") as ws:
            ws.receive_text()


def test_caller_utterance_gets_spoken_reply_and_is_persisted(client, monkeypatch):
    call_id = seed_call()
    use_providers(
        monkeypatch,
        MockAsyncSTT(["Haan, appointment book kar dijiye"]),
        MockAsyncLLM(
            [
                [],  # no opening line
                ["Theek hai, site visit book kar diya. ", ToolRequest("book_appointment", {}), ToolRequest("end_call")],
            ]
        ),
    )

    received = []
    with connect(client, call_id) as ws:
        start_and_wait_for_opening(ws, "S1")
        for msg in speech(400) + quiet(800):
            ws.send_text(msg)
        # Agent speaks, then ends the call, which closes the socket.
        try:
            while True:
                received.append(json.loads(ws.receive_text()))
        except WebSocketDisconnect:
            pass

    media_frames = [m for m in received if m["event"] == "playAudio"]
    assert len(media_frames) > 0

    with SessionLocal() as db:
        call = db.get(Call, call_id)
        assert call.status == CallStatus.COMPLETED
        assert call.outcome == "appointment_booked"
        assert call.billed_seconds > 0
        turns = db.query(CallTurn).filter_by(call_id=call_id).order_by(CallTurn.id).all()
        assert [t.role for t in turns] == ["caller", "agent"]
        assert turns[0].text == "Haan, appointment book kar dijiye"
        assert db.get(Lead, call.lead_id).status == LeadStatus.DONE


def call_turns(call_id: int) -> list[CallTurn]:
    with SessionLocal() as db:
        return db.query(CallTurn).filter_by(call_id=call_id).order_by(CallTurn.id).all()


def finished_call(call_id: int) -> Call | None:
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        return call if call.status == CallStatus.COMPLETED else None


def test_caller_barges_in_and_agent_is_cleared(client, monkeypatch, eventually):
    call_id = seed_call()
    long_reply = "Ji bilkul, main aapko poori jaankari deti hoon. Site visit ka time aap bata sakte hain. Aur kuch."
    use_providers(
        monkeypatch,
        MockAsyncSTT(["Pehli baat"]),
        MockAsyncLLM([[], [long_reply]]),
    )

    with connect(client, call_id) as ws:
        start_and_wait_for_opening(ws, "S2")
        for msg in speech(400) + quiet(800):
            ws.send_text(msg)

        # Wait until the agent is audibly speaking.
        first = json.loads(ws.receive_text())
        assert first["event"] == "playAudio"

        # Caller talks over the agent.
        for msg in speech(400):
            ws.send_text(msg)

        seen_clear = False
        for _ in range(500):
            msg = json.loads(ws.receive_text())
            if msg["event"] == "clearAudio":
                seen_clear = True
                break
        assert seen_clear, "agent playback should be cleared on barge-in"
        ws.send_text(json.dumps({"event": "stop"}))

    turns = eventually(lambda: call_turns(call_id))
    # The interrupted reply is not recorded as spoken; the caller's words are.
    assert [t.role for t in turns] == ["caller"]
    assert turns[0].text == "Pehli baat"


def test_stop_without_speech_still_finalizes(client, monkeypatch, eventually):
    call_id = seed_call()
    use_providers(monkeypatch, MockAsyncSTT([]), MockAsyncLLM([[]]))
    with connect(client, call_id) as ws:
        ws.send_text(json.dumps({"event": "start", "stream_sid": "S3"}))
        ws.send_text(json.dumps({"event": "stop"}))
        # Wait while the socket is open: leaving the block makes the test client cancel the handler.
        call = eventually(lambda: finished_call(call_id))

    assert call is not None, "the call should be saved as completed"
    assert call.outcome == "no_resolution"


def test_stream_session_predefined_greeting(client, monkeypatch, eventually):
    call_id = seed_call()
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        call.captured_fields = {"greeting": "Namaskaram, ela unnaru?", "mode": "ai_agent"}
        db.commit()

    use_providers(monkeypatch, MockAsyncSTT([]), MockAsyncLLM([[]]))
    with connect(client, call_id) as ws:
        ws.send_text(json.dumps({"event": "start", "stream_sid": "S4"}))
        ws.send_text(json.dumps({"event": "stop"}))
        turns = eventually(lambda: call_turns(call_id))

    assert len(turns) >= 1
    assert turns[0].role == "agent"
    assert turns[0].text == "Namaskaram, ela unnaru?"
