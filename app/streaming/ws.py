"""WebSocket endpoint the telephony provider connects to for each live call."""

import hashlib
import hmac

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, CallTurn, Lead
from app.services.call_engine import CallResult, finalize_call
from app.streaming import factory
from app.streaming.audio import FRAME_MS
from app.streaming.protocol import parse_event
from app.streaming.session import StreamSession
from datetime import datetime

router = APIRouter()

_call_modes: dict[int, str] = {}


def set_call_mode(call_id: int, mode: str) -> None:
    _call_modes[call_id] = mode


def stream_token(call_id: int) -> str:
    key = get_settings().stream_secret.encode()
    return hmac.new(key, str(call_id).encode(), hashlib.sha256).hexdigest()


@router.websocket("/v1/stream/{call_id}")
async def media_stream(websocket: WebSocket, call_id: int, token: str = ""):
    if not hmac.compare_digest(stream_token(call_id), token):
        await websocket.close(code=4401)
        return
    await websocket.accept()

    session: StreamSession | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            event = parse_event(raw)
            if session is None and event.kind != "start":
                print(f"[WS] pre-session event: {event.kind}")
            if event.kind == "start" and session is None:
                print(f"[WS] START event, stream_sid={event.stream_sid}, raw_start={event.raw.get('start', {})}")
                session = _open_session(call_id, event.stream_sid or "", websocket)
                await session.start()
            elif event.kind == "media" and session is not None and event.audio:
                await session.on_audio(event.audio)
            elif event.kind == "stop":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if session is not None:
            await session.cancel()
        _persist(call_id, session)


def _open_session(call_id: int, stream_sid: str, websocket: WebSocket) -> StreamSession:
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        employee = db.get(AIEmployee, call.employee_id)
        stt, llm, tts = factory.build_providers(employee)
        mode = _call_modes.pop(call_id, "campaign")

        holder: dict[str, StreamSession] = {}

        async def send(message: str) -> None:
            await websocket.send_text(message)

        async def end() -> None:
            _persist(call_id, holder["session"])
            await websocket.close()

        session = StreamSession(
            stt=stt,
            llm=llm,
            tts=tts,
            script=employee.script,
            language=employee.language,
            voice_tier=employee.voice_tier.value,
            stream_sid=stream_sid,
            send=send,
            on_end=end,
            mode=mode,
        )
        holder["session"] = session
        return session


def _persist(call_id: int, session: StreamSession | None) -> None:
    """Write transcript, outcome, and billing once the stream closes."""
    if session is not None and session.persisted:
        return
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        if call is None:
            return
        if session is None:
            call.status = CallStatus.FAILED
            db.commit()
            return

        for role, text in session.turns:
            db.add(CallTurn(call_id=call.id, role=role, text=text))

        lead = db.get(Lead, call.lead_id) if call.lead_id else None
        employee = db.get(AIEmployee, call.employee_id)
        result = CallResult(
            outcome=session.outcome(),
            summary=f"{len(session.turns)} turns; outcome={session.outcome()}",
            actions=session.actions,
        )
        connected_seconds = session.frames_received * FRAME_MS // 1000
        finalize_call(db, call, employee, lead, result, connected_seconds=connected_seconds, now=datetime.utcnow())
        db.commit()
        if session is not None:
            session.persisted = True
