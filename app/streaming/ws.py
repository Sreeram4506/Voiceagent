"""WebSocket endpoint the telephony provider connects to for each live call."""

import asyncio
import hashlib
import hmac
import os
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, CallTurn, Lead
from app.services.call_engine import CallResult, finalize_call
from app.streaming import factory
from app.streaming.audio import FRAME_MS
from app.streaming.protocol import parse_event
from app.streaming.realtime_session import RealtimeSession
from app.streaming.session import StreamSession
from datetime import datetime

router = APIRouter()

_call_modes: dict[int, str] = {}
# Realtime sessions started while the phone rings, waiting for the audio stream to connect.
_prepared: dict[int, asyncio.Task] = {}
PREPARED_TTL_SECONDS = 45  # ring timeout is 30s


def set_call_mode(call_id: int, mode: str) -> None:
    _call_modes[call_id] = mode


def stream_token(call_id: int) -> str:
    key = get_settings().stream_secret.encode()
    return hmac.new(key, str(call_id).encode(), hashlib.sha256).hexdigest()


@dataclass
class CallContext:
    employee: AIEmployee
    extras: dict
    mode: str
    greeting: str


def _load_context(call_id: int) -> CallContext | None:
    from app.models import AppSetting

    with SessionLocal() as db:
        call = db.get(Call, call_id)
        if call is None:
            return None
        employee = db.get(AIEmployee, call.employee_id)
        row = db.get(AppSetting, f"t{employee.tenant_id}_employee_extras")
        all_extras = row.value if row else {}
        extras = all_extras.get(str(employee.id), {}) if isinstance(all_extras, dict) else {}
        fields = call.captured_fields or {}
        return CallContext(
            employee=employee,
            extras=extras,
            mode=_call_modes.get(call_id, fields.get("mode", "ai_agent")),
            greeting=fields.get("greeting", "") or extras.get("greeting", ""),
        )


def _start_realtime(ctx: CallContext | None) -> RealtimeSession | None:
    if ctx is None or not factory.use_realtime(ctx.employee.voice_tier.value, ctx.mode):
        return None
    session = RealtimeSession(
        api_key=os.environ["OPENAI_API_KEY"],
        script=ctx.employee.script,
        language=ctx.employee.language,
        greeting=ctx.greeting,
        voice_gender=ctx.extras.get("voice_gender", "female"),
    )
    session.connect()
    return session


def prepare_call(call_id: int) -> None:
    """The phone is ringing or was just answered: get the agent ready to speak before the audio stream connects."""
    if call_id in _prepared:
        return

    async def prepare() -> RealtimeSession | None:
        return _start_realtime(await asyncio.to_thread(_load_context, call_id))

    task = asyncio.create_task(prepare())
    _prepared[call_id] = task
    asyncio.get_running_loop().call_later(PREPARED_TTL_SECONDS, _discard_unclaimed, call_id, task)


def discard_prepared(call_id: int) -> None:
    """The call ended without an audio stream (no answer, busy): drop the waiting session."""
    task = _prepared.get(call_id)
    if task is not None:
        _discard_unclaimed(call_id, task)


def _discard_unclaimed(call_id: int, task: asyncio.Task) -> None:
    if _prepared.get(call_id) is not task:
        return
    del _prepared[call_id]

    async def close() -> None:
        session = await task
        if session is not None:
            await session.cancel()

    asyncio.create_task(close())


@router.websocket("/v1/stream/{call_id}")
async def media_stream(websocket: WebSocket, call_id: int, token: str = ""):
    if not hmac.compare_digest(stream_token(call_id), token):
        await websocket.close(code=4401)
        return
    await websocket.accept()

    session: StreamSession | RealtimeSession | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            event = parse_event(raw)
            if session is None and event.kind != "start":
                print(f"[WS] pre-session event: {event.kind}")
            if event.kind == "start" and session is None:
                print(f"[WS] START event, stream_sid={event.stream_sid}, raw_start={event.raw.get('start', {})}")
                session = await _open_session(call_id, event.stream_sid or "", websocket)
            elif event.kind == "media" and session is not None and event.audio:
                await session.on_audio(event.audio)
            elif event.kind == "stop":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _call_modes.pop(call_id, None)
        if session is not None:
            await session.cancel()
        _persist(call_id, session)


async def _open_session(call_id: int, stream_sid: str, websocket: WebSocket) -> StreamSession | RealtimeSession:
    holder: dict[str, StreamSession | RealtimeSession] = {}

    async def send(message: str) -> None:
        await websocket.send_text(message)

    async def end() -> None:
        _persist(call_id, holder["session"])
        await websocket.close()

    ctx: CallContext | None = None
    prepared = _prepared.pop(call_id, None)
    realtime = await prepared if prepared is not None else None
    if realtime is None:
        ctx = _load_context(call_id)
        realtime = _start_realtime(ctx)

    if realtime is not None:
        if await realtime.wait_ready(timeout=4.0):
            holder["session"] = realtime
            await realtime.attach(stream_sid, send, end)
            return realtime
        print("[WS] realtime voice unavailable, falling back to the STT/LLM/TTS pipeline")
        await realtime.cancel()

    ctx = ctx or _load_context(call_id)
    if ctx is None:
        raise RuntimeError(f"unknown call {call_id}")
    stt, llm, tts = factory.build_providers(ctx.employee, extras=ctx.extras)
    session = StreamSession(
        stt=stt,
        llm=llm,
        tts=tts,
        script=ctx.employee.script,
        language=ctx.employee.language,
        voice_tier=ctx.employee.voice_tier.value,
        stream_sid=stream_sid,
        send=send,
        on_end=end,
        mode=ctx.mode,
        greeting=ctx.greeting,
    )
    holder["session"] = session
    await session.start()
    return session


def _persist(call_id: int, session: StreamSession | RealtimeSession | None) -> None:
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
