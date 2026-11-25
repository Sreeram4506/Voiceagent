"""WebSocket endpoint the telephony provider connects to for each live call."""

import asyncio
import base64
import functools
import hashlib
import hmac
import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.db import SessionLocal
from app.models import AIEmployee, Call, CallStatus, CallTurn, Lead
from app.services.call_engine import CallResult, finalize_call
from app.streaming import factory
from app.streaming.audio import FRAME_MS
from app.streaming.gemini_live_session import GEMINI_MODEL, GeminiLiveSession
from app.streaming.memory import load_memory, memory_to_prompt, update_memory
from app.streaming.protocol import parse_event
from app.streaming.realtime_session import MODEL, MODEL_BY_TIER, RealtimeSession, call_details, persona
from app.streaming.recorded import RecordedAudioSession, _to_pcm_8k
from app.streaming.session import StreamSession, pop_sentences
from datetime import datetime

router = APIRouter()

DEFAULT_TIMEZONE = "Asia/Kolkata"

_call_modes: dict[int, str] = {}
_call_lead_ids: dict[int, int | None] = {}
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
    caller_name: str = ""
    caller_notes: str = ""
    timezone: str = DEFAULT_TIMEZONE
    agent_mode: str = "ai_agent"
    lead_id: int | None = None
    memory_context: str = ""


def _load_context(call_id: int) -> CallContext | None:
    from app.models import AppSetting

    with SessionLocal() as db:
        call = db.get(Call, call_id)
        if call is None:
            return None
        employee = db.get(AIEmployee, call.employee_id)
        row = db.get(AppSetting, f"t{employee.tenant_id}:employee_extras")
        all_extras = row.value if row else {}
        extras = all_extras.get(str(employee.id), {}) if isinstance(all_extras, dict) else {}
        fields = call.captured_fields or {}
        lead = db.get(Lead, call.lead_id) if call.lead_id else None
        org = db.get(AppSetting, f"t{employee.tenant_id}:org")
        org_settings = org.value if org and isinstance(org.value, dict) else {}
        lead_id = lead.id if lead else None
        mem = load_memory(lead_id)
        return CallContext(
            employee=employee,
            extras=extras,
            mode=_call_modes.get(call_id, fields.get("mode", "ai_agent")),
            greeting=fields.get("greeting", "") or extras.get("greeting", ""),
            caller_name=str((lead.name if lead else "") or fields.get("name") or ""),
            caller_notes=str(((lead.metadata_json or {}).get("notes") if lead else "") or fields.get("notes") or ""),
            timezone=str(org_settings.get("timezone") or DEFAULT_TIMEZONE),
            agent_mode=str(extras.get("agent_mode") or "ai_agent"),
            lead_id=lead_id,
            memory_context=memory_to_prompt(mem),
        )


def _caller_line(caller_name: str) -> list[str]:
    name = " ".join(caller_name.split())
    return [f"You are speaking with {name}. Use their name respectfully now and then."] if name else []


def _local_time(timezone: str) -> datetime:
    """The time where the caller is, so the agent knows the day and whether it is morning or evening."""
    try:
        return datetime.now(ZoneInfo(timezone))
    except Exception:
        return datetime.now(ZoneInfo(DEFAULT_TIMEZONE))


def _number(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _persona_settings(ctx: CallContext) -> dict:
    return {
        "name": ctx.employee.name or "",
        "speaking_style": str(ctx.extras.get("speaking_style") or "humanized"),
        "creativity": _number(ctx.extras.get("temperature"), 0.85),
        "language_style": str(ctx.extras.get("language_style") or "natural_mix"),
    }


def _start_realtime(ctx: CallContext | None) -> RealtimeSession | None:
    if ctx is None or not factory.use_realtime(ctx.employee.voice_tier.value, ctx.mode):
        return None
    call = {
        "script": ctx.employee.script,
        "language": ctx.employee.language,
        "greeting": ctx.greeting,
        "voice_gender": ctx.extras.get("voice_gender", "female"),
        "speech_rate": _number(ctx.extras.get("speech_rate"), 1.0),
        "caller_name": ctx.caller_name,
        "caller_notes": ctx.caller_notes,
        "local_time": _local_time(ctx.timezone),
        "memory_context": ctx.memory_context,
        **_persona_settings(ctx),
    }
    if factory.voice_engine() == "gemini":
        session = GeminiLiveSession(api_key=os.environ["GEMINI_API_KEY"], model=GEMINI_MODEL, **call)
    else:
        model = MODEL_BY_TIER.get(ctx.employee.voice_tier.value, MODEL)
        session = RealtimeSession(api_key=os.environ["OPENAI_API_KEY"], model=model, **call)
    session.connect()
    return session


def _classic_script(ctx: CallContext) -> str:
    persona_settings = {k: v for k, v in _persona_settings(ctx).items() if k != "language_style"}
    return "\n\n".join(part for part in (
        ctx.employee.script,
        call_details(ctx.caller_name, ctx.caller_notes, _local_time(ctx.timezone)),
        "\n".join(persona(**persona_settings) + _caller_line(ctx.caller_name)),
        ctx.memory_context,
    ) if part)


async def _prepare_classic(ctx: CallContext) -> dict:
    """Write and voice the opening line while the phone rings, so the agent speaks the moment it is answered."""
    stt, llm, tts = factory.build_providers(ctx.employee, extras=ctx.extras)
    text = ctx.greeting.strip()
    if not text:
        written = ""
        async for item in llm.stream_reply(_classic_script(ctx), [], ""):
            if isinstance(item, str):
                written += item
        sentences, rest = pop_sentences(written + " ")
        text = (sentences[0] if sentences else rest).strip()
    pcm = await tts.synthesize(text, ctx.employee.language, ctx.employee.voice_tier.value) if text else b""
    print(f"[WS] opening line ready before pickup: {text!r}")
    return {"type": "classic", "ctx": ctx, "providers": (stt, llm, tts), "opening": (text, pcm)}


def _stream_session(ctx: CallContext, stream_sid: str, send, end, providers=None, opening=None) -> StreamSession:
    stt, llm, tts = providers or factory.build_providers(ctx.employee, extras=ctx.extras)
    return StreamSession(
        stt=stt,
        llm=llm,
        tts=tts,
        script=_classic_script(ctx),
        language=ctx.employee.language,
        voice_tier=ctx.employee.voice_tier.value,
        stream_sid=stream_sid,
        send=send,
        on_end=end,
        mode=ctx.mode,
        greeting=ctx.greeting,
        opening=opening,
    )


def prepare_call(call_id: int) -> None:
    """The phone is ringing or was just answered: get the agent ready to speak before the audio stream connects."""
    if call_id in _prepared:
        return

    async def prepare() -> RealtimeSession | dict | None:
        ctx = await asyncio.to_thread(_load_context, call_id)
        if ctx is not None:
            _call_lead_ids[call_id] = ctx.lead_id
        if ctx is not None and ctx.agent_mode == "recorded_audio":
            audio = ctx.extras.get("audio_greeting")
            if audio and audio.get("data"):
                raw = base64.b64decode(audio["data"])
                pcm = await asyncio.to_thread(_to_pcm_8k, raw, audio.get("content_type", "audio/mpeg"))
                print(f"[WS] prepared recorded audio: {len(pcm)} bytes PCM")
                return {"type": "recorded", "pcm": pcm, "filename": audio.get("filename", "greeting.mp3")}
            return None
        realtime = _start_realtime(ctx)
        if realtime is not None or ctx is None or ctx.mode == "broadcast":
            return realtime
        try:
            return await _prepare_classic(ctx)
        except Exception as e:
            print(f"[WS] could not prepare the opening line ({type(e).__name__}: {e}); it will be made on pickup")
            return None

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
        result = await task
        if isinstance(result, RealtimeSession):
            await result.cancel()
            print(f"[Cost] call {call_id} not answered, greeting only: {result.usage_report()}")

    asyncio.create_task(close())


@router.websocket("/v1/stream/{call_id}")
async def media_stream(websocket: WebSocket, call_id: int, token: str = ""):
    if not hmac.compare_digest(stream_token(call_id), token):
        await websocket.close(code=4401)
        return
    await websocket.accept()

    session: StreamSession | RealtimeSession | RecordedAudioSession | None = None
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
        try:
            if session is not None:
                await session.cancel()
        finally:
            # Even if this handler is being cancelled, the call is still saved and billed.
            await _persist_once(call_id, session)
            _call_lead_ids.pop(call_id, None)


async def _open_session(call_id: int, stream_sid: str, websocket: WebSocket) -> StreamSession | RealtimeSession | RecordedAudioSession:
    holder: dict[str, StreamSession | RealtimeSession | RecordedAudioSession] = {}

    async def send(message: str) -> None:
        await websocket.send_text(message)

    async def end() -> None:
        try:
            await _persist_once(call_id, holder["session"])
        except Exception as e:
            print(f"[WS] persist error in end callback: {e}")
        await websocket.close()

    prepared = _prepared.pop(call_id, None)
    result = await prepared if prepared is not None else None

    if isinstance(result, dict) and result.get("type") == "recorded":
        print(f"[WS] recorded_audio mode (prepared) — playing {result['filename']}")
        session = RecordedAudioSession(
            pcm=result["pcm"],
            filename=result["filename"],
            stream_sid=stream_sid,
            send=send,
            on_end=end,
        )
        holder["session"] = session
        await session.start()
        return session

    if isinstance(result, dict) and result.get("type") == "classic":
        session = _stream_session(result["ctx"], stream_sid, send, end, result["providers"], result["opening"])
        holder["session"] = session
        await session.start()
        return session

    realtime = result if isinstance(result, RealtimeSession) else None

    ctx: CallContext | None = None
    if realtime is None:
        ctx = await asyncio.to_thread(_load_context, call_id)
        if ctx is not None:
            _call_lead_ids[call_id] = ctx.lead_id
        if ctx is not None and ctx.agent_mode == "recorded_audio":
            audio = ctx.extras.get("audio_greeting")
            if audio and audio.get("data"):
                print(f"[WS] recorded_audio mode (inline) — playing {audio.get('filename', 'audio')}")
                session = RecordedAudioSession(
                    audio_b64=audio["data"],
                    content_type=audio.get("content_type", "audio/mpeg"),
                    filename=audio.get("filename", "greeting.mp3"),
                    stream_sid=stream_sid,
                    send=send,
                    on_end=end,
                )
                holder["session"] = session
                await session.start()
                return session
            print("[WS] recorded_audio mode but no audio file — falling back to AI")
        realtime = _start_realtime(ctx)

    if realtime is not None:
        if await realtime.wait_ready(timeout=4.0):
            holder["session"] = realtime
            await realtime.attach(stream_sid, send, end)
            return realtime
        print("[WS] realtime voice unavailable, falling back to the STT/LLM/TTS pipeline")
        await realtime.cancel()

    ctx = ctx or await asyncio.to_thread(_load_context, call_id)
    if ctx is None:
        raise RuntimeError(f"unknown call {call_id}")
    session = _stream_session(ctx, stream_sid, send, end)
    holder["session"] = session
    await session.start()
    return session


LiveSession = StreamSession | RealtimeSession | RecordedAudioSession


@dataclass(frozen=True)
class CallRecord:
    """What a finished call leaves behind, copied off the live session so another thread can save it."""

    turns: tuple[tuple[str, str], ...]
    outcome: str
    actions: tuple[str, ...]
    connected_seconds: int
    cost_usd: float | None = None
    usage_report: str = ""
    provider: str = "OpenAI"  # who ran the call, named next to its cost in the summary


def _record_of(session: LiveSession) -> CallRecord:
    """Read the session on the event loop thread, where its live state is safe to read."""
    realtime = isinstance(session, RealtimeSession)
    return CallRecord(
        turns=tuple(session.turns),
        outcome=session.outcome(),
        actions=tuple(session.actions),
        connected_seconds=session.frames_received * FRAME_MS // 1000,
        cost_usd=session.cost_usd() if realtime else None,
        usage_report=session.usage_report() if realtime else "",
        provider="Gemini" if isinstance(session, GeminiLiveSession) else "OpenAI",
    )


def _in_thread(fn, *args) -> asyncio.Future:
    """Start fn in the default thread pool right away, so it runs even if the caller is then cancelled."""
    return asyncio.get_running_loop().run_in_executor(None, functools.partial(fn, *args))


async def _persist_once(call_id: int, session: LiveSession | None) -> None:
    """Save the call in a worker thread, off the event loop that carries every live call's audio.

    The end-of-call callback and the socket closing can both get here at once; they share one save, so a
    call is never written or billed twice. A save that failed is tried again by the next caller.
    """
    if session is None:
        await asyncio.shield(_in_thread(_persist, call_id, None))
        return
    if session.persisted:
        return
    task: asyncio.Future | None = getattr(session, "persist_task", None)
    if task is None or task.done():  # first save, or the previous one failed
        task = _in_thread(_save, call_id, session, _record_of(session))
        session.persist_task = task
    await asyncio.shield(task)


def _save(call_id: int, session: LiveSession, record: CallRecord) -> None:
    _persist(call_id, record)
    session.persisted = True


def _persist(call_id: int, record: CallRecord | None) -> None:
    """Write transcript, outcome, billing, and conversation memory once the stream closes. Blocking."""
    with SessionLocal() as db:
        call = db.get(Call, call_id)
        if call is None:
            return
        if record is None:
            call.status = CallStatus.FAILED
            db.commit()
            return

        for role, text in record.turns:
            db.add(CallTurn(call_id=call.id, role=role, text=text))

        lead = db.get(Lead, call.lead_id) if call.lead_id else None
        employee = db.get(AIEmployee, call.employee_id)
        summary = f"{len(record.turns)} turns; outcome={record.outcome}"
        if record.usage_report:
            print(f"[Cost] call {call_id} {record.usage_report}")
        if record.cost_usd is not None:
            summary += f"; {record.provider} ${record.cost_usd:.4f}"
        result = CallResult(outcome=record.outcome, summary=summary, actions=list(record.actions))
        finalize_call(db, call, employee, lead, result, connected_seconds=record.connected_seconds,
                      now=datetime.utcnow())
        db.commit()

    # Update per-lead conversation memory (outside the main transaction)
    lead_id = _call_lead_ids.pop(call_id, None)
    if lead_id and record.turns:
        try:
            update_memory(lead_id, list(record.turns), record.outcome)
        except Exception as e:
            print(f"[WS] memory update error: {e}")
