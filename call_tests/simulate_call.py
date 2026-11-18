"""Simulate a Plivo phone call against the local server and record how the agent sounds.

The caller is a Telugu Edge TTS voice streamed in real time over the same WebSocket Plivo
uses. Writes <label>.wav (both sides mixed, as heard on the phone) and <label>.json
(transcript, per-turn response gaps, filler-word counts, barge-ins).

Usage (local server must be running on :8000 with VA_DATABASE_URL=sqlite:///local_test.db):
    VA_DATABASE_URL=sqlite:///local_test.db python call_tests/simulate_call.py baseline
"""

import asyncio
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from datetime import time as dtime
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if not os.environ.get("VA_DATABASE_URL", "").startswith("sqlite"):
    sys.exit("Refusing to run: set VA_DATABASE_URL=sqlite:///local_test.db so nothing touches production.")

from app.db import SessionLocal  # noqa: E402
from app.models import AIEmployee, AppSetting, Call, CallStatus, CallTurn, Lead, LeadStatus, Tenant, VoiceTier  # noqa: E402
from app.streaming.edge_tts_provider import EdgeTTS  # noqa: E402
from app.streaming.ws import stream_token  # noqa: E402

BASE = "http://localhost:8000"
RATE = 8000
FRAME = 320  # 20 ms of 8 kHz 16-bit mono
OUT = Path(__file__).resolve().parent

SCRIPT = (
    "You are Suma from Green Valley Properties, Hyderabad. You are calling about DTCP-approved open plots "
    "at Shadnagar, 30 minutes from the airport. Plots start at 18 lakhs for 150 square yards, with bank loans "
    "available. Your goal is to book a site visit this weekend; free pickup is available from Shamshabad."
)
CALLER_LINES = [
    "హా చెప్పండి, ఎవరు మాట్లాడుతున్నారు?",
    "ఓకే, ఆ ప్లాట్ రేట్ ఎంత ఉంటుంది?",
    "కొంచెం ఎక్కువ అనిపిస్తుంది. లొకేషన్ సరిగ్గా ఎక్కడ?",
    "సరే, ఈ ఆదివారం ఉదయం వచ్చి చూస్తాను.",
    "ఓకే థాంక్యూ, బై.",
]
FILLERS = [
    r"\buh+\b", r"\bum+\b", r"\bhmm+\b", r"\bactually\b", r"\bi mean\b", r"\bso,", r"\banyway\b",
    "అంటే", "ఉమ్", "హ్మ్", "అవునా", "సరే సరే", "ఓహో", "అలాగా", "అయ్యో", "ఏంటంటే", "మరి ", r"ఆ\.\.\.",
    "అవును అవును", "హా హా",
]


def seed(tier: VoiceTier) -> int:
    now = datetime.utcnow()
    with SessionLocal() as db:
        tenant = db.query(Tenant).order_by(Tenant.id).first()
        employee = AIEmployee(
            tenant_id=tenant.id, name="Suma", language="te", voice_tier=tier, script=SCRIPT,
            window_start=dtime(0, 0), window_end=dtime(23, 59),
        )
        db.add(employee)
        db.flush()
        key = f"t{tenant.id}:employee_extras"
        row = db.get(AppSetting, key)
        extras = dict(row.value) if row else {}
        extras[str(employee.id)] = {"voice_gender": "female", "speaking_style": "humanized", "speech_rate": 1.0}
        if row:
            row.value = extras
        else:
            db.add(AppSetting(key=key, value=extras))
        lead = Lead(tenant_id=tenant.id, employee_id=employee.id, phone_e164="+910000000000", name="Ravi",
                    status=LeadStatus.CALLING, next_attempt_at=now)
        db.add(lead)
        db.flush()
        call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=employee.id,
                    status=CallStatus.CONNECTED, started_at=now)
        db.add(call)
        db.commit()
        return call.id


class Line:
    """Tracks what the phone would be playing, so the simulated caller waits its turn like a person."""

    def __init__(self, t0: float):
        self.t0 = t0
        self.play_until = t0
        self.last_chunk = 0.0
        self.agent: list[list] = []  # [start_s, pcm]
        self.caller: list[list] = []
        self.clears: list[float] = []
        self.waiting_since: float | None = None
        self.gaps: list[float] = []
        self.chunks = 0
        self.closed = False

    def on_agent_audio(self, pcm: bytes) -> None:
        now = time.monotonic()
        start = max(now, self.play_until)
        self.play_until = start + len(pcm) / (RATE * 2)
        self.last_chunk = now
        self.chunks += 1
        self.agent.append([start - self.t0, pcm])
        if self.waiting_since is not None:
            self.gaps.append(round(start - self.waiting_since, 2))
            self.waiting_since = None

    def on_clear(self) -> None:
        now = time.monotonic()
        cut = now - self.t0
        self.clears.append(round(cut, 2))
        kept = []
        for start, pcm in self.agent:
            if start >= cut:
                continue
            keep = int((cut - start) * RATE) * 2
            kept.append([start, pcm[:keep]])
        self.agent = kept
        self.play_until = now

    def agent_done(self) -> bool:
        now = time.monotonic()
        return self.chunks > 0 and now > self.play_until + 0.4 and now - self.last_chunk > 0.6


async def caller_audio() -> list[bytes]:
    tts = EdgeTTS(gender="male")
    return [await tts.synthesize(line, "te", "standard") for line in CALLER_LINES]


async def run(label: str, tier: VoiceTier) -> dict:
    lines_pcm = await caller_audio()
    call_id = seed(tier)
    async with httpx.AsyncClient(timeout=10) as http:
        await http.post(f"{BASE}/v1/plivo/ring/{call_id}")
        await asyncio.sleep(2.0)  # the phone rings for a couple of seconds
        await http.post(f"{BASE}/v1/plivo/answer/{call_id}")

    url = f"ws://localhost:8000/v1/stream/{call_id}?token={stream_token(call_id)}"
    outbound: asyncio.Queue[bytes] = asyncio.Queue()

    async with websockets.connect(url, max_size=None) as ws:
        line = Line(time.monotonic())
        line.waiting_since = line.t0  # time to first word after pickup
        await ws.send(json.dumps({"event": "start", "start": {
            "streamId": f"sim-{call_id}", "callId": "sim", "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 8000}}}))

        async def receive() -> None:
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("event") == "playAudio":
                        line.on_agent_audio(base64.b64decode(msg["media"]["payload"]))
                    elif msg.get("event") == "clearAudio":
                        line.on_clear()
            except websockets.ConnectionClosed:
                pass
            line.closed = True

        async def pump() -> None:
            """Send one 20 ms frame every 20 ms, like a phone line: caller speech when queued, else silence."""
            nxt = time.monotonic()
            silence = bytes(FRAME)
            while not line.closed:
                frame = silence if outbound.empty() else outbound.get_nowait()
                try:
                    await ws.send(json.dumps({"event": "media", "media": {"payload": base64.b64encode(frame).decode()}}))
                except websockets.ConnectionClosed:
                    return
                nxt += 0.02
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))

        async def wait_for(cond, timeout: float) -> bool:
            end = time.monotonic() + timeout
            while time.monotonic() < end and not line.closed:
                if cond():
                    return True
                await asyncio.sleep(0.05)
            return cond()

        recv_task = asyncio.create_task(receive())
        pump_task = asyncio.create_task(pump())

        await wait_for(line.agent_done, 20)
        for pcm in lines_pcm:
            if line.closed:
                break
            await asyncio.sleep(0.5)  # a person takes a beat before answering
            line.caller.append([time.monotonic() - line.t0, pcm])
            for i in range(0, len(pcm), FRAME):
                outbound.put_nowait(pcm[i:i + FRAME].ljust(FRAME, b"\0"))
            await wait_for(outbound.empty, 30)
            line.waiting_since = time.monotonic()
            chunks_before = line.chunks
            await wait_for(lambda: line.chunks > chunks_before, 12)
            await wait_for(line.agent_done, 30)

        await asyncio.sleep(2.5)
        if not line.closed:
            await ws.send(json.dumps({"event": "stop"}))
        line.closed = True
        pump_task.cancel()
        await asyncio.gather(recv_task, return_exceptions=True)

    await asyncio.sleep(1.0)  # let the server persist the transcript
    with SessionLocal() as db:
        turns = [(t.role, t.text) for t in db.query(CallTurn).filter_by(call_id=call_id).order_by(CallTurn.id)]
        outcome = db.get(Call, call_id).outcome

    write_mix(OUT / f"{label}.wav", line)
    agent_lines = [text for role, text in turns if role == "agent"]
    fillers = {text: sum(len(re.findall(p, text.lower())) for p in FILLERS) for text in agent_lines}
    report = {
        "label": label,
        "model_tier": tier.value,
        "call_id": call_id,
        "outcome": outcome,
        "first_word_after_pickup_s": line.gaps[0] if line.gaps else None,
        "reply_gaps_s": line.gaps[1:],
        "avg_reply_gap_s": round(sum(line.gaps[1:]) / len(line.gaps[1:]), 2) if line.gaps[1:] else None,
        "barge_ins": line.clears,
        "filler_words_total": sum(fillers.values()),
        "filler_words_per_reply": fillers,
        "transcript": turns,
    }
    (OUT / f"{label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def write_mix(path: Path, line: Line) -> None:
    segments = [(s, p) for s, p in line.agent] + [(s, p) for s, p in line.caller]
    end = max((s + len(p) / (RATE * 2) for s, p in segments), default=0) + 0.5
    mix = np.zeros(int(end * RATE) + 1, dtype=np.int32)
    for start, pcm in segments:
        samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.int32)
        i = int(start * RATE)
        mix[i:i + len(samples)] += samples[: len(mix) - i]
    sf.write(str(path), np.clip(mix, -32768, 32767).astype(np.int16), RATE, subtype="PCM_16")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "call"
    tier = VoiceTier(sys.argv[2]) if len(sys.argv) > 2 else VoiceTier.PREMIUM
    result = asyncio.run(run(label, tier))
    print(json.dumps({k: v for k, v in result.items() if k != "filler_words_per_reply"}, ensure_ascii=False, indent=2))
