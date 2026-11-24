"""Phone-call scenarios against the local server, with a caller who behaves like a real person.

The caller is an LLM persona that listens to what the agent said (transcribed), decides what to
say, and speaks it with an expressive OpenAI voice over the same WebSocket Plivo uses. Scenarios
cover the awkward moments: the caller cuts the agent off, hangs up mid-sentence, is busy, says
it's a wrong number, or says nothing at all.

Writes call_tests/scenarios/<run>/<scenario>.wav (both sides, as heard on the phone) and .json
(transcript, timings, who hung up, outcome, reply lengths).

Usage (local server on :8000 with VA_DATABASE_URL=sqlite:///local_test.db, PLIVO_AUTH_ID empty):
    VA_DATABASE_URL=sqlite:///local_test.db python call_tests/scenario_call.py run1 [scenario ...]
SIM_VOICE_ENGINE=sarvam runs the agent on Sarvam's voice (the server needs SARVAM_API_KEY).
"""

import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from app.streaming.ws import stream_token  # noqa: E402

BASE = "http://localhost:8000"
RATE = 8000
FRAME = 320  # 20 ms of 8 kHz 16-bit mono
OPENAI = "https://api.openai.com/v1"
CALLER_MODEL = "gpt-4.1-mini"
CALLER_VOICE_MODEL = "gpt-4o-mini-tts"
TRANSCRIBE_MODEL = "gpt-4o-transcribe"
REPLY_RECEIVED_S = 0.6  # no new agent audio for this long: the reply has fully arrived (it plays slower)
AGENT_DONE_S = 1.2  # silence after the agent's audio finishes before a person decides it's their turn
MAX_TURNS = 9

AGENT_NAME = "Suma"
AGENT_BRIEF = (  # the script production agent #1 should carry: the voice-agent pitch, with how to say the name
    "You are Suma from Indraam (say it ఇంద్రామ్), a tech company with offices in the US, Hyderabad and Guntur. "
    "You are calling business owners about Indraam's AI voice agent: it phones their customers for them in Telugu, "
    "Hindi or English and talks like a person. It answers customers' questions, books appointments and visits, calls "
    "back people who didn't pick up, and can send a recorded voice message to a whole list. Every call's transcript and "
    "result shows in a dashboard. It suits hospitals and clinics, real estate, car dealers, coaching centres and shops: "
    "for a clinic it books patient appointments and reminds them; for real estate it calls leads and books site visits. "
    "It calls hundreds of people a day without extra staff. Price depends on how many calls they need; the team gives "
    "the exact quote in the free demo, so never quote a number. Your goal is to book a free 15-minute demo where our "
    "team shows the agent calling for their business."
)


@dataclass(frozen=True)
class Scenario:
    name: str
    persona: str  # who the caller is and how they behave, for the caller LLM
    caller_name: str = "Ravi"
    # turn number (1 = the caller's first reply) -> what the caller does on that turn
    moves: dict = field(default_factory=dict)
    silent: bool = False
    expect: str = ""


SCENARIOS = {s.name: s for s in [
    Scenario(
        "interested_books",
        "You run a small dental clinic in Guntur. You are curious but careful with money. Ask what they do, ask the "
        "price, say it feels a bit costly, then agree to a meeting on Saturday afternoon if the answers are good. "
        "Say bye when it's settled.",
        expect="answers each question briefly, handles the price concern, books Saturday, says bye and hangs up",
    ),
    Scenario(
        "busy_call_later",
        "You are driving and in a hurry. On your first reply say you are driving and ask them to call in the evening. "
        "If they keep talking, say 'OK OK, evening call cheyyandi' and say bye.",
        expect="offers a call back in one line, says bye, hangs up itself",
    ),
    Scenario(
        "hangs_up_midway",
        "You own a clothing shop. Ask what this is about, then ask whether it can really speak Telugu like a person.",
        moves={2: "hangup_mid"},
        expect="caller hangs up while the agent is talking; call ends cleanly and the transcript is saved",
    ),
    Scenario(
        "cuts_in",
        "You run a restaurant. You are impatient. Ask what this is about. While they answer your second question you "
        "cut them off and ask the price straight away. Then say you'll think about it and say bye.",
        moves={2: "interrupt"},
        expect="stops talking the moment the caller cuts in and answers the new question",
    ),
    Scenario(
        "not_interested",
        "You are a hospital admin. You already have a calling team at the front desk. Say you're not interested. If they "
        "try again, firmly say 'vaddu andi, cut cheyyandi'.",
        expect="one gentle try at most, then a short goodbye and hangs up",
    ),
    Scenario(
        "wrong_number",
        "Your name is Suresh, not Ravi. Say they have the wrong number. If they ask, say you don't own a business.",
        caller_name="Ravi",
        expect="apologises briefly and hangs up",
    ),
    Scenario(
        "silent_line",
        "",
        silent=True,
        expect="checks 'are you there' twice, then says goodbye and hangs up",
    ),
]}

CALLER_SYSTEM = """You are role-playing a real person in Andhra Pradesh who has just picked up a phone call from an unknown number.
{persona}

How you talk: everyday spoken Telugu the way people in Guntur and Hyderabad really talk on the phone, mixing in English
words freely (OK, rate, meeting, Instagram, busy, cut). Short, natural lines, 3-15 words, like a real person: casual,
sometimes a little impatient, using small words like "హా", "ఆ", "ఏంటి", "అవునా", "సరే". Write Telugu in Telugu script
and English words in English letters. Never sound like a customer-service script.

Reply with JSON only: {{"say": "<what you say next>", "done": <true once you have said bye or want to end the call>}}"""

# Plain "casual" was transcribed best (5% character error); "fast, distracted" styles garbled the Telugu.
CALLER_VOICE_STYLE = "A real person answering the phone, casual and natural, not a presenter."


def char_error(said: str, heard: str) -> float:
    """How far what a listener heard is from what the agent meant to say (0 = perfectly clear)."""
    a, b = re.sub(r"[\s.,!?\-–]", "", said.lower()), re.sub(r"[\s.,!?\-–]", "", heard.lower())
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return round(row[-1] / max(1, len(a)), 3)


def openai_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ["OPENAI_API_KEY"]


def seed(scenario: Scenario, voice_gender: str, language_style: str) -> int:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with SessionLocal() as db:
        tenant = db.query(Tenant).order_by(Tenant.id).first()
        employee = AIEmployee(
            tenant_id=tenant.id, name=AGENT_NAME, language="te", voice_tier=VoiceTier.PREMIUM, script=AGENT_BRIEF,
            window_start=dtime(0, 0), window_end=dtime(23, 59),
        )
        db.add(employee)
        db.flush()
        key = f"t{tenant.id}:employee_extras"
        row = db.get(AppSetting, key)
        extras = dict(row.value) if row else {}
        extras[str(employee.id)] = {"voice_gender": voice_gender, "speaking_style": "humanized",
                                    "speech_rate": float(os.environ.get("SIM_SPEECH_RATE", "1.0")),
                                    "language_style": language_style,
                                    "voice_engine": os.environ.get("SIM_VOICE_ENGINE", "openai")}
        if row:
            row.value = extras
        else:
            db.add(AppSetting(key=key, value=extras))
        lead = Lead(tenant_id=tenant.id, employee_id=employee.id, phone_e164="+910000000000",
                    name=scenario.caller_name, status=LeadStatus.CALLING, next_attempt_at=now)
        db.add(lead)
        db.flush()
        call = Call(tenant_id=tenant.id, lead_id=lead.id, employee_id=employee.id,
                    status=CallStatus.CONNECTED, started_at=now)
        db.add(call)
        db.commit()
        return call.id


class Caller:
    """The person on the other end: hears the agent (via transcription), thinks, and speaks."""

    def __init__(self, scenario: Scenario, http: httpx.AsyncClient):
        self.scenario = scenario
        self.http = http
        self.history = [{"role": "system", "content": CALLER_SYSTEM.format(persona=scenario.persona)}]

    async def hear(self, pcm: bytes) -> str:
        if len(pcm) < RATE * 2 // 5:  # under 0.2 s: nothing worth transcribing
            return ""
        buf = io.BytesIO()
        sf.write(buf, np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2"), RATE, format="WAV", subtype="PCM_16")
        r = await self.http.post(f"{OPENAI}/audio/transcriptions", files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                                 data={"model": TRANSCRIBE_MODEL, "language": "te"})
        r.raise_for_status()
        return r.json().get("text", "").strip()

    async def reply(self, agent_said: str, note: str = "") -> tuple[str, bool]:
        heard = agent_said or "(you couldn't make out what they said)"
        self.history.append({"role": "user", "content": f"They said: {heard}{note}"})
        r = await self.http.post(f"{OPENAI}/chat/completions", json={
            "model": CALLER_MODEL, "messages": self.history, "temperature": 0.9,
            "response_format": {"type": "json_object"}})
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        self.history.append({"role": "assistant", "content": content})
        data = json.loads(content)
        return str(data.get("say", "")).strip(), bool(data.get("done"))

    async def speak(self, text: str) -> bytes:
        r = await self.http.post(f"{OPENAI}/audio/speech", json={
            "model": CALLER_VOICE_MODEL, "voice": "coral", "input": text,
            "instructions": CALLER_VOICE_STYLE, "response_format": "pcm"})
        r.raise_for_status()
        pcm24 = np.frombuffer(r.content, dtype="<i2").astype(np.float32)
        idx = np.arange(0, len(pcm24), 3)  # 24 kHz -> 8 kHz
        return pcm24[idx].astype("<i2").tobytes()


class Line:
    """What the phone is playing, so the caller waits its turn (or deliberately doesn't)."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.play_until = self.t0
        self.last_chunk = 0.0
        self.agent: list[list] = []  # [start_s, pcm]
        self.caller: list[list] = []
        self.clears: list[float] = []
        self.chunks = 0
        self.closed = False
        self.turn_audio = bytearray()
        self.reply_started_at: float | None = None

    def now(self) -> float:
        return time.monotonic() - self.t0

    def on_agent_audio(self, pcm: bytes) -> None:
        now = time.monotonic()
        start = max(now, self.play_until)
        if self.reply_started_at is None:
            self.reply_started_at = start
        self.play_until = start + len(pcm) / (RATE * 2)
        self.last_chunk = now
        self.chunks += 1
        self.turn_audio += pcm
        self.agent.append([start - self.t0, pcm])

    def on_clear(self) -> None:
        now = time.monotonic()
        cut = now - self.t0
        self.clears.append(round(cut, 2))
        self.agent = [[s, p[: int((cut - s) * RATE) * 2]] for s, p in self.agent if s < cut]
        self.play_until = now

    def reply_received(self) -> bool:
        return self.chunks > 0 and time.monotonic() - self.last_chunk > REPLY_RECEIVED_S

    def agent_done(self) -> bool:
        now = time.monotonic()
        return self.chunks > 0 and now > self.play_until + AGENT_DONE_S and now - self.last_chunk > AGENT_DONE_S


async def wait_for(line: Line, cond, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end and not line.closed:
        if cond():
            return True
        await asyncio.sleep(0.05)
    return cond()


async def hang_up(ws, http: httpx.AsyncClient, call_id: int, line: Line) -> None:
    """What Plivo does when the callee puts the phone down: stop the stream, then the hangup callback."""
    if not line.closed:
        line.closed = True
        try:
            await ws.send(json.dumps({"event": "stop"}))
            await ws.close()
        except websockets.ConnectionClosed:
            pass
    await http.post(f"{BASE}/v1/plivo/status/{call_id}", data={
        "CallStatus": "completed", "HangupCauseName": "Normal Hangup", "HangupSource": "Callee",
        "Duration": str(int(line.now())), "BillDuration": str(int(line.now()))})


async def run(scenario: Scenario, out_dir: Path, voice_gender: str, language_style: str) -> dict:
    events: list[str] = []
    call_id = seed(scenario, voice_gender, language_style)
    headers = {"Authorization": f"Bearer {openai_key()}"}
    async with httpx.AsyncClient(timeout=20) as local, httpx.AsyncClient(timeout=60, headers=headers) as api:
        caller = Caller(scenario, api)
        await local.post(f"{BASE}/v1/plivo/ring/{call_id}")
        await asyncio.sleep(2.0)  # it rings for a couple of seconds
        await local.post(f"{BASE}/v1/plivo/answer/{call_id}")

        url = f"ws://localhost:8000/v1/stream/{call_id}?token={stream_token(call_id)}"
        outbound: asyncio.Queue[bytes] = asyncio.Queue()
        hung_up_by = "nobody"
        first_word_s = None
        gaps: list[float] = []
        interrupt_stop_s = None

        async with websockets.connect(url, max_size=None) as ws:
            line = Line()
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

            def say(pcm: bytes) -> None:
                line.caller.append([line.now(), pcm])
                for i in range(0, len(pcm), FRAME):
                    outbound.put_nowait(pcm[i:i + FRAME].ljust(FRAME, b"\0"))

            recv_task = asyncio.create_task(receive())
            pump_task = asyncio.create_task(pump())

            await wait_for(line, lambda: line.chunks > 0, 15)
            if line.reply_started_at:
                first_word_s = round(line.reply_started_at - line.t0, 2)

            if scenario.silent:
                await wait_for(line, lambda: False, 40)  # say nothing; see what the agent does
                hung_up_by = "agent" if line.closed else "nobody"
            else:
                turn = 0
                done = False
                while not line.closed and turn < MAX_TURNS:
                    turn += 1
                    move = scenario.moves.get(turn, "")
                    await wait_for(line, line.reply_received, 30)
                    heard = await caller.hear(bytes(line.turn_audio))
                    events.append(f"{line.now():6.1f}s agent: {heard}")
                    if done:  # caller already said bye; the agent's goodbye has arrived
                        break
                    note = " (While they talk next, you will cut in.)" if move == "interrupt" else ""
                    text, done = await caller.reply(heard, note)
                    pcm = await caller.speak(text)
                    if move == "interrupt":  # ready before the agent answers, so the cut-in lands mid-sentence
                        cut_text, _ = await caller.reply(
                            "(they start a long answer)",
                            " (Cut them off mid-sentence: 'ఆగండి ఆగండి' and ask the price straight away.)")
                        cut_pcm = await caller.speak(cut_text)
                    await wait_for(line, line.agent_done, 30)
                    if line.closed:
                        break
                    await asyncio.sleep(random.uniform(0.2, 0.6))  # a person takes a beat
                    line.turn_audio = bytearray()
                    line.reply_started_at = None
                    say(pcm)
                    events.append(f"{line.now():6.1f}s caller: {text}")
                    await wait_for(line, outbound.empty, 30)
                    spoke_at = time.monotonic()
                    chunks = line.chunks
                    got_reply = await wait_for(line, lambda: line.chunks > chunks, 12)
                    if got_reply and line.reply_started_at:
                        gaps.append(round(line.reply_started_at - spoke_at, 2))

                    if move == "hangup_mid" and got_reply:
                        await asyncio.sleep(1.5)  # a second and a half into the agent's answer...
                        events.append(f"{line.now():6.1f}s caller HANGS UP mid-sentence")
                        await hang_up(ws, local, call_id, line)
                        hung_up_by = "caller (mid-sentence)"
                        break
                    if move == "interrupt" and got_reply:
                        await asyncio.sleep(0.8)
                        line.turn_audio = bytearray()
                        line.reply_started_at = None
                        clears = len(line.clears)
                        cut_at = time.monotonic()
                        say(cut_pcm)
                        events.append(f"{line.now():6.1f}s caller CUTS IN: {cut_text}")
                        if await wait_for(line, lambda: len(line.clears) > clears, 5):
                            interrupt_stop_s = round(time.monotonic() - cut_at, 2)
                        await wait_for(line, outbound.empty, 30)
                if line.closed and hung_up_by == "nobody":
                    hung_up_by = "agent"
                elif not line.closed:
                    await wait_for(line, lambda: False, 6)  # give the agent a moment to hang up after bye
                    if line.closed:
                        hung_up_by = "agent"
                    else:
                        await hang_up(ws, local, call_id, line)
                        hung_up_by = "caller (after bye)" if done else "caller (gave up)"

            line.closed = True
            pump_task.cancel()
            done_tasks, _ = await asyncio.wait({recv_task}, timeout=5)
            if not done_tasks:
                recv_task.cancel()
        if hung_up_by == "agent":
            await local.post(f"{BASE}/v1/plivo/status/{call_id}", data={
                "CallStatus": "completed", "HangupCauseName": "Normal Hangup", "HangupSource": "API",
                "Duration": str(int(line.now()))})

    for _ in range(30):  # the server saves the transcript once its OpenAI socket has closed
        await asyncio.sleep(0.5)
        with SessionLocal() as db:
            if db.get(Call, call_id).summary:
                break
    with SessionLocal() as db:
        turns = [(t.role, t.text) for t in db.query(CallTurn).filter_by(call_id=call_id).order_by(CallTurn.id)]
        call = db.get(Call, call_id)
        summary = call.summary or ""
        found = re.search(r"outcome=(\w+)", summary)
        outcome = found.group(1) if found else call.outcome  # the hangup callback overwrites call.outcome

    out_dir.mkdir(parents=True, exist_ok=True)
    write_mix(out_dir / f"{scenario.name}.wav", line)
    agent_only = Line()
    agent_only.agent = line.agent
    write_mix(out_dir / f"{scenario.name}_agent.wav", agent_only)
    agent_lines = [text for role, text in turns if role == "agent"]
    heard = [e.split(" agent: ", 1)[1] for e in events if " agent: " in e]
    report = {
        "scenario": scenario.name,
        "expect": scenario.expect,
        "call_id": call_id,
        "voice_gender": voice_gender,
        "language_style": language_style,
        "outcome": outcome,
        "hung_up_by": hung_up_by,
        "length_s": round(line.now(), 1),
        "first_word_after_pickup_s": first_word_s,
        "reply_gaps_s": gaps,
        "avg_reply_gap_s": round(sum(gaps) / len(gaps), 2) if gaps else None,
        "agent_stopped_after_cut_in_s": interrupt_stop_s,
        "agent_words_per_reply": [len(t.split()) for t in agent_lines],
        "agent_unclear": char_error(" ".join(agent_lines), " ".join(heard)) if heard else None,
        "openers": [t.split()[0] if t.split() else "" for t in agent_lines],
        "cost": summary,
        "transcript": turns,
        "timeline": events,
    }
    (out_dir / f"{scenario.name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def write_mix(path: Path, line: Line) -> None:
    segments = list(line.agent) + list(line.caller)
    end = max((s + len(p) / (RATE * 2) for s, p in segments), default=0) + 0.5
    mix = np.zeros(int(end * RATE) + 1, dtype=np.int32)
    for start, pcm in segments:
        samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.int32)
        i = int(start * RATE)
        mix[i:i + len(samples)] += samples[: len(mix) - i]
    sf.write(str(path), np.clip(mix, -32768, 32767).astype(np.int16), RATE, subtype="PCM_16")


async def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    names = sys.argv[2:] or list(SCENARIOS)
    voice = os.environ.get("SIM_VOICE_GENDER", "male")
    style = os.environ.get("SIM_LANGUAGE_STYLE", "english_heavy")
    out_dir = Path(__file__).resolve().parent / "scenarios" / label
    for name in names:
        report = await run(SCENARIOS[name], out_dir, voice, style)
        brief = {k: report[k] for k in ("scenario", "outcome", "hung_up_by", "length_s", "first_word_after_pickup_s",
                                         "avg_reply_gap_s", "agent_stopped_after_cut_in_s", "agent_words_per_reply",
                                         "agent_unclear")}
        print(json.dumps(brief, ensure_ascii=False))
        for role, text in report["transcript"]:
            print(f"   {role:>6}: {text}")
        print(f"   cost: {report['cost']}")


if __name__ == "__main__":
    asyncio.run(main())
