"""Seed realistic client data: employees, leads, campaigns, calls, appointments."""
import os, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone, time as dtime
import random

# Load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent))

from app.db import SessionLocal, init_db
from app.models import (
    AIEmployee, Call, CallDirection, CallStatus, CallTurn,
    Campaign, CampaignStatus, ContactList, Lead, LeadStatus,
    LedgerEntry, LedgerKind, Tenant, TenantUser, Appointment,
    AppointmentStatus, VoiceTier, ActivityLog,
)
from app.auth import hash_new_password, ensure_default_tenant

init_db()
ensure_default_tenant()

now = datetime.now(timezone.utc)

LEADS = [
    ("Ravi Kumar",        "9876543210"),
    ("Priya Sharma",      "9123456789"),
    ("Suresh Reddy",      "9988776655"),
    ("Anitha Devi",       "9876501234"),
    ("Venkat Rao",        "9000112233"),
    ("Lakshmi Naidu",     "9900887766"),
    ("Mohan Das",         "9811223344"),
    ("Kavitha Rao",       "9700998877"),
    ("Ramesh Babu",       "9654321098"),
    ("Sunita Verma",      "9823456780"),
    ("Arjun Singh",       "9712345678"),
    ("Deepa Nair",        "9601234567"),
    ("Kiran Kumar",       "9512345670"),
    ("Meena Krishnan",    "9445678901"),
    ("Ashok Pillai",      "9334567890"),
    ("Swathi Reddy",      "9256789012"),
    ("Naresh Chandra",    "9178901234"),
    ("Pooja Gupta",       "9087654321"),
    ("Sunil Yadav",       "8976543210"),
    ("Girija Devi",       "8865432109"),
]

AGENTS = [
    {
        "name": "Suma Telugu",
        "language": "te",
        "script": "నమస్కారం అండి! నేను సుమ. మీ రుణ పునరుద్ధరణ గురించి మాట్లాడవచ్చా?",
        "voice_tier": VoiceTier.PREMIUM,
        "phone_number": "+918065591949",  # real Plivo caller ID
        "max_concurrency": 40,
        "window_start": dtime(9, 0),
        "window_end": dtime(21, 0),
        "max_attempts": 3,
        "paused": False,
    },
    {
        "name": "Arya Hindi",
        "language": "hi",
        "script": "नमस्ते! मैं आर्या हूँ। क्या मैं आपसे लोन के बारे में बात कर सकती हूँ?",
        "voice_tier": VoiceTier.STANDARD,
        "phone_number": None,  # will use fallback when dialing
        "max_concurrency": 20,
        "window_start": dtime(10, 0),
        "window_end": dtime(20, 0),
        "max_attempts": 3,
        "paused": False,
    },
    {
        "name": "Priya English",
        "language": "en",
        "script": "Hello! I'm Priya from SUMA Finance. May I speak with you about your loan renewal?",
        "voice_tier": VoiceTier.CUSTOM,
        "phone_number": None,
        "max_concurrency": 15,
        "window_start": dtime(9, 30),
        "window_end": dtime(19, 30),
        "max_attempts": 4,
        "paused": False,
    },
]

CAMPAIGNS = [
    ("October Loan Renewal", "Appointment Booking", "te"),
    ("MSME Outreach Q4",     "Lead Qualification",  "hi"),
    ("Premium Customer Re-engagement", "Survey", "en"),
]

with SessionLocal() as db:
    # ── Tenant ───────────────────────────────────────────────────────────────
    tenant = db.query(Tenant).first()
    if not tenant:
        print("No tenant found — run ensure_default_tenant first"); sys.exit(1)
    tid = tenant.id
    print(f"Tenant: {tenant.name} (id={tid})")

    # ── Top-up credits ───────────────────────────────────────────────────────
    db.add(LedgerEntry(tenant_id=tid, kind=LedgerKind.TOPUP, amount_paise=500_000, note="seed topup 5000 credits"))

    # ── Employees ────────────────────────────────────────────────────────────
    existing_emps = {e.name: e for e in db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).all()}
    # Also index by phone_number to detect conflicts
    existing_by_phone = {e.phone_number: e for e in existing_emps.values() if e.phone_number}
    emp_ids = []
    for a in AGENTS:
        # If phone_number is already owned by another agent, rename that agent
        if a["phone_number"] and a["phone_number"] in existing_by_phone:
            e = existing_by_phone[a["phone_number"]]
            old_name = e.name
            e.name = a["name"]
            e.language = a["language"]
            e.script = a["script"]
            e.voice_tier = a["voice_tier"]
            e.max_concurrency = a["max_concurrency"]
            e.window_start = a["window_start"]
            e.window_end = a["window_end"]
            e.max_attempts = a["max_attempts"]
            e.paused = a["paused"]
            db.flush()
            existing_emps[a["name"]] = e
            existing_emps.pop(old_name, None)
            print(f"  Renamed/updated agent: {old_name} → {a['name']} (id={e.id})")
        elif a["name"] not in existing_emps:
            e = AIEmployee(
                tenant_id=tid,
                name=a["name"],
                language=a["language"],
                script=a["script"],
                voice_tier=a["voice_tier"],
                phone_number=a["phone_number"],
                max_concurrency=a["max_concurrency"],
                window_start=a["window_start"],
                window_end=a["window_end"],
                max_attempts=a["max_attempts"],
                paused=a["paused"],
            )
            db.add(e)
            db.flush()
            existing_emps[a["name"]] = e
            print(f"  Created agent: {a['name']} (id={e.id})")
        else:
            e = existing_emps[a["name"]]
            e.paused = False
            e.window_start = a["window_start"]
            e.window_end = a["window_end"]
            db.flush()
            print(f"  Updated agent: {a['name']} (id={e.id})")
        emp_ids.append(existing_emps[a["name"]].id)

    primary_emp_id = emp_ids[0]

    # ── Contact List ─────────────────────────────────────────────────────────
    cl = ContactList(tenant_id=tid, name="October 2026 Leads", filename="oct2026.xlsx", contact_count=len(LEADS))
    db.add(cl)
    db.flush()

    # ── Leads ────────────────────────────────────────────────────────────────
    added_leads = []
    for i, (name, phone) in enumerate(LEADS):
        e164 = "+91" + phone.lstrip("+91")
        # skip if already exists
        if db.query(Lead).filter(Lead.tenant_id == tid, Lead.phone_e164 == e164).first():
            continue
        status = random.choice([LeadStatus.DONE, LeadStatus.DONE, LeadStatus.NEW, LeadStatus.FAILED])
        emp_id = emp_ids[i % len(emp_ids)]
        lead = Lead(
            tenant_id=tid,
            employee_id=emp_id,
            phone_e164=e164,
            name=name,
            status=status,
            source=f"list:{cl.id}",
            next_attempt_at=now - timedelta(hours=random.randint(1, 48)),
            attempts=random.randint(1, 3) if status != LeadStatus.NEW else 0,
            last_called_at=(now - timedelta(hours=random.randint(1, 24))) if status != LeadStatus.NEW else None,
        )
        db.add(lead)
        db.flush()
        added_leads.append((lead.id, emp_id, name, e164, status))

    print(f"  Added {len(added_leads)} leads")

    # ── Campaigns ────────────────────────────────────────────────────────────
    for i, (cname, objective, lang) in enumerate(CAMPAIGNS):
        if db.query(Campaign).filter(Campaign.tenant_id == tid, Campaign.name == cname).first():
            continue
        emp_id = emp_ids[i % len(emp_ids)]
        c_status = CampaignStatus.RUNNING if i == 0 else (CampaignStatus.COMPLETED if i == 1 else CampaignStatus.DRAFT)
        c = Campaign(
            tenant_id=tid,
            employee_id=emp_id,
            contact_list_id=cl.id if i == 0 else None,
            name=cname,
            objective=objective,
            language=lang,
            status=c_status,
            concurrency=20,
            window_start="09:30",
            window_end="20:00",
            max_retries=3,
            retry_interval_hours=2,
            launched_at=now - timedelta(days=random.randint(1, 10)) if c_status != CampaignStatus.DRAFT else None,
        )
        db.add(c)
        db.flush()
        print(f"  Created campaign: {cname} ({c_status.value})")

    # ── Calls + Appointments ─────────────────────────────────────────────────
    call_statuses = [CallStatus.COMPLETED, CallStatus.COMPLETED, CallStatus.NO_ANSWER, CallStatus.BUSY, CallStatus.COMPLETED]
    call_count = 0
    appt_count = 0
    for lead_id, emp_id, name, e164, lead_status in added_leads[:15]:
        if lead_status == LeadStatus.NEW:
            continue
        started = now - timedelta(hours=random.randint(2, 72), minutes=random.randint(0, 59))
        c_status = random.choice(call_statuses)
        duration = random.randint(45, 320) if c_status == CallStatus.COMPLETED else 0
        outcome = random.choice(["appointment_booked", "callback_requested", "not_interested", None])
        call = Call(
            tenant_id=tid,
            lead_id=lead_id,
            employee_id=emp_id,
            direction=CallDirection.OUTBOUND,
            status=c_status,
            outcome=outcome,
            billed_seconds=duration,
            started_at=started,
            ended_at=started + timedelta(seconds=duration) if duration else None,
            provider_call_id=f"seed-{lead_id}-{call_count}",
        )
        db.add(call)
        db.flush()
        call_count += 1

        if outcome == "appointment_booked":
            appt = Appointment(
                tenant_id=tid,
                call_id=call.id,
                lead_id=lead_id,
                employee_id=emp_id,
                contact_name=name,
                phone=e164,
                scheduled_at=now + timedelta(days=random.randint(1, 7), hours=random.randint(9, 17)),
                notes="Booked via voice agent - seed data",
                status=AppointmentStatus.CONFIRMED,
            )
            db.add(appt)
            appt_count += 1

    print(f"  Added {call_count} calls, {appt_count} appointments")

    # ── Activity log ─────────────────────────────────────────────────────────
    db.add(ActivityLog(tenant_id=tid, actor="system", action="seed.completed",
                       detail=f"{len(added_leads)} leads, {call_count} calls, {appt_count} appts"))

    db.commit()
    print("\nSeed complete!")
    print(f"  Tunnel:   https://amp-lasting-unsigned-ground.trycloudflare.com")
    print(f"  Login:    admin@suma.ai / suma2026")
    print(f"  Agents:   {len(emp_ids)} configured")
    print(f"  Leads:    {len(added_leads)} added")
    print(f"  Calls:    {call_count} seeded")
    print(f"  Appts:    {appt_count} seeded")
