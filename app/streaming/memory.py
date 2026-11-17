"""Per-lead conversation memory: remember key details across calls.

Stores summaries in Lead.metadata_json["conversation_memory"] so repeat
callers get a more personal experience.
"""

from __future__ import annotations

from app.db import SessionLocal
from app.models import Lead


def load_memory(lead_id: int | None) -> dict:
    if not lead_id:
        return {}
    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        if not lead:
            return {}
        meta = lead.metadata_json or {}
        return meta.get("conversation_memory", {})


def memory_to_prompt(memory: dict) -> str:
    if not memory:
        return ""
    lines = ["# What you remember from previous conversations with this person"]

    count = memory.get("calls_count", 0)
    if count:
        lines.append(f"- You have spoken with them {count} time(s) before.")

    last = memory.get("last_outcome")
    if last:
        lines.append(f"- Last call: {last}")

    topics = memory.get("topics", [])
    if topics:
        lines.append(f"- They mentioned: {', '.join(topics[-5:])}")

    prefs = memory.get("preferences", [])
    if prefs:
        lines.append(f"- Preferences: {', '.join(prefs[-3:])}")

    notes = memory.get("relationship_notes", [])
    if notes:
        lines.append(f"- Personal notes: {', '.join(notes[-3:])}")

    lines.append(
        "Reference these naturally when relevant — mentioning something you remember "
        "builds trust. Don't recite a list."
    )
    return "\n".join(lines)


def update_memory(lead_id: int | None, turns: list[tuple[str, str]], outcome: str) -> None:
    """Update the lead's conversation memory after a call ends."""
    if not lead_id:
        return
    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        if not lead:
            return
        meta = lead.metadata_json or {}
        mem = dict(meta.get("conversation_memory", {}))
        mem["calls_count"] = mem.get("calls_count", 0) + 1
        if outcome:
            mem["last_outcome"] = outcome[:300]
        # keep the last few caller lines as topic hints
        caller_lines = [t for role, t in turns if role == "caller" and t.strip()]
        if caller_lines:
            topics = mem.get("topics", [])
            for line in caller_lines[-3:]:
                snippet = line.strip()[:120]
                if snippet and snippet not in topics:
                    topics.append(snippet)
            mem["topics"] = topics[-8:]
        meta["conversation_memory"] = mem
        lead.metadata_json = meta
        db.commit()
