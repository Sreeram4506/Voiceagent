"""The per-campaign 'keep recordings' toggle: campaigns record only when opted in; test calls still record."""

import wave

from app.dashboard_ops import _serialize_campaign, _with_keep_recordings
from app.db import SessionLocal
from app.models import AIEmployee, Campaign, CampaignStatus, Lead, LeadStatus, Tenant, VoiceTier
from app.streaming.call_recorder import recorder_from_env
from app.streaming.ws import _should_record


def _tenant(db) -> int:
    t = Tenant(name="Acme", api_key_hash="x", webhook_secret="s", webhook_url="https://hooks.example.com/in")
    db.add(t)
    db.flush()
    agent = AIEmployee(tenant_id=t.id, name="Agent", language="te", voice_tier=VoiceTier.PREMIUM, script="x")
    db.add(agent)
    db.flush()
    return t.id


def _agent_id(db, tid: int) -> int:
    return db.query(AIEmployee).filter(AIEmployee.tenant_id == tid).first().id


def _campaign(db, tid: int, keep: bool | None) -> Campaign:
    config = {} if keep is None else {"keep_recordings": keep}
    c = Campaign(tenant_id=tid, name="C", status=CampaignStatus.RUNNING, config=config)
    db.add(c)
    db.flush()
    return c


def _lead(db, tid: int, campaign_id: int | None) -> Lead:
    lead = Lead(tenant_id=tid, employee_id=_agent_id(db, tid), phone_e164="+918800000001", name="L",
                status=LeadStatus.NEW, campaign_id=campaign_id)
    db.add(lead)
    db.flush()
    return lead


def test_recorder_is_off_when_the_call_is_not_wanted_even_with_a_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VA_RECORDINGS_DIR", str(tmp_path))
    assert recorder_from_env(False) is None
    assert recorder_from_env(True) is not None


def test_a_campaign_call_records_only_when_the_campaign_opted_in():
    with SessionLocal() as db:
        tid = _tenant(db)
        opted_in = _lead(db, tid, _campaign(db, tid, True).id)
        opted_out = _lead(db, tid, _campaign(db, tid, False).id)
        not_set = _lead(db, tid, _campaign(db, tid, None).id)
        assert _should_record(db, opted_in) is True
        assert _should_record(db, opted_out) is False
        assert _should_record(db, not_set) is False  # off by default


def test_an_ad_hoc_test_call_with_no_campaign_still_records():
    with SessionLocal() as db:
        tid = _tenant(db)
        assert _should_record(db, _lead(db, tid, None)) is True
        assert _should_record(db, None) is True


def test_create_payload_folds_the_toggle_into_config():
    assert _with_keep_recordings({"keep_recordings": True}) == {"keep_recordings": True}
    assert _with_keep_recordings({"keep_recordings": False}) == {"keep_recordings": False}
    assert _with_keep_recordings({}) == {}  # untouched when the toggle is absent
    assert _with_keep_recordings({"config": {"x": 1}, "keep_recordings": True}) == {"x": 1, "keep_recordings": True}


def test_serialized_campaign_exposes_keep_recordings():
    with SessionLocal() as db:
        tid = _tenant(db)
        c = _campaign(db, tid, True)
        db.commit()
        assert _serialize_campaign(db, c, tid)["keep_recordings"] is True
        off = _campaign(db, tid, None)
        db.commit()
        assert _serialize_campaign(db, off, tid)["keep_recordings"] is False
