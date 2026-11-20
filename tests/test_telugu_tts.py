"""Our own Piper Telugu voice: spoken-form text, voice lookup, and when the factory picks it."""

from app.streaming import factory, piper_tts
from app.streaming.piper_fetch import voice_urls
from app.streaming.telugu_text import normalize_telugu


class Agent:
    language = "te"
    voice_tier = "standard"


def test_rupee_amounts_get_the_currency_word():
    assert normalize_telugu("మీ ఈఎమ్ ₹5,000 అండి") == "మీ ఈఎమ్ 5,000 రూపాయలు అండి"
    assert normalize_telugu("Rs. 1,50,000 కట్టాలి") == "1,50,000 రూపాయలు కట్టాలి"


def test_dates_are_said_with_the_month_name():
    assert normalize_telugu("15/08/2026 న") == "15 ఆగస్టు 2026 న"


def test_phone_numbers_are_read_digit_by_digit_even_before_a_comma():
    said = normalize_telugu("నా నంబర్ 98765 43210, కాల్ చేయండి.")
    assert said == "నా నంబర్ తొమ్మిది ఎనిమిది ఏడు ఆరు ఐదు నాలుగు మూడు రెండు ఒకటి సున్నా, కాల్ చేయండి."


def test_amounts_and_decimals_are_not_mistaken_for_phone_numbers():
    assert normalize_telugu("1,50,000 లేదా 2.5 లక్షలు, 2026 లో") == "1,50,000 లేదా 2.5 లక్షలు, 2026 లో"


def test_anusvara_is_said_m_at_word_end_and_in_spelled_letters():
    assert normalize_telugu("నమస్కారం అండి, సంతోషం.") == "నమస్కారమ్ అండి, సంతోషమ్."
    assert normalize_telugu("ఈఎంఐ") == "ఈఎమ్ ఐ"
    assert normalize_telugu("అంటే సంతోషంగా") == "అంటే సంతోషంగా"  # inside a word ం stays


def test_prepare_text_leaves_no_english_letters():
    text = piper_tts.prepare_text("OK sir, Dr. Rao EMI 5000")
    assert not any("a" <= ch.lower() <= "z" for ch in text)
    assert "డాక్టర్" in text


def test_long_replies_are_voiced_sentence_by_sentence():
    assert piper_tts._split_phrases("సరే అండి. రేపు కాల్ చేస్తాను! బై") == ["సరే అండి.", "రేపు కాల్ చేస్తాను!", "బై"]


def test_custom_voice_file_is_found_from_piper_model(tmp_path, monkeypatch):
    model = tmp_path / "te_IN-custom-medium.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(piper_tts, "PIPER_MODEL", str(model))
    assert piper_tts.model_path() == model


def test_public_voice_is_found_anywhere_under_the_voices_dir(tmp_path, monkeypatch):
    model = tmp_path / "te" / "te_IN" / "padmavathi" / "medium" / "te_IN-padmavathi-medium.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    monkeypatch.setattr(piper_tts, "PIPER_MODEL", "")
    monkeypatch.setattr(piper_tts, "PIPER_VOICES_DIR", str(tmp_path))
    monkeypatch.setattr(piper_tts, "PIPER_VOICE", "te_IN-padmavathi-medium")
    assert piper_tts.model_path() == model


def test_factory_uses_piper_for_telugu_when_switched_on(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(piper_tts, "PIPER_MODEL", str(model))
    monkeypatch.setenv("VA_TTS_ENGINE", "piper")
    tts = factory._build_tts(Agent(), openai_key=None, extras={"speech_rate": 1.3})
    assert isinstance(tts, piper_tts.PiperTTS)
    assert tts.length_scale == round(piper_tts.PIPER_LENGTH_SCALE / 1.3, 3)  # the agent's faster pace carries over


def test_factory_stays_on_edge_when_the_piper_voice_is_missing(tmp_path, monkeypatch):
    from app.streaming.edge_tts_provider import EdgeTTS

    monkeypatch.setattr(piper_tts, "PIPER_MODEL", str(tmp_path / "missing.onnx"))
    monkeypatch.setenv("VA_TTS_ENGINE", "piper")
    assert isinstance(factory._build_tts(Agent(), openai_key=None, extras={}), EdgeTTS)


def test_factory_keeps_edge_by_default(monkeypatch):
    from app.streaming.edge_tts_provider import EdgeTTS

    monkeypatch.delenv("VA_TTS_ENGINE", raising=False)
    assert isinstance(factory._build_tts(Agent(), openai_key=None, extras={}), EdgeTTS)


def test_voice_is_loaded_at_server_start_only_when_piper_is_on(tmp_path, monkeypatch):
    import threading

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setattr(piper_tts, "PIPER_MODEL", str(model))
    loaded = threading.Event()
    monkeypatch.setattr(piper_tts, "_synthesize_pcm8k", lambda text: loaded.set())

    monkeypatch.delenv("VA_TTS_ENGINE", raising=False)
    assert piper_tts.warmup() is False
    monkeypatch.setenv("VA_TTS_ENGINE", "piper")
    assert piper_tts.warmup() is True
    assert loaded.wait(timeout=5)


def test_missing_voice_is_downloaded_at_server_start_then_loaded(tmp_path, monkeypatch):
    import threading

    from app.streaming import piper_fetch

    model = tmp_path / "voice.onnx"
    monkeypatch.setattr(piper_tts, "PIPER_MODEL", str(model))
    monkeypatch.setenv("VA_TTS_ENGINE", "piper")
    monkeypatch.setattr(piper_fetch, "main", lambda: model.write_bytes(b"onnx") and 0)
    loaded = threading.Event()
    monkeypatch.setattr(piper_tts, "_synthesize_pcm8k", lambda text: loaded.set())

    assert piper_tts.warmup() is True
    assert loaded.wait(timeout=5)
    assert model.exists()


def test_fetch_urls_for_public_and_custom_voices():
    assert voice_urls("te_IN-padmavathi-medium")[0].endswith("/te/te_IN/padmavathi/medium/te_IN-padmavathi-medium.onnx")
    assert voice_urls("ignored", "https://x.test/v.onnx") == ["https://x.test/v.onnx", "https://x.test/v.onnx.json"]
