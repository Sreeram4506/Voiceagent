"""The mini-model prompt: short, human, spoken language, and not tied to any one kind of business."""

import asyncio

import pytest

from app.streaming.realtime_session import RealtimeSession

MINI = "gpt-realtime-mini"
FULL = "gpt-realtime-1.5"
REAL_ESTATE_WORDS = ("ప్లాట్", "సైట్ విజిట్", "గజాలు", "లక్షలు", "plot", "site visit", "square yards")
NEWSREADER_WORDS = ("ధన్యవాదాలు", "సదుపాయం", "ప్రారంభం", "అందుబాటులో", "నిమిత్తం", "పరిష్కారాలు")


def prompt(language: str = "te", model: str = MINI, script: str = "Call about our services.", **kwargs) -> str:
    return RealtimeSession("key", script, language, model=model, **kwargs)._instructions


def section(text: str, heading: str) -> str:
    """The body of one '# heading' section of the prompt."""
    start = text.index(f"# {heading}")
    end = text.find("\n# ", start + 1)
    return text[start:end if end != -1 else None]


def test_mini_prompt_is_not_written_for_one_kind_of_business():
    text = prompt(script="")
    assert not [word for word in REAL_ESTATE_WORDS if word in text]


def test_mini_samples_use_spoken_telugu_not_newsreader_words():
    samples = section(prompt(), "Sample phrases")
    assert samples.count('"') >= 8  # several example lines to copy the style from
    assert not [word for word in NEWSREADER_WORDS if word in samples]


def test_mini_bans_newsreader_words_and_keeps_names_in_english():
    text = prompt()
    assert "ధన్యవాదాలు" in section(text, "How you talk")  # named only to forbid it
    assert "product and service names" in text


def test_mini_keeps_replies_short_and_varied():
    text = prompt()
    assert "about ten words" in text
    assert "never start two replies the same way" in text.lower()


def test_mini_says_the_booking_back_and_books_in_the_same_reply():
    assert "in the same reply" in section(prompt(), "Actions")


@pytest.mark.parametrize("language", ["te", "hi", "en", "ta", "kn"])
def test_every_language_gets_the_short_prompt_on_mini(language):
    mini, full = prompt(language), prompt(language, model=FULL)
    assert len(mini) < len(full) * 0.6
    assert "# Actions" in mini


def test_mini_follows_the_agents_language_style():
    assert "English words people really say" in prompt(language_style="natural_mix")
    pure = prompt(language_style="pure")
    assert "pure" in pure.lower() and "English words people really say" not in pure
    assert "lean on English" in prompt(language_style="english_heavy")


def test_full_models_keep_the_long_prompt():
    assert "# Reading the caller" in prompt(model=FULL)
    assert "# Reading the caller" not in prompt()


def test_greeting_is_requested_with_the_session_settings_not_after_them():
    """Waiting for session.updated before asking for the greeting cost ~0.7 s at pickup."""

    async def scenario():
        session = RealtimeSession("key", "Call about our services.", "te", model=MINI, greeting="నమస్కారం అండి")
        sent: list[dict] = []

        async def capture(event: dict) -> None:
            sent.append(event)

        session._to_openai = capture
        await session._configure()
        assert [e["type"] for e in sent] == ["session.update", "response.create"]
        assert "నమస్కారం అండి" in sent[-1]["response"]["instructions"]

        await session._on_event({"type": "session.updated"})
        assert len(sent) == 2  # no second greeting
        assert session._ready.is_set()

    asyncio.run(scenario())


def test_mini_hangs_up_on_a_wrong_number_or_a_firm_no():
    flow = section(prompt(), "Flow")
    assert "wrong number" in flow
    assert "accept it the first time" in flow


def test_mini_books_the_day_the_caller_said():
    assert "exactly the day and time they said" in section(prompt(), "Actions")


def test_mini_goodbye_ends_with_bye_and_never_a_wrong_greeting():
    actions = section(prompt(), "Actions")
    assert '"bye"' in actions
    assert "good night" in prompt().lower()


def test_telugu_is_always_respectful():
    assert "పోరా" in section(prompt(), "Who you are")  # named only to forbid it


def test_english_heavy_telugu_has_its_own_samples():
    heavy = section(prompt(language_style="english_heavy"), "Sample phrases")
    mixed = section(prompt(language_style="natural_mix"), "Sample phrases")
    assert heavy != mixed
    assert "అన్నమాట" in heavy and "as well as" in heavy


def test_opening_gives_mini_a_line_to_follow():
    session = RealtimeSession("key", "Call about our services.", "te", model=MINI, caller_name="Ravi")
    opening = session._greeting_request()["response"]["instructions"]
    assert "ఒక్క నిమిషం" in opening and "Ravi" in opening


def test_mini_never_makes_up_a_price():
    assert "NEVER MAKE UP A PRICE OR NUMBER" in section(prompt(), "Flow")


def test_mini_checks_they_are_free_then_says_why_then_waits_for_interest():
    flow = section(prompt(), "Flow")
    assert "never again" in flow  # one introduction only
    assert "if they are free" in flow.lower() and "why you are calling" in flow
    assert "ask when they will be free" in flow
    assert "only after they show interest" in flow
