"""
Tests for ai.py's resilience behavior: the Claude client is always mocked
here (no real API calls, no network, no cost) so these run instantly and
deterministically. The point isn't testing Claude's output quality -- that's
inherently non-deterministic -- it's proving the bot never crashes or goes
silent when the API call fails, which is the exact bug this project hit in
production.
"""

import json

import pytest

import ai
import db

# Captured at import time, BEFORE any test's autouse conftest.py fixture
# (_passthrough_narration) gets a chance to monkeypatch the ai.narrate_reply
# ATTRIBUTE to a pure passthrough lambda for the rest of this test suite --
# monkeypatch only ever reassigns that attribute, never mutates the function
# object itself, so this reference stays the real implementation throughout.
# Needed because these are the one place in the suite that tests
# narrate_reply's OWN internals (its prompt, its API call shape) rather than
# treating it as an opaque dependency the way every other test file does.
_REAL_NARRATE_REPLY = ai.narrate_reply


class _FakeContentBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeContentBlock(text)]


class _FakeClient:
    """Stands in for anthropic.Anthropic. `create_return` can be a string
    (returned as the response text) or an Exception instance/class (raised
    instead), so tests can simulate both success and failure paths."""

    def __init__(self, create_return):
        self._create_return = create_return
        self.messages = self

    def create(self, **kwargs):
        if isinstance(self._create_return, Exception):
            raise self._create_return
        return _FakeResponse(self._create_return)


@pytest.fixture(autouse=True)
def _reset_client_singleton(monkeypatch):
    # ai.py caches the client in a module-level global; make sure each test
    # starts from a clean slate regardless of test order.
    monkeypatch.setattr(ai, "_client", None)
    yield


def _mock_client(monkeypatch, create_return):
    fake = _FakeClient(create_return)
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    return fake


# ---------- parse_message: happy path, log_expense ----------

def test_parse_message_returns_parsed_log_expense(monkeypatch):
    payload = {
        "intent": "log_expense",
        "expenses": [{"amount": 12.5, "currency": None, "description": "lunch",
                       "category": "Food", "is_claimable": False}],
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "clarification_question": None, "casual_reply": None,
    }
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.parse_message("spent 12.50 on lunch", [])
    assert result["intent"] == "log_expense"
    assert result["expenses"][0]["amount"] == 12.5


def test_parse_message_strips_markdown_code_fence(monkeypatch):
    payload = {"intent": "log_expense",
               "expenses": [{"amount": 5, "currency": None, "description": "coffee",
                              "category": "Food", "is_claimable": False}],
               "target_expense_id": None, "correction_action": None, "days_ago": None,
               "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
               "clarification_question": None, "casual_reply": None}
    _mock_client(monkeypatch, f"```json\n{json.dumps(payload)}\n```")
    result = ai.parse_message("coffee 5", [])
    assert result["expenses"][0]["amount"] == 5


def test_parse_message_returns_multiple_expenses_in_one_message(monkeypatch):
    """Regression test for a real user complaint: '$5 for lunch and $5 for
    coffee' must come back as two separate items, not just the first one."""
    payload = {
        "intent": "log_expense",
        "expenses": [
            {"amount": 5, "currency": None, "description": "lunch", "category": "Food", "is_claimable": False},
            {"amount": 5, "currency": None, "description": "coffee", "category": "Food", "is_claimable": False},
        ],
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "clarification_question": None, "casual_reply": None,
    }
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.parse_message("$5 for lunch and $5 for coffee", [])
    assert result["intent"] == "log_expense"
    assert len(result["expenses"]) == 2
    assert {e["description"] for e in result["expenses"]} == {"lunch", "coffee"}


def test_parse_message_passes_recent_expenses_into_the_prompt(monkeypatch):
    """Regression guard for the actual reported bug: without seeing recent
    expenses, the model has no way to know which entry a correction like
    "that was for yesterday" refers to. Just checks the context is actually
    sent to the API, not what the model does with it (that's not testable
    without a real call)."""
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps({
        "intent": "casual", "amount": None, "currency": None, "description": None, "category": None,
        "is_claimable": None, "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "clarification_question": None, "casual_reply": "hey!",
    }))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    recent = [{"id": 42, "amount": 100, "currency": "SGD", "description": "parking cashcard top-up",
               "category": "Transport", "expense_date": "2026-07-27", "is_claimable": False}]
    ai.parse_message("hi", recent)
    sent_content = captured["messages"][0]["content"]
    assert "42" in sent_content
    assert "parking cashcard top-up" in sent_content


def test_parse_message_tells_the_model_todays_actual_date(monkeypatch):
    """Root-cause regression guard for the real bug: 'on 18 September' and
    similar explicit/absolute dates silently logged onto today because the
    model was never told what today's date IS -- only ever asked to compute
    offsets FROM today. Pins db.today_str() to a known value and checks that
    exact date (and weekday) actually reaches the sent prompt content."""
    monkeypatch.setattr(db, "today_str", lambda: "2026-09-19")
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps({
        "intent": "casual", "amount": None, "currency": None, "description": None, "category": None,
        "is_claimable": None, "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "clarification_question": None, "casual_reply": "hey!",
    }))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    ai.parse_message("hi", [])
    sent_content = captured["messages"][0]["content"]
    assert "2026-09-19" in sent_content
    assert "Saturday" in sent_content


# ---------- parse_message: casual intent ----------

def test_parse_message_returns_casual_reply_for_non_expense(monkeypatch):
    payload = {"intent": "casual", "amount": None, "currency": None, "description": None,
               "category": None, "is_claimable": None,
               "target_expense_id": None, "correction_action": None, "days_ago": None,
               "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
               "clarification_question": None, "casual_reply": "Hey! Doing well -- anything to log?"}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.parse_message("hey how's it going", [])
    assert result["intent"] == "casual"
    assert result["casual_reply"] == "Hey! Doing well -- anything to log?"


# ---------- parse_message: correction intent ----------

def test_parse_message_returns_correction_targeting_a_recent_id(monkeypatch):
    payload = {"intent": "correction", "amount": None, "currency": None, "description": None,
               "category": None, "is_claimable": None,
               "target_expense_id": 42, "correction_action": "edit_date", "days_ago": 2,
               "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
               "clarification_question": None, "casual_reply": None}
    _mock_client(monkeypatch, json.dumps(payload))
    recent = [{"id": 42, "amount": 100, "currency": "SGD", "description": "parking cashcard top-up",
               "category": "Transport", "expense_date": "2026-07-27", "is_claimable": False}]
    result = ai.parse_message("sorry that was from two days ago", recent)
    assert result["intent"] == "correction"
    assert result["target_expense_id"] == 42
    assert result["correction_action"] == "edit_date"
    assert result["days_ago"] == 2


# ---------- parse_message: resilience (the actual production bug) ----------

def test_parse_message_never_raises_on_api_failure(monkeypatch):
    """This is the regression test for the real bug: an unhandled exception
    from the Claude call (auth failure, network blip, dependency mismatch --
    exactly what happened with the anthropic/httpx version conflict) must
    never propagate out of parse_message. It must come back as a normal,
    gracefully-worded clarification dict instead."""
    _mock_client(monkeypatch, TypeError("Client.__init__() got an unexpected keyword argument 'proxies'"))
    result = ai.parse_message("test message", [])
    assert result["intent"] == "clarification"
    assert result["clarification_question"]  # some non-empty message, not None
    assert result["casual_reply"] is None


def test_parse_message_falls_back_on_invalid_json(monkeypatch):
    _mock_client(monkeypatch, "this is not json at all")
    result = ai.parse_message("garbled", [])
    assert result["intent"] == "clarification"


def test_parse_message_requests_enough_tokens_for_a_large_bulk_paste(monkeypatch):
    """Regression guard for a real production incident: Andrew pasted a
    starter list of 24 subscriptions in one message and the bot replied with
    a generic "didn't quite catch that" fallback instead of logging any of
    them. Root cause: the full parse_message response schema has ~84
    top-level fields (most null on any given call) PLUS one object per
    subscription -- for 24 subscriptions the complete, valid response comes
    to roughly 1700-2200 tokens, well over the max_tokens=1500 cap that was
    in place at the time. The real API call got cut off mid-object, so
    _safe_json's own json.loads and best-effort recovery both failed on the
    truncated JSON and silently fell back to a single generic clarification
    -- for all 24 subscriptions at once, not just the ones past some limit.
    This pins max_tokens comfortably above that observed worst case so a
    large bulk paste (subscriptions, tasks, or expenses) has real room to
    complete instead of being cut off into invalid JSON."""
    fake = _mock_recording_client(
        monkeypatch, json.dumps({"intent": "casual", "casual_reply": "ok", "clarification_question": None})
    )
    ai.parse_message("doesn't matter for this test", [])
    assert fake.calls[0]["max_tokens"] >= 4000


def test_parse_message_passes_recent_reminders_into_the_prompt(monkeypatch):
    """Regression guard, same reasoning as recent_expenses above: without
    seeing the standing daily reminders list, the model has no way to know
    which reminder "took my hair pills" or "stop reminding me about that"
    refers to (target_domain="reminder" correction, see ai.py's PARSE_
    SYSTEM_PROMPT and correction.py's _handle_reminder_correction)."""
    fake = _mock_recording_client(monkeypatch, json.dumps({
        "intent": "casual", "clarification_question": None, "casual_reply": "hey!",
    }))
    recent_reminders = [{"id": 7, "description": "take hair pills", "last_done_date": None}]
    ai.parse_message("hi", recent_reminders=recent_reminders)
    sent_content = fake.calls[0]["messages"][0]["content"]
    assert "take hair pills" in sent_content
    assert '"id": 7' in sent_content


def test_parse_system_prompt_documents_the_reminder_correction_domain():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert '"reminder"' in prompt
    assert "done for TODAY only" in prompt


# ---------- categorize: resilience ----------

def test_categorize_returns_model_choice_when_valid(monkeypatch):
    _mock_client(monkeypatch, "Transport")
    assert ai.categorize("uber ride") == "Transport"


def test_categorize_defaults_to_other_on_invalid_category(monkeypatch):
    _mock_client(monkeypatch, "NotARealCategory")
    assert ai.categorize("mystery purchase") == "Other"


def test_categorize_accepts_gifts_and_hobbies_categories(monkeypatch):
    """Regression guard for the category expansion: 'Gifts & Occasions' and
    'Hobbies & Collectibles' must be accepted as valid, not fall back to
    'Other' just because they're multi-word/new."""
    import config
    assert "Gifts & Occasions" in config.CATEGORIES
    assert "Hobbies & Collectibles" in config.CATEGORIES

    _mock_client(monkeypatch, "Gifts & Occasions")
    assert ai.categorize("wedding ang pow") == "Gifts & Occasions"

    _mock_client(monkeypatch, "Hobbies & Collectibles")
    assert ai.categorize("pokemon card booster box") == "Hobbies & Collectibles"


def test_categorize_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, ConnectionError("network blip"))
    assert ai.categorize("anything") == "Other"


# ---------- photo classification: current-time context ----------
# Regression coverage for a real bad interaction: a "daily activity" screen
# showing a full day's totals, sent a few minutes after local midnight, got
# logged onto the NEW day even though that much activity couldn't have
# accumulated since rollover -- the screen was still showing the day that
# just ended. Fixed by giving the model the current local time alongside
# today's date, plus an explicit reasoning rule for this exact shape of
# screenshot (see PHOTO_CLASSIFY_SYSTEM_PROMPT). These tests only prove the
# time context actually reaches the model call -- the model's own reasoning
# from it isn't something a unit test can verify.

# ---------- photo classification: TOTAL calories, not just the Move ring ----------
# Regression guard for a real observed bug: an Apple Health/Fitness activity
# screenshot shows at least two different calorie numbers -- the Move ring's
# own big, bold headline figure against its goal (e.g. "772/700KCAL", active
# calories only, excludes resting/BMR burn) and a much smaller "TOTAL X,XXX
# KCAL" line printed just below the Move chart (the real day's total). The
# bot kept logging calories_burned from the Move ring's headline number
# (772) instead of the real total (2,762) sitting right below it, every
# time -- not bad luck, a genuine gap in what the prompt told the model to
# look for. Like the calorie_calibration regression above, a unit test can't
# verify the model's own vision reasoning, only that the instruction telling
# it to prefer the TOTAL line actually exists in the prompt it's given.

def test_photo_classify_prompt_prefers_total_over_move_ring_figure():
    prompt = ai.PHOTO_CLASSIFY_SYSTEM_PROMPT
    assert "TOTAL" in prompt
    assert "772/700KCAL" in prompt  # the actual real-world example that broke
    assert "2,762 KCAL" in prompt
    assert "EXCLUDES resting/BMR burn" in prompt


class _CapturingFakeClient:
    """Like _FakeClient, but also records the kwargs each create() call
    received, so a test can inspect exactly what was sent to Claude."""

    def __init__(self, create_return):
        self._create_return = create_return
        self.messages = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._create_return)


def test_current_local_time_str_returns_hh_mm_format():
    import re
    assert re.fullmatch(r"\d{2}:\d{2}", ai._current_local_time_str())


def test_extract_from_photo_tells_the_model_the_current_time(monkeypatch):
    fake = _CapturingFakeClient(json.dumps({"kind": "unclear"}))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    ai.extract_from_photo(b"fake-image-bytes", caption=None)
    text_block = next(b["text"] for b in fake.calls[0]["messages"][0]["content"] if b["type"] == "text")
    assert "It's currently" in text_block


def test_extract_from_photos_tells_the_model_the_current_time(monkeypatch):
    fake = _CapturingFakeClient(json.dumps({"kind": "unclear"}))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    ai.extract_from_photos([b"img1", b"img2"], caption=None)
    text_block = next(b["text"] for b in fake.calls[0]["messages"][0]["content"] if b["type"] == "text")
    assert "It's currently" in text_block


# ---------- meal calorie calibration ----------
# Regression guard against drift: this calibration text used to be three
# separately hand-maintained copies (parse_message's log_meal rules,
# MEAL_ESTIMATE_SYSTEM_PROMPT, PHOTO_CLASSIFY_SYSTEM_PROMPT) -- now a single
# shared constant referenced by all three, so they can't quietly diverge.

def test_calorie_calibration_note_present_in_all_three_prompts():
    assert ai.MEAL_CALORIE_CALIBRATION_NOTE in ai.PARSE_SYSTEM_PROMPT
    assert ai.MEAL_CALORIE_CALIBRATION_NOTE in ai.MEAL_ESTIMATE_SYSTEM_PROMPT
    assert ai.MEAL_CALORIE_CALIBRATION_NOTE in ai.PHOTO_CLASSIFY_SYSTEM_PROMPT


# ---------- _safe_json / _parse_json_or_none: tolerating stray prose ----------
# Regression coverage for a real bad interaction: a genuinely edge-case-y
# request (correcting a workout/lift field before edit_workout/edit_lift
# existed) occasionally got a response with a stray sentence of prose
# wrapped around an otherwise-valid JSON object, which the old strict
# json.loads(whole_string) failed on entirely -- producing a generic
# "didn't catch that" clarification even though a real, correctly-shaped
# JSON object was right there in the response.

def test_safe_json_parses_clean_json():
    assert ai._safe_json('{"intent": "casual"}') == {"intent": "casual"}


def test_safe_json_recovers_json_with_leading_prose():
    raw = 'Sure, here is the correction:\n{"intent": "casual", "casual_reply": "ok"}'
    assert ai._safe_json(raw) == {"intent": "casual", "casual_reply": "ok"}


def test_safe_json_recovers_json_with_trailing_prose():
    raw = '{"intent": "casual"}\nLet me know if that looks right!'
    assert ai._safe_json(raw) == {"intent": "casual"}


def test_safe_json_falls_back_to_clarification_on_genuinely_broken_json():
    raw = "I'm not sure what you mean by that."
    result = ai._safe_json(raw)
    assert result["intent"] == "clarification"
    assert result["clarification_question"] is not None


def test_parse_json_or_none_recovers_json_with_stray_prose():
    raw = 'Here you go: {"activity": "run", "duration_min": 30} -- hope that helps!'
    assert ai._parse_json_or_none(raw) == {"activity": "run", "duration_min": 30}


def test_parse_json_or_none_returns_none_on_genuinely_broken_json():
    assert ai._parse_json_or_none("not json at all") is None


def test_calorie_calibration_note_generalizes_beyond_the_named_dish_list():
    """The old wording only biased a short named-dish list upward; the
    broadened version has to apply to restaurant/hawker/fried food as a
    general category, not just those examples."""
    note = ai.MEAL_CALORIE_CALIBRATION_NOTE
    assert "restaurant" in note.lower()
    assert "hawker" in note.lower()
    assert "mala" in note.lower()  # the original named examples are kept, not dropped


# ---------- narration model tier: casual conversation and narration calls ----------
# CLAUDE_NARRATION_MODEL exists so genuinely writing (conversation, cross-domain
# narration) can use a stronger model than fast structured extraction needs --
# these tests lock in that extraction calls stay on CLAUDE_MODEL while narration
# calls move to CLAUDE_NARRATION_MODEL, even when a test deliberately sets them
# to two different values (the default has them equal, which wouldn't catch a
# call site accidentally left on the wrong constant).

class _RecordingFakeClient:
    """Like _FakeClient, but also records every kwargs dict passed to
    create() so a test can assert on model/messages without re-deriving
    them from the response alone."""

    def __init__(self, create_return):
        self._create_return = create_return
        self.messages = self
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._create_return, Exception):
            raise self._create_return
        return _FakeResponse(self._create_return)


def _mock_recording_client(monkeypatch, create_return):
    fake = _RecordingFakeClient(create_return)
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    return fake


def test_answer_casually_uses_the_narration_model_not_the_extraction_one(monkeypatch):
    """The whole point of splitting casual conversation out of parse_message:
    it should use CLAUDE_NARRATION_MODEL, not CLAUDE_MODEL -- set them to
    different values here so the default (where they happen to be equal)
    can't hide a call site left on the wrong constant."""
    monkeypatch.setattr(ai.config, "CLAUDE_NARRATION_MODEL", "narration-model-x")
    monkeypatch.setattr(ai.config, "CLAUDE_MODEL", "extraction-model-y")
    fake = _mock_recording_client(monkeypatch, "Hey! Good to hear from you.")
    reply = ai.answer_casually("hey what's up", [], [], {})
    assert reply == "Hey! Good to hear from you."
    assert fake.calls[0]["model"] == "narration-model-x"


def test_answer_casually_includes_conversation_history_memory_and_snapshot(monkeypatch):
    fake = _mock_recording_client(monkeypatch, "noted")
    recent_messages = [{"role": "user", "content": "I hit a new PR on squats today"}]
    memory_list = [{"label": "gym", "category": "plan", "content": "Fitness First Bugis Tue/Thu"}]
    today_snapshot = {"balance": {"available_today": 42.0}, "today": {"meals": {"count": 1}}}
    ai.answer_casually("nice right?", recent_messages, memory_list, today_snapshot)
    sent = fake.calls[0]["messages"][0]["content"]
    assert "new PR on squats" in sent
    assert "Fitness First Bugis" in sent
    assert "42.0" in sent


def test_answer_casually_system_prompt_forbids_claiming_actions():
    """Guards the guardrail itself -- this call has no way to write to the
    DB, so its own system prompt must keep telling the model never to claim
    it did (the exact real bug this discipline exists for: a casual reply
    once falsely claimed to have "corrected" a meal entry that was never
    actually edited)."""
    assert "NEVER claim" in ai.CASUAL_SYSTEM_PROMPT
    assert "performed" in ai.CASUAL_SYSTEM_PROMPT


def test_parse_message_extraction_stays_on_claude_model(monkeypatch):
    """Regression guard for the model-tier split: parse_message is pure
    classification/extraction and must stay on CLAUDE_MODEL even when
    CLAUDE_NARRATION_MODEL is set to something else."""
    monkeypatch.setattr(ai.config, "CLAUDE_NARRATION_MODEL", "narration-model-x")
    monkeypatch.setattr(ai.config, "CLAUDE_MODEL", "extraction-model-y")
    payload = {
        "intent": "casual", "expenses": None, "target_expense_id": None, "correction_action": None,
        "days_ago": None, "new_currency": None, "new_amount": None, "new_description": None,
        "new_category": None, "clarification_question": None, "casual_reply": "hi",
    }
    fake = _mock_recording_client(monkeypatch, json.dumps(payload))
    ai.parse_message("hey", [])
    assert fake.calls[0]["model"] == "extraction-model-y"


def test_answer_with_rundown_uses_the_narration_model(monkeypatch):
    monkeypatch.setattr(ai.config, "CLAUDE_NARRATION_MODEL", "narration-model-x")
    monkeypatch.setattr(ai.config, "CLAUDE_MODEL", "extraction-model-y")
    fake = _mock_recording_client(monkeypatch, "You're doing fine this week.")
    ai.answer_with_rundown({"window_days": 7, "balance": {}, "meals": {"count": 0},
                             "workouts": {"count": 0}, "vitals": {"checkins": 0}})
    assert fake.calls[0]["model"] == "narration-model-x"


# ---------- event correction: reschedule (not delete) ----------
# Regression guards for a real, actively harmful bug: a date-correction
# message against an event ("correct both to 2026-09-23") used to have no
# matching action at all (only "delete" existed for events), and the model
# chose "delete" -- silently destroying real scheduled events instead of
# moving them. See correction.py's EVENT_DOMAIN_ACTIONS for the full story.

def test_parse_system_prompt_offers_reschedule_for_events():
    assert '"reschedule"' in ai.PARSE_SYSTEM_PROMPT
    assert "new_event_in_days" in ai.PARSE_SYSTEM_PROMPT


def test_parse_system_prompt_warns_against_deleting_instead_of_rescheduling():
    """Guards the guardrail itself -- the exact real bug this prevents was
    the model picking 'delete' for a message that was actually asking to
    move an event's date."""
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "2026-09-23" in prompt  # the actual real-world example that broke
    assert "never use \"delete\" as a stand-in for" in prompt


def test_clarify_fallback_includes_new_event_in_days():
    result = ai._clarify_fallback("huh?")
    assert result["new_event_in_days"] is None


# ---------- event reschedule correction: matching the right leg of a batch ----------
# Regression guards for a real observed bug: an itinerary logged several
# events at once (a flight-out date, a return date, a landing date -- one
# add_event row per leg, per the add_event rule), and one leg was logged
# with the wrong date. A reply-style correction naming the leg's OLD (wrong)
# date, e.g. "Saturday morning is October 3 not October 5", got "I'm not
# sure which event you mean" instead of rescheduling the right one --
# the "default to the assistant's last logged item" exception (written
# only for edit_date) doesn't by itself say (a) it also covers reschedule,
# the event-domain equivalent, and (b) how to pick ONE specific item out of
# several the assistant just logged in the same turn, or that a message
# naming the item's OLD/current date is a strong, safe match signal (as
# opposed to matching on the NEW date, which is the real no-op trap the
# original exception warns about).

def test_parse_system_prompt_extends_last_logged_item_exception_to_reschedule():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "IMPORTANT exception for edit_date and" in prompt
    assert "its event-domain equivalent, reschedule" in prompt


def test_parse_system_prompt_prefers_matching_the_old_value_over_the_new_one():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "October 3 not October 5" in prompt  # the actual real-world example that broke
    assert "searching that domain's recent list for the item that currently" in prompt


def test_parse_system_prompt_covers_disambiguating_within_a_just_logged_batch():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "logged MULTIPLE" in prompt
    assert "treat the whole batch as the" in prompt


# ---------- log_lift vs remember: not primed by Morrow's own wording ----------
# Regression guards for a real observed bug: a fully-described completed
# workout got filed as "remember" instead of "log_lift" purely because
# Morrow's own preceding reply had used the word "remember" in a follow-up
# question -- so the real session never landed as queryable/correctable data,
# only a vague memory paragraph. See ai.py's log_lift/remember sections for
# the full story.

def test_parse_system_prompt_warns_log_lift_against_remember_priming():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert 'never "remember", even if Morrow\'s OWN immediately preceding message' in prompt


def test_parse_system_prompt_warns_remember_against_its_own_priming():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert 'never classify a message as "remember" just' in prompt
    assert "Morrow's OWN preceding reply used the word \"remember\"" in prompt


# ---------- casual conversation grounded in real lift data ----------
# Regression guards for a second real observed bug: gym-routine questions
# ("what's my push day at Visa look like") were answered by freely narrating
# from the memory-table prose blob, so exact sets/reps/weight drifted between
# successive near-identical questions in the same conversation -- the same
# "real numbers in, never guessed" violation day_stats/rundown already fixed
# elsewhere. See lifts._recent_lifts_for_narration and ai.answer_casually.

def test_answer_casually_includes_recent_lifts_in_the_prompt(monkeypatch):
    fake = _mock_recording_client(monkeypatch, "Last Visa pull day: pull-ups 10x3.")
    recent_lifts = [{"exercise": "pull-ups", "location": "Visa gym",
                      "sets": [{"reps": 10, "load": None}] * 3, "effort": None,
                      "context_notes": None, "lift_date": "2026-09-22"}]
    ai.answer_casually("what's my push day at visa look like", [], [], {}, recent_lifts)
    sent = fake.calls[0]["messages"][0]["content"]
    assert "Visa gym" in sent
    assert "pull-ups" in sent


def test_answer_casually_recent_lifts_defaults_to_empty_list(monkeypatch):
    """Backward-compat: existing call sites that don't pass recent_lifts at
    all must keep working, same as every other optional param here."""
    fake = _mock_recording_client(monkeypatch, "hey!")
    reply = ai.answer_casually("hi", [], [], {})
    assert reply == "hey!"
    assert "recent_lifts:\n[]" in fake.calls[0]["messages"][0]["content"]


def test_casual_system_prompt_grounds_lift_questions_in_real_data():
    prompt = ai.CASUAL_SYSTEM_PROMPT
    assert "recent_lifts" in prompt
    assert "never in the durable-memory prose" in prompt


# ---------- literal backslash-n formatting bug ----------
# Regression guard for a real observed bug: at least two of Morrow's replies
# contained the literal two characters "\n" as visible text instead of an
# actual line break, likely from the model echoing JSON-serialized
# conversation history verbatim. See tg_html.py's normalization fix and this
# prompt-level instruction (defense in depth, same pattern as the event
# reschedule fix -- a prompt instruction alone already failed once).

def test_telegram_formatting_note_warns_against_literal_backslash_n():
    assert "backslash-n" in ai.TELEGRAM_FORMATTING_NOTE
    assert "never the two literal characters" in ai.TELEGRAM_FORMATTING_NOTE


def test_answer_with_day_stats_uses_the_narration_model(monkeypatch):
    monkeypatch.setattr(ai.config, "CLAUDE_NARRATION_MODEL", "narration-model-x")
    monkeypatch.setattr(ai.config, "CLAUDE_MODEL", "extraction-model-y")
    fake = _mock_recording_client(monkeypatch, "Quiet day.")
    ai.answer_with_day_stats({"day": db.today_str(), "is_today": True, "meals": {"count": 0},
                               "workouts": {"count": 0}, "net_calories": None, "vitals": None})
    assert fake.calls[0]["model"] == "narration-model-x"


# ---------- subscriptions: frequency/renewal-date schema (not billing_day) ----------
# Regression guards for the Round 3 schema redesign -- Andrew's real
# subscriptions list included quarterly/annual/biweekly items the old
# monthly-only billing_day design couldn't represent at all (see db.py's
# "subscriptions" section docstring), plus card/notes/is_claimable detail
# he explicitly asked to track for future analytics.

def test_parse_system_prompt_log_subscription_uses_frequency_not_billing_day():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "billing_day" not in prompt
    assert '"renews_in_days"' in prompt
    assert '"frequency": "weekly" | "biweekly" | "monthly" | "quarterly" | "annual"' in prompt


def test_parse_system_prompt_log_subscription_covers_card_notes_claimable():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert '"card": short string or null' in prompt
    assert '"is_claimable": true or null' in prompt


def test_parse_system_prompt_edit_subscription_uses_new_schema_fields():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "new_subscription_frequency" in prompt
    assert "new_subscription_renews_in_days" in prompt
    assert "new_subscription_card" in prompt
    assert "new_subscription_notes" in prompt
    assert "new_subscription_is_claimable" in prompt


# ---------- recurring tasks (not just daily reminders) ----------
# Regression guards for the other Round 3 feature: a to-do can now repeat
# at weekly/biweekly/monthly/quarterly/annual cadences (not just the
# existing daily-only add_reminder), e.g. "add a task to claim my gym
# membership every month". See db.py's RECURRENCE_FREQUENCIES and
# mark_task_done's cycle-advance behavior.

def test_parse_system_prompt_log_task_has_a_recurrence_field():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert '"recurrence": "weekly" | "biweekly" | "monthly" | "quarterly" | "annual" | null' in prompt


def test_parse_system_prompt_edit_task_has_recurrence_correction_fields():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "new_task_recurrence" in prompt
    assert "remove_recurrence" in prompt


def test_extract_task_system_prompt_also_covers_recurrence():
    """/addtask (extract_task) should have the same recurrence capability as
    the natural-language log_task path, not a narrower one."""
    assert "recurrence" in ai.TASK_EXTRACT_SYSTEM_PROMPT


# ---------- casual conversation grounded in a real month-to-date total ----------
# Regression guards for a real gap: "what's my total monthly expenditure" /
# "how much have I spent this month" had no reliable real-number answer --
# it risked landing on "trend" (which only computes a first/last/min/max/
# change trajectory, the wrong SHAPE of answer for a flat total) or on
# "casual" with no month-to-date figure in today_snapshot to ground it.
# Fixed by adding db.get_month_to_date_total to today_snapshot (see
# handlers._casual_reply_text) and a CRITICAL classification note steering
# plain total/sum questions to "casual" instead of "trend".

def test_casual_system_prompt_describes_month_to_date_snapshot_field():
    prompt = ai.CASUAL_SYSTEM_PROMPT
    assert '"month_to_date"' in prompt
    assert "what's my total spend this month" in prompt


def test_parse_system_prompt_routes_a_plain_total_question_to_casual_not_trend():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "how much have I spent this month" in prompt
    assert "is NOT \"trend\", even when it's about spending" in prompt


def test_answer_casually_includes_month_to_date_in_the_prompt(monkeypatch):
    fake = _mock_recording_client(monkeypatch, "You've spent $420.69 so far this month.")
    today_snapshot = {"month_to_date": {"total": 420.69, "days_elapsed": 12, "month_start": "2026-10-01"}}
    ai.answer_casually("what's my total spend this month?", [], [], today_snapshot)
    sent = fake.calls[0]["messages"][0]["content"]
    assert "420.69" in sent


# ---------- narrate_reply: restyle only, never invent ----------
#
# Regression guards for a real, confirmed production bug: a message that
# mixed a weight check-in with a food mention ("Check in 75.3 kg this
# morning, breakfast was half a kaya toast and iced latte home made") only
# ever logged the vitals check-in (log_vitals is a single intent -- the food
# mention never reached db.add_meal). The deterministic reply handed to
# narrate_reply correctly said only "Logged: #85 75.3kg (2026-10-01)" -- but
# narrate_reply, seeing the user's own breakfast mention in recent_messages,
# fabricated an entire extra paragraph confirming a fake meal log with an
# invented calorie estimate AND an invented "Today's running total: ~415
# kcal", none of which was ever real (no meal row was ever inserted). Andrew
# only discovered this hours later when the real, deterministic daily total
# didn't match what he'd been told. See NARRATE_REPLY_SYSTEM_PROMPT's rule 4
# for the fix -- an explicit prohibition using this exact incident.

def test_narrate_reply_system_prompt_forbids_inventing_an_unlogged_item():
    prompt = ai.NARRATE_REPLY_SYSTEM_PROMPT
    assert "Do NOT fill that gap" in prompt
    assert "Today's running total: ~415 kcal" in prompt  # the real fabricated number, pinned as a negative example


def test_narrate_reply_sends_the_deterministic_text_and_context(monkeypatch):
    fake = _mock_recording_client(monkeypatch, "Logged -- #85 75.3kg, nice drop since last time.")
    today_snapshot = {"balance": {"balance": 10.0}, "today": {}}
    reply = _REAL_NARRATE_REPLY(
        "Logged: #85 75.3kg (2026-10-01)", recent_messages=[{"role": "user", "content": "75.3kg this morning"}],
        memory_list=[], today_snapshot=today_snapshot, recent_lifts=[],
    )
    assert reply == "Logged -- #85 75.3kg, nice drop since last time."
    assert fake.calls[0]["model"] == ai.config.CLAUDE_NARRATION_MODEL
    sent = fake.calls[0]["messages"][0]["content"]
    assert "Logged: #85 75.3kg (2026-10-01)" in sent
    assert "75.3kg this morning" in sent


def test_narrate_reply_never_raises_on_api_failure(monkeypatch):
    """replies._narrate_text is what actually catches this (narrate_reply
    itself has no try/except -- see its own docstring: the caller falls
    back to the deterministic text unchanged), but this pins that
    narrate_reply's real signature/behavior on failure is "raises", not
    "swallows", so a future refactor can't quietly change which layer is
    responsible for the fallback without a test noticing."""
    _mock_client(monkeypatch, RuntimeError("API down"))
    with pytest.raises(RuntimeError):
        _REAL_NARRATE_REPLY("Logged: #85 75.3kg (2026-10-01)")
