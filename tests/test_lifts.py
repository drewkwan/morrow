"""
Tests for the lifts domain (Phase A of the coaching-engine work): db.py CRUD
(including the sets-as-JSON round trip), ai.py's extract_lift resilience and
parse_message's recent_lifts context injection, and bot.py's natural-
language/command logging (single exercise and a multi-exercise session in
one message) plus the edit_date/edit_lift/delete correction surface (see
correction.py's _DOMAIN_OPS and LIFT_DOMAIN_ACTIONS), including undo. Same
discipline as the rest of the suite: no real network, no real Claude calls
(ai._get_client is always mocked), throwaway SQLite per test.
"""

import asyncio
import json

import ai
import bot
import db
from conftest import CHAT
from test_ai import _FakeClient, _mock_client
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------- db.py: lifts ----------

def test_add_and_get_lift_roundtrip():
    sets = [{"reps": 10, "load": None}, {"reps": 10, "load": None}, {"reps": 8, "load": None}]
    lift_id = db.add_lift(CHAT, "pull-ups", "AF Wheelock", sets, effort="felt strong")
    row = db.get_lift(CHAT, lift_id)
    assert row["exercise"] == "pull-ups"
    assert row["location"] == "AF Wheelock"
    assert row["sets"] == sets
    assert row["effort"] == "felt strong"
    assert row["lift_date"] == db.today_str()


def test_get_recent_lifts_is_most_recent_first():
    db.add_lift(CHAT, "pull-ups", "AF Wheelock", [{"reps": 10, "load": None}])
    db.add_lift(CHAT, "bench press", "AF Biopolis", [{"reps": 8, "load": "80kg"}])
    exercises = [r["exercise"] for r in db.get_recent_lifts(CHAT)]
    assert exercises == ["bench press", "pull-ups"]


def test_edit_lift_date():
    lift_id = db.add_lift(CHAT, "squat", "AF Wheelock", [{"reps": 5, "load": "90kg"}])
    updated = db.edit_lift_date(CHAT, lift_id, "2026-09-01")
    assert updated["lift_date"] == "2026-09-01"


def test_edit_lift_only_touches_fields_passed():
    """Regression test for a real bad interaction: a mis-typed v-bar-rows
    set (one set logged, a second one missed) had no way to be fixed
    without deleting and relogging the whole exercise. edit_lift's _UNSET
    "only touch what's passed" discipline (same as edit_meal) must leave
    every other field exactly as it was."""
    lift_id = db.add_lift(CHAT, "v bar rows", "Visa gym", [{"reps": 8, "load": "setting 10"}],
                           effort="grindy", context_notes="usually does 3 sets")
    new_sets = [{"reps": 8, "load": "setting 10"}, {"reps": 8, "load": "setting 12"}]
    updated = db.edit_lift(CHAT, lift_id, new_sets=new_sets)
    assert updated["sets"] == new_sets
    assert updated["exercise"] == "v bar rows"  # untouched
    assert updated["location"] == "Visa gym"  # untouched
    assert updated["effort"] == "grindy"  # untouched
    assert updated["context_notes"] == "usually does 3 sets"  # untouched


def test_edit_lift_returns_none_for_missing_lift():
    assert db.edit_lift(CHAT, 999999, new_effort="to failure") is None


def test_delete_and_restore_lift_roundtrip():
    sets = [{"reps": 8, "load": "35kg"}]
    lift_id = db.add_lift(CHAT, "v-bar row", "AF Wheelock", sets, context_notes="no warm-up set")
    deleted = db.delete_lift(CHAT, lift_id)
    assert db.get_lift(CHAT, lift_id) is None
    restored = db.restore_deleted_lift(CHAT, deleted)
    assert restored["exercise"] == "v-bar row"
    assert restored["sets"] == sets
    assert restored["context_notes"] == "no warm-up set"
    assert restored["id"] != lift_id


# ---------- ai.py: extract_lift / parse_message ----------

def test_extract_lift_happy_path(monkeypatch):
    payload = {"exercise": "pull-ups", "location": "AF Wheelock",
               "sets": [{"reps": 10, "load": None}, {"reps": 10, "load": None}, {"reps": 8, "load": None}],
               "effort": None, "context_notes": None}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_lift("pull-ups 10, 10, 8 at wheelock")
    assert result["exercise"] == "pull-ups"
    assert result["location"] == "AF Wheelock"
    assert len(result["sets"]) == 3


def test_extract_lift_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, ConnectionError("network blip"))
    result = ai.extract_lift("bench press")
    assert result["exercise"] == "bench press"  # raw text preserved rather than lost
    assert result["sets"] == [{"reps": None, "load": None}]


def test_parse_message_passes_recent_lifts_into_the_prompt(monkeypatch):
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps({"intent": "casual", "casual_reply": "hey!"}))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    recent_lifts = [{"id": 3, "exercise": "bench press", "location": "AF Biopolis",
                      "sets": [{"reps": 4, "load": "80kg"}], "lift_date": "2026-09-10"}]
    ai.parse_message("hi", [], [], [], [], [], [], [], [], recent_lifts)
    sent = captured["messages"][0]["content"]
    assert "bench press" in sent
    assert "AF Biopolis" in sent


# ---------- bot.py: natural-language, command, correction/undo ----------

def test_natural_language_log_lift_single_exercise(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_lift",
            "lifts": [{"exercise": "pull-ups", "location": "AF Wheelock",
                       "sets": [{"reps": 10, "load": None}, {"reps": 10, "load": None}, {"reps": 8, "load": None}],
                       "effort": "felt strong", "context_notes": None, "logged_days_ago": None}],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="pull-ups 10, 10, 8 at wheelock, felt strong")
    _run(bot.handle_text(update, FakeContext()))
    assert any("pull-ups" in r and "AF Wheelock" in r for r in update.message.replies)
    row = db.get_recent_lifts(CHAT)[0]
    assert row["exercise"] == "pull-ups"
    assert row["location"] == "AF Wheelock"
    assert row["effort"] == "felt strong"


def test_natural_language_log_lift_multiple_exercises_in_one_message(monkeypatch):
    """Regression guard mirroring log_task's multi-item discipline: a whole
    session described in one message ("pull-ups 10x3, then v-bar rows 35kg
    8x3, then lat pulldown 70kg 8x3") must produce three separate lifts
    rows, not one merged or truncated entry."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_lift",
            "lifts": [
                {"exercise": "pull-ups", "location": "AF Wheelock",
                 "sets": [{"reps": 10, "load": None}] * 3, "effort": None, "context_notes": None,
                 "logged_days_ago": None},
                {"exercise": "v-bar row", "location": "AF Wheelock",
                 "sets": [{"reps": 8, "load": "35kg"}] * 3, "effort": None, "context_notes": None,
                 "logged_days_ago": None},
                {"exercise": "lat pulldown", "location": "AF Wheelock",
                 "sets": [{"reps": 8, "load": "70kg"}] * 3, "effort": None, "context_notes": None,
                 "logged_days_ago": None},
            ],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="pull-ups 10x3, then v-bar rows 35kg 8x3, then lat pulldown 70kg 8x3")
    _run(bot.handle_text(update, FakeContext()))
    exercises = {r["exercise"] for r in db.get_recent_lifts(CHAT)}
    assert exercises == {"pull-ups", "v-bar row", "lat pulldown"}
    reply = update.message.replies[0]
    assert "pull-ups" in reply and "v-bar row" in reply and "lat pulldown" in reply


# ---------- richer log confirmations: same-exercise comparison ----------

def test_last_lift_text_compares_against_most_recent_same_exercise():
    db.add_lift(CHAT, "pull-ups", "AF Wheelock",
                [{"reps": 10, "load": None}, {"reps": 10, "load": None}, {"reps": 8, "load": None}])
    new_id = db.add_lift(CHAT, "pull-ups", "AF Wheelock", [{"reps": 12, "load": None}] * 3)
    text = bot._last_lift_text(CHAT, "pull-ups", exclude_id=new_id)
    assert text is not None
    assert "Last time" in text
    assert "10" in text  # the PRIOR session's reps, not the one just logged


def test_last_lift_text_is_case_insensitive_and_ignores_other_exercises():
    db.add_lift(CHAT, "V-Bar Row", "AF Wheelock", [{"reps": 8, "load": "35kg"}] * 3)
    other_id = db.add_lift(CHAT, "lat pulldown", "AF Wheelock", [{"reps": 8, "load": "70kg"}] * 3)
    assert bot._last_lift_text(CHAT, "lat pulldown", exclude_id=other_id) is None  # no prior lat pulldown
    new_id = db.add_lift(CHAT, "v-bar row", "AF Wheelock", [{"reps": 8, "load": "40kg"}] * 3)
    text = bot._last_lift_text(CHAT, "v-bar row", exclude_id=new_id)
    assert text is not None and "35kg" in text  # matched "V-Bar Row" case-insensitively


def test_last_lift_text_returns_none_for_a_brand_new_exercise():
    new_id = db.add_lift(CHAT, "hack squat", "AF Wheelock", [{"reps": 10, "load": "60kg"}] * 3)
    assert bot._last_lift_text(CHAT, "hack squat", exclude_id=new_id) is None


def test_natural_language_log_lift_shows_comparison_to_last_session(monkeypatch):
    """End-to-end: a second pull-ups session, logged naturally, should show
    the prior session's sets alongside the fresh confirmation -- the kind
    of thing a real training partner would notice, not just 'Logged'."""
    db.get_or_create_user(CHAT)
    db.add_lift(CHAT, "pull-ups", "AF Wheelock", [{"reps": 10, "load": None}] * 3, effort="tough")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_lift",
            "lifts": [{"exercise": "pull-ups", "location": "AF Wheelock",
                       "sets": [{"reps": 12, "load": None}] * 3, "effort": "felt easier",
                       "context_notes": None, "logged_days_ago": None}],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="pull-ups 12x3 at wheelock, felt easier")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Logged:" in reply
    assert "Last time" in reply


def test_loglift_command_uses_extract_lift(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "extract_lift", lambda desc: {
        "exercise": "bench press", "location": "AF Biopolis",
        "sets": [{"reps": 4, "load": "80kg"}] * 3, "effort": None, "context_notes": None,
    })
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["bench", "press", "80kg", "4x3", "at", "biopolis"]
    from lifts import loglift_cmd
    _run(loglift_cmd(update, context))
    assert any("bench press" in r for r in update.message.replies)
    row = db.get_recent_lifts(CHAT)[0]
    assert row["exercise"] == "bench press"
    assert row["sets"] == [{"reps": 4, "load": "80kg"}] * 3


def test_recentlifts_command_lists_recent():
    db.get_or_create_user(CHAT)
    db.add_lift(CHAT, "squat", "AF Wheelock", [{"reps": 5, "load": "90kg"}])
    update = FakeUpdate(CHAT)
    from lifts import recentlifts
    _run(recentlifts(update, FakeContext()))
    assert any("squat" in r for r in update.message.replies)


def test_recentlifts_command_with_none_logged():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    from lifts import recentlifts
    _run(recentlifts(update, FakeContext()))
    assert any("No lifts logged yet" in r for r in update.message.replies)


def test_correction_can_delete_a_lift_by_domain(monkeypatch):
    db.get_or_create_user(CHAT)
    lift_id = db.add_lift(CHAT, "pull-ups", "AF Wheelock", [{"reps": 10, "load": None}])

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "lift", "target_expense_id": lift_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="delete that pull-ups set, logged it by mistake")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_lift(CHAT, lift_id) is None
    assert any("Deleted" in r for r in update.message.replies)


def test_undo_reverts_a_lift_deletion(monkeypatch):
    db.get_or_create_user(CHAT)
    lift_id = db.add_lift(CHAT, "bench press", "AF Biopolis", [{"reps": 4, "load": "80kg"}])

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "lift", "target_expense_id": lift_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    delete_update = FakeUpdate(CHAT, text="delete that bench press")
    _run(bot.handle_text(delete_update, context))
    assert db.get_lift(CHAT, lift_id) is None

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_recent_lifts(CHAT)[0]["exercise"] == "bench press"


def test_correction_can_edit_a_lift_date_and_undo(monkeypatch):
    db.get_or_create_user(CHAT)
    lift_id = db.add_lift(CHAT, "squat", "AF Wheelock", [{"reps": 5, "load": "90kg"}])
    original_date = db.get_lift(CHAT, lift_id)["lift_date"]

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "lift", "target_expense_id": lift_id,
            "correction_action": "edit_date", "days_ago": 1,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="that squat was actually yesterday")
    _run(bot.handle_text(edit_update, context))
    assert db.get_lift(CHAT, lift_id)["lift_date"] != original_date

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_lift(CHAT, lift_id)["lift_date"] == original_date


def test_correction_can_edit_a_lifts_sets_and_undo(monkeypatch):
    """Regression test for a real bad interaction: correcting a v-bar-rows
    entry's sets used to have no supported action at all, and the model
    would sometimes wrongly emit correction_action="edit_date" instead
    (producing a nonsense "which day?" question) rather than a proper
    field edit."""
    db.get_or_create_user(CHAT)
    lift_id = db.add_lift(CHAT, "v bar rows", "Visa gym", [{"reps": 8, "load": "setting 10"}])
    original_sets = db.get_lift(CHAT, lift_id)["sets"]
    corrected_sets = [{"reps": 8, "load": "setting 10"}, {"reps": 8, "load": "setting 12"}]

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "lift", "target_expense_id": lift_id,
            "correction_action": "edit_lift", "new_lift_sets": corrected_sets,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="the v bar rows were actually 10x8x1 and 12x8x2")
    _run(bot.handle_text(edit_update, context))
    assert db.get_lift(CHAT, lift_id)["sets"] == corrected_sets

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_lift(CHAT, lift_id)["sets"] == original_sets


# ---------- narration grounding: real data for gym-routine questions ----------
# Regression guards for a real observed bug: gym-routine questions ("what's
# my push day at Visa look like") were answered by freely narrating from the
# memory-table prose blob instead of real logged rows, so exact sets/reps/
# weight drifted between successive near-identical questions in the same
# conversation. See ai.answer_casually and handlers._casual_reply_text.

def test_recent_lifts_for_narration_returns_full_row_detail():
    from lifts import _recent_lifts_for_narration
    db.add_lift(CHAT, "pull-ups", "Visa gym", [{"reps": 10, "load": None}] * 3,
                effort="felt strong", context_notes="no warm-up")
    rows = _recent_lifts_for_narration(CHAT)
    assert len(rows) == 1
    row = rows[0]
    assert row["exercise"] == "pull-ups"
    assert row["location"] == "Visa gym"
    assert row["sets"] == [{"reps": 10, "load": None}] * 3
    assert row["effort"] == "felt strong"
    assert row["context_notes"] == "no warm-up"
    assert row["lift_date"] == db.today_str()


def test_recent_lifts_for_narration_is_most_recent_first_and_respects_limit():
    from lifts import _recent_lifts_for_narration
    for i in range(3):
        db.add_lift(CHAT, f"exercise {i}", "Visa gym", [{"reps": 10, "load": None}])
    rows = _recent_lifts_for_narration(CHAT, limit=2)
    assert len(rows) == 2
    assert rows[0]["exercise"] == "exercise 2"  # most recent first


def test_casual_reply_text_passes_recent_lifts_to_answer_casually(monkeypatch):
    """End-to-end: the 'casual' intent dispatch must ground answer_casually
    in real logged lifts, not just conversation history/memory -- otherwise
    a gym-routine question still gets freely narrated from prose."""
    import handlers

    db.get_or_create_user(CHAT)
    db.add_lift(CHAT, "pull-ups", "Visa gym", [{"reps": 10, "load": None}] * 3)
    captured = {}

    def fake_answer_casually(message, recent_messages=None, memory_list=None, today_snapshot=None,
                              recent_lifts=None):
        captured["recent_lifts"] = recent_lifts
        return "Last Visa pull day was pull-ups 10x3."

    monkeypatch.setattr(handlers.ai, "answer_casually", fake_answer_casually)
    _run(handlers._casual_reply_text(CHAT, "what's my push day at visa look like", [], [], None))
    assert captured["recent_lifts"]
    assert captured["recent_lifts"][0]["exercise"] == "pull-ups"


def test_correction_edit_lift_with_nothing_set_asks_what_to_fix(monkeypatch):
    db.get_or_create_user(CHAT)
    lift_id = db.add_lift(CHAT, "squat", "Visa gym", [{"reps": 5, "load": "90kg"}])

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "lift", "target_expense_id": lift_id,
            "correction_action": "edit_lift",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="fix that squat")
    _run(bot.handle_text(update, FakeContext()))
    assert "What should I fix" in update.message.replies[-1]
