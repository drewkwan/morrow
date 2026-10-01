"""
Tests for the vitals addition: db.py CRUD, ai.py's extraction resilience,
and bot.py's natural-language/command logging and cross-domain correction,
including undo. Same discipline as test_meals_workouts.py: no real network,
no real Claude calls (ai._get_client is always mocked), throwaway SQLite
per test.
"""

import asyncio
import datetime as dt

import ai
import bot
import db
from conftest import CHAT
from test_ai import _mock_client  # reuse the existing fake client
from test_meals_workouts import FakeContext, FakeUpdate  # reuse the fake Telegram objects


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------- db.py: vitals ----------

def test_add_and_get_vitals_roundtrip():
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=5.5, knee_pain=2, notes="felt tight")
    row = db.get_vitals(CHAT, vitals_id)
    assert row["weight_kg"] == 76.6
    assert row["sleep_hours"] == 5.5
    assert row["knee_pain"] == 2
    assert row["notes"] == "felt tight"
    assert row["vitals_date"] == db.today_str()


def test_add_vitals_only_sets_mentioned_fields():
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6)
    row = db.get_vitals(CHAT, vitals_id)
    assert row["weight_kg"] == 76.6
    assert row["sleep_hours"] is None
    assert row["knee_pain"] is None


def test_edit_vitals_date_clamps_future_to_today():
    vitals_id = db.add_vitals(CHAT, weight_kg=76.0)
    future = (dt.date.today() + dt.timedelta(days=5)).isoformat()
    row = db.edit_vitals_date(CHAT, vitals_id, future)
    assert row["vitals_date"] == db.today_str()


def test_edit_vitals_only_touches_fields_passed():
    """Regression test for a real gap: a mis-typed weight/sleep/knee number
    had no way to be fixed without deleting and relogging the whole
    check-in. edit_vitals' _UNSET "only touch what's passed" discipline
    (same as edit_meal/edit_workout/edit_lift) must leave every other field
    exactly as it was."""
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=5.5, knee_pain=2, notes="felt tight")
    updated = db.edit_vitals(CHAT, vitals_id, new_weight_kg=76.0)
    assert updated["weight_kg"] == 76.0
    assert updated["sleep_hours"] == 5.5  # untouched
    assert updated["knee_pain"] == 2  # untouched
    assert updated["notes"] == "felt tight"  # untouched


def test_edit_vitals_returns_none_for_missing_vitals():
    assert db.edit_vitals(CHAT, 999999, new_weight_kg=76.0) is None


def test_delete_and_restore_vitals_roundtrip():
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=5.5, knee_pain=2)
    deleted = db.delete_vitals(CHAT, vitals_id)
    assert db.get_vitals(CHAT, vitals_id) is None
    restored = db.restore_deleted_vitals(CHAT, deleted)
    assert restored["weight_kg"] == 76.6
    assert restored["knee_pain"] == 2
    assert restored["id"] != vitals_id  # fresh id, same as delete_meal's documented behavior


def test_delete_most_recent_vitals():
    db.add_vitals(CHAT, weight_kg=76.6)
    db.add_vitals(CHAT, weight_kg=76.4)
    deleted = db.delete_most_recent_vitals(CHAT)
    assert deleted["weight_kg"] == 76.4
    assert db.get_recent_vitals(CHAT)[0]["weight_kg"] == 76.6


# ---------- ai.py: extract_vitals / parse_message ----------

def test_extract_vitals_happy_path(monkeypatch):
    import json
    payload = {"weight_kg": 76.6, "sleep_hours": 5.5, "knee_pain": 2, "notes": None}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_vitals("weight 76.6, slept 5.5 hours, knee 2/10")
    assert result["weight_kg"] == 76.6
    assert result["knee_pain"] == 2


def test_extract_vitals_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, ConnectionError("network blip"))
    result = ai.extract_vitals("weight 76.6")
    assert result["weight_kg"] is None
    assert result["notes"] == "weight 76.6"  # raw text preserved rather than lost


def test_parse_message_passes_recent_vitals_into_the_prompt(monkeypatch):
    import json
    from test_ai import _FakeClient
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps({"intent": "casual", "casual_reply": "hey!"}))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    recent_vitals = [{"id": 4, "weight_kg": 76.6, "sleep_hours": 5.5,
                       "knee_pain": 2, "vitals_date": "2026-09-02"}]
    ai.parse_message("hi", [], [], [], recent_vitals)
    sent = captured["messages"][0]["content"]
    assert "76.6" in sent


# ---------- bot.py: natural-language, command, and correction/undo ----------

def test_natural_language_log_vitals(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None,
                            recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {"intent": "log_vitals", "weight_kg": 76.6, "sleep_hours": 5.5,
                "knee_pain": 2, "vitals_notes": None,
                "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="weight 76.6, slept 5.5 hours, knee 2/10")
    _run(bot.handle_text(update, FakeContext()))
    assert any("76.6" in r for r in update.message.replies)
    assert db.get_recent_vitals(CHAT)[0]["weight_kg"] == 76.6


# ---------- richer log confirmations: trend vs previous check-in ----------

def test_vitals_trend_text_reports_weight_and_sleep_deltas():
    db.add_vitals(CHAT, weight_kg=77.0, sleep_hours=6.0)
    new_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=7.0)
    row = db.get_vitals(CHAT, new_id)
    text = bot._vitals_trend_text(CHAT, row)
    assert text is not None
    assert "-0.4kg" in text
    assert "+1.0h" in text


def test_vitals_trend_text_skips_a_field_missing_from_either_side():
    """A partial check-in (weight only, no sleep this time) must never get
    a misleading sleep comparison against a field it didn't actually
    report -- same discipline as db.py's edit functions only touching
    fields actually passed."""
    db.add_vitals(CHAT, weight_kg=77.0, sleep_hours=6.0)
    new_id = db.add_vitals(CHAT, weight_kg=76.6)  # no sleep_hours this time
    row = db.get_vitals(CHAT, new_id)
    text = bot._vitals_trend_text(CHAT, row)
    assert text is not None
    assert "weight" in text
    assert "sleep" not in text


def test_vitals_trend_text_returns_none_for_a_first_ever_checkin():
    new_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=7.0)
    row = db.get_vitals(CHAT, new_id)
    assert bot._vitals_trend_text(CHAT, row) is None


def test_vitals_trend_text_ignores_a_backfilled_row_inserted_after_a_newer_one():
    """Regression test for a real reported bug: a historical backfill
    inserts old-dated rows LAST, so they get higher ids than genuinely
    more recent live-logged rows. get_recent_vitals used to order by
    "id DESC" alone, so a July/August backfill row could outrank a real
    September check-in in "recent" -- observed live as the trend line
    comparing a new weigh-in against a stale 11 Sept row instead of a
    real, more recent one from just a day or two before. Reproduces that
    exact id-vs-date mismatch here."""
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date="2026-07-01")  # earliest by id AND date
    db.add_vitals(CHAT, weight_kg=75.0, vitals_date="2026-09-20")  # a genuine, more recent live check-in
    # Simulates the backfill step: inserted LAST (highest id so far), but dated
    # well before the Sept 20 entry above -- the exact id-vs-date mismatch.
    db.add_vitals(CHAT, weight_kg=72.0, vitals_date="2026-07-15")

    new_id = db.add_vitals(CHAT, weight_kg=76.0, vitals_date="2026-09-29")
    row = db.get_vitals(CHAT, new_id)
    text = bot._vitals_trend_text(CHAT, row)

    assert text is not None
    assert "(2026-09-20)" in text  # the real most-recent-by-DATE row, not 07-15 or 07-01
    assert "+1.0kg" in text  # 76.0 - 75.0, i.e. compared against the Sept 20 row


def test_get_recent_vitals_orders_by_date_not_insertion_order():
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date="2026-07-01")
    db.add_vitals(CHAT, weight_kg=75.0, vitals_date="2026-09-20")
    db.add_vitals(CHAT, weight_kg=72.0, vitals_date="2026-07-15")  # inserted last, dated earliest of the three
    dates = [v["vitals_date"] for v in db.get_recent_vitals(CHAT)]
    assert dates == sorted(dates, reverse=True)


def test_natural_language_log_vitals_shows_trend_against_last_checkin(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=77.0, sleep_hours=6.0)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None,
                            recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {"intent": "log_vitals", "weight_kg": 76.6, "sleep_hours": 7.0,
                "knee_pain": None, "vitals_notes": None,
                "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="weight 76.6, slept 7 hours")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Logged:" in reply
    assert "Since last check-in" in reply


def test_natural_language_log_vitals_backdates_with_logged_days_ago(monkeypatch):
    """Same fix as the meal/workout/expense cases: 'I forgot to log it, but
    last night I weighed 76.6' used to always land on today -- see ai.py's
    logged_days_ago rule."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None,
                            recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {"intent": "log_vitals", "weight_kg": 76.6, "sleep_hours": None,
                "knee_pain": None, "vitals_notes": None, "logged_days_ago": 1,
                "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="forgot to log it, but last night I weighed 76.6")
    _run(bot.handle_text(update, FakeContext()))

    expected_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    logged = db.get_recent_vitals(CHAT)[0]
    assert logged["vitals_date"] == expected_date
    assert logged["vitals_date"] != db.today_str()
    assert expected_date in update.message.replies[-1]  # _vitals_line always shows the date


def test_logvitals_command_uses_extract_vitals(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "extract_vitals", lambda desc: {
        "weight_kg": 76.6, "sleep_hours": 5.5, "knee_pain": 2, "notes": None,
    })
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["weight", "76.6,", "knee", "2/10"]
    _run(bot.logvitals_cmd(update, context))
    assert any("76.6" in r for r in update.message.replies)
    assert db.get_recent_vitals(CHAT)[0]["knee_pain"] == 2


def test_correction_can_target_vitals_by_domain(monkeypatch):
    """Cross-domain version of the meal/workout correction tests: a vitals
    check-in can be deleted by natural language too, dispatched via
    target_domain, exercising the shared _DOMAIN_OPS registry path."""
    db.get_or_create_user(CHAT)
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=5.5, knee_pain=2)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None,
                            recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "vitals", "target_expense_id": vitals_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="delete that check-in, logged it twice")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_vitals(CHAT, vitals_id) is None
    assert any("Deleted" in r for r in update.message.replies)


def test_undo_reverts_a_vitals_deletion(monkeypatch):
    db.get_or_create_user(CHAT)
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=5.5, knee_pain=2)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None,
                            recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "vitals", "target_expense_id": vitals_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    delete_update = FakeUpdate(CHAT, text="delete that check-in")
    _run(bot.handle_text(delete_update, context))
    assert db.get_vitals(CHAT, vitals_id) is None

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_recent_vitals(CHAT)[0]["weight_kg"] == 76.6


def test_undo_reverts_a_vitals_date_edit(monkeypatch):
    """Exercises the edit_date branch of _revert_last_correction's
    _DOMAIN_OPS-registry path (delete is covered above; edit_date wasn't
    exercised yet for any of the three simple domains)."""
    db.get_or_create_user(CHAT)
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6)
    original_date = db.get_vitals(CHAT, vitals_id)["vitals_date"]

    def fake_parse_message(text, recent_expenses=None, recent_meals=None,
                            recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "vitals", "target_expense_id": vitals_id,
            "correction_action": "edit_date", "days_ago": 1,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="that check-in was actually yesterday")
    _run(bot.handle_text(edit_update, context))
    assert db.get_vitals(CHAT, vitals_id)["vitals_date"] != original_date

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_vitals(CHAT, vitals_id)["vitals_date"] == original_date


# ---------- correction.py: edit_vitals dispatch ----------

def test_correction_can_edit_vitals_weight_and_undo(monkeypatch):
    """Regression test for a real gap: 'that was 76.0kg not 76.6' used to
    have no supported action at all for a vitals target (only edit_date and
    delete existed), the same shape of bug edit_workout/edit_lift were
    added to fix for their own domains."""
    db.get_or_create_user(CHAT)
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6, sleep_hours=5.5, knee_pain=2)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "vitals", "target_expense_id": vitals_id,
            "correction_action": "edit_vitals", "new_vitals_weight_kg": 76.0,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="that was 76.0kg not 76.6")
    _run(bot.handle_text(edit_update, context))
    assert db.get_vitals(CHAT, vitals_id)["weight_kg"] == 76.0
    assert db.get_vitals(CHAT, vitals_id)["sleep_hours"] == 5.5  # untouched

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_vitals(CHAT, vitals_id)["weight_kg"] == 76.6


def test_correction_edit_vitals_with_nothing_set_asks_what_to_fix(monkeypatch):
    db.get_or_create_user(CHAT)
    vitals_id = db.add_vitals(CHAT, weight_kg=76.6)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "vitals", "target_expense_id": vitals_id,
            "correction_action": "edit_vitals",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="fix that check-in")
    _run(bot.handle_text(update, FakeContext()))
    assert "What should I fix" in update.message.replies[-1]
