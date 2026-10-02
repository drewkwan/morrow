"""
Tests for the daily-reminders domain: db.py CRUD (including the
"done today" / restore-tomorrow semantics that make it different from
tasks.py's permanent "done"), reminders.py's commands and undo, the
natural-language add_reminder/show_reminders intents, and the morning
briefing's "still pending today" filtering. Same discipline as
test_tasks.py: no real network, no real Claude calls (ai._get_client is
always mocked), throwaway SQLite per test.
"""

import asyncio
import datetime as dt

import bot
import db
from conftest import CHAT
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------- db.py: daily_reminders ----------

def test_add_and_get_reminder_roundtrip():
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    row = db.get_reminder(CHAT, reminder_id)
    assert row["description"] == "take hair pills"
    assert row["last_done_date"] is None


def test_get_active_reminders_returns_creation_order():
    db.add_reminder(CHAT, "take hair pills")
    db.add_reminder(CHAT, "stretch before bed")
    rows = db.get_active_reminders(CHAT)
    assert [r["description"] for r in rows] == ["take hair pills", "stretch before bed"]


def test_mark_reminder_done_today_sets_todays_date():
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    updated = db.mark_reminder_done_today(CHAT, reminder_id)
    assert updated["last_done_date"] == db.today_str()


def test_unmark_reminder_done_today_clears_it():
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    db.mark_reminder_done_today(CHAT, reminder_id)
    updated = db.unmark_reminder_done_today(CHAT, reminder_id)
    assert updated["last_done_date"] is None


def test_a_reminder_marked_done_yesterday_is_not_done_today():
    """The core semantic that makes this different from a to-do's permanent
    "done": last_done_date from a PRIOR day doesn't count as done today --
    nothing has to explicitly reset it, today_str() just moves on."""
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    with db.get_conn() as conn:
        conn.execute("UPDATE daily_reminders SET last_done_date = ? WHERE id = ?", (yesterday, reminder_id))
    row = db.get_reminder(CHAT, reminder_id)
    assert row["last_done_date"] != db.today_str()


def test_delete_and_restore_reminder_roundtrip():
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    db.mark_reminder_done_today(CHAT, reminder_id)
    deleted = db.delete_reminder(CHAT, reminder_id)
    assert db.get_reminder(CHAT, reminder_id) is None

    restored = db.restore_deleted_reminder(CHAT, deleted)
    assert restored["description"] == "take hair pills"
    assert restored["last_done_date"] == db.today_str()  # carries through delete/restore, not just dropped


# ---------- formatting._reminder_line ----------

def test_reminder_line_tags_done_today():
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    row = db.get_reminder(CHAT, reminder_id)
    assert "[done today]" not in bot._reminder_line(row)

    db.mark_reminder_done_today(CHAT, reminder_id)
    row = db.get_reminder(CHAT, reminder_id)
    assert "[done today]" in bot._reminder_line(row)


# ---------- /addreminder, /reminders ----------

def test_addreminder_command():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["take", "hair", "pills"]
    _run(bot.addreminder_cmd(update, context))
    assert any("take hair pills" in r for r in update.message.replies)
    assert db.get_active_reminders(CHAT)[0]["description"] == "take hair pills"


def test_reminders_command_lists_active_reminders():
    db.get_or_create_user(CHAT)
    db.add_reminder(CHAT, "take hair pills")
    update = FakeUpdate(CHAT)
    _run(bot.reminders_cmd(update, FakeContext()))
    assert any("take hair pills" in r for r in update.message.replies)


def test_reminders_command_with_none_set():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(bot.reminders_cmd(update, FakeContext()))
    assert any("No daily reminders" in r for r in update.message.replies)


# ---------- /donereminder + undo ----------

def test_donereminder_command_and_undo():
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = [str(reminder_id)]
    _run(bot.donereminder_cmd(update, context))
    assert db.get_reminder(CHAT, reminder_id)["last_done_date"] == db.today_str()
    assert any("Marked done for today" in r for r in update.message.replies)

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_reminder(CHAT, reminder_id)["last_done_date"] is None


def test_donereminder_command_with_unknown_id():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["9999"]
    _run(bot.donereminder_cmd(update, context))
    assert any("Couldn't find" in r for r in update.message.replies)


# ---------- /removereminder + undo ----------

def test_removereminder_command_and_undo():
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = [str(reminder_id)]
    _run(bot.removereminder_cmd(update, context))
    assert db.get_reminder(CHAT, reminder_id) is None
    assert any("Removed" in r for r in update.message.replies)

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_active_reminders(CHAT)[0]["description"] == "take hair pills"


def test_removereminder_command_with_unknown_id():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["9999"]
    _run(bot.removereminder_cmd(update, context))
    assert any("Couldn't find" in r for r in update.message.replies)


# ---------- natural language: add_reminder / show_reminders ----------

def test_natural_language_add_reminder(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "add_reminder", "reminder_description": "take hair pills",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="remind me every day to take my hair pills")
    _run(bot.handle_text(update, FakeContext()))
    assert any("take hair pills" in r for r in update.message.replies)
    assert db.get_active_reminders(CHAT)[0]["description"] == "take hair pills"


def test_natural_language_add_reminder_without_a_description_asks_instead_of_guessing(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "add_reminder", "reminder_description": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="remind me every day")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_active_reminders(CHAT) == []
    assert any("What should I remind you" in r for r in update.message.replies)


def test_natural_language_show_reminders(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_reminder(CHAT, "take hair pills")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "show_reminders",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="what are my daily reminders")
    _run(bot.handle_text(update, FakeContext()))
    assert any("take hair pills" in r for r in update.message.replies)


# ---------- morning briefing integration ----------

def test_morning_briefing_includes_pending_reminders():
    db.get_or_create_user(CHAT)
    db.add_reminder(CHAT, "take hair pills")
    payload = bot._morning_briefing_payload(CHAT)
    assert [r["description"] for r in payload["reminders_pending"]] == ["take hair pills"]
    text = bot._morning_briefing_text(payload)
    assert "Daily reminders still to do today" in text
    assert "take hair pills" in text


def test_morning_briefing_omits_a_reminder_already_done_today():
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    db.mark_reminder_done_today(CHAT, reminder_id)
    payload = bot._morning_briefing_payload(CHAT)
    assert payload["reminders_pending"] == []
    text = bot._morning_briefing_text(payload)
    assert "Daily reminders" not in text


def test_morning_briefing_omits_the_section_entirely_with_no_reminders_set():
    db.get_or_create_user(CHAT)
    payload = bot._morning_briefing_payload(CHAT)
    text = bot._morning_briefing_text(payload)
    assert "Daily reminders" not in text


def test_a_reminder_done_today_is_pending_again_the_next_day(monkeypatch):
    """The whole point of a DAILY reminder, as opposed to a to-do: once a
    new local day starts (see db._now_local_date, BOT_TIMEZONE-aware), it's
    pending again automatically -- no separate rollover job resets it."""
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")
    db.mark_reminder_done_today(CHAT, reminder_id)
    assert bot._morning_briefing_payload(CHAT)["reminders_pending"] == []

    tomorrow = dt.date.today() + dt.timedelta(days=1)
    monkeypatch.setattr(db, "_now_local_date", lambda: tomorrow)
    payload = bot._morning_briefing_payload(CHAT)
    assert [r["description"] for r in payload["reminders_pending"]] == ["take hair pills"]


# ---------- natural language: correction (mark_done / delete) ----------
#
# Regression coverage for a real, explicit ask: "I don't want to have to do
# fuckin telegram commands, I want to do my shit in plain chat text" --
# mark_done/delete used to be /donereminder-/removereminder-only (see this
# module's own docstring for why that was a deliberate, temporary scope cut).
# These mirror test_tasks.py's NL mark_done/delete correction tests, but
# target_domain="reminder" instead of "task", and confirm the EXISTING
# _revert_last_correction "reminder" branch (built for the slash commands
# above) also covers the natural-language path without any changes of its
# own -- same snapshot shape, same undo code.

def _reminder_correction(target_id, action, **extra):
    return {
        "intent": "correction", "target_domain": "reminder", "target_expense_id": target_id,
        "correction_action": action,
        "clarification_question": None, "casual_reply": None,
        **extra,
    }


def test_natural_language_mark_reminder_done_for_today(monkeypatch):
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        assert recent_reminders == [{"id": reminder_id, "description": "take hair pills", "last_done_date": None}]
        return _reminder_correction(reminder_id, "mark_done")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="took my hair pills")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_reminder(CHAT, reminder_id)["last_done_date"] == db.today_str()
    assert any("Marked done for today" in r for r in update.message.replies)


def test_natural_language_mark_reminder_done_undoes_back_to_pending(monkeypatch):
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _reminder_correction(reminder_id, "mark_done")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    _run(bot.handle_text(FakeUpdate(CHAT, text="took my hair pills"), context))
    assert db.get_reminder(CHAT, reminder_id)["last_done_date"] == db.today_str()

    _run(bot.handle_text(FakeUpdate(CHAT, text="undo"), context))
    assert db.get_reminder(CHAT, reminder_id)["last_done_date"] is None


def test_natural_language_delete_reminder(monkeypatch):
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _reminder_correction(reminder_id, "delete")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="stop reminding me to take hair pills")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_reminder(CHAT, reminder_id) is None
    assert any("Removed" in r for r in update.message.replies)


def test_natural_language_delete_reminder_undoes_the_removal(monkeypatch):
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _reminder_correction(reminder_id, "delete")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    _run(bot.handle_text(FakeUpdate(CHAT, text="stop reminding me to take hair pills"), context))
    assert db.get_reminder(CHAT, reminder_id) is None

    _run(bot.handle_text(FakeUpdate(CHAT, text="undo"), context))
    assert db.get_active_reminders(CHAT)[0]["description"] == "take hair pills"


def test_natural_language_reminder_correction_with_unknown_id_asks_instead_of_guessing(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_reminder(CHAT, "take hair pills")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _reminder_correction(9999, "mark_done")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="done with that other reminder")
    _run(bot.handle_text(update, FakeContext()))
    assert any("not sure which daily reminder" in r for r in update.message.replies)


def test_natural_language_reminder_correction_rejects_an_unsupported_action(monkeypatch):
    """Reminders only support mark_done/delete -- no edit_date/edit_task-
    style flexible edit yet (see _handle_reminder_correction's docstring).
    A stray/unexpected correction_action should ask rather than silently
    doing nothing or crashing on a KeyError."""
    db.get_or_create_user(CHAT)
    reminder_id = db.add_reminder(CHAT, "take hair pills")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _reminder_correction(reminder_id, "edit_date")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="that reminder should be dated yesterday")
    _run(bot.handle_text(update, FakeContext()))
    assert any("isn't supported yet" in r for r in update.message.replies)
    assert db.get_reminder(CHAT, reminder_id)["last_done_date"] is None  # untouched
