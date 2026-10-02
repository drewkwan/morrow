"""
Tests for the scheduled-events domain: db.py CRUD (including
get_upcoming_events' date/time ordering and its "today forward only"
filtering), formatting._event_line, ai.py's extract_event resilience,
events.py's commands (including reschedule) and undo, the natural-language
add_event/show_events intents (single event and a multi-day batch, mirroring
log_task's list discipline), and the morning briefing's "Coming up" section
(including its lookahead-window cutoff). Same discipline as test_tasks.py/
test_reminders.py: no real network, no real Claude calls (ai._get_client is
always mocked), throwaway SQLite per test.
"""

import asyncio
import datetime as dt
import json

import ai
import bot
import db
from conftest import CHAT
from test_ai import _FakeClient, _mock_client
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _in_days(n):
    return (dt.date.today() + dt.timedelta(days=n)).isoformat()


# ---------- db.py: events ----------

def test_add_and_get_event_roundtrip():
    event_id = db.add_event(CHAT, "Dinner with Mel", _in_days(3), event_time="19:00", notes="Their place")
    row = db.get_event(CHAT, event_id)
    assert row["title"] == "Dinner with Mel"
    assert row["event_date"] == _in_days(3)
    assert row["event_time"] == "19:00"
    assert row["notes"] == "Their place"


def test_get_upcoming_events_orders_by_date_then_time():
    db.add_event(CHAT, "later day", _in_days(5))
    db.add_event(CHAT, "same day, later time", _in_days(1), event_time="18:00")
    db.add_event(CHAT, "same day, earlier time", _in_days(1), event_time="09:00")
    db.add_event(CHAT, "same day, no time", _in_days(1))
    titles = [r["title"] for r in db.get_upcoming_events(CHAT)]
    assert titles == ["same day, earlier time", "same day, later time", "same day, no time", "later day"]


def test_get_upcoming_events_excludes_the_past():
    db.add_event(CHAT, "yesterday's thing", (dt.date.today() - dt.timedelta(days=1)).isoformat())
    db.add_event(CHAT, "today's thing", _in_days(0))
    titles = [r["title"] for r in db.get_upcoming_events(CHAT)]
    assert titles == ["today's thing"]


def test_edit_event_date_reschedules():
    event_id = db.add_event(CHAT, "Company Tennis", _in_days(2))
    updated = db.edit_event_date(CHAT, event_id, _in_days(5))
    assert updated["event_date"] == _in_days(5)


def test_delete_and_restore_event_roundtrip():
    event_id = db.add_event(CHAT, "Dentist", _in_days(2), event_time="15:00", notes="bring insurance card")
    deleted = db.delete_event(CHAT, event_id)
    assert db.get_event(CHAT, event_id) is None
    restored = db.restore_deleted_event(CHAT, deleted)
    assert restored["title"] == "Dentist"
    assert restored["event_time"] == "15:00"
    assert restored["notes"] == "bring insurance card"
    assert restored["id"] != event_id


# ---------- formatting._event_line ----------

def test_event_line_includes_time_and_notes_when_present():
    row = {"id": 1, "title": "Dinner with Mel", "event_date": "2026-09-21", "event_time": "19:00",
           "notes": "their place"}
    line = bot._event_line(row)
    assert "#1 Dinner with Mel" in line
    assert "2026-09-21 19:00" in line
    assert "their place" in line


def test_event_line_omits_time_and_notes_when_absent():
    row = {"id": 2, "title": "Pull day", "event_date": "2026-09-21", "event_time": None, "notes": None}
    line = bot._event_line(row)
    assert line == "#2 Pull day (scheduled 2026-09-21)"


# ---------- ai.py: extract_event ----------

def test_extract_event_happy_path(monkeypatch):
    payload = {"title": "Dinner with Mel", "event_in_days": 6, "event_time": "19:00", "notes": None}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_event("dinner with Mel next Monday at 7pm")
    assert result["title"] == "Dinner with Mel"
    assert result["event_in_days"] == 6
    assert result["event_time"] == "19:00"


def test_extract_event_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, ConnectionError("network blip"))
    result = ai.extract_event("dinner with Mel")
    assert result["title"] == "dinner with Mel"  # raw text preserved rather than lost
    assert result["event_in_days"] is None


def test_extract_event_tells_the_model_todays_actual_date(monkeypatch):
    """Same root-cause guard as test_ai.py's parse_message version: an
    explicit event date like "on 18 September" needs today's real date to
    compute event_in_days against -- can't be done from a relative word alone."""
    monkeypatch.setattr(db, "today_str", lambda: "2026-09-19")
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps(
        {"title": "Dinner with Mel", "event_in_days": None, "event_time": None, "notes": None}
    ))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    ai.extract_event("dinner with Mel")
    sent_content = captured["messages"][0]["content"]
    assert "2026-09-19" in sent_content
    assert "Saturday" in sent_content


# ---------- /addevent, /events ----------

def test_addevent_command_uses_extract_event_and_computes_date(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "extract_event", lambda desc: {
        "title": "Dinner with Mel", "event_in_days": 6, "event_time": "19:00", "notes": None,
    })
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["dinner", "with", "Mel", "next", "Monday"]
    _run(bot.addevent_cmd(update, context))
    assert any("Dinner with Mel" in r for r in update.message.replies)
    row = db.get_upcoming_events(CHAT)[0]
    assert row["event_date"] == _in_days(6)
    assert row["event_time"] == "19:00"


def test_addevent_command_with_no_extractable_day_asks_instead_of_guessing(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "extract_event", lambda desc: {
        "title": "Dinner with Mel", "event_in_days": None, "event_time": None, "notes": None,
    })
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["dinner", "with", "Mel"]
    _run(bot.addevent_cmd(update, context))
    assert db.get_upcoming_events(CHAT) == []
    assert any("What day" in r for r in update.message.replies)


def test_events_command_lists_upcoming():
    db.get_or_create_user(CHAT)
    db.add_event(CHAT, "Dinner with Mel", _in_days(3))
    update = FakeUpdate(CHAT)
    _run(bot.events_cmd(update, FakeContext()))
    assert any("Dinner with Mel" in r for r in update.message.replies)


def test_events_command_with_none_set():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(bot.events_cmd(update, FakeContext()))
    assert any("Nothing on your schedule" in r for r in update.message.replies)


# ---------- /rescheduleevent + undo ----------

def test_rescheduleevent_command_and_undo():
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "Company Tennis", _in_days(2))
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = [str(event_id), "5"]
    _run(bot.rescheduleevent_cmd(update, context))
    assert db.get_event(CHAT, event_id)["event_date"] == _in_days(5)
    assert any("Rescheduled" in r for r in update.message.replies)

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_event(CHAT, event_id)["event_date"] == _in_days(2)


def test_rescheduleevent_command_with_unknown_id():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["9999", "1"]
    _run(bot.rescheduleevent_cmd(update, context))
    assert any("Couldn't find" in r for r in update.message.replies)


# ---------- /removeevent + undo ----------

def test_removeevent_command_and_undo():
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "Company Tennis", _in_days(2))
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = [str(event_id)]
    _run(bot.removeevent_cmd(update, context))
    assert db.get_event(CHAT, event_id) is None
    assert any("Removed" in r for r in update.message.replies)

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_upcoming_events(CHAT)[0]["title"] == "Company Tennis"


def test_removeevent_command_with_unknown_id():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["9999"]
    _run(bot.removeevent_cmd(update, context))
    assert any("Couldn't find" in r for r in update.message.replies)


# ---------- natural-language correction: "X is done" clears an event ----------
#
# Regression tests for a real observed bug: an event has no "done" state,
# but "5 is done" against an event id used to confidently misfire as a task
# correction instead (no task with that id existed, so it always fell
# through to the generic, wrong-sounding "I'm not sure which to-do you
# mean"). Fixed by giving the model an upcoming-events list to match
# against and treating "done"/"already happened" as correction_action=
# "delete" for target_domain="event" (see correction.py's EVENT_DOMAIN_ACTIONS
# and ai.py's target_domain="event" prompt rules).

def test_correction_can_clear_an_event_by_domain(monkeypatch):
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "X-ray for foot", _in_days(0), event_time="08:30")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "event", "target_expense_id": event_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="5 is done")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_event(CHAT, event_id) is None
    assert any("Deleted" in r for r in update.message.replies)


def test_undo_reverts_an_event_correction_deletion(monkeypatch):
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "X-ray for foot", _in_days(0), event_time="08:30")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "event", "target_expense_id": event_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    delete_update = FakeUpdate(CHAT, text="5 is done")
    _run(bot.handle_text(delete_update, context))
    assert db.get_event(CHAT, event_id) is None

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_upcoming_events(CHAT)[0]["title"] == "X-ray for foot"


def test_event_correction_with_unmatched_id_says_not_sure(monkeypatch):
    """The id genuinely doesn't match any upcoming event -- a different
    failure mode from "found it, but that action isn't supported" (see the
    next test), and deliberately worded differently (see
    correction.py's _handle_simple_domain_correction)."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "event", "target_expense_id": 9999,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="9999 is done")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "not sure which event" in reply
    assert "Found that" not in reply


def test_event_correction_rejects_edit_date_action(monkeypatch):
    """"edit_date" specifically (backward-looking days_ago math) is still
    NOT a supported action for events, even though "reschedule" (forward-
    looking new_event_in_days -- see the section below) now is -- a
    correction attempting edit_date against an event must hit the "found
    it, but that's not supported" branch, not silently succeed or look like
    the event wasn't found at all."""
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "Company Tennis", _in_days(2))

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "event", "target_expense_id": event_id,
            "correction_action": "edit_date", "days_ago": 0,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="move that to today instead")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Found that event" in reply
    assert "isn't supported yet" in reply
    assert db.get_event(CHAT, event_id)["event_date"] == _in_days(2)  # untouched


# ---------- natural-language correction: "reschedule" moves an event ----------
#
# Regression tests for a real, actively harmful bug: before "reschedule"
# existed, a date-correction message against an event ("correct both to
# 2026-09-23", "push day should be 2026-09-23") had no matching action at
# all -- only "delete" was ever supported for events -- and the model chose
# "delete", silently destroying two real scheduled events instead of moving
# them. See correction.py's EVENT_DOMAIN_ACTIONS and ai.py's
# target_domain="event" prompt rules for the fix.

def test_correction_can_reschedule_an_event_via_natural_language(monkeypatch):
    """The exact real scenario: an event landed on the wrong day and the
    user corrects it by name -- this must move the event, not delete it."""
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "Push day in the office gym", _in_days(1), notes="quick session after work")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "event", "target_expense_id": event_id,
            "correction_action": "reschedule", "new_event_in_days": 0,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="push day should be today, not tomorrow")
    _run(bot.handle_text(update, FakeContext()))

    row = db.get_event(CHAT, event_id)
    assert row is not None  # NOT deleted
    assert row["event_date"] == _in_days(0)
    reply = update.message.replies[-1]
    assert "Rescheduled" in reply
    assert "Deleted" not in reply


def test_undo_reverts_an_event_reschedule_correction(monkeypatch):
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "Coworking session with Scott", _in_days(1), event_time="22:00")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "event", "target_expense_id": event_id,
            "correction_action": "reschedule", "new_event_in_days": 0,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    reschedule_update = FakeUpdate(CHAT, text="correct that to today")
    _run(bot.handle_text(reschedule_update, context))
    assert db.get_event(CHAT, event_id)["event_date"] == _in_days(0)

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_event(CHAT, event_id)["event_date"] == _in_days(1)


def test_event_reschedule_correction_with_no_day_given_asks_which_day(monkeypatch):
    db.get_or_create_user(CHAT)
    event_id = db.add_event(CHAT, "Dinner with Mel", _in_days(2))

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "event", "target_expense_id": event_id,
            "correction_action": "reschedule", "new_event_in_days": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="that date's wrong for dinner with Mel")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "which day" in reply.lower()
    assert db.get_event(CHAT, event_id)["event_date"] == _in_days(2)  # untouched
    assert db.get_event(CHAT, event_id) is not None  # NOT deleted


def test_rescheduling_two_events_in_the_same_conversation_neither_gets_deleted(monkeypatch):
    """The exact shape of the real transcript: two events both corrected to
    the same day, one after another -- both must survive as reschedules,
    not deletes."""
    db.get_or_create_user(CHAT)
    gym_id = db.add_event(CHAT, "Push day in the office gym", _in_days(1))
    coworking_id = db.add_event(CHAT, "Coworking session with Scott", _in_days(1), event_time="22:00")

    responses = iter([
        {"intent": "correction", "target_domain": "event", "target_expense_id": coworking_id,
         "correction_action": "reschedule", "new_event_in_days": 0,
         "clarification_question": None, "casual_reply": None},
        {"intent": "correction", "target_domain": "event", "target_expense_id": gym_id,
         "correction_action": "reschedule", "new_event_in_days": 0,
         "clarification_question": None, "casual_reply": None},
    ])

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return next(responses)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    _run(bot.handle_text(FakeUpdate(CHAT, text="sorry correct both to today"), context))
    _run(bot.handle_text(FakeUpdate(CHAT, text="push day should be today too"), context))

    assert db.get_event(CHAT, gym_id) is not None
    assert db.get_event(CHAT, coworking_id) is not None
    assert db.get_event(CHAT, gym_id)["event_date"] == _in_days(0)
    assert db.get_event(CHAT, coworking_id)["event_date"] == _in_days(0)


# ---------- handlers._recent_events_for_ai ----------

def test_recent_events_for_ai_only_includes_upcoming():
    import handlers
    db.get_or_create_user(CHAT)
    db.add_event(CHAT, "past thing", (dt.date.today() - dt.timedelta(days=1)).isoformat())
    db.add_event(CHAT, "Dinner with Mel", _in_days(3))
    rows = handlers._recent_events_for_ai(CHAT)
    assert [r["event_title"] for r in rows] == ["Dinner with Mel"]
    assert {"id", "event_title", "event_date", "event_time"} <= rows[0].keys()


# ---------- natural language: add_event / show_events ----------

def test_natural_language_add_single_event_computes_date_deterministically(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "add_event",
            "events": [{"event_title": "Dinner with Mel", "event_in_days": 6, "event_time": None,
                        "event_notes": None}],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="dinner with Mel next Monday")
    _run(bot.handle_text(update, FakeContext()))
    assert any("Dinner with Mel" in r for r in update.message.replies)
    row = db.get_upcoming_events(CHAT)[0]
    assert row["event_date"] == _in_days(6)


def test_natural_language_add_event_with_no_events_asks_instead_of_guessing(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {"intent": "add_event", "events": [], "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="add that to my schedule")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_upcoming_events(CHAT) == []
    assert any("didn't catch" in r for r in update.message.replies)


def test_natural_language_add_a_weeks_worth_of_events_logs_every_one(monkeypatch):
    """Mirrors log_task's multi-item regression test -- a week's workout plan
    named day by day must produce one event per day, not one merged or
    dropped entry."""
    db.get_or_create_user(CHAT)
    plan = [
        ("Pull day in the gym", 0), ("Run intervals", 1), ("Company Tennis", 2),
        ("Push day in the gym", 3), ("Easy 5km", 4), ("IPPT time trial", 5), ("Rest", 6),
    ]

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "add_event",
            "events": [{"event_title": t, "event_in_days": n, "event_time": None, "event_notes": None}
                       for t, n in plan],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="ok cool can you log that in to my schedule for the week")
    _run(bot.handle_text(update, FakeContext()))

    reply = update.message.replies[-1]
    assert "Added 7 to your schedule" in reply
    assert "Rest" in reply, "the last item must not be dropped"

    upcoming_titles = {r["title"] for r in db.get_upcoming_events(CHAT)}
    assert upcoming_titles == {t for t, _ in plan}


def test_natural_language_show_events(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_event(CHAT, "Dinner with Mel", _in_days(3))

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {"intent": "show_events", "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="what's on my schedule")
    _run(bot.handle_text(update, FakeContext()))
    assert any("Dinner with Mel" in r for r in update.message.replies)


# ---------- morning briefing integration ----------

def test_morning_briefing_includes_upcoming_events_within_the_lookahead_window():
    db.get_or_create_user(CHAT)
    db.add_event(CHAT, "Dinner with Mel", _in_days(3))
    payload = bot._morning_briefing_payload(CHAT)
    assert [e["title"] for e in payload["events_upcoming"]] == ["Dinner with Mel"]
    text = bot._morning_briefing_text(payload)
    assert "Coming up" in text
    assert "Dinner with Mel" in text


def test_morning_briefing_excludes_events_beyond_the_lookahead_window():
    db.get_or_create_user(CHAT)
    db.add_event(CHAT, "Something far off", _in_days(30))
    payload = bot._morning_briefing_payload(CHAT)
    assert payload["events_upcoming"] == []
    text = bot._morning_briefing_text(payload)
    assert "Coming up" not in text


def test_morning_briefing_omits_the_events_section_entirely_with_nothing_scheduled():
    db.get_or_create_user(CHAT)
    payload = bot._morning_briefing_payload(CHAT)
    text = bot._morning_briefing_text(payload)
    assert "Coming up" not in text
