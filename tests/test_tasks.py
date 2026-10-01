"""
Tests for the to-dos (tasks) domain: db.py CRUD (including get_open_tasks's
soonest-due-first ordering vs. get_recent_tasks's creation order), ai.py's
extract_task resilience and parse_message's recent_tasks context injection,
and bot.py's natural-language/command logging, show_tasks, and the
mark_done/delete-only correction surface (deliberately narrower than
meal/workout/vitals' edit_date/delete -- see PARSE_SYSTEM_PROMPT's
correction rules for why a forward-looking due date can't reuse days_ago's
backward-only math), including undo. Same discipline as test_vitals.py: no
real network, no real Claude calls (ai._get_client is always mocked),
throwaway SQLite per test.
"""

import asyncio
import datetime as dt
import json

import ai
import bot
import db
from conftest import CHAT
from test_ai import _mock_client, _FakeClient
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _in_days(n):
    return (dt.date.today() + dt.timedelta(days=n)).isoformat()


# ---------- db.py: tasks ----------

def test_add_and_get_task_roundtrip():
    task_id = db.add_task(CHAT, "call the dentist", due_at=_in_days(1), notes="ask about the crown")
    row = db.get_task(CHAT, task_id)
    assert row["title"] == "call the dentist"
    assert row["due_at"] == _in_days(1)
    assert row["notes"] == "ask about the crown"
    assert row["done"] == 0


def test_get_open_tasks_orders_soonest_due_first_with_undated_last():
    db.add_task(CHAT, "far out", due_at=_in_days(10))
    db.add_task(CHAT, "no date")
    db.add_task(CHAT, "soonest", due_at=_in_days(1))
    titles = [r["title"] for r in db.get_open_tasks(CHAT)]
    assert titles == ["soonest", "far out", "no date"]


def test_get_open_tasks_excludes_done():
    task_id = db.add_task(CHAT, "finish report")
    db.mark_task_done(CHAT, task_id)
    assert db.get_open_tasks(CHAT) == []


def test_get_recent_tasks_is_creation_order_desc():
    db.add_task(CHAT, "first")
    db.add_task(CHAT, "second")
    titles = [r["title"] for r in db.get_recent_tasks(CHAT)]
    assert titles == ["second", "first"]


def test_edit_task_due():
    task_id = db.add_task(CHAT, "renew passport")
    updated = db.edit_task_due(CHAT, task_id, _in_days(30))
    assert updated["due_at"] == _in_days(30)


def test_mark_and_unmark_task_done_roundtrip():
    task_id = db.add_task(CHAT, "buy milk")
    done = db.mark_task_done(CHAT, task_id)
    assert done["done"] == 1
    reopened = db.unmark_task_done(CHAT, task_id)
    assert reopened["done"] == 0


def test_delete_and_restore_task_roundtrip():
    task_id = db.add_task(CHAT, "walk the dog", due_at=_in_days(0), notes="evening")
    deleted = db.delete_task(CHAT, task_id)
    assert db.get_task(CHAT, task_id) is None
    restored = db.restore_deleted_task(CHAT, deleted)
    assert restored["title"] == "walk the dog"
    assert restored["notes"] == "evening"
    assert restored["id"] != task_id


# ---------- ai.py: extract_task / parse_message ----------

def test_extract_task_happy_path(monkeypatch):
    payload = {"title": "call the dentist", "due_in_days": 1, "due_time": "17:00", "notes": None}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_task("call the dentist tomorrow 5pm")
    assert result["title"] == "call the dentist"
    assert result["due_in_days"] == 1
    assert result["due_time"] == "17:00"


def test_extract_task_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, ConnectionError("network blip"))
    result = ai.extract_task("call the dentist")
    assert result["title"] == "call the dentist"  # raw text preserved rather than lost
    assert result["due_in_days"] is None


def test_extract_task_tells_the_model_todays_actual_date(monkeypatch):
    """Same root-cause guard as test_ai.py's parse_message version: an
    explicit due date like "due 18 September" needs today's real date to
    compute due_in_days against -- can't be done from a relative word alone."""
    monkeypatch.setattr(db, "today_str", lambda: "2026-09-19")
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps(
        {"title": "renew passport", "due_in_days": None, "due_time": None, "notes": None}
    ))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    ai.extract_task("renew passport")
    sent_content = captured["messages"][0]["content"]
    assert "2026-09-19" in sent_content
    assert "Saturday" in sent_content


def test_parse_message_passes_recent_tasks_into_the_prompt(monkeypatch):
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps({"intent": "casual", "casual_reply": "hey!"}))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    recent_tasks = [{"id": 7, "title": "call the dentist", "due_at": "2026-09-10"}]
    ai.parse_message("hi", [], [], [], [], recent_tasks)
    sent = captured["messages"][0]["content"]
    assert "call the dentist" in sent


# ---------- bot.py: natural-language, command, show_tasks, correction/undo ----------

def test_natural_language_log_task_computes_due_date_deterministically(monkeypatch):
    """The model only ever supplies a day-count -- bot.py, not ai.py, turns
    that into an actual calendar date, same discipline as edit_date/days_ago."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_task",
            "tasks": [{"title": "call the dentist", "due_in_days": 1, "due_time": None, "notes": None}],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="remind me to call the dentist tomorrow")
    _run(bot.handle_text(update, FakeContext()))
    assert any("call the dentist" in r for r in update.message.replies)
    row = db.get_open_tasks(CHAT)[0]
    assert row["title"] == "call the dentist"
    assert row["due_at"] == _in_days(1)


def test_natural_language_log_task_with_no_due_date_leaves_due_at_null(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_task",
            "tasks": [{"title": "buy milk", "due_in_days": None, "due_time": None, "notes": None}],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="add buy milk to my list")
    _run(bot.handle_text(update, FakeContext()))
    row = db.get_open_tasks(CHAT)[0]
    assert row["due_at"] is None


def test_natural_language_log_task_with_many_todos_logs_every_one(monkeypatch):
    """Regression test for a real bad interaction: a message numbering 13
    separate to-dos only produced one, title-less "to-do" entry -- because
    parse_message's log_task fields used to be singular (one title/due_at
    slot per message) instead of a list like log_expense's "expenses". Every
    named to-do must now be logged, each with its own title, not merged or
    dropped."""
    db.get_or_create_user(CHAT)
    titles = [
        "Book flight from san francisco to new york",
        "Book hotel in New York",
        "Look at New York hotels",
        "Send Shardul my San Francisco schedule",
        "Message Sri about staying with him in san francisco",
        "Message Hubert about scheduling San Francisco hangout",
        "Reply Lana about New York sched",
        "Message Jeremiah about visiting",
        "Message Chris about visiting",
        "Order Simba and Stella's stuff to Shardul's",
        "Schedule Shelby hangout",
        "Cancel IPPT and inform Yan",
        "Patch remaining RMP servers and run regressions",
    ]

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_task",
            "tasks": [{"title": t, "due_in_days": None, "due_time": None, "notes": None} for t in titles],
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="Can you add the following tasks to my tasklist: ...")
    _run(bot.handle_text(update, FakeContext()))

    reply = update.message.replies[-1]
    assert "Added 13 to-dos" in reply
    assert "Patch remaining RMP servers and run regressions" in reply, "the last item must not be dropped"

    open_titles = {r["title"] for r in db.get_open_tasks(CHAT)}
    assert open_titles == set(titles)


def test_addtask_command_uses_extract_task_and_computes_due_time(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "extract_task", lambda desc: {
        "title": "submit the report", "due_in_days": 0, "due_time": "17:00", "notes": None,
    })
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["submit", "the", "report", "by", "5pm"]
    _run(bot.addtask_cmd(update, context))
    assert any("submit the report" in r for r in update.message.replies)
    row = db.get_open_tasks(CHAT)[0]
    assert row["due_at"] == f"{_in_days(0)} 17:00"


def test_natural_language_show_tasks_lists_open_todos(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_task(CHAT, "renew passport")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {"intent": "show_tasks", "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="what's on my list?")
    _run(bot.handle_text(update, FakeContext()))
    assert any("renew passport" in r for r in update.message.replies)


def test_tasks_command_lists_open_todos():
    db.get_or_create_user(CHAT)
    db.add_task(CHAT, "walk the dog")
    update = FakeUpdate(CHAT)
    _run(bot.tasks_cmd(update, FakeContext()))
    assert any("walk the dog" in r for r in update.message.replies)


def test_tasks_command_when_empty():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(bot.tasks_cmd(update, FakeContext()))
    assert any("Nothing on your to-do list" in r for r in update.message.replies)


def test_correction_can_mark_a_task_done_by_domain(monkeypatch):
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "call the dentist")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "task", "target_expense_id": task_id,
            "correction_action": "mark_done", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="I finished calling the dentist")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_task(CHAT, task_id)["done"] == 1
    assert any("Marked done" in r for r in update.message.replies)


def test_correction_can_delete_a_task_by_domain(monkeypatch):
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "duplicate reminder")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "task", "target_expense_id": task_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="delete that reminder, added it twice")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_task(CHAT, task_id) is None
    assert any("Deleted" in r for r in update.message.replies)


def test_task_correction_rejects_unsupported_edit_date_action(monkeypatch):
    """edit_date's backward-only days_ago math is still the wrong tool for a
    to-do's forward-looking due date -- that's what edit_task (due_in_days/
    due_time) is for instead (see correction.py's TASK_DOMAIN_ACTIONS). An
    edit_date correction_action should hit the same 'not sure / not
    supported' branch as a genuinely unmatched id."""
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "renew passport")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "task", "target_expense_id": task_id,
            "correction_action": "edit_date", "days_ago": 1,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="actually that's due tomorrow")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_task(CHAT, task_id)["due_at"] is None  # untouched
    assert any("isn't supported yet" in r for r in update.message.replies)


def test_done_task_is_not_a_valid_correction_target():
    """recent_task_ids (fed to the AI and enforced server-side) come from
    get_open_tasks -- a done task shouldn't be reachable by a fresh
    'mark that done' style correction."""
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "already done")
    db.mark_task_done(CHAT, task_id)
    assert task_id not in {r["id"] for r in bot._recent_tasks_for_ai(CHAT)}


def test_undo_reverts_a_task_mark_done(monkeypatch):
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "call the dentist")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "task", "target_expense_id": task_id,
            "correction_action": "mark_done", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    done_update = FakeUpdate(CHAT, text="finished calling the dentist")
    _run(bot.handle_text(done_update, context))
    assert db.get_task(CHAT, task_id)["done"] == 1

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_task(CHAT, task_id)["done"] == 0


def test_undo_reverts_a_task_deletion(monkeypatch):
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "duplicate reminder")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "task", "target_expense_id": task_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    delete_update = FakeUpdate(CHAT, text="delete that reminder")
    _run(bot.handle_text(delete_update, context))
    assert db.get_task(CHAT, task_id) is None

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_open_tasks(CHAT)[0]["title"] == "duplicate reminder"


def test_done_command_and_undo():
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "buy milk")
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = [str(task_id)]
    _run(bot.done_cmd(update, context))
    assert db.get_task(CHAT, task_id)["done"] == 1
    assert any("Marked done" in r for r in update.message.replies)

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_task(CHAT, task_id)["done"] == 0


def test_done_command_with_unknown_id():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["9999"]
    _run(bot.done_cmd(update, context))
    assert any("Couldn't find" in r for r in update.message.replies)
