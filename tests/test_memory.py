"""
Tests for the durable-memory + rolling-conversation-history feature: db.py's
messages/memory CRUD, ai.py's context-injection for parse_message, and
bot.py's "remember"/"forget"/"show_memory" intents plus the /memory and
/forget commands -- including undo. Same discipline as
test_meals_workouts.py / test_vitals.py: no real network, no real Claude
calls (ai._get_client is always mocked), throwaway SQLite per test.
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


# ---------- db.py: messages ----------

def test_add_and_get_recent_messages_chronological_order():
    db.add_message(CHAT, "user", "hey")
    db.add_message(CHAT, "morrow", "hi there!")
    db.add_message(CHAT, "user", "what's up")
    rows = db.get_recent_messages(CHAT)
    assert [r["content"] for r in rows] == ["hey", "hi there!", "what's up"]
    assert [r["role"] for r in rows] == ["user", "morrow", "user"]


def test_get_recent_messages_respects_limit_and_stays_chronological():
    for i in range(5):
        db.add_message(CHAT, "user", f"msg {i}")
    rows = db.get_recent_messages(CHAT, limit=3)
    # newest 3, but still oldest-first within that window
    assert [r["content"] for r in rows] == ["msg 2", "msg 3", "msg 4"]


# ---------- db.py: memory ----------

def test_set_and_get_memory_by_label():
    db.set_memory(CHAT, "Tuesday gym", "Fitness First Bugis, legs and back", category="plan")
    row = db.get_memory_by_label(CHAT, "tuesday gym")  # case-insensitive
    assert row["content"] == "Fitness First Bugis, legs and back"
    assert row["category"] == "plan"


def test_set_memory_upserts_by_label_case_insensitively():
    """The whole point of the design: 'remember X' twice edits the same
    row instead of piling up near-duplicates."""
    first_id = db.set_memory(CHAT, "Weight goal", "Hit 75kg by December")
    second_id = db.set_memory(CHAT, "weight goal", "Hit 73kg by December instead")
    assert first_id == second_id
    rows = db.get_memory_list(CHAT)
    assert len(rows) == 1
    assert rows[0]["content"] == "Hit 73kg by December instead"


def test_set_memory_keeps_old_category_when_not_given_again():
    db.set_memory(CHAT, "Diet", "No shellfish", category="preference")
    db.set_memory(CHAT, "Diet", "No shellfish or peanuts")
    row = db.get_memory_by_label(CHAT, "Diet")
    assert row["category"] == "preference"
    assert row["content"] == "No shellfish or peanuts"


def test_get_memory_list_most_recently_updated_first():
    db.set_memory(CHAT, "First", "one")
    db.set_memory(CHAT, "Second", "two")
    db.set_memory(CHAT, "First", "one, updated")  # touches First's updated_at again
    labels = [r["label"] for r in db.get_memory_list(CHAT)]
    assert labels[0] == "First"


def test_delete_memory_by_label_and_restore_roundtrip():
    db.set_memory(CHAT, "Bugis gym", "Tue/Thu legs and back", category="plan")
    deleted = db.delete_memory_by_label(CHAT, "bugis gym")
    assert deleted["content"] == "Tue/Thu legs and back"
    assert db.get_memory_by_label(CHAT, "Bugis gym") is None

    restored = db.restore_deleted_memory(CHAT, deleted)
    assert restored["label"] == "Bugis gym"
    assert restored["content"] == "Tue/Thu legs and back"


def test_delete_memory_by_label_returns_none_when_missing():
    assert db.delete_memory_by_label(CHAT, "nonexistent") is None


# ---------- ai.py: parse_message context injection ----------

def test_parse_message_includes_recent_messages_and_memory_in_the_prompt(monkeypatch):
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps({"intent": "casual", "casual_reply": "hey!"}))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)

    recent_messages = [{"role": "user", "content": "I'm heading to Fitness First Bugis"}]
    memory_list = [{"label": "Bugis gym", "category": "plan", "content": "Tue/Thu legs and back"}]
    ai.parse_message("what's my split today?", [], [], [], [], [], recent_messages, memory_list)
    sent = captured["messages"][0]["content"]
    assert "Fitness First Bugis" in sent
    assert "legs and back" in sent


def test_parse_message_remember_intent_extraction(monkeypatch):
    payload = {
        "intent": "remember", "memory_label": "Bugis gym", "memory_content": "Tue/Thu legs and back",
        "memory_category": "plan", "clarification_question": None, "casual_reply": None,
    }
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.parse_message("remember I go to Fitness First Bugis Tue/Thu for legs and back")
    assert result["intent"] == "remember"
    assert result["memory_label"] == "Bugis gym"


# ---------- bot.py: natural-language remember/forget/show_memory ----------

def _no_op_extra_fields():
    # Deliberately excludes memory_label/memory_content/memory_category --
    # tests that need those set them explicitly in the dict literal, and
    # spreading this AFTER those keys would silently overwrite them back to
    # None (dict-unpacking order bites here, same as it would in ai.py).
    return {
        "amount": None, "currency": None, "description": None, "category": None, "is_claimable": None,
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
    }


def test_natural_language_remember_saves_to_db(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "remember", "memory_label": "Bugis gym", "memory_content": "Tue/Thu legs and back",
            "memory_category": "plan", "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="remember I go to Fitness First Bugis Tue/Thu for legs and back")
    _run(bot.handle_text(update, FakeContext()))
    assert any("Bugis gym" in r for r in update.message.replies)
    row = db.get_memory_by_label(CHAT, "Bugis gym")
    assert row["content"] == "Tue/Thu legs and back"


def test_natural_language_forget_deletes_from_db(monkeypatch):
    db.get_or_create_user(CHAT)
    db.set_memory(CHAT, "Bugis gym", "Tue/Thu legs and back")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "forget", "memory_label": "Bugis gym",
            "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="forget the Bugis gym plan")
    _run(bot.handle_text(update, FakeContext()))
    assert any("Forgot" in r for r in update.message.replies)
    assert db.get_memory_by_label(CHAT, "Bugis gym") is None


def test_undo_reverts_a_forget(monkeypatch):
    db.get_or_create_user(CHAT)
    db.set_memory(CHAT, "Bugis gym", "Tue/Thu legs and back")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "forget", "memory_label": "Bugis gym",
            "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    forget_update = FakeUpdate(CHAT, text="forget the Bugis gym plan")
    _run(bot.handle_text(forget_update, context))
    assert db.get_memory_by_label(CHAT, "Bugis gym") is None

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    restored = db.get_memory_by_label(CHAT, "Bugis gym")
    assert restored is not None
    assert restored["content"] == "Tue/Thu legs and back"


def test_show_memory_answers_directly_with_real_saved_list(monkeypatch):
    """Same discipline as show_balance/show_recent -- the real list, not
    the model's guess at what's remembered."""
    db.get_or_create_user(CHAT)
    db.set_memory(CHAT, "Bugis gym", "Tue/Thu legs and back", category="plan")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "show_memory", "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="what do you remember about me?")
    _run(bot.handle_text(update, FakeContext()))
    assert update.message.replies[-1] == bot._memory_text(CHAT)
    assert "Bugis gym" in update.message.replies[-1]


def test_remember_without_label_or_content_asks_instead_of_saving_junk(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "remember", "memory_label": None, "memory_content": None, "memory_category": None,
            "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="remember that")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_memory_list(CHAT) == []
    assert "remember" in update.message.replies[-1].lower()


# ---------- bot.py: /memory and /forget commands ----------

def test_memory_command_lists_saved_items():
    db.get_or_create_user(CHAT)
    db.set_memory(CHAT, "Bugis gym", "Tue/Thu legs and back", category="plan")
    update = FakeUpdate(CHAT)
    _run(bot.memory_cmd(update, FakeContext()))
    assert "Bugis gym" in update.message.replies[-1]


def test_memory_command_with_nothing_saved():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(bot.memory_cmd(update, FakeContext()))
    assert "don't have anything saved" in update.message.replies[-1]


def test_forget_command_deletes_and_supports_undo():
    db.get_or_create_user(CHAT)
    db.set_memory(CHAT, "Bugis gym", "Tue/Thu legs and back")
    context = FakeContext()
    context.args = ["Bugis", "gym"]
    update = FakeUpdate(CHAT)
    _run(bot.forget_cmd(update, context))
    assert db.get_memory_by_label(CHAT, "Bugis gym") is None
    assert "Forgot" in update.message.replies[-1]

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_memory_by_label(CHAT, "Bugis gym") is not None


def test_forget_command_with_unknown_label():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["nonexistent"]
    update = FakeUpdate(CHAT)
    _run(bot.forget_cmd(update, context))
    assert "Nothing saved" in update.message.replies[-1]


# ---------- bot.py: rolling conversation history is actually populated ----------

def test_handle_text_logs_both_sides_of_the_conversation(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "casual", "casual_reply": "Hey, how's it going?",
            "clarification_question": None, **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="hey there")
    _run(bot.handle_text(update, FakeContext()))

    rows = db.get_recent_messages(CHAT)
    assert [r["role"] for r in rows] == ["user", "morrow"]
    assert rows[0]["content"] == "hey there"
    assert rows[1]["content"] == "Hey, how's it going?"
