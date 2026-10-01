"""
Tests for recurring monthly subscriptions: db.py's subscriptions CRUD +
day_matches_billing_day, subscriptions.py's commands, and
subscriptions_tick (the daily auto-logging job) -- including its
idempotency guard (never double-charges the same billing cycle) and its
billing-day clamping for months shorter than the configured day.
"""

import asyncio
import datetime as dt

import bot
import db
import subscriptions
from conftest import CHAT
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeBot:
    """Minimal stand-in for context.bot, just enough for
    subscriptions_tick's context.bot.send_message(...) calls."""
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


class FakeTickContext:
    def __init__(self):
        self.bot = FakeBot()


# ---------- db.py: day_matches_billing_day ----------

def test_day_matches_billing_day_exact():
    assert db.day_matches_billing_day(dt.date(2026, 10, 25), 25) is True
    assert db.day_matches_billing_day(dt.date(2026, 10, 24), 25) is False


def test_day_matches_billing_day_clamps_to_last_day_of_short_month():
    """A billing_day of 31 doesn't exist in a 30-day month -- it should
    clamp to that month's real last day rather than silently never firing
    that month."""
    assert db.day_matches_billing_day(dt.date(2026, 4, 30), 31) is True
    assert db.day_matches_billing_day(dt.date(2026, 4, 29), 31) is False


def test_day_matches_billing_day_clamps_for_february():
    assert db.day_matches_billing_day(dt.date(2026, 2, 28), 31) is True  # 2026 is not a leap year


# ---------- db.py: subscriptions CRUD ----------

def test_add_and_get_subscription():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25, category="Entertainment")
    row = db.get_subscription(CHAT, sub_id)
    assert row["name"] == "Netflix"
    assert row["amount"] == 15.98
    assert row["billing_day"] == 25
    assert row["category"] == "Entertainment"
    assert row["active"] == 1
    assert row["last_logged_month"] is None


def test_get_subscriptions_orders_by_billing_day_then_name():
    db.add_subscription(CHAT, "Gym", 80, "SGD", 1)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    db.add_subscription(CHAT, "Spotify", 11.98, "SGD", 1)
    names = [s["name"] for s in db.get_subscriptions(CHAT)]
    assert names == ["Gym", "Spotify", "Netflix"]


def test_delete_subscription_removes_it():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    row = db.delete_subscription(CHAT, sub_id)
    assert row["name"] == "Netflix"
    assert db.get_subscription(CHAT, sub_id) is None


def test_delete_subscription_returns_none_for_unknown_id():
    assert db.delete_subscription(CHAT, 9999) is None


def test_mark_subscription_logged_sets_month():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    db.mark_subscription_logged(sub_id, "2026-10")
    assert db.get_subscription(CHAT, sub_id)["last_logged_month"] == "2026-10"


def test_edit_subscription_only_touches_the_given_field():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25, category="Entertainment")
    updated = db.edit_subscription(CHAT, sub_id, new_amount=17.98)
    assert updated["amount"] == 17.98
    assert updated["name"] == "Netflix"  # untouched
    assert updated["billing_day"] == 25  # untouched
    assert updated["category"] == "Entertainment"  # untouched


def test_edit_subscription_can_change_several_fields_at_once():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    updated = db.edit_subscription(CHAT, sub_id, new_name="Netflix Premium", new_billing_day=5)
    assert updated["name"] == "Netflix Premium"
    assert updated["billing_day"] == 5
    assert updated["amount"] == 15.98  # untouched


def test_edit_subscription_clamps_billing_day():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    updated = db.edit_subscription(CHAT, sub_id, new_billing_day=45)
    assert updated["billing_day"] == 31


def test_edit_subscription_never_touches_last_logged_month():
    """A price/billing-day correction must not retroactively re-trigger (or
    re-skip) this month's auto-post -- see db.edit_subscription's
    docstring."""
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    db.mark_subscription_logged(sub_id, "2026-10")
    db.edit_subscription(CHAT, sub_id, new_amount=17.98)
    assert db.get_subscription(CHAT, sub_id)["last_logged_month"] == "2026-10"


def test_edit_subscription_returns_none_for_unknown_id():
    assert db.edit_subscription(CHAT, 9999, new_amount=17.98) is None


def test_restore_deleted_subscription_brings_it_back():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25, category="Entertainment")
    db.mark_subscription_logged(sub_id, "2026-10")
    row = db.delete_subscription(CHAT, sub_id)
    restored = db.restore_deleted_subscription(CHAT, row)
    assert restored["name"] == "Netflix"
    assert restored["billing_day"] == 25
    # Preserving this matters: undoing a delete made right after this
    # month's auto-post must not make it look un-logged and eligible to
    # double-charge the same cycle on the next tick.
    assert restored["last_logged_month"] == "2026-10"


# ---------- subscriptions_tick ----------

def test_subscriptions_tick_auto_logs_on_billing_day():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", dt.date.today().day, category="Entertainment")
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    expenses = db.get_recent_expenses(CHAT)
    assert len(expenses) == 1
    assert expenses[0]["description"] == "Netflix"
    assert expenses[0]["amount"] == 15.98
    assert expenses[0]["category"] == "Entertainment"
    assert len(context.bot.sent) == 1


def test_subscriptions_tick_is_idempotent_within_the_same_month():
    """Regression guard: a bot restart or a job re-running the same day
    must never double-charge the same subscription in one billing cycle."""
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    db.mark_subscription_logged(sub_id, dt.date.today().strftime("%Y-%m"))
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    assert db.get_recent_expenses(CHAT) == []


def test_subscriptions_tick_skips_subscriptions_not_due_today():
    db.get_or_create_user(CHAT)
    not_today = 1 if dt.date.today().day != 1 else 15
    db.add_subscription(CHAT, "Gym", 80, "SGD", not_today)
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    assert db.get_recent_expenses(CHAT) == []


def test_subscriptions_tick_notifies_the_chat():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", dt.date.today().day)
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    assert len(context.bot.sent) == 1
    assert context.bot.sent[0][0] == CHAT
    assert "Netflix" in context.bot.sent[0][1]


# ---------- commands ----------

def test_addsubscription_cmd_logs_and_replies():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Netflix", "15.98", "25", "Entertainment"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 1
    assert rows[0]["name"] == "Netflix"
    assert rows[0]["billing_day"] == 25
    assert "Netflix" in update.message.replies[-1]


def test_addsubscription_cmd_accepts_currency():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Spotify", "9.99", "USD", "1"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert db.get_subscriptions(CHAT)[0]["currency"] == "USD"


def test_addsubscription_cmd_rejects_bad_amount():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Netflix", "banana", "25"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert "Usage:" in update.message.replies[-1] or "amount" in update.message.replies[-1].lower()


def test_addsubscription_cmd_rejects_out_of_range_billing_day():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Netflix", "15.98", "45"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert "between 1 and 31" in update.message.replies[-1]


def test_subscriptions_cmd_lists_and_totals():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    db.add_subscription(CHAT, "Gym", 80, "SGD", 1)
    update = FakeUpdate(CHAT)
    _run(subscriptions.subscriptions_cmd(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Netflix" in reply
    assert "Gym" in reply
    assert "95.98" in reply  # 15.98 + 80


def test_subscriptions_cmd_empty_state():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(subscriptions.subscriptions_cmd(update, FakeContext()))
    assert "No active subscriptions" in update.message.replies[-1]


def test_removesubscription_cmd_removes_it():
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)
    context = FakeContext()
    context.args = [str(sub_id)]
    update = FakeUpdate(CHAT)
    _run(subscriptions.removesubscription_cmd(update, context))
    assert db.get_subscription(CHAT, sub_id) is None
    assert "Netflix" in update.message.replies[-1]


def test_removesubscription_cmd_unknown_id():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["9999"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.removesubscription_cmd(update, context))
    assert "don't have" in update.message.replies[-1]


# ---------- correction.py: editing/undoing a subscription ----------

def test_correction_can_edit_a_subscription_and_undo(monkeypatch):
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25, category="Entertainment")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "subscription", "target_expense_id": sub_id,
            "correction_action": "edit_subscription", "new_subscription_amount": 17.98,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="Netflix went up to $17.98")
    _run(bot.handle_text(edit_update, context))
    assert db.get_subscription(CHAT, sub_id)["amount"] == 17.98

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_subscription(CHAT, sub_id)["amount"] == 15.98


def test_correction_edit_subscription_with_nothing_set_asks_what_to_fix(monkeypatch):
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "subscription", "target_expense_id": sub_id,
            "correction_action": "edit_subscription",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="fix that subscription")
    _run(bot.handle_text(update, FakeContext()))
    assert "What should I fix" in update.message.replies[-1]


def test_correction_subscription_has_no_edit_date_action(monkeypatch):
    """A subscription has no logged date (just a billing_day, which is part
    of edit_subscription) -- edit_date must be reported as an unsupported
    action, not silently misapplied."""
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", 25)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "subscription", "target_expense_id": sub_id,
            "correction_action": "edit_date", "days_ago": 1,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="move that to yesterday")
    _run(bot.handle_text(update, FakeContext()))
    assert "isn't supported yet" in update.message.replies[-1]


# ---------- natural-language "log_subscription" intent ----------

def _fake_parse_message_subscription(items):
    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_subscription", "subscriptions": items,
            "clarification_question": None, "casual_reply": None,
        }
    return fake_parse_message


def test_natural_language_log_subscription_single_item(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_subscription([{"name": "Netflix", "amount": 15.98, "currency": "SGD",
                                            "billing_day": 25, "category": "Entertainment"}])
    )
    update = FakeUpdate(CHAT, text="I pay $15.98 for Netflix on the 25th")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 1
    assert rows[0]["name"] == "Netflix"
    assert rows[0]["billing_day"] == 25
    assert "Netflix" in update.message.replies[-1]


def test_natural_language_log_subscription_bulk_list_in_one_message(monkeypatch):
    """The real point of this intent: registering a whole starter list at
    once instead of one /addsubscription per line."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_subscription([
            {"name": "Netflix", "amount": 15.98, "currency": "SGD", "billing_day": 25},
            {"name": "Spotify", "amount": 11.98, "currency": "SGD", "billing_day": 1},
            {"name": "Gym", "amount": 80, "currency": "SGD", "billing_day": 1},
        ])
    )
    update = FakeUpdate(CHAT, text="Netflix 15.98 the 25th, Spotify 11.98 the 1st, gym 80 the 1st")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 3
    assert "3 subscriptions" in update.message.replies[-1]


def test_natural_language_log_subscription_skips_item_missing_billing_day_but_keeps_the_rest(monkeypatch):
    """A big pasted list with one bad line shouldn't lose the rest of it --
    see _log_subscriptions_and_reply's docstring."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_subscription([
            {"name": "Netflix", "amount": 15.98, "currency": "SGD", "billing_day": 25},
            {"name": "Mystery Charge", "amount": 9.99, "currency": "SGD", "billing_day": None},
        ])
    )
    update = FakeUpdate(CHAT, text="Netflix 15.98 the 25th, and some mystery charge for 9.99")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 1
    assert rows[0]["name"] == "Netflix"
    reply = update.message.replies[-1]
    assert "Netflix" in reply
    assert "Mystery Charge" in reply
    assert "billing day" in reply.lower()


def test_natural_language_log_subscription_missing_amount_asks_rather_than_guessing(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_subscription([{"name": "Netflix", "amount": None}])
    )
    update = FakeUpdate(CHAT, text="add Netflix as a subscription")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_subscriptions(CHAT) == []
    assert "didn't catch a subscription" in update.message.replies[-1]
