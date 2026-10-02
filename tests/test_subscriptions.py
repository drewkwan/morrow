"""
Tests for recurring subscriptions: db.py's advance_date_by_frequency +
subscriptions CRUD (frequency/next_renewal_date/card/notes/is_claimable,
replacing the old monthly-only billing_day design -- see db.py's
"subscriptions" section docstring), subscriptions.py's commands and the two
jobs (subscriptions_tick's auto-logging + idempotency guard,
subscriptions_digest_tick's weekly renewal heads-up), and correction.py's
flexible edit_subscription action including undo. Same discipline as
test_vitals.py: no real network, no real Claude calls (ai._get_client is
always mocked), throwaway SQLite per test.
"""

import asyncio
import datetime as dt

import bot
import config
import db
import subscriptions
from conftest import CHAT
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _in_days(n):
    return (dt.date.today() + dt.timedelta(days=n)).isoformat()


class FakeBot:
    """Minimal stand-in for context.bot, just enough for
    subscriptions_tick/subscriptions_digest_tick's context.bot.send_message(...) calls."""
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


class FakeTickContext:
    def __init__(self):
        self.bot = FakeBot()


# ---------- db.py: advance_date_by_frequency ----------

def test_advance_date_by_frequency_weekly():
    assert db.advance_date_by_frequency(dt.date(2026, 10, 1), "weekly") == dt.date(2026, 10, 8)


def test_advance_date_by_frequency_biweekly():
    assert db.advance_date_by_frequency(dt.date(2026, 10, 1), "biweekly") == dt.date(2026, 10, 15)


def test_advance_date_by_frequency_monthly():
    assert db.advance_date_by_frequency(dt.date(2026, 10, 1), "monthly") == dt.date(2026, 11, 1)


def test_advance_date_by_frequency_quarterly():
    assert db.advance_date_by_frequency(dt.date(2026, 10, 1), "quarterly") == dt.date(2027, 1, 1)


def test_advance_date_by_frequency_annual():
    assert db.advance_date_by_frequency(dt.date(2026, 10, 1), "annual") == dt.date(2027, 10, 1)


def test_advance_date_by_frequency_clamps_to_short_month():
    """The 31st of a monthly subscription must land on April's real last
    day (30th), not overflow into May -- the same clamping discipline
    day_matches_billing_day already used for salary pay days."""
    assert db.advance_date_by_frequency(dt.date(2026, 3, 31), "monthly") == dt.date(2026, 4, 30)


def test_advance_date_by_frequency_clamps_for_february():
    assert db.advance_date_by_frequency(dt.date(2026, 1, 31), "monthly") == dt.date(2026, 2, 28)


def test_advance_date_by_frequency_rejects_unknown_frequency():
    try:
        db.advance_date_by_frequency(dt.date(2026, 10, 1), "daily")
        assert False, "should have raised"
    except ValueError:
        pass


# ---------- db.py: subscriptions CRUD ----------

def test_add_and_get_subscription():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25",
                                  category="Entertainment", card="Citibank", notes="shared plan",
                                  is_claimable=False)
    row = db.get_subscription(CHAT, sub_id)
    assert row["name"] == "Netflix"
    assert row["amount"] == 15.98
    assert row["frequency"] == "monthly"
    assert row["next_renewal_date"] == "2026-10-25"
    assert row["category"] == "Entertainment"
    assert row["card"] == "Citibank"
    assert row["notes"] == "shared plan"
    assert row["is_claimable"] == 0
    assert row["active"] == 1
    assert row["last_logged_date"] is None


def test_add_subscription_falls_back_to_monthly_for_unknown_frequency():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "fortnightly", "2026-10-25")
    assert db.get_subscription(CHAT, sub_id)["frequency"] == "monthly"


def test_get_subscriptions_orders_by_next_renewal_date_then_name():
    db.add_subscription(CHAT, "Gym", 80, "SGD", "monthly", "2026-11-01")
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
    db.add_subscription(CHAT, "Spotify", 11.98, "SGD", "monthly", "2026-11-01")
    names = [s["name"] for s in db.get_subscriptions(CHAT)]
    assert names == ["Netflix", "Gym", "Spotify"]


def test_delete_subscription_removes_it():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
    row = db.delete_subscription(CHAT, sub_id)
    assert row["name"] == "Netflix"
    assert db.get_subscription(CHAT, sub_id) is None


def test_delete_subscription_returns_none_for_unknown_id():
    assert db.delete_subscription(CHAT, 9999) is None


def test_advance_subscription_sets_last_logged_date_and_next_renewal_date():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
    db.advance_subscription(sub_id, "2026-10-25", "2026-11-25")
    row = db.get_subscription(CHAT, sub_id)
    assert row["last_logged_date"] == "2026-10-25"
    assert row["next_renewal_date"] == "2026-11-25"


def test_edit_subscription_only_touches_the_given_field():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25",
                                  category="Entertainment")
    updated = db.edit_subscription(CHAT, sub_id, new_amount=17.98)
    assert updated["amount"] == 17.98
    assert updated["name"] == "Netflix"  # untouched
    assert updated["next_renewal_date"] == "2026-10-25"  # untouched
    assert updated["category"] == "Entertainment"  # untouched


def test_edit_subscription_can_change_several_fields_at_once():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
    updated = db.edit_subscription(CHAT, sub_id, new_name="Netflix Premium", new_next_renewal_date="2026-10-05")
    assert updated["name"] == "Netflix Premium"
    assert updated["next_renewal_date"] == "2026-10-05"
    assert updated["amount"] == 15.98  # untouched


def test_edit_subscription_can_change_frequency_card_notes_claimable():
    sub_id = db.add_subscription(CHAT, "Anytime Fitness", 105, "SGD", "monthly", "2026-10-03")
    updated = db.edit_subscription(
        CHAT, sub_id, new_frequency="quarterly", new_card="OCBC", new_notes="claim from work",
        new_is_claimable=True,
    )
    assert updated["frequency"] == "quarterly"
    assert updated["card"] == "OCBC"
    assert updated["notes"] == "claim from work"
    assert updated["is_claimable"] == 1


def test_edit_subscription_rejects_unknown_frequency_leaves_it_unchanged():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
    updated = db.edit_subscription(CHAT, sub_id, new_frequency="fortnightly")
    assert updated["frequency"] == "monthly"


def test_edit_subscription_never_touches_last_logged_date():
    """A price/frequency/date correction must not retroactively re-trigger
    (or re-skip) a cycle that already auto-posted -- see
    db.edit_subscription's docstring."""
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
    db.advance_subscription(sub_id, "2026-10-25", "2026-11-25")
    db.edit_subscription(CHAT, sub_id, new_amount=17.98)
    assert db.get_subscription(CHAT, sub_id)["last_logged_date"] == "2026-10-25"


def test_edit_subscription_returns_none_for_unknown_id():
    assert db.edit_subscription(CHAT, 9999, new_amount=17.98) is None


def test_restore_deleted_subscription_brings_it_back_with_every_field():
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25",
                                  category="Entertainment", card="Citibank", notes="shared", is_claimable=True)
    db.advance_subscription(sub_id, "2026-10-25", "2026-11-25")
    row = db.delete_subscription(CHAT, sub_id)
    restored = db.restore_deleted_subscription(CHAT, row)
    assert restored["name"] == "Netflix"
    assert restored["next_renewal_date"] == "2026-11-25"
    assert restored["card"] == "Citibank"
    assert restored["notes"] == "shared"
    assert restored["is_claimable"] == 1
    # Preserving this matters: undoing a delete made right after an
    # auto-post must not make it look un-logged and eligible to
    # double-charge the same cycle on the next tick.
    assert restored["last_logged_date"] == "2026-10-25"


# ---------- subscriptions_tick ----------

def test_subscriptions_tick_auto_logs_on_renewal_date():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", db.today_str(), category="Entertainment")
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    expenses = db.get_recent_expenses(CHAT)
    assert len(expenses) == 1
    assert expenses[0]["description"] == "Netflix"
    assert expenses[0]["amount"] == 15.98
    assert expenses[0]["category"] == "Entertainment"
    assert len(context.bot.sent) == 1


def test_subscriptions_tick_advances_next_renewal_date_by_frequency():
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "HBO Max", 37.98, "SGD", "quarterly", db.today_str())
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    today = dt.date.fromisoformat(db.today_str())
    expected_next = db.advance_date_by_frequency(today, "quarterly")
    assert db.get_subscription(CHAT, sub_id)["next_renewal_date"] == expected_next.isoformat()


def test_subscriptions_tick_is_idempotent_within_the_same_day():
    """Regression guard: a bot restart or a job re-running the same day
    must never double-charge the same subscription in one cycle -- the
    tick now relies on next_renewal_date itself already having moved past
    today (see db.advance_subscription), not a separate last_logged_month
    string."""
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", db.today_str())
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    assert len(db.get_recent_expenses(CHAT)) == 1

    second_context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(second_context))
    assert len(db.get_recent_expenses(CHAT)) == 1  # still just one
    assert second_context.bot.sent == []


def test_subscriptions_tick_skips_subscriptions_not_due_today():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Gym", 80, "SGD", "monthly", _in_days(5))
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    assert db.get_recent_expenses(CHAT) == []


def test_subscriptions_tick_passes_through_is_claimable():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Anytime Fitness", 105, "SGD", "monthly", db.today_str(), is_claimable=True)
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    assert db.get_recent_expenses(CHAT)[0]["is_claimable"] == 1


def test_subscriptions_tick_notifies_the_chat():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", db.today_str())
    context = FakeTickContext()
    _run(subscriptions.subscriptions_tick(context))
    assert len(context.bot.sent) == 1
    assert context.bot.sent[0][0] == CHAT
    assert "Netflix" in context.bot.sent[0][1]


# ---------- subscriptions_digest_tick ----------

def test_subscriptions_digest_tick_includes_renewals_within_the_lookahead_window():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly",
                         _in_days(config.SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS))
    context = FakeTickContext()
    _run(subscriptions.subscriptions_digest_tick(context))
    assert len(context.bot.sent) == 1
    assert "Netflix" in context.bot.sent[0][1]


def test_subscriptions_digest_tick_excludes_renewals_outside_the_window():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "AIA Life Insurance", 2821, "SGD", "annual",
                         _in_days(config.SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS + 1))
    context = FakeTickContext()
    _run(subscriptions.subscriptions_digest_tick(context))
    assert context.bot.sent == []


def test_subscriptions_digest_tick_sends_nothing_when_nothing_is_upcoming():
    db.get_or_create_user(CHAT)
    context = FakeTickContext()
    _run(subscriptions.subscriptions_digest_tick(context))
    assert context.bot.sent == []


# ---------- commands ----------

def test_addsubscription_cmd_logs_and_replies():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Netflix", "15.98", "monthly", "2026-10-25", "Entertainment"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 1
    assert rows[0]["name"] == "Netflix"
    assert rows[0]["frequency"] == "monthly"
    assert rows[0]["next_renewal_date"] == "2026-10-25"
    assert rows[0]["category"] == "Entertainment"
    assert "Netflix" in update.message.replies[-1]


def test_addsubscription_cmd_accepts_currency():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Spotify", "9.99", "USD", "monthly", "2026-11-01"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert db.get_subscriptions(CHAT)[0]["currency"] == "USD"


def test_addsubscription_cmd_accepts_non_monthly_frequency():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["HBO Max", "37.98", "quarterly", "2027-01-01"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert db.get_subscriptions(CHAT)[0]["frequency"] == "quarterly"


def test_addsubscription_cmd_rejects_bad_amount():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Netflix", "banana", "monthly", "2026-10-25"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert "Usage:" in update.message.replies[-1] or "amount" in update.message.replies[-1].lower()


def test_addsubscription_cmd_rejects_unknown_frequency():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Netflix", "15.98", "fortnightly", "2026-10-25"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert "Frequency must be one of" in update.message.replies[-1]


def test_addsubscription_cmd_rejects_bad_date():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["Netflix", "15.98", "monthly", "not-a-date"]
    update = FakeUpdate(CHAT)
    _run(subscriptions.addsubscription_cmd(update, context))
    assert "date" in update.message.replies[-1].lower()


def test_subscriptions_cmd_lists_and_shows_monthly_equivalent_total():
    db.get_or_create_user(CHAT)
    db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
    db.add_subscription(CHAT, "Gym", 80, "SGD", "monthly", "2026-11-01")
    update = FakeUpdate(CHAT)
    _run(subscriptions.subscriptions_cmd(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Netflix" in reply
    assert "Gym" in reply
    assert "95.98" in reply  # 15.98 + 80, both monthly -> no conversion needed


def test_subscriptions_cmd_empty_state():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(subscriptions.subscriptions_cmd(update, FakeContext()))
    assert "No active subscriptions" in update.message.replies[-1]


def test_removesubscription_cmd_removes_it():
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")
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
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25",
                                  category="Entertainment")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
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


def test_correction_can_edit_a_subscriptions_frequency_and_renewal_date(monkeypatch):
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "HBO Max", 37.98, "SGD", "monthly", "2026-10-25")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "subscription", "target_expense_id": sub_id,
            "correction_action": "edit_subscription", "new_subscription_frequency": "quarterly",
            "new_subscription_renews_in_days": 10,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="that's actually quarterly, renews in 10 days")
    _run(bot.handle_text(update, FakeContext()))
    row = db.get_subscription(CHAT, sub_id)
    assert row["frequency"] == "quarterly"
    assert row["next_renewal_date"] == _in_days(10)


def test_correction_can_mark_a_subscription_claimable(monkeypatch):
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Anytime Fitness", 105, "SGD", "monthly", "2026-10-03")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "subscription", "target_expense_id": sub_id,
            "correction_action": "edit_subscription", "new_subscription_is_claimable": True,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="I should actually claim that gym membership from my company")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_subscription(CHAT, sub_id)["is_claimable"] == 1


def test_correction_edit_subscription_with_nothing_set_asks_what_to_fix(monkeypatch):
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
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
    """A subscription has no logged date (just a forward-looking
    next_renewal_date, which is part of edit_subscription) -- edit_date
    must be reported as an unsupported action, not silently misapplied."""
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "subscription", "target_expense_id": sub_id,
            "correction_action": "edit_date", "days_ago": 1,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="move that to yesterday")
    _run(bot.handle_text(update, FakeContext()))
    assert "isn't supported yet" in update.message.replies[-1]


def test_correction_can_delete_a_subscription_and_undo(monkeypatch):
    db.get_or_create_user(CHAT)
    sub_id = db.add_subscription(CHAT, "Netflix", 15.98, "SGD", "monthly", "2026-10-25")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "subscription", "target_expense_id": sub_id,
            "correction_action": "delete",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    delete_update = FakeUpdate(CHAT, text="cancel my Netflix subscription")
    _run(bot.handle_text(delete_update, context))
    assert db.get_subscription(CHAT, sub_id) is None

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_subscriptions(CHAT)[0]["name"] == "Netflix"


# ---------- natural-language "log_subscription" intent ----------

def _fake_parse_message_subscription(items):
    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
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
                                            "frequency": "monthly", "renews_in_days": 24,
                                            "category": "Entertainment"}])
    )
    update = FakeUpdate(CHAT, text="I pay $15.98 for Netflix, renews in 24 days")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 1
    assert rows[0]["name"] == "Netflix"
    assert rows[0]["next_renewal_date"] == _in_days(24)
    assert "Netflix" in update.message.replies[-1]


def test_natural_language_log_subscription_captures_card_notes_and_claimable(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_subscription([{
            "name": "Anytime Fitness", "amount": 105, "currency": "SGD", "frequency": "monthly",
            "renews_in_days": 2, "category": "Health", "card": "OCBC",
            "notes": "I should actually claim this from my company", "is_claimable": True,
        }])
    )
    update = FakeUpdate(CHAT, text="Anytime Fitness $105/mo on OCBC, I should claim this from my company")
    _run(bot.handle_text(update, FakeContext()))
    row = db.get_subscriptions(CHAT)[0]
    assert row["card"] == "OCBC"
    assert row["is_claimable"] == 1
    assert "claim" in row["notes"].lower()


def test_natural_language_log_subscription_bulk_list_in_one_message(monkeypatch):
    """The real point of this intent: registering a whole starter list at
    once instead of one /addsubscription per line -- Andrew's actual
    workflow of pasting his full real subscriptions list in one message."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_subscription([
            {"name": "Netflix", "amount": 15.98, "currency": "SGD", "frequency": "monthly", "renews_in_days": 13},
            {"name": "Spotify", "amount": 11.98, "currency": "SGD", "frequency": "monthly", "renews_in_days": 15},
            {"name": "HBO Max", "amount": 37.98, "currency": "SGD", "frequency": "quarterly",
             "renews_in_days": 92},
        ])
    )
    update = FakeUpdate(CHAT, text="Netflix 15.98 monthly, Spotify 11.98 monthly, HBO Max 37.98 every 3 months")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 3
    assert "3 subscriptions" in update.message.replies[-1]
    names = {r["name"] for r in rows}
    assert names == {"Netflix", "Spotify", "HBO Max"}


def test_natural_language_log_subscription_defaults_missing_renewal_date_instead_of_dropping_the_item(monkeypatch):
    """Unlike the old billing_day-required design, a missing renewal date no
    longer sinks the item -- it falls back to the next occurrence of its
    frequency starting today (see _next_renewal_date_from_days)."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_subscription([
            {"name": "Mystery Charge", "amount": 9.99, "currency": "SGD", "frequency": "monthly",
             "renews_in_days": None},
        ])
    )
    update = FakeUpdate(CHAT, text="some mystery charge for 9.99 a month")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_subscriptions(CHAT)
    assert len(rows) == 1
    today = dt.date.fromisoformat(db.today_str())
    assert rows[0]["next_renewal_date"] == db.advance_date_by_frequency(today, "monthly").isoformat()


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
