"""
Tests for income/deductions/net worth: db.py's CRUD + totals, income.py's
commands and income_tick (the daily recurring-salary auto-post job), the
natural-language "log_income"/"log_deduction" intents' handlers.py
dispatch, and the golden-string prompt coverage in ai.py. Deliberately
mirrors test_subscriptions.py's structure -- the two features share the
same "recurring auto-post, idempotent per month" design.
"""

import asyncio
import datetime as dt

import ai
import bot
import db
import income
from conftest import CHAT
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


class FakeTickContext:
    def __init__(self):
        self.bot = FakeBot()


def _no_op_extra_fields():
    return {
        "amount": None, "currency": None, "description": None, "category": None, "is_claimable": None,
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "memory_label": None, "memory_content": None, "memory_category": None,
    }


# ---------- db.py: day_matches_billing_day ----------
#
# Used only by income_tick's recurring-salary pay day now -- subscriptions
# moved to the more general advance_date_by_frequency (see
# test_subscriptions.py) once they needed non-monthly cadences, but a
# salary's pay day stays this simpler plain-monthly concept by design (see
# day_matches_billing_day's own docstring).

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


# ---------- db.py: income_config ----------

def test_set_and_get_income_config():
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20, stock_rate=0.05)
    config_row = db.get_income_config(CHAT)
    assert config_row["gross_amount"] == 6000
    assert config_row["pay_day"] == 25
    assert config_row["cpf_rate"] == 0.20
    assert config_row["stock_rate"] == 0.05
    assert config_row["active"] == 1
    assert config_row["last_logged_month"] is None


def test_set_income_config_upserts():
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20)
    db.set_income_config(CHAT, 6500, "SGD", 28, cpf_rate=0.20, stock_rate=0.05)
    config_row = db.get_income_config(CHAT)
    assert config_row["gross_amount"] == 6500
    assert config_row["pay_day"] == 28
    assert config_row["stock_rate"] == 0.05


def test_set_income_config_preserves_last_logged_month_across_a_reset():
    """Regression guard: re-setting a salary (a raise, a new pay day)
    mid-month must not risk a double auto-post for the cycle that already
    ran -- see db.set_income_config's docstring."""
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20)
    db.mark_income_config_logged(CHAT, "2026-10")
    db.set_income_config(CHAT, 6500, "SGD", 25, cpf_rate=0.20)
    assert db.get_income_config(CHAT)["last_logged_month"] == "2026-10"


def test_clear_income_config_removes_it():
    db.set_income_config(CHAT, 6000, "SGD", 25)
    db.clear_income_config(CHAT)
    assert db.get_income_config(CHAT) is None


def test_get_income_config_returns_none_when_never_set():
    db.get_or_create_user(CHAT)
    assert db.get_income_config(CHAT) is None


def test_edit_income_config_only_touches_the_given_field():
    """Regression guard for the real bug edit_income_config exists to
    prevent: a raise reported with only a new gross figure must NOT reset
    cpf_rate/stock_rate to set_income_config's zero defaults."""
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20, stock_rate=0.05)
    updated = db.edit_income_config(CHAT, new_gross_amount=7000)
    assert updated["gross_amount"] == 7000
    assert updated["pay_day"] == 25
    assert updated["cpf_rate"] == 0.20  # preserved, not reset to 0
    assert updated["stock_rate"] == 0.05  # preserved, not reset to 0


def test_edit_income_config_can_change_several_fields_at_once():
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20, stock_rate=0.05)
    updated = db.edit_income_config(CHAT, new_gross_amount=8000, new_pay_day=1, new_cpf_rate=0.22)
    assert updated["gross_amount"] == 8000
    assert updated["pay_day"] == 1
    assert updated["cpf_rate"] == 0.22
    assert updated["stock_rate"] == 0.05  # still preserved


def test_edit_income_config_preserves_last_logged_month():
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20)
    db.mark_income_config_logged(CHAT, "2026-10")
    db.edit_income_config(CHAT, new_gross_amount=7000)
    assert db.get_income_config(CHAT)["last_logged_month"] == "2026-10"


def test_edit_income_config_returns_none_when_no_config_exists():
    db.get_or_create_user(CHAT)
    assert db.edit_income_config(CHAT, new_gross_amount=7000) is None


# ---------- db.py: income ledger ----------

def test_add_and_get_income():
    income_id = db.add_income(CHAT, "bonus", 850, "SGD", description="year-end bonus",
                               gross_amount=1000, cpf_amount=150)
    row = db.get_income(CHAT, income_id)
    assert row["source"] == "bonus"
    assert row["net_amount"] == 850
    assert row["gross_amount"] == 1000
    assert row["cpf_amount"] == 150
    assert row["description"] == "year-end bonus"


def test_get_recent_income_orders_by_date_not_insertion_order():
    """Same real lesson as get_recent_meals/get_recent_vitals/get_recent_
    workouts: ordering by id alone breaks once any entry is backdated or
    auto-posted out of strict insertion order."""
    db.add_income(CHAT, "salary", 4500, "SGD", income_date="2026-07-01")
    db.add_income(CHAT, "bonus", 850, "SGD", income_date="2026-09-20")
    # Inserted last (highest id), but dated before the Sept 20 entry above.
    db.add_income(CHAT, "other", 200, "SGD", income_date="2026-07-15")
    dates = [r["income_date"] for r in db.get_recent_income(CHAT)]
    assert dates == sorted(dates, reverse=True)


def test_delete_income_removes_it():
    income_id = db.add_income(CHAT, "bonus", 500, "SGD")
    row = db.delete_income(CHAT, income_id)
    assert row["net_amount"] == 500
    assert db.get_income(CHAT, income_id) is None


def test_delete_income_returns_none_for_unknown_id():
    assert db.delete_income(CHAT, 9999) is None


def test_get_total_income_sums_net_amount_base():
    db.add_income(CHAT, "salary", 4500, "SGD")
    db.add_income(CHAT, "bonus", 850, "SGD")
    assert db.get_total_income(CHAT) == 5350


def test_edit_income_date_moves_it():
    income_id = db.add_income(CHAT, "bonus", 500, "SGD", income_date="2026-09-01")
    updated = db.edit_income_date(CHAT, income_id, "2026-09-05")
    assert updated["income_date"] == "2026-09-05"


def test_edit_income_date_clamps_to_today():
    income_id = db.add_income(CHAT, "bonus", 500, "SGD")
    far_future = (dt.date.fromisoformat(db.today_str()) + dt.timedelta(days=5)).isoformat()
    updated = db.edit_income_date(CHAT, income_id, far_future)
    assert updated["income_date"] == db.today_str()


def test_edit_income_only_touches_the_given_field():
    income_id = db.add_income(CHAT, "bonus", 500, "SGD", description="old bonus")
    updated = db.edit_income(CHAT, income_id, new_amount=600)
    assert updated["net_amount"] == 600
    assert updated["source"] == "bonus"  # untouched
    assert updated["description"] == "old bonus"  # untouched


def test_edit_income_can_change_source_and_description():
    income_id = db.add_income(CHAT, "bonus", 500, "SGD")
    updated = db.edit_income(CHAT, income_id, new_source="other", new_description="freelance")
    assert updated["source"] == "other"
    assert updated["description"] == "freelance"
    assert updated["net_amount"] == 500  # untouched


def test_restore_deleted_income_brings_it_back():
    income_id = db.add_income(CHAT, "bonus", 500, "SGD", description="bonus", gross_amount=600, cpf_amount=100)
    row = db.delete_income(CHAT, income_id)
    restored = db.restore_deleted_income(CHAT, row)
    assert restored["net_amount"] == 500
    assert restored["gross_amount"] == 600
    assert restored["cpf_amount"] == 100
    assert restored["description"] == "bonus"


# ---------- db.py: deductions ----------

def test_add_and_get_deduction():
    deduction_id = db.add_deduction(CHAT, "income tax", 800, "SGD")
    row = db.get_deduction(CHAT, deduction_id)
    assert row["label"] == "income tax"
    assert row["amount"] == 800


def test_get_recent_deductions_orders_by_date_not_insertion_order():
    db.add_deduction(CHAT, "tax", 800, "SGD", deduction_date="2026-07-01")
    db.add_deduction(CHAT, "CPF top-up", 500, "SGD", deduction_date="2026-09-20")
    db.add_deduction(CHAT, "tax (late)", 100, "SGD", deduction_date="2026-07-15")
    dates = [r["deduction_date"] for r in db.get_recent_deductions(CHAT)]
    assert dates == sorted(dates, reverse=True)


def test_delete_deduction_removes_it():
    deduction_id = db.add_deduction(CHAT, "income tax", 800, "SGD")
    row = db.delete_deduction(CHAT, deduction_id)
    assert row["label"] == "income tax"
    assert db.get_deduction(CHAT, deduction_id) is None


def test_get_total_deductions_sums_amount_base():
    db.add_deduction(CHAT, "tax", 800, "SGD")
    db.add_deduction(CHAT, "CPF top-up", 500, "SGD")
    assert db.get_total_deductions(CHAT) == 1300


def test_edit_deduction_date_moves_it():
    deduction_id = db.add_deduction(CHAT, "tax", 800, "SGD", deduction_date="2026-09-01")
    updated = db.edit_deduction_date(CHAT, deduction_id, "2026-09-05")
    assert updated["deduction_date"] == "2026-09-05"


def test_edit_deduction_only_touches_the_given_field():
    deduction_id = db.add_deduction(CHAT, "tax", 800, "SGD")
    updated = db.edit_deduction(CHAT, deduction_id, new_amount=600)
    assert updated["amount"] == 600
    assert updated["label"] == "tax"  # untouched


def test_restore_deleted_deduction_brings_it_back():
    deduction_id = db.add_deduction(CHAT, "income tax", 800, "SGD")
    row = db.delete_deduction(CHAT, deduction_id)
    restored = db.restore_deleted_deduction(CHAT, row)
    assert restored["label"] == "income tax"
    assert restored["amount"] == 800


# ---------- db.py: net worth ----------

def test_get_total_spend_reuses_get_daily_totals():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 50, "SGD", "groceries", "Groceries")
    db.add_expense(CHAT, 30, "SGD", "claimable lunch", "Food", is_claimable=True)
    # Claimable spend is excluded, same as get_daily_totals' own discipline.
    assert db.get_total_spend(CHAT) == 50


def test_get_net_worth_computes_real_figures():
    db.get_or_create_user(CHAT)
    db.add_income(CHAT, "salary", 4500, "SGD")
    db.add_deduction(CHAT, "tax", 800, "SGD")
    db.add_expense(CHAT, 200, "SGD", "groceries", "Groceries")
    net_worth = db.get_net_worth(CHAT)
    assert net_worth["total_income"] == 4500
    assert net_worth["total_deductions"] == 800
    assert net_worth["total_spend"] == 200
    assert net_worth["net_worth"] == 3500  # 4500 - 800 - 200


def test_get_net_worth_is_zero_with_nothing_logged():
    db.get_or_create_user(CHAT)
    net_worth = db.get_net_worth(CHAT)
    assert net_worth == {"total_income": 0, "total_deductions": 0, "total_spend": 0, "net_worth": 0}


# ---------- income.py: _parse_rate / _compute_net_salary ----------

def test_parse_rate_accepts_percent_sign():
    assert income._parse_rate("20%") == 0.2


def test_parse_rate_accepts_bare_fraction():
    assert income._parse_rate("0.2") == 0.2


def test_parse_rate_accepts_bare_number_over_one_as_a_percentage():
    assert income._parse_rate("20") == 0.2


def test_compute_net_salary():
    config_row = {"gross_amount": 6000, "cpf_rate": 0.20, "stock_rate": 0.05}
    assert income._compute_net_salary(config_row) == 6000 * 0.75


# ---------- income_tick ----------

def test_income_tick_auto_posts_on_pay_day():
    db.get_or_create_user(CHAT)
    db.set_income_config(CHAT, 6000, "SGD", dt.date.today().day, cpf_rate=0.20, stock_rate=0.05)
    context = FakeTickContext()
    _run(income.income_tick(context))
    rows = db.get_recent_income(CHAT)
    assert len(rows) == 1
    assert rows[0]["source"] == "salary"
    assert rows[0]["gross_amount"] == 6000
    assert rows[0]["cpf_amount"] == 1200
    assert rows[0]["stock_amount"] == 300
    assert rows[0]["net_amount"] == 4500
    assert len(context.bot.sent) == 1


def test_income_tick_is_idempotent_within_the_same_month():
    db.get_or_create_user(CHAT)
    db.set_income_config(CHAT, 6000, "SGD", dt.date.today().day, cpf_rate=0.20)
    db.mark_income_config_logged(CHAT, dt.date.today().strftime("%Y-%m"))
    context = FakeTickContext()
    _run(income.income_tick(context))
    assert db.get_recent_income(CHAT) == []


def test_income_tick_skips_when_pay_day_is_not_today():
    db.get_or_create_user(CHAT)
    not_today = 1 if dt.date.today().day != 1 else 15
    db.set_income_config(CHAT, 6000, "SGD", not_today, cpf_rate=0.20)
    context = FakeTickContext()
    _run(income.income_tick(context))
    assert db.get_recent_income(CHAT) == []


def test_income_tick_skips_when_no_config_set():
    db.get_or_create_user(CHAT)
    context = FakeTickContext()
    _run(income.income_tick(context))
    assert db.get_recent_income(CHAT) == []


def test_income_tick_skips_after_config_cleared():
    db.get_or_create_user(CHAT)
    db.set_income_config(CHAT, 6000, "SGD", dt.date.today().day, cpf_rate=0.20)
    db.clear_income_config(CHAT)
    context = FakeTickContext()
    _run(income.income_tick(context))
    assert db.get_recent_income(CHAT) == []


# ---------- commands: /setincome, /incomeconfig, /clearincome ----------

def test_setincome_cmd_sets_config_and_replies():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["6000", "25", "20%", "5%"]
    update = FakeUpdate(CHAT)
    _run(income.setincome_cmd(update, context))
    config_row = db.get_income_config(CHAT)
    assert config_row["gross_amount"] == 6000
    assert config_row["pay_day"] == 25
    assert config_row["cpf_rate"] == 0.20
    assert config_row["stock_rate"] == 0.05
    assert "Gross" in update.message.replies[-1]


def test_setincome_cmd_accepts_currency():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["6000", "USD", "25"]
    update = FakeUpdate(CHAT)
    _run(income.setincome_cmd(update, context))
    assert db.get_income_config(CHAT)["currency"] == "USD"


def test_setincome_cmd_rejects_bad_pay_day():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["6000", "45"]
    update = FakeUpdate(CHAT)
    _run(income.setincome_cmd(update, context))
    assert "between 1 and 31" in update.message.replies[-1]


def test_incomeconfig_cmd_shows_current_setup():
    db.get_or_create_user(CHAT)
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20, stock_rate=0.05)
    update = FakeUpdate(CHAT)
    _run(income.incomeconfig_cmd(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "6,000" in reply or "6000" in reply
    assert "20.0%" in reply


def test_incomeconfig_cmd_empty_state():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(income.incomeconfig_cmd(update, FakeContext()))
    assert "No recurring salary" in update.message.replies[-1]


def test_clearincome_cmd_clears_config():
    db.get_or_create_user(CHAT)
    db.set_income_config(CHAT, 6000, "SGD", 25)
    update = FakeUpdate(CHAT)
    _run(income.clearincome_cmd(update, FakeContext()))
    assert db.get_income_config(CHAT) is None
    assert "cleared" in update.message.replies[-1].lower()


# ---------- commands: /addincome, /recentincome, /removeincome ----------

def test_addincome_cmd_logs_and_replies():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["500", "freelance", "gig"]
    update = FakeUpdate(CHAT)
    _run(income.addincome_cmd(update, context))
    rows = db.get_recent_income(CHAT)
    assert len(rows) == 1
    assert rows[0]["net_amount"] == 500
    assert rows[0]["description"] == "freelance gig"


def test_recentincome_cmd_lists_entries():
    db.add_income(CHAT, "bonus", 500, "SGD", description="bonus")
    update = FakeUpdate(CHAT)
    _run(income.recentincome_cmd(update, FakeContext()))
    assert "bonus" in update.message.replies[-1]


def test_recentincome_cmd_empty_state():
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    _run(income.recentincome_cmd(update, FakeContext()))
    assert "No income" in update.message.replies[-1]


def test_removeincome_cmd_removes_it():
    income_id = db.add_income(CHAT, "bonus", 500, "SGD")
    context = FakeContext()
    context.args = [str(income_id)]
    update = FakeUpdate(CHAT)
    _run(income.removeincome_cmd(update, context))
    assert db.get_income(CHAT, income_id) is None


# ---------- commands: /adddeduction, /recentdeductions, /removededuction ----------

def test_adddeduction_cmd_logs_and_replies():
    db.get_or_create_user(CHAT)
    context = FakeContext()
    context.args = ["800", "income", "tax"]
    update = FakeUpdate(CHAT)
    _run(income.adddeduction_cmd(update, context))
    rows = db.get_recent_deductions(CHAT)
    assert len(rows) == 1
    assert rows[0]["amount"] == 800
    assert rows[0]["label"] == "income tax"


def test_recentdeductions_cmd_lists_entries():
    db.add_deduction(CHAT, "income tax", 800, "SGD")
    update = FakeUpdate(CHAT)
    _run(income.recentdeductions_cmd(update, FakeContext()))
    assert "income tax" in update.message.replies[-1]


def test_removededuction_cmd_removes_it():
    deduction_id = db.add_deduction(CHAT, "income tax", 800, "SGD")
    context = FakeContext()
    context.args = [str(deduction_id)]
    update = FakeUpdate(CHAT)
    _run(income.removededuction_cmd(update, context))
    assert db.get_deduction(CHAT, deduction_id) is None


# ---------- /networth ----------

def test_networth_cmd_shows_real_figures():
    db.get_or_create_user(CHAT)
    db.add_income(CHAT, "salary", 4500, "SGD")
    db.add_deduction(CHAT, "tax", 800, "SGD")
    db.add_expense(CHAT, 200, "SGD", "groceries", "Groceries")
    update = FakeUpdate(CHAT)
    _run(income.networth_cmd(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "3,500" in reply or "3500" in reply


# ---------- ai.py: golden-string coverage for the new intents ----------

def test_parse_system_prompt_covers_log_income_and_log_deduction():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert '"log_income"' in prompt
    assert '"log_deduction"' in prompt
    assert "real NET take-home figure" in prompt


def test_clarify_fallback_includes_income_and_deductions():
    result = ai._clarify_fallback("huh?")
    assert result["income"] is None
    assert result["deductions"] is None


# ---------- natural-language "log_income"/"log_deduction" intents ----------

def _fake_parse_message_income(items):
    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_income", "clarification_question": None, "casual_reply": None,
            "income": items, **_no_op_extra_fields(),
        }
    return fake_parse_message


def _fake_parse_message_deduction(items):
    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "log_deduction", "clarification_question": None, "casual_reply": None,
            "deductions": items, **_no_op_extra_fields(),
        }
    return fake_parse_message


def test_natural_language_log_income_single_item(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_income([{"amount": 500, "currency": "SGD", "source": "bonus",
                                      "description": "bonus"}])
    )
    update = FakeUpdate(CHAT, text="got a $500 bonus today")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_recent_income(CHAT)
    assert len(rows) == 1
    assert rows[0]["net_amount"] == 500
    assert "500" in update.message.replies[-1]


def test_natural_language_log_income_multiple_items(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_income([
            {"amount": 500, "currency": "SGD", "source": "bonus", "description": "bonus"},
            {"amount": 200, "currency": "SGD", "source": "other", "description": "side gig"},
        ])
    )
    update = FakeUpdate(CHAT, text="got a $500 bonus and $200 from a side gig")
    _run(bot.handle_text(update, FakeContext()))
    assert len(db.get_recent_income(CHAT)) == 2
    assert "2 income entries" in update.message.replies[-1]


def test_natural_language_log_income_with_gross_breakdown(monkeypatch):
    """The real distinguishing case from ai.py's log_income field docs:
    amount must be the NET figure, gross/cpf/stock are optional breakdown
    detail kept alongside it."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_income([{"amount": 850, "currency": "SGD", "source": "bonus",
                                      "gross_amount": 1000, "cpf_amount": 150}])
    )
    update = FakeUpdate(CHAT, text="got a $1000 bonus, $150 went to CPF so $850 hit my account")
    _run(bot.handle_text(update, FakeContext()))
    row = db.get_recent_income(CHAT)[0]
    assert row["net_amount"] == 850
    assert row["gross_amount"] == 1000
    assert row["cpf_amount"] == 150


def test_natural_language_log_income_missing_amount_asks_rather_than_guessing(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_income([{"amount": None}]))
    update = FakeUpdate(CHAT, text="got some money")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_recent_income(CHAT) == []
    assert "didn't catch an amount" in update.message.replies[-1]


def test_natural_language_log_deduction_single_item(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_deduction([{"amount": 800, "currency": "SGD", "label": "income tax"}])
    )
    update = FakeUpdate(CHAT, text="paid $800 income tax today")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_recent_deductions(CHAT)
    assert len(rows) == 1
    assert rows[0]["amount"] == 800
    assert rows[0]["label"] == "income tax"


def test_natural_language_log_deduction_does_not_affect_daily_spending_target(monkeypatch):
    """The core guarantee of this intent -- see ai.py's log_deduction
    description: a deduction must never touch the daily spending target
    math that log_expense feeds."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    status_before = db.get_status(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_deduction([{"amount": 800, "currency": "SGD", "label": "income tax"}])
    )
    update = FakeUpdate(CHAT, text="paid $800 income tax today")
    _run(bot.handle_text(update, FakeContext()))
    status_after = db.get_status(CHAT)
    assert status_after["spent_today"] == status_before["spent_today"]
    assert status_after["available_today"] == status_before["available_today"]


def test_natural_language_log_deduction_missing_amount_asks_rather_than_guessing(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_deduction([{"amount": None}]))
    update = FakeUpdate(CHAT, text="paid some tax")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_recent_deductions(CHAT) == []
    assert "didn't catch an amount" in update.message.replies[-1]


def test_natural_language_log_income_logs_to_conversation_history(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        _fake_parse_message_income([{"amount": 500, "currency": "SGD", "source": "bonus"}])
    )
    update = FakeUpdate(CHAT, text="got a $500 bonus today")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_recent_messages(CHAT)
    assert rows[-1]["role"] == "morrow"


# ---------- correction.py: editing income/deduction entries ----------

def test_correction_can_edit_an_income_entry_and_undo(monkeypatch):
    db.get_or_create_user(CHAT)
    income_id = db.add_income(CHAT, "bonus", 500, "SGD")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "income", "target_expense_id": income_id,
            "correction_action": "edit_income", "new_income_amount": 600,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="that bonus was actually $600")
    _run(bot.handle_text(edit_update, context))
    assert db.get_income(CHAT, income_id)["net_amount"] == 600

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_income(CHAT, income_id)["net_amount"] == 500


def test_correction_edit_income_with_nothing_set_asks_what_to_fix(monkeypatch):
    db.get_or_create_user(CHAT)
    income_id = db.add_income(CHAT, "bonus", 500, "SGD")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "income", "target_expense_id": income_id,
            "correction_action": "edit_income",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="fix that income entry")
    _run(bot.handle_text(update, FakeContext()))
    assert "What should I fix" in update.message.replies[-1]


def test_correction_can_edit_a_deduction_and_undo(monkeypatch):
    db.get_or_create_user(CHAT)
    deduction_id = db.add_deduction(CHAT, "tax", 800, "SGD")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "deduction", "target_expense_id": deduction_id,
            "correction_action": "edit_deduction", "new_deduction_amount": 900,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="that tax payment was actually $900")
    _run(bot.handle_text(edit_update, context))
    assert db.get_deduction(CHAT, deduction_id)["amount"] == 900

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_deduction(CHAT, deduction_id)["amount"] == 800


# ---------- correction.py: "I got a raise" (edit_income_config) ----------

def test_correction_can_edit_income_config_raise_and_undo(monkeypatch):
    """The real distinguishing case from ai.py's edit_income_config rule:
    a raise changes the ONGOING recurring config, never adds a new income
    row the way log_income would."""
    db.set_income_config(CHAT, 6000, "SGD", 25, cpf_rate=0.20, stock_rate=0.05)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "income_config",
            "correction_action": "edit_income_config", "new_income_config_gross_amount": 7000,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="I got a raise, now making $7000")
    _run(bot.handle_text(edit_update, context))
    config_row = db.get_income_config(CHAT)
    assert config_row["gross_amount"] == 7000
    assert config_row["cpf_rate"] == 0.20  # preserved
    assert db.get_recent_income(CHAT) == []  # never adds a new income row

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_income_config(CHAT)["gross_amount"] == 6000


def test_correction_edit_income_config_with_no_config_asks_to_setincome_first(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "income_config",
            "correction_action": "edit_income_config", "new_income_config_gross_amount": 7000,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="I got a raise, now making $7000")
    _run(bot.handle_text(update, FakeContext()))
    assert "/setincome" in update.message.replies[-1]


def test_correction_edit_income_config_with_nothing_given_asks_for_gross_salary(monkeypatch):
    db.set_income_config(CHAT, 6000, "SGD", 25)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "income_config", "correction_action": "edit_income_config",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="I got a raise")
    _run(bot.handle_text(update, FakeContext()))
    assert "new gross salary" in update.message.replies[-1]


# ---------- ai.py: golden-string coverage for editing + log_subscription ----------

def test_parse_system_prompt_covers_correction_edits_for_income_domains():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert '"edit_subscription"' in prompt
    assert '"edit_income"' in prompt
    assert '"edit_deduction"' in prompt
    assert '"edit_income_config"' in prompt
    assert '"income_config"' in prompt


def test_parse_system_prompt_distinguishes_log_income_from_edit_income_config():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert "CRITICAL distinction from \"log_income\"" in prompt
    assert "it only changes what future auto-posts will use" in prompt


def test_parse_system_prompt_covers_log_subscription():
    prompt = ai.PARSE_SYSTEM_PROMPT
    assert '"log_subscription"' in prompt
    assert '"subscriptions"' in prompt


# ---------- casual conversation grounded in real net worth ----------

def test_casual_reply_today_snapshot_includes_net_worth(monkeypatch):
    """Regression guard: a genuine 'what's my net worth' question must be
    answerable from real figures, not guessed -- see handlers._casual_
    reply_text's docstring and ai.py's today_snapshot net_worth note."""
    db.get_or_create_user(CHAT)
    db.add_income(CHAT, "salary", 4500, "SGD")
    captured = {}

    def fake_answer_casually(message, recent_messages, memory_list, today_snapshot, recent_lifts):
        captured["today_snapshot"] = today_snapshot
        return "You've got real money saved up."

    monkeypatch.setattr(bot.ai, "answer_casually", fake_answer_casually)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        lambda *a, **kw: {
            "intent": "casual", "clarification_question": None, "casual_reply": "fallback",
            **_no_op_extra_fields(),
        }
    )
    update = FakeUpdate(CHAT, text="what's my net worth looking like?")
    _run(bot.handle_text(update, FakeContext()))
    assert captured["today_snapshot"]["net_worth"]["total_income"] == 4500


def test_casual_reply_today_snapshot_includes_month_to_date_total(monkeypatch):
    """Regression guard: a genuine 'what's my total spend this month'
    question must be answerable from a real calendar-month-to-date sum, not
    guessed or misrouted to the 'trend' intent's day-by-day trajectory shape
    -- see handlers._casual_reply_text's docstring and PARSE_SYSTEM_PROMPT's
    "trend" paragraph's CRITICAL total-vs-trajectory distinction."""
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 42.50, "SGD", "groceries", "Groceries")
    captured = {}

    def fake_answer_casually(message, recent_messages, memory_list, today_snapshot, recent_lifts):
        captured["today_snapshot"] = today_snapshot
        return "You've spent $42.50 so far this month."

    monkeypatch.setattr(bot.ai, "answer_casually", fake_answer_casually)
    monkeypatch.setattr(
        bot.ai, "parse_message",
        lambda *a, **kw: {
            "intent": "casual", "clarification_question": None, "casual_reply": "fallback",
            **_no_op_extra_fields(),
        }
    )
    update = FakeUpdate(CHAT, text="what's my total monthly expenditure on expenses?")
    _run(bot.handle_text(update, FakeContext()))
    assert captured["today_snapshot"]["month_to_date"]["total"] == 42.50
