"""
Core money-math tests: rollover, streaks, budget alerts, currency conversion,
and balance-safe undo/edit/delete.
"""

from datetime import date, timedelta

import db
from conftest import CHAT


def _insert_on(chat_id, day, amount, currency="SGD", category="Food", is_claimable=False):
    """Backdoor insert for a specific expense_date, bypassing add_expense's
    'always today' behavior so we can simulate past days."""
    amount_base = amount if currency == "SGD" else amount  # tests that need real conversion override fx
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO expenses (chat_id, amount, currency, amount_base, description, category, "
            "is_claimable, expense_date) VALUES (?, ?, ?, ?, 'x', ?, ?, ?)",
            (chat_id, amount, currency, amount_base, category, int(is_claimable), day.isoformat()),
        )


# ---------- daily target / user bootstrap ----------

def test_new_user_gets_default_target():
    user = db.get_or_create_user(CHAT)
    assert user["daily_target"] == 100.0
    assert user["balance"] == 0.0
    assert user["current_streak"] == 0


def test_set_daily_target_updates_existing_user():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 250)
    assert db.get_or_create_user(CHAT)["daily_target"] == 250


# ---------- rollover ----------

def test_rollover_credits_leftover_to_balance():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = date.today() - timedelta(days=1)
    _insert_on(CHAT, yesterday, 30)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (yesterday.isoformat(), CHAT),
        )
    result = db.ensure_rollover(CHAT)
    assert len(result["rollovers"]) == 1
    assert result["rollovers"][0] == (yesterday.isoformat(), 70.0)  # 100 target - 30 spent
    assert db.get_or_create_user(CHAT)["balance"] == 70.0


def test_rollover_overspend_reduces_balance():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = date.today() - timedelta(days=1)
    _insert_on(CHAT, yesterday, 150)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (yesterday.isoformat(), CHAT),
        )
    db.ensure_rollover(CHAT)
    assert db.get_or_create_user(CHAT)["balance"] == -50.0  # 100 - 150


def test_rollover_handles_multi_day_offline_gap():
    """If the bot was offline for 3 days, ensure_rollover walks forward one
    day at a time using the *current* daily_target for each missed day."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    three_days_ago = date.today() - timedelta(days=3)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (three_days_ago.isoformat(), CHAT),
        )
    # spend on two of the three missed days, nothing on the third
    _insert_on(CHAT, three_days_ago, 40)
    _insert_on(CHAT, three_days_ago + timedelta(days=1), 120)  # overspend day
    result = db.ensure_rollover(CHAT)
    assert len(result["rollovers"]) == 3
    # 60 (under) + -20 (over) + 100 (nothing spent) = 140
    assert db.get_or_create_user(CHAT)["balance"] == 140.0


def test_ensure_rollover_is_noop_same_day():
    db.get_or_create_user(CHAT)
    result = db.ensure_rollover(CHAT)
    assert result["rollovers"] == []


# ---------- manual balance adjustment (the one escape hatch from derived balance) ----------

def test_adjust_balance_adds_a_deficit():
    """Real motivating case: expense history from before some date is gone
    (lost/reset data), so the automatic rollover has nothing to derive that
    period's real overspend from. adjust_balance is the manual correction."""
    db.get_or_create_user(CHAT)
    result = db.adjust_balance(CHAT, -1135.89)
    assert result["old_balance"] == 0.0
    assert result["new_balance"] == -1135.89
    assert db.get_or_create_user(CHAT)["balance"] == -1135.89


def test_adjust_balance_adds_a_credit_on_top_of_existing_balance():
    db.get_or_create_user(CHAT)
    db.adjust_balance(CHAT, 50)
    result = db.adjust_balance(CHAT, 25)
    assert result["old_balance"] == 50
    assert result["new_balance"] == 75


def test_adjust_balance_catches_up_rollover_first():
    """A stale balance should be caught up to today before the delta is
    applied, so the adjustment lands on the current number, not a stale
    pre-rollover one."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = date.today() - timedelta(days=1)
    _insert_on(CHAT, yesterday, 30)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (yesterday.isoformat(), CHAT),
        )
    result = db.adjust_balance(CHAT, -10)
    assert result["old_balance"] == 70.0  # rolled over first (100 - 30)
    assert result["new_balance"] == 60.0


def test_set_balance_restores_an_exact_value():
    db.get_or_create_user(CHAT)
    db.adjust_balance(CHAT, -1135.89)
    db.set_balance(CHAT, 0.0)
    assert db.get_or_create_user(CHAT)["balance"] == 0.0


# ---------- streaks ----------

def test_streak_increments_on_days_within_target():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    three_days_ago = date.today() - timedelta(days=3)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (three_days_ago.isoformat(), CHAT),
        )
    for i in range(3):
        _insert_on(CHAT, three_days_ago + timedelta(days=i), 50)  # always under target
    db.ensure_rollover(CHAT)
    user = db.get_or_create_user(CHAT)
    assert user["current_streak"] == 3
    assert user["best_streak"] == 3


def test_streak_resets_to_zero_on_overspend_day():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    two_days_ago = date.today() - timedelta(days=2)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ?, current_streak = 5, best_streak = 5 WHERE chat_id = ?",
            (two_days_ago.isoformat(), CHAT),
        )
    _insert_on(CHAT, two_days_ago, 50)       # under target: streak -> 6
    _insert_on(CHAT, two_days_ago + timedelta(days=1), 150)  # over target: streak -> 0
    db.ensure_rollover(CHAT)
    user = db.get_or_create_user(CHAT)
    assert user["current_streak"] == 0
    assert user["best_streak"] == 6  # best is preserved even after the reset


# ---------- budget alert ----------

def test_alert_fires_once_after_crossing_threshold():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    _insert_on(CHAT, date.today(), 95)  # 95% of 100 target, threshold is 90%
    assert db.maybe_alert(CHAT) is True
    assert db.maybe_alert(CHAT) is False  # already fired today, no repeat


def test_alert_does_not_fire_below_threshold():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    _insert_on(CHAT, date.today(), 50)
    assert db.maybe_alert(CHAT) is False


# ---------- currency conversion ----------

def test_add_expense_converts_foreign_currency(monkeypatch):
    import fx
    monkeypatch.setattr(fx, "get_rate", lambda f, t: 1.35)  # 1 USD = 1.35 SGD
    db.get_or_create_user(CHAT)
    expense_id = db.add_expense(CHAT, 20, "USD", "taxi", "Transport")
    row = db.get_expense(CHAT, expense_id)
    assert row["amount"] == 20
    assert row["currency"] == "USD"
    assert row["amount_base"] == 27.0  # 20 * 1.35


def test_add_expense_same_currency_skips_conversion():
    db.get_or_create_user(CHAT)
    expense_id = db.add_expense(CHAT, 20, "SGD", "lunch", "Food")
    row = db.get_expense(CHAT, expense_id)
    assert row["amount_base"] == 20  # no fx.get_rate call needed (would fail via conftest guard)


def test_add_expense_records_no_fallback_on_a_successful_conversion(monkeypatch):
    import fx
    monkeypatch.setattr(fx, "get_rate", lambda f, t: 1.35)
    db.get_or_create_user(CHAT)
    expense_id = db.add_expense(CHAT, 20, "USD", "taxi", "Transport")
    row = db.get_expense(CHAT, expense_id)
    assert row["fx_fallback"] == 0


def test_add_expense_flags_fx_fallback_and_logs_at_1to1_when_the_rate_lookup_fails(monkeypatch):
    """Regression test for a real production incident: a transient
    Frankfurter failure silently produced a materially wrong amount_base
    (a USD lunch landed in the SGD total completely unconverted) with
    nothing in the data to say the real rate hadn't actually been used.
    add_expense must still succeed (never block logging -- see fx.
    to_base_checked's docstring) but now flags the row so the confirmation
    reply can warn instead of presenting it as a normal conversion."""
    import fx

    def _always_fails(from_ccy, to_ccy):
        raise RuntimeError("simulated Frankfurter outage")

    monkeypatch.setattr(fx, "get_rate", _always_fails)
    db.get_or_create_user(CHAT)
    expense_id = db.add_expense(CHAT, 115, "USD", "lunch", "Food")
    row = db.get_expense(CHAT, expense_id)
    assert row["amount"] == 115
    assert row["amount_base"] == 115  # unconverted 1:1 fallback, not a guessed rate
    assert row["fx_fallback"] == 1


def test_add_expense_defaults_is_subscription_to_false():
    db.get_or_create_user(CHAT)
    expense_id = db.add_expense(CHAT, 20, "SGD", "lunch", "Food")
    row = db.get_expense(CHAT, expense_id)
    assert row["is_subscription"] == 0


# ---------- is_subscription: excluded from the daily target/streak, not from real spend ----------
#
# Real production bug this guards against: subscriptions.subscriptions_tick
# auto-posts a due subscription as a real expense (see its own docstring),
# and that used to count against the daily spending target and the
# within-budget streak exactly like a discretionary purchase -- a Netflix
# or phone-bill renewal could silently blow a streak the user never
# actually overspent on. is_subscription=True expenses must be excluded
# from spent_today/available_today/the streak (db._spent_on), while still
# counting everywhere real spend matters: month-to-date total, category
# insights, net worth, and /recent.

def test_subscription_expense_does_not_count_toward_spent_today():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    db.add_expense(CHAT, 69.00, "SGD", "Singtel", "Bills & Utilities", is_subscription=True)
    db.add_expense(CHAT, 6.99, "SGD", "Twitch", "Entertainment", is_subscription=True)
    status = db.get_status(CHAT)
    assert status["spent_today"] == 0
    assert status["available_today"] == 100


def test_subscription_expense_does_not_break_the_streak(monkeypatch):
    """A subscription dated YESTERDAY that exceeds the daily target must
    not reset the streak on today's rollover -- the exact real-world
    shape of the bug (a bill renewal on a day with nothing else spent)."""
    from datetime import date, timedelta
    yesterday = date.today() - timedelta(days=1)
    monkeypatch.setattr(db, "_now_local_date", lambda: yesterday)
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    db.add_expense(CHAT, 150.00, "SGD", "Annual hosting renewal", "Bills & Utilities", is_subscription=True)
    monkeypatch.setattr(db, "_now_local_date", lambda: date.today())
    status = db.get_status(CHAT)
    assert status["current_streak"] == 1  # yesterday counted as within budget, not blown


def test_subscription_expense_still_counts_toward_month_to_date_total():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 69.00, "SGD", "Singtel", "Bills & Utilities", is_subscription=True)
    totals = db.get_month_to_date_total(CHAT)
    assert totals["total"] == 69.00


def test_subscription_expense_still_counts_toward_net_worth():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 69.00, "SGD", "Singtel", "Bills & Utilities", is_subscription=True)
    net_worth = db.get_net_worth(CHAT)
    assert net_worth["total_spend"] == 69.00
    assert net_worth["net_worth"] == -69.00


def test_subscription_expense_still_counts_toward_category_totals():
    from datetime import date, timedelta
    db.get_or_create_user(CHAT)
    today = db.today_str()
    tomorrow = (date.fromisoformat(today) + timedelta(days=1)).isoformat()
    db.add_expense(CHAT, 69.00, "SGD", "Singtel", "Bills & Utilities", is_subscription=True)
    totals = db.get_category_totals(CHAT, today, tomorrow)
    assert totals == [{"category": "Bills & Utilities", "total": 69.00, "n": 1}]


# ---------- balance-safe undo / edit / delete ----------

def test_delete_past_day_expense_credits_balance_back():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = date.today() - timedelta(days=1)
    _insert_on(CHAT, yesterday, 30)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (yesterday.isoformat(), CHAT),
        )
    db.ensure_rollover(CHAT)  # balance becomes 70
    expense = db.get_recent_expenses(CHAT, limit=1)[0]
    db.delete_expense(CHAT, expense["id"])
    # deleting a past expense refunds its amount_base back into balance
    assert db.get_or_create_user(CHAT)["balance"] == 100.0  # 70 + 30


def test_delete_today_expense_does_not_touch_balance():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    expense_id = db.add_expense(CHAT, 30, "SGD", "lunch", "Food")
    db.delete_expense(CHAT, expense_id)
    assert db.get_or_create_user(CHAT)["balance"] == 0.0  # unaffected; today's spend is live, not rolled


def test_edit_past_day_expense_adjusts_balance_by_delta():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = date.today() - timedelta(days=1)
    _insert_on(CHAT, yesterday, 30)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (yesterday.isoformat(), CHAT),
        )
    db.ensure_rollover(CHAT)  # balance = 70
    expense = db.get_recent_expenses(CHAT, limit=1)[0]
    db.edit_expense(CHAT, expense["id"], new_amount=50)  # was 30, now 50 -> spent 20 more
    assert db.get_or_create_user(CHAT)["balance"] == 50.0  # 70 - 20


def test_undo_removes_most_recent_only():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 10, "SGD", "coffee", "Food")
    second_id = db.add_expense(CHAT, 20, "SGD", "lunch", "Food")
    removed = db.delete_most_recent(CHAT)
    assert removed["id"] == second_id
    remaining = db.get_recent_expenses(CHAT)
    assert len(remaining) == 1
    assert remaining[0]["description"] == "coffee"


# ---------- claimables ----------

def test_claimables_are_separate_from_balance():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 300, "SGD", "team dinner", "Food", is_claimable=True)
    status = db.get_status(CHAT)
    assert status["pending_claimable"] == 300
    assert status["balance"] == 0.0  # claimables never touch the daily allowance


def test_clear_claimables_marks_all_as_claimed():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 100, "SGD", "flight", "Travel", is_claimable=True)
    db.add_expense(CHAT, 50, "SGD", "cab", "Transport", is_claimable=True)
    count, total = db.clear_claimables(CHAT)
    assert count == 2
    assert total == 150
    assert db.get_pending_claimables(CHAT) == []


# ---------- edit_expense_date: the "wrong day" correction ----------
# This is a real user-reported scenario: logging something around midnight
# and getting the day wrong ("that was for yesterday, not today"). Balance
# math differs depending on whether each side of the move is "today" (live,
# nothing stored yet) or an already-rolled-over past day (baked into the
# single cumulative `balance` number) -- see edit_expense_date's docstring.

def test_edit_date_today_to_past_decreases_balance():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    expense_id = db.add_expense(CHAT, 30, "SGD", "cashcard top-up", "Transport")
    yesterday = date.today() - timedelta(days=1)
    row = db.edit_expense_date(CHAT, expense_id, yesterday.isoformat())
    assert row["expense_date"] == yesterday.isoformat()
    # moving 30 out of "today" (live) into an already-rolled past day means
    # that day's leftover -- and therefore balance -- drops by 30
    assert db.get_or_create_user(CHAT)["balance"] == -30.0
    # today's live spend no longer includes it
    assert db.get_status(CHAT)["spent_today"] == 0.0


def test_edit_date_past_to_today_increases_balance():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = date.today() - timedelta(days=1)
    _insert_on(CHAT, yesterday, 30)
    expense = db.get_recent_expenses(CHAT, limit=1)[0]
    db.edit_expense_date(CHAT, expense["id"], date.today().isoformat())
    # removing 30 from an already-rolled day frees up 30 of leftover -> balance up
    assert db.get_or_create_user(CHAT)["balance"] == 30.0
    # and it now counts against today's live spend instead
    assert db.get_status(CHAT)["spent_today"] == 30.0


def test_edit_date_past_to_past_nets_zero_balance_change():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    three_days_ago = date.today() - timedelta(days=3)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (three_days_ago.isoformat(), CHAT),
        )
    _insert_on(CHAT, three_days_ago, 20)
    two_days_ago = three_days_ago + timedelta(days=1)
    _insert_on(CHAT, two_days_ago, 30)
    db.ensure_rollover(CHAT)  # rolls 3 days: balance = (100-20)+(100-30)+(100-0) = 250
    assert db.get_or_create_user(CHAT)["balance"] == 250.0

    # move the 30 from two_days_ago to three_days_ago -- both already rolled,
    # so the total balance shouldn't change at all
    moved = next(e for e in db.get_recent_expenses(CHAT, limit=10) if e["amount"] == 30)
    db.edit_expense_date(CHAT, moved["id"], three_days_ago.isoformat())
    assert db.get_or_create_user(CHAT)["balance"] == 250.0
    assert db.get_expense(CHAT, moved["id"])["expense_date"] == three_days_ago.isoformat()


def test_edit_date_same_date_is_noop():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    expense_id = db.add_expense(CHAT, 10, "SGD", "coffee", "Food")
    db.edit_expense_date(CHAT, expense_id, date.today().isoformat())
    assert db.get_or_create_user(CHAT)["balance"] == 0.0


def test_edit_date_claimable_never_touches_balance():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    expense_id = db.add_expense(CHAT, 200, "SGD", "flight", "Travel", is_claimable=True)
    yesterday = date.today() - timedelta(days=1)
    row = db.edit_expense_date(CHAT, expense_id, yesterday.isoformat())
    assert row["expense_date"] == yesterday.isoformat()
    assert db.get_or_create_user(CHAT)["balance"] == 0.0  # claimables never touch balance
    assert db.get_status(CHAT)["pending_claimable"] == 200


def test_edit_date_future_is_clamped_to_today():
    db.get_or_create_user(CHAT)
    expense_id = db.add_expense(CHAT, 10, "SGD", "coffee", "Food")
    tomorrow = date.today() + timedelta(days=1)
    row = db.edit_expense_date(CHAT, expense_id, tomorrow.isoformat())
    assert row["expense_date"] == date.today().isoformat()  # clamped, not tomorrow
    assert db.get_or_create_user(CHAT)["balance"] == 0.0  # same-day no-op after clamping


def test_edit_date_returns_none_for_unknown_expense():
    db.get_or_create_user(CHAT)
    assert db.edit_expense_date(CHAT, 99999, date.today().isoformat()) is None


# ---------- restore_deleted_expense: one-step "undo the deletion" ----------

def test_restore_deleted_expense_reverses_past_day_delete():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = date.today() - timedelta(days=1)
    _insert_on(CHAT, yesterday, 30)
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_rollover_date = ? WHERE chat_id = ?",
            (yesterday.isoformat(), CHAT),
        )
    db.ensure_rollover(CHAT)  # balance = 70
    deleted = db.delete_expense(CHAT, db.get_recent_expenses(CHAT, limit=1)[0]["id"])
    assert db.get_or_create_user(CHAT)["balance"] == 100.0  # 70 + 30 refunded
    restored = db.restore_deleted_expense(CHAT, deleted)
    assert restored["amount"] == 30
    assert restored["expense_date"] == yesterday.isoformat()
    assert db.get_or_create_user(CHAT)["balance"] == 70.0  # back to pre-delete balance


def test_restore_deleted_expense_reverses_today_delete_with_no_balance_change():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    expense_id = db.add_expense(CHAT, 20, "SGD", "coffee", "Food")
    deleted = db.delete_expense(CHAT, expense_id)
    assert db.get_or_create_user(CHAT)["balance"] == 0.0
    db.restore_deleted_expense(CHAT, deleted)
    assert db.get_or_create_user(CHAT)["balance"] == 0.0  # today's spend is live either way
    assert db.get_status(CHAT)["spent_today"] == 20.0


def test_restore_deleted_expense_preserves_claimable_flag():
    db.get_or_create_user(CHAT)
    expense_id = db.add_expense(CHAT, 300, "SGD", "team dinner", "Food", is_claimable=True)
    deleted = db.delete_expense(CHAT, expense_id)
    restored = db.restore_deleted_expense(CHAT, deleted)
    assert restored["is_claimable"] == 1
    assert db.get_status(CHAT)["pending_claimable"] == 300


# ---------- month-to-date ----------

def test_month_to_date_total_only_counts_current_month():
    """The rolling `balance` is supplemented, not replaced, by a fresh
    month-to-date figure -- this must reset to zero on the 1st and ignore
    spend from prior months entirely, computed straight from expense_date
    with no stored running total (so it needs no schema change)."""
    db.get_or_create_user(CHAT)
    today = date.today()
    first_of_month = today.replace(day=1)
    last_month_day = first_of_month - timedelta(days=1)

    _insert_on(CHAT, last_month_day, 999)  # must be excluded
    _insert_on(CHAT, first_of_month, 30)
    db.add_expense(CHAT, 20, "SGD", "today thing", "Food")  # today

    mtd = db.get_month_to_date_total(CHAT)
    assert mtd["total"] == 50.0
    assert mtd["month_start"] == first_of_month.isoformat()
    assert mtd["today"] == today.isoformat()
    assert mtd["days_elapsed"] == (today - first_of_month).days + 1


def test_month_to_date_total_ignores_claimable_expenses():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 300, "SGD", "team dinner", "Food", is_claimable=True)
    db.add_expense(CHAT, 20, "SGD", "coffee", "Food")
    assert db.get_month_to_date_total(CHAT)["total"] == 20.0


def test_month_to_date_total_zero_with_no_expenses():
    db.get_or_create_user(CHAT)
    assert db.get_month_to_date_total(CHAT)["total"] == 0.0


# ---------- proactive insight dedup ----------

def test_was_insight_sent_recently_is_false_before_anything_is_recorded():
    db.get_or_create_user(CHAT)
    assert db.was_insight_sent_recently(CHAT, "expense:category_spike:Dining") is False


def test_record_and_check_insight_sent_roundtrip():
    db.get_or_create_user(CHAT)
    db.record_insight_sent(CHAT, "expense:category_spike:Dining")
    assert db.was_insight_sent_recently(CHAT, "expense:category_spike:Dining") is True
    # A different dedup_key must not be suppressed by an unrelated one.
    assert db.was_insight_sent_recently(CHAT, "vitals:weight_trend") is False


def test_was_insight_sent_recently_respects_the_within_days_window():
    """Regression guard for the exact reason this table exists: an insight
    recorded outside the throttle window must be treated as fresh again,
    not suppressed forever."""
    db.get_or_create_user(CHAT)
    old_date = (date.today() - timedelta(days=10)).isoformat()
    db.record_insight_sent(CHAT, "lift:stale:squat", sent_date=old_date)
    assert db.was_insight_sent_recently(CHAT, "lift:stale:squat", within_days=7) is False
    assert db.was_insight_sent_recently(CHAT, "lift:stale:squat", within_days=14) is True


# ---------- init_db: forward-compatible migrations against an existing db ----------

def test_init_db_migrates_an_existing_tasks_table_to_add_recurrence_frequency():
    """Regression guard for a real production incident: adding a column to
    the `tasks` CREATE TABLE statement does nothing for a database that
    already has a `tasks` table from before that column existed --
    CREATE TABLE IF NOT EXISTS is a no-op there. Without the matching
    _add_column_if_missing call, every read of a pre-existing task row
    (e.g. tasks._recent_tasks_for_ai, which runs on EVERY handle_text call)
    raised KeyError: 'recurrence_frequency' against a live database that
    was never actually given the new column. Simulates that "old-shape"
    production table directly, then re-runs init_db() the same way the
    app does on every startup, and checks the migration actually ran."""
    with db.get_conn() as conn:
        conn.execute("DROP TABLE tasks")
        conn.execute("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                due_at TEXT,
                done INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("INSERT INTO tasks (chat_id, title) VALUES (?, ?)", (CHAT, "pre-existing to-do"))

    db.init_db()  # the same call app.py makes on every startup

    row = db.get_recent_tasks(CHAT, limit=1)[0]
    assert row["title"] == "pre-existing to-do"
    assert row["recurrence_frequency"] is None  # column exists now, old row reads as "not recurring"
    # And the column is actually usable going forward, not just present.
    task_id = db.add_task(CHAT, "new recurring to-do", recurrence_frequency="monthly")
    assert db.get_task(CHAT, task_id)["recurrence_frequency"] == "monthly"


def test_init_db_migrates_an_existing_subscriptions_table_to_the_new_schema():
    """Regression guard for TWO real, sequential production incidents, both
    the same root cause as the tasks one above -- the subscriptions schema
    redesign (billing_day/last_logged_month -> frequency/next_renewal_date/
    card/notes/is_claimable/last_logged_date) was built on the wrong
    assumption that no subscriptions table existed in production yet. It
    did, from an earlier push, in the OLD shape, and CREATE TABLE IF NOT
    EXISTS silently left it there:

    1. Every new subscriptions function broke with 'sqlite3.OperationalError:
       no such column: next_renewal_date' -- fixed by the six
       _add_column_if_missing calls above.
    2. That fix alone wasn't enough: billing_day in the REAL production
       table is NOT NULL with no DEFAULT (an earlier version of this test
       wrongly assumed it was nullable, which is exactly why it didn't catch
       this), and add_subscription never sets it, so every insert still
       broke -- this time with 'sqlite3.IntegrityError: NOT NULL constraint
       failed: subscriptions.billing_day'. Fixed by
       _drop_subscriptions_legacy_columns actually rebuilding the table.

    Simulates that old-shape table directly, with billing_day NOT NULL and a
    pre-existing row (the way production actually had it), re-runs
    init_db(), and checks every new column is present and usable AND that
    billing_day/last_logged_month are actually gone, not just worked around."""
    with db.get_conn() as conn:
        conn.execute("DROP TABLE subscriptions")
        conn.execute(f"""
            CREATE TABLE subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT '{db.config.BASE_CURRENCY}',
                amount_base REAL NOT NULL,
                category TEXT,
                billing_day INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                last_logged_month TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "INSERT INTO subscriptions (chat_id, name, amount, currency, amount_base, billing_day) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (CHAT, "pre-existing sub", 15.98, "SGD", 15.98, 25),
        )

    db.init_db()  # the same call app.py makes on every startup

    row = db.get_subscriptions(CHAT)[0]
    assert row["name"] == "pre-existing sub"
    assert row["frequency"] == "monthly"  # DEFAULT applied to the pre-existing row
    assert row["is_claimable"] == 0
    assert row["next_renewal_date"] == db.today_str()  # backfilled placeholder, never left NULL
    assert row["card"] is None
    assert row["last_logged_date"] is None
    assert set(row.keys()) >= {
        "id", "chat_id", "name", "amount", "currency", "amount_base", "category",
        "frequency", "next_renewal_date", "card", "notes", "is_claimable",
        "active", "last_logged_date", "created_at",
    }
    assert "billing_day" not in row.keys()  # the whole point: actually gone, not just unused
    assert "last_logged_month" not in row.keys()

    # init_db() must also be idempotent: running it again against the
    # now-clean table (no billing_day left to find) must not error.
    db.init_db()

    # And every new column is actually usable going forward, not just
    # present -- this INSERT is exactly what crashed with "NOT NULL
    # constraint failed: subscriptions.billing_day" before the fix, since
    # add_subscription never supplies that column.
    sub_id = db.add_subscription(CHAT, "new sub", 9.99, "SGD", "quarterly", "2027-01-01", card="OCBC")
    assert db.get_subscription(CHAT, sub_id)["card"] == "OCBC"
