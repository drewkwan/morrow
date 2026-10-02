"""
Tests for Morrow's cross-domain synthesis ("rundown"): db.py's date-range
readers for meals/workouts/vitals, bot.py's _rundown_payload computing real
7-day figures, ai.py's answer_with_rundown narrating them, and the
natural-language "rundown" intent + /rundown command wiring, including the
never-silent fallback when the Claude synthesis call itself fails. Same
discipline as test_memory.py / test_vitals.py: no real network, no real
Claude calls (ai._get_client is always mocked), throwaway SQLite per test.
"""

import asyncio
import datetime as dt

import ai
import bot
import db
from conftest import CHAT
from test_ai import _mock_client
from test_meals_workouts import FakeContext, FakeUpdate


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _days_ago(n):
    return (dt.date.today() - dt.timedelta(days=n)).isoformat()


# ---------- db.py: date-range readers ----------

def test_get_meals_in_range_respects_bounds():
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO meals (chat_id, meal_date, items, calories_estimate) VALUES (?, ?, '[]', 400)",
            (CHAT, _days_ago(10)),
        )
        conn.execute(
            "INSERT INTO meals (chat_id, meal_date, items, calories_estimate) VALUES (?, ?, '[]', 500)",
            (CHAT, _days_ago(2)),
        )
    window_start = _days_ago(6)
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    rows = db.get_meals_in_range(CHAT, window_start, tomorrow)
    assert len(rows) == 1
    assert rows[0]["calories_estimate"] == 500


def test_get_workouts_in_range_respects_bounds():
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO workouts (chat_id, workout_date, activity) VALUES (?, ?, 'old run')",
            (CHAT, _days_ago(30)),
        )
        conn.execute(
            "INSERT INTO workouts (chat_id, workout_date, activity) VALUES (?, ?, 'tennis')",
            (CHAT, _days_ago(1)),
        )
    rows = db.get_workouts_in_range(CHAT, _days_ago(6), (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert [r["activity"] for r in rows] == ["tennis"]


def test_get_vitals_in_range_is_oldest_first():
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO vitals (chat_id, vitals_date, weight_kg) VALUES (?, ?, 77.0)",
            (CHAT, _days_ago(5)),
        )
        conn.execute(
            "INSERT INTO vitals (chat_id, vitals_date, weight_kg) VALUES (?, ?, 76.4)",
            (CHAT, _days_ago(1)),
        )
    rows = db.get_vitals_in_range(CHAT, _days_ago(6), (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert [r["weight_kg"] for r in rows] == [77.0, 76.4]  # oldest first


# ---------- bot.py: _rundown_payload ----------

def test_rundown_payload_computes_real_figures():
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    db.add_expense(CHAT, 20, "SGD", "lunch", "Food")
    db.add_meal(CHAT, "lunch", ["mango"], 80, 120, 100)
    db.add_workout(CHAT, "tennis", duration_min=60)
    db.add_vitals(CHAT, weight_kg=76.0, sleep_hours=7)
    db.add_vitals(CHAT, weight_kg=75.5, sleep_hours=6)

    payload = bot._rundown_payload(CHAT)
    assert payload["window_days"] == 7
    assert payload["balance"]["spent_today"] == 20
    assert payload["meals"]["count"] == 1
    assert payload["meals"]["total_calories_estimate"] == 100
    assert payload["workouts"]["count"] == 1
    assert payload["workouts"]["activities"] == ["tennis"]
    assert payload["vitals"]["checkins"] == 2
    assert payload["vitals"]["latest_weight_kg"] == 75.5
    assert payload["vitals"]["weight_change_kg"] == -0.5
    assert payload["vitals"]["avg_sleep_hours"] == 6.5


def test_rundown_payload_nulls_out_empty_sections_rather_than_zeroing():
    """No meals/workouts/vitals logged this week -- the payload should say
    so with None/empty, not fabricate a misleading 0."""
    db.get_or_create_user(CHAT)
    payload = bot._rundown_payload(CHAT)
    assert payload["meals"]["count"] == 0
    assert payload["meals"]["total_calories_estimate"] is None
    assert payload["vitals"]["checkins"] == 0
    assert payload["vitals"]["latest_weight_kg"] is None
    assert payload["vitals"]["weight_change_kg"] is None
    assert payload["vitals"]["avg_sleep_hours"] is None
    assert payload["workouts"]["activities"] == []


def test_rundown_payload_excludes_data_outside_the_window():
    db.get_or_create_user(CHAT)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO workouts (chat_id, workout_date, activity) VALUES (?, ?, 'ancient gym session')",
            (CHAT, _days_ago(60)),
        )
    payload = bot._rundown_payload(CHAT)
    assert payload["workouts"]["count"] == 0


# ---------- ai.py: answer_with_rundown ----------

def test_answer_with_rundown_returns_model_text(monkeypatch):
    _mock_client(monkeypatch, "You're on a good streak and weight's trending down slightly.")
    payload = {"window_days": 7, "balance": {}, "meals": {}, "workouts": {}, "vitals": {}}
    text = ai.answer_with_rundown(payload)
    assert "streak" in text


def test_answer_with_rundown_raises_on_api_failure(monkeypatch):
    """Unlike parse_message/extract_*, answer_with_rundown does NOT
    self-guard -- bot.py's _rundown_reply_text is responsible for the
    fallback, matching answer_with_trends's existing division of labor."""
    _mock_client(monkeypatch, ConnectionError("network blip"))
    try:
        ai.answer_with_rundown({"window_days": 7})
        assert False, "expected the ConnectionError to propagate"
    except ConnectionError:
        pass


# ---------- bot.py: natural-language + command + fallback ----------

def _no_op_extra_fields():
    return {
        "amount": None, "currency": None, "description": None, "category": None, "is_claimable": None,
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "memory_label": None, "memory_content": None, "memory_category": None,
    }


def test_natural_language_rundown_replies_with_synthesis(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_workout(CHAT, "tennis", duration_min=60)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "rundown", "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    monkeypatch.setattr(bot.ai, "answer_with_rundown", lambda payload: "You played tennis this week, nice.")
    update = FakeUpdate(CHAT, text="how am I doing this week?")
    _run(bot.handle_text(update, FakeContext()))
    assert update.message.replies[-1] == "You played tennis this week, nice."


def test_natural_language_rundown_reply_is_not_narrated_a_second_time(monkeypatch):
    """Regression guard: handlers.py must pass narrate=False for the
    'rundown' intent (see replies._reply's docstring) -- ai.answer_with_rundown
    output is already a full narration, and running it back through
    ai.narrate_reply would be redundant and risks a second pass quietly
    drifting from the first pass's real numbers. Uses a non-identity fake
    narrate_reply (unlike the autouse passthrough fixture) so a regression
    here can actually be detected."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "rundown", "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    monkeypatch.setattr(bot.ai, "answer_with_rundown", lambda payload: "You played tennis this week, nice.")
    monkeypatch.setattr(bot.ai, "narrate_reply", lambda text, *a, **kw: f"RE-NARRATED: {text}")
    update = FakeUpdate(CHAT, text="how am I doing this week?")
    _run(bot.handle_text(update, FakeContext()))
    assert update.message.replies[-1] == "You played tennis this week, nice."


def test_natural_language_rundown_logs_to_conversation_history(monkeypatch):
    """Regression guard: _reply (not a bare reply_text) must be used so the
    rundown reply lands in the rolling messages table like every other
    handle_text branch."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "rundown", "clarification_question": None, "casual_reply": None,
            **_no_op_extra_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    monkeypatch.setattr(bot.ai, "answer_with_rundown", lambda payload: "All quiet this week.")
    update = FakeUpdate(CHAT, text="give me a rundown")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_recent_messages(CHAT)
    assert rows[-1]["role"] == "morrow"
    assert rows[-1]["content"] == "All quiet this week."


def test_rundown_command_uses_real_data(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=75.0)
    captured = {}

    def fake_answer_with_rundown(payload):
        captured["payload"] = payload
        return "Weight's holding steady."

    monkeypatch.setattr(bot.ai, "answer_with_rundown", fake_answer_with_rundown)
    update = FakeUpdate(CHAT)
    _run(bot.rundown_cmd(update, FakeContext()))
    assert update.message.replies[-1] == "Weight's holding steady."
    assert captured["payload"]["vitals"]["latest_weight_kg"] == 75.0


def test_rundown_falls_back_to_raw_breakdown_when_synthesis_fails(monkeypatch):
    """Never silent -- mirrors /summary's existing fallback discipline."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    db.add_workout(CHAT, "tennis")

    def fake_answer_with_rundown(payload):
        raise ConnectionError("network blip")

    monkeypatch.setattr(bot.ai, "answer_with_rundown", fake_answer_with_rundown)
    update = FakeUpdate(CHAT)
    _run(bot.rundown_cmd(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Balance" in reply
    assert "tennis" in reply


# ---------- rundown.py: _resolve_day ----------

def test_resolve_day_none_or_zero_means_today():
    assert bot._resolve_day(None) == db.today_str()
    assert bot._resolve_day(0) == db.today_str()


def test_resolve_day_counts_back_from_today():
    assert bot._resolve_day(1) == _days_ago(1)


def test_resolve_day_is_not_clamped():
    """Regression test for a real reported bug: _resolve_day used to
    silently clamp anything past 14 days back to exactly 14 days ago, so
    asking about a day months back would silently answer about the wrong
    day instead. day_stats has no recency limit now -- db.py's own
    get_meals_in_range/get_workouts_in_range/get_vitals_in_range are
    themselves unbounded, so there's no reason for this to be capped."""
    assert bot._resolve_day(999) == _days_ago(999)
    assert bot._resolve_day(76) == _days_ago(76)


# ---------- rundown.py: _day_stats_payload ----------

def test_day_stats_payload_computes_real_figures_for_one_day():
    db.get_or_create_user(CHAT)
    yesterday = _days_ago(1)
    db.add_meal(CHAT, "lunch", ["mango", "rice"], 400, 600, 500, meal_date=yesterday)
    db.add_workout(CHAT, "tennis", calories_burned=800, workout_date=yesterday)
    db.add_vitals(CHAT, weight_kg=76.1, vitals_date=yesterday)
    # A meal logged TODAY shouldn't leak into yesterday's payload.
    db.add_meal(CHAT, "dinner", ["noodles"], 300, 500, 400)

    payload = bot._day_stats_payload(CHAT, yesterday)
    assert payload["day"] == yesterday
    assert payload["is_today"] is False
    assert payload["meals"]["count"] == 1
    assert payload["meals"]["items"] == ["mango", "rice"]
    assert payload["meals"]["total_calories_estimate"] == 500
    assert payload["workouts"]["total_calories_burned"] == 800
    assert payload["net_calories"] == -300  # 500 in - 800 out
    assert payload["vitals"]["weight_kg"] == 76.1


def test_day_stats_payload_nulls_out_net_when_nothing_logged_at_all():
    db.get_or_create_user(CHAT)
    payload = bot._day_stats_payload(CHAT, db.today_str())
    assert payload["meals"]["count"] == 0
    assert payload["workouts"]["count"] == 0
    assert payload["net_calories"] is None
    assert payload["vitals"] is None


def test_day_stats_payload_net_is_meaningful_with_only_one_side_logged():
    """A day with meals but no workout burn logged shouldn't null out net --
    it's just calories_in - 0, a real (if partial) number."""
    db.get_or_create_user(CHAT)
    db.add_meal(CHAT, "lunch", ["mango"], 400, 600, 500)
    payload = bot._day_stats_payload(CHAT, db.today_str())
    assert payload["net_calories"] == 500


# ---------- ai.py: answer_with_day_stats ----------

def test_answer_with_day_stats_returns_model_text(monkeypatch):
    _mock_client(monkeypatch, "You ate about 500 kcal and burned 800, so a solid deficit.")
    payload = {"day": db.today_str(), "is_today": True, "meals": {}, "workouts": {}, "net_calories": None,
               "vitals": None}
    text = ai.answer_with_day_stats(payload)
    assert "deficit" in text


def test_answer_with_day_stats_raises_on_api_failure(monkeypatch):
    """Same division of labor as answer_with_rundown -- the reply-text
    wrapper (not this function) is responsible for the fallback."""
    _mock_client(monkeypatch, ConnectionError("network blip"))
    try:
        ai.answer_with_day_stats({"day": db.today_str()})
        assert False, "expected the ConnectionError to propagate"
    except ConnectionError:
        pass


# ---------- rundown.py: _day_stats_fallback_text ----------

def test_day_stats_fallback_text_never_silent_on_empty_day():
    db.get_or_create_user(CHAT)
    payload = bot._day_stats_payload(CHAT, db.today_str())
    text = bot._day_stats_fallback_text(payload)
    assert "nothing logged" in text


# ---------- natural-language "day_stats" intent + /daystats command ----------

def _fake_parse_message_day_stats(days_ago):
    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "day_stats", "clarification_question": None, "casual_reply": None,
            "day_stats_days_ago": days_ago,
            **_no_op_extra_fields(),
        }
    return fake_parse_message


def test_natural_language_day_stats_replies_with_synthesis_for_the_right_day(monkeypatch):
    db.get_or_create_user(CHAT)
    yesterday = _days_ago(1)
    db.add_meal(CHAT, "lunch", ["mala"], 900, 1200, 1050, meal_date=yesterday)
    captured = {}

    def fake_answer_with_day_stats(payload):
        captured["payload"] = payload
        return "Yesterday you had about 1050 kcal, mostly from the mala."

    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_day_stats(1))
    monkeypatch.setattr(bot.ai, "answer_with_day_stats", fake_answer_with_day_stats)
    update = FakeUpdate(CHAT, text="stats from last night")
    _run(bot.handle_text(update, FakeContext()))
    assert update.message.replies[-1] == "Yesterday you had about 1050 kcal, mostly from the mala."
    assert captured["payload"]["day"] == yesterday
    assert captured["payload"]["meals"]["total_calories_estimate"] == 1050


def test_natural_language_day_stats_reply_is_not_narrated_a_second_time(monkeypatch):
    """Same regression guard as rundown's, for narrate=False on the
    'day_stats' intent -- see that test's docstring."""
    db.get_or_create_user(CHAT)

    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_day_stats(0))
    monkeypatch.setattr(bot.ai, "answer_with_day_stats", lambda payload: "Nothing logged today yet.")
    monkeypatch.setattr(bot.ai, "narrate_reply", lambda text, *a, **kw: f"RE-NARRATED: {text}")
    update = FakeUpdate(CHAT, text="stats for today")
    _run(bot.handle_text(update, FakeContext()))
    assert update.message.replies[-1] == "Nothing logged today yet."


def test_natural_language_day_stats_logs_to_conversation_history(monkeypatch):
    db.get_or_create_user(CHAT)

    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_day_stats(0))
    monkeypatch.setattr(bot.ai, "answer_with_day_stats", lambda payload: "Nothing logged today yet.")
    update = FakeUpdate(CHAT, text="stats for today")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_recent_messages(CHAT)
    assert rows[-1]["role"] == "morrow"
    assert rows[-1]["content"] == "Nothing logged today yet."


def test_daystats_command_defaults_to_today(monkeypatch):
    db.get_or_create_user(CHAT)
    captured = {}

    def fake_answer_with_day_stats(payload):
        captured["payload"] = payload
        return "Today's numbers."

    monkeypatch.setattr(bot.ai, "answer_with_day_stats", fake_answer_with_day_stats)
    update = FakeUpdate(CHAT)
    _run(bot.daystats_cmd(update, FakeContext()))
    assert update.message.replies[-1] == "Today's numbers."
    assert captured["payload"]["day"] == db.today_str()


def test_daystats_command_accepts_yesterday_and_n(monkeypatch):
    db.get_or_create_user(CHAT)
    captured = {}

    def fake_answer_with_day_stats(payload):
        captured.setdefault("days", []).append(payload["day"])
        return "ok"

    monkeypatch.setattr(bot.ai, "answer_with_day_stats", fake_answer_with_day_stats)
    context1 = FakeContext()
    context1.args = ["yesterday"]
    _run(bot.daystats_cmd(FakeUpdate(CHAT), context1))
    context2 = FakeContext()
    context2.args = ["3"]
    _run(bot.daystats_cmd(FakeUpdate(CHAT), context2))
    assert captured["days"] == [_days_ago(1), _days_ago(3)]


def test_daystats_command_rejects_bad_argument(monkeypatch):
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["banana"]
    _run(bot.daystats_cmd(update, context))
    assert "Usage:" in update.message.replies[-1]


def test_day_stats_falls_back_to_raw_breakdown_when_synthesis_fails(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_meal(CHAT, "lunch", ["mango"], 400, 600, 500)

    def fake_answer_with_day_stats(payload):
        raise ConnectionError("network blip")

    monkeypatch.setattr(bot.ai, "answer_with_day_stats", fake_answer_with_day_stats)
    update = FakeUpdate(CHAT)
    _run(bot.daystats_cmd(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "500" in reply


# ---------- rundown.py: _resolve_trend_range ----------

def test_resolve_trend_range_none_start_uses_beginning_placeholder():
    """Regression test for a real reported bug: "summarise my weight
    progression since the beginning" used to be misread as a today-scoped
    question. trend_start_days_ago=None means "since the beginning" -- there's
    no real day-count for that, so it resolves to a placeholder start date old
    enough to predate any real Morrow data, never today."""
    start, end = bot._resolve_trend_range(None, None)
    assert start == "2000-01-01"
    assert end == (dt.date.today() + dt.timedelta(days=1)).isoformat()


def test_resolve_trend_range_counts_back_from_today():
    start, end = bot._resolve_trend_range(10, None)
    assert start == _days_ago(10)
    assert end == (dt.date.today() + dt.timedelta(days=1)).isoformat()


def test_resolve_trend_range_end_days_ago_bounds_before_today():
    start, end = bot._resolve_trend_range(10, 3)
    assert start == _days_ago(10)
    assert end == (dt.date.today() - dt.timedelta(days=2)).isoformat()  # _days_ago(3) + 1 day


# ---------- rundown.py: _trend_series ----------

def test_trend_series_orders_by_date_not_insertion_order():
    """Same real bug as get_recent_vitals/get_recent_meals/get_recent_workouts
    (a historical backfill inserts old-dated rows LAST, with the newest ids)
    -- _trend_series must sort explicitly by date, not trust the id-ordered
    rows get_vitals_in_range hands back."""
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date=_days_ago(60))
    db.add_vitals(CHAT, weight_kg=75.0, vitals_date=_days_ago(5))
    # Inserted last (highest id), but dated well before the entry above.
    db.add_vitals(CHAT, weight_kg=78.0, vitals_date=_days_ago(30))
    series = bot._trend_series(CHAT, "weight", "2000-01-01", (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert [d for d, _ in series] == sorted(d for d, _ in series)


def test_trend_series_calories_in_sums_per_day_from_meals():
    db.get_or_create_user(CHAT)
    yesterday = _days_ago(1)
    db.add_meal(CHAT, "lunch", ["mango"], 400, 600, 500, meal_date=yesterday)
    db.add_meal(CHAT, "dinner", ["rice"], 300, 500, 400, meal_date=yesterday)
    db.add_meal(CHAT, "lunch", ["egg"], 100, 200, 150, meal_date=db.today_str())
    series = bot._trend_series(CHAT, "calories_in", "2000-01-01",
                                (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert dict(series) == {yesterday: 900, db.today_str(): 150}


def test_trend_series_spending_uses_get_daily_totals():
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 20, "SGD", "lunch", "Food")
    series = bot._trend_series(CHAT, "spending", "2000-01-01",
                                (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert dict(series) == {db.today_str(): 20}


# ---------- rundown.py: _trend_payload ----------

def test_trend_payload_computes_real_first_last_min_max_change():
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date=_days_ago(30))
    db.add_vitals(CHAT, weight_kg=82.0, vitals_date=_days_ago(15))  # the real max
    db.add_vitals(CHAT, weight_kg=75.5, vitals_date=_days_ago(1))  # the real min and last

    payload = bot._trend_payload(CHAT, "weight", "2000-01-01",
                                  (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert payload["metric"] == "weight"
    assert payload["unit"] == "kg"
    assert payload["count"] == 3
    assert payload["start_date"] == _days_ago(30)  # real earliest date found, not the placeholder
    assert payload["end_date"] == _days_ago(1)
    assert payload["first"] == {"date": _days_ago(30), "value": 80.0}
    assert payload["last"] == {"date": _days_ago(1), "value": 75.5}
    assert payload["min"] == {"date": _days_ago(1), "value": 75.5}
    assert payload["max"] == {"date": _days_ago(15), "value": 82.0}
    assert payload["change"] == -4.5  # 75.5 - 80.0


def test_trend_payload_nulls_out_when_nothing_logged():
    db.get_or_create_user(CHAT)
    payload = bot._trend_payload(CHAT, "weight", "2000-01-01",
                                  (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert payload["count"] == 0
    assert payload["first"] is None
    assert payload["last"] is None
    assert payload["min"] is None
    assert payload["max"] is None
    assert payload["change"] is None
    assert payload["start_date"] is None
    assert payload["end_date"] is None


def test_trend_payload_change_is_none_with_only_one_data_point():
    """A single reading has no real "change" to report -- null, not 0
    (0 would falsely claim "no movement")."""
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=76.0)
    payload = bot._trend_payload(CHAT, "weight", "2000-01-01",
                                  (dt.date.today() + dt.timedelta(days=1)).isoformat())
    assert payload["count"] == 1
    assert payload["change"] is None
    assert payload["first"] == payload["last"]


# ---------- ai.py: answer_with_trend ----------

def test_answer_with_trend_returns_model_text(monkeypatch):
    _mock_client(monkeypatch, "You're down 4.5kg since the beginning, nice steady progress.")
    payload = {"metric": "weight", "unit": "kg", "start_date": None, "end_date": None, "count": 0,
               "first": None, "last": None, "min": None, "max": None, "change": None}
    text = ai.answer_with_trend(payload)
    assert "4.5kg" in text


def test_answer_with_trend_raises_on_api_failure(monkeypatch):
    """Same division of labor as answer_with_rundown/answer_with_day_stats --
    the reply-text wrapper (not this function) is responsible for the
    fallback."""
    _mock_client(monkeypatch, ConnectionError("network blip"))
    try:
        ai.answer_with_trend({"metric": "weight"})
        assert False, "expected the ConnectionError to propagate"
    except ConnectionError:
        pass


# ---------- rundown.py: _trend_fallback_text ----------

def test_trend_fallback_text_never_silent_on_empty_range():
    payload = bot._trend_payload(CHAT, "weight", "2000-01-01",
                                  (dt.date.today() + dt.timedelta(days=1)).isoformat())
    text = bot._trend_fallback_text(payload)
    assert "Nothing logged" in text


def test_trend_fallback_text_reports_real_change():
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date=_days_ago(10))
    db.add_vitals(CHAT, weight_kg=75.5, vitals_date=_days_ago(1))
    payload = bot._trend_payload(CHAT, "weight", "2000-01-01",
                                  (dt.date.today() + dt.timedelta(days=1)).isoformat())
    text = bot._trend_fallback_text(payload)
    assert "-4.5kg" in text


# ---------- natural-language "trend" intent + /trend command ----------

def _fake_parse_message_trend(metric, start_days_ago=None, end_days_ago=None):
    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "trend", "clarification_question": None, "casual_reply": None,
            "trend_metric": metric, "trend_start_days_ago": start_days_ago, "trend_end_days_ago": end_days_ago,
            **_no_op_extra_fields(),
        }
    return fake_parse_message


def test_natural_language_trend_replies_with_synthesis_for_the_right_metric(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date=_days_ago(30))
    db.add_vitals(CHAT, weight_kg=75.5, vitals_date=_days_ago(1))
    captured = {}

    def fake_answer_with_trend(payload):
        captured["payload"] = payload
        return "You've lost 4.5kg since the beginning, solid progress."

    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_trend("weight"))
    monkeypatch.setattr(bot.ai, "answer_with_trend", fake_answer_with_trend)
    update = FakeUpdate(CHAT, text="summarise my weight progression since the beginning")
    _run(bot.handle_text(update, FakeContext()))
    assert update.message.replies[-1] == "You've lost 4.5kg since the beginning, solid progress."
    assert captured["payload"]["metric"] == "weight"
    assert captured["payload"]["count"] == 2
    assert captured["payload"]["change"] == -4.5


def test_natural_language_trend_reply_is_not_narrated_a_second_time(monkeypatch):
    """Same regression guard as rundown/day_stats's, for narrate=False on the
    'trend' intent -- see those tests' docstrings."""
    db.get_or_create_user(CHAT)

    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_trend("weight"))
    monkeypatch.setattr(bot.ai, "answer_with_trend", lambda payload: "Nothing logged for that yet.")
    monkeypatch.setattr(bot.ai, "narrate_reply", lambda text, *a, **kw: f"RE-NARRATED: {text}")
    update = FakeUpdate(CHAT, text="how's my weight trended")
    _run(bot.handle_text(update, FakeContext()))
    assert update.message.replies[-1] == "Nothing logged for that yet."


def test_natural_language_trend_logs_to_conversation_history(monkeypatch):
    db.get_or_create_user(CHAT)

    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_trend("sleep"))
    monkeypatch.setattr(bot.ai, "answer_with_trend", lambda payload: "Sleep's been steady.")
    update = FakeUpdate(CHAT, text="how's my sleep trended")
    _run(bot.handle_text(update, FakeContext()))
    rows = db.get_recent_messages(CHAT)
    assert rows[-1]["role"] == "morrow"
    assert rows[-1]["content"] == "Sleep's been steady."


def test_natural_language_trend_asks_rather_than_guesses_when_metric_missing(monkeypatch):
    """Regression guard: if the model classifies as 'trend' but doesn't
    settle on one of the six covered metrics, Morrow should ask which one
    rather than silently defaulting to something that could be wrong."""
    db.get_or_create_user(CHAT)

    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_trend(None))
    update = FakeUpdate(CHAT, text="how have I been trending")
    _run(bot.handle_text(update, FakeContext()))
    assert "which" in update.message.replies[-1].lower()


def test_trend_command_defaults_to_since_the_beginning(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date=_days_ago(500))
    captured = {}

    def fake_answer_with_trend(payload):
        captured["payload"] = payload
        return "ok"

    monkeypatch.setattr(bot.ai, "answer_with_trend", fake_answer_with_trend)
    context = FakeContext()
    context.args = ["weight"]
    _run(bot.trend_cmd(FakeUpdate(CHAT), context))
    assert captured["payload"]["start_date"] == _days_ago(500)  # the old reading wasn't excluded


def test_trend_command_accepts_n_days(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date=_days_ago(500))
    db.add_vitals(CHAT, weight_kg=75.0, vitals_date=_days_ago(3))
    captured = {}

    def fake_answer_with_trend(payload):
        captured["payload"] = payload
        return "ok"

    monkeypatch.setattr(bot.ai, "answer_with_trend", fake_answer_with_trend)
    context = FakeContext()
    context.args = ["weight", "10"]
    _run(bot.trend_cmd(FakeUpdate(CHAT), context))
    # Bounded to the last 10 days -- the 500-days-ago reading is excluded.
    assert captured["payload"]["count"] == 1
    assert captured["payload"]["first"]["value"] == 75.0


def test_trend_command_rejects_unknown_metric(monkeypatch):
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["bench_press"]
    _run(bot.trend_cmd(update, context))
    assert "Unknown metric" in update.message.replies[-1]


def test_trend_command_rejects_bad_range_argument(monkeypatch):
    db.get_or_create_user(CHAT)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["weight", "banana"]
    _run(bot.trend_cmd(update, context))
    assert "Usage:" in update.message.replies[-1]


def test_trend_falls_back_to_raw_breakdown_when_synthesis_fails(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_vitals(CHAT, weight_kg=80.0, vitals_date=_days_ago(10))
    db.add_vitals(CHAT, weight_kg=75.5, vitals_date=_days_ago(1))

    def fake_answer_with_trend(payload):
        raise ConnectionError("network blip")

    monkeypatch.setattr(bot.ai, "answer_with_trend", fake_answer_with_trend)
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["weight"]
    _run(bot.trend_cmd(update, context))
    reply = update.message.replies[-1]
    assert "-4.5kg" in reply
