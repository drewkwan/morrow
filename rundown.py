"""
Cross-domain synthesis: /rundown, and the 'day_stats' intent. Real,
deterministically-computed figures -- across all four domains for the
trailing window (_rundown_payload), or across meals/workouts/vitals for one
named day (_day_stats_payload) -- handed to Claude only to narrate, never to
invent a number. Same "never let the model guess a number" discipline as
finance._balance_text.

_day_stats_payload exists specifically because a single-day "what did I eat
yesterday" / "calories in vs out for Saturday" question used to fall through
to ai.py's generic 'casual' intent, where the model was handed raw recent
meals/workouts (not even day-filtered) and asked to sum and subtract them
itself, in freeform text, from scratch, every time it was asked -- a real
observed bug where the same question got a different, sometimes internally
contradictory (even sign-flipped) answer on each successive ask, and once
even backfilled a fake "I corrected that entry" explanation for its own
inconsistency instead of just re-reading the DB. day_stats guarantees a
real, identical-every-time DB read backs the answer instead.
"""

import logging
from datetime import date, timedelta

from telegram import Update
from telegram.ext import ContextTypes

import ai
import db
from access import _reject_if_not_allowed

logger = logging.getLogger(__name__)

RUNDOWN_WINDOW_DAYS = 7


def _rundown_payload(chat_id: int) -> dict:
    """Real, deterministically-computed figures across all four domains for
    the last RUNDOWN_WINDOW_DAYS (inclusive of today) -- fed to
    ai.answer_with_rundown for synthesis. Shared by the natural-language
    'rundown' intent and /rundown, same discipline as finance._balance_text /
    finance._recent_text: one implementation, not two."""
    today = date.fromisoformat(db.today_str())
    window_start = today - timedelta(days=RUNDOWN_WINDOW_DAYS - 1)
    tomorrow = today + timedelta(days=1)

    status = db.get_status(chat_id)

    meals = db.get_meals_in_range(chat_id, window_start.isoformat(), tomorrow.isoformat())
    total_calories = sum(m["calories_estimate"] or 0 for m in meals)
    total_water = sum(m["water_ml"] or 0 for m in meals)

    workouts = db.get_workouts_in_range(chat_id, window_start.isoformat(), tomorrow.isoformat())

    vitals = db.get_vitals_in_range(chat_id, window_start.isoformat(), tomorrow.isoformat())
    weights = [v["weight_kg"] for v in vitals if v["weight_kg"] is not None]
    sleep_hours = [v["sleep_hours"] for v in vitals if v["sleep_hours"] is not None]
    knee_pain = [v["knee_pain"] for v in vitals if v["knee_pain"] is not None]

    return {
        "window_days": RUNDOWN_WINDOW_DAYS,
        "balance": {
            "available_today": round(status["available_today"], 2),
            "spent_today": round(status["spent_today"], 2),
            "daily_target": status["daily_target"],
            "current_streak": status["current_streak"],
        },
        "meals": {
            "count": len(meals),
            "total_calories_estimate": total_calories if meals else None,
            "total_water_ml": total_water if meals else None,
        },
        "workouts": {
            "count": len(workouts),
            "activities": [w["activity"] for w in workouts if w["activity"]],
        },
        "vitals": {
            "checkins": len(vitals),
            "latest_weight_kg": weights[-1] if weights else None,
            "weight_change_kg": round(weights[-1] - weights[0], 2) if len(weights) >= 2 else None,
            "avg_sleep_hours": round(sum(sleep_hours) / len(sleep_hours), 2) if sleep_hours else None,
            "avg_knee_pain": round(sum(knee_pain) / len(knee_pain), 2) if knee_pain else None,
        },
    }


def _rundown_fallback_text(payload: dict) -> str:
    """Raw, deterministic rendering used only if the Claude synthesis call
    itself fails -- never silent, same discipline as summary.summary's
    fallback. Skips a section entirely when there's nothing in it, same as
    the narrative prompt is told to."""
    lines = [f"Last {payload['window_days']} days:"]
    bal = payload["balance"]
    lines.append(
        f"Balance: {bal['available_today']} available today (streak {bal['current_streak']}d)"
    )
    meals = payload["meals"]
    if meals["count"]:
        lines.append(f"Meals: {meals['count']} logged, ~{meals['total_calories_estimate']} cal")
    workouts = payload["workouts"]
    if workouts["count"]:
        activities = ", ".join(workouts["activities"]) or "unspecified"
        lines.append(f"Workouts: {workouts['count']} ({activities})")
    vitals = payload["vitals"]
    if vitals["checkins"]:
        weight_line = f", latest weight {vitals['latest_weight_kg']}kg" if vitals["latest_weight_kg"] else ""
        lines.append(f"Vitals: {vitals['checkins']} check-in(s){weight_line}")
    return "\n".join(lines)


async def _rundown_reply_text(chat_id: int) -> str:
    """Shared by /rundown and the natural-language 'rundown' intent, same
    one-implementation discipline as finance._balance_text/_recent_text."""
    payload = _rundown_payload(chat_id)
    try:
        return ai.answer_with_rundown(payload)
    except Exception:
        logger.exception("AI rundown synthesis failed, falling back to raw breakdown")
        return _rundown_fallback_text(payload)


async def rundown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    await update.message.reply_text(await _rundown_reply_text(chat_id))


async def daystats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/daystats [today|yesterday|N] -- command-path equivalent of the
    natural-language 'day_stats' intent, same "real numbers in" discipline
    as /rundown vs the 'rundown' intent. Defaults to today with no
    argument. N is a plain day count (0 = today, 1 = yesterday, ...), same
    convention as everywhere else in this app."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    arg = context.args[0].lower() if context.args else "today"
    if arg == "today":
        days_ago = 0
    elif arg == "yesterday":
        days_ago = 1
    else:
        try:
            days_ago = max(0, int(arg))
        except ValueError:
            await update.message.reply_text("Usage: /daystats [today|yesterday|N] -- N = days ago")
            return
    await update.message.reply_text(await _day_stats_reply_text(chat_id, days_ago))


def _resolve_day(days_ago: int | None) -> str:
    """Converts the 'day_stats' intent's day_stats_days_ago (a plain
    day-count, never a date -- same discipline as ai.py's logged_days_ago
    for photo logging) into a real ISO date deterministically. Unlike
    nutrition._target_date_from_days_ago (which returns None meaning
    "leave the log on today, the DB default"), this always returns a
    concrete date, since a day_stats query has nothing to default onto --
    it has to know exactly which day to look up. days_ago None or 0 means
    today.

    Deliberately NOT capped. This used to silently clamp to 14 days back --
    so "what did I log on July 15" (76 days ago) would silently resolve to
    a DIFFERENT day 14 days back and answer about that instead, with no
    indication a substitution happened. That's worse than just answering
    honestly: db.get_meals_in_range/get_workouts_in_range/get_vitals_in_range
    are themselves unbounded (see db.py), so there's no real reason to cap
    this -- an old or out-of-range day just comes back as "nothing logged
    that day" from _day_stats_payload, which is the truthful answer."""
    days_ago = max(0, int(days_ago)) if days_ago else 0
    return (date.fromisoformat(db.today_str()) - timedelta(days=days_ago)).isoformat()


def _day_stats_payload(chat_id: int, day: str) -> dict:
    """Real, deterministically-computed figures for exactly ONE day --
    meals (with the actual logged items, not just a total), workouts,
    lifts count, and vitals -- fed to ai.answer_with_day_stats for
    synthesis. This exists specifically so a "what did I eat/burn
    yesterday" question is answered from a real DB read instead of the
    model re-deriving totals (and re-subtracting them) from raw recent-item
    context on every turn -- see handlers.py's 'day_stats' intent branch.
    Same "real numbers in, never guessed" discipline as _rundown_payload,
    just scoped to a single named day instead of a trailing window."""
    next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()

    meals = db.get_meals_in_range(chat_id, day, next_day)
    total_calories_in = sum(m["calories_estimate"] or 0 for m in meals)
    total_water = sum(m["water_ml"] or 0 for m in meals)

    workouts = db.get_workouts_in_range(chat_id, day, next_day)
    total_calories_out = sum(w["calories_burned"] or 0 for w in workouts)

    vitals = db.get_vitals_in_range(chat_id, day, next_day)

    # Net is only meaningful when there's at least one real number on
    # either side -- an empty day (nothing logged at all) shouldn't report
    # "net 0 kcal", which reads as "you ate nothing and burned nothing" (a
    # real fact) rather than "nothing was logged" (the actual situation).
    net_calories = (
        round(total_calories_in - total_calories_out, 0) if (meals or workouts) else None
    )

    return {
        "day": day,
        "is_today": day == db.today_str(),
        "meals": {
            "count": len(meals),
            "items": [item for m in meals for item in (m.get("items") or [])],
            "total_calories_estimate": total_calories_in if meals else None,
            "total_water_ml": total_water if meals else None,
        },
        "workouts": {
            "count": len(workouts),
            "activities": [w["activity"] for w in workouts if w["activity"]],
            "total_calories_burned": total_calories_out if workouts else None,
        },
        "net_calories": net_calories,
        "vitals": vitals[0] if vitals else None,
    }


def _day_stats_fallback_text(payload: dict) -> str:
    """Raw, deterministic rendering used only if the Claude synthesis call
    itself fails -- never silent, same discipline as _rundown_fallback_text."""
    label = "Today" if payload["is_today"] else payload["day"]
    lines = [f"{label}:"]
    meals = payload["meals"]
    if meals["count"]:
        items = ", ".join(meals["items"]) or "unspecified items"
        lines.append(f"Ate: ~{meals['total_calories_estimate']:.0f} kcal ({items})")
        if meals["total_water_ml"]:
            lines.append(f"Water: {meals['total_water_ml']:.0f}ml")
    else:
        lines.append("Ate: nothing logged")
    workouts = payload["workouts"]
    if workouts["count"]:
        activities = ", ".join(workouts["activities"]) or "unspecified"
        burned = (
            f", ~{workouts['total_calories_burned']:.0f} kcal burned"
            if workouts["total_calories_burned"] else ""
        )
        lines.append(f"Trained: {activities}{burned}")
    if payload["net_calories"] is not None:
        sign = "-" if payload["net_calories"] < 0 else ""
        lines.append(f"Net: {sign}{abs(payload['net_calories']):.0f} kcal")
    vitals = payload["vitals"]
    if vitals and vitals.get("weight_kg"):
        lines.append(f"Weight: {vitals['weight_kg']}kg")
    return "\n".join(lines)


async def _day_stats_reply_text(chat_id: int, days_ago: int | None) -> str:
    """Shared entry point for the natural-language 'day_stats' intent --
    resolves the day-count to a real date, pulls the real numbers, and
    hands them to Claude only to narrate (see _day_stats_payload)."""
    day = _resolve_day(days_ago)
    payload = _day_stats_payload(chat_id, day)
    try:
        return ai.answer_with_day_stats(payload)
    except Exception:
        logger.exception("AI day_stats synthesis failed, falling back to raw breakdown")
        return _day_stats_fallback_text(payload)


# "Since the beginning" (trend_start_days_ago is null) has no real day-count --
# there's no way to know how far back logging started without querying the
# data itself. This placeholder is just a lower bound old enough that no real
# Morrow data could predate it; _trend_payload always reports the REAL
# earliest date actual data was found on, never this placeholder itself.
TREND_BEGINNING_PLACEHOLDER = "2000-01-01"

TREND_METRICS = ("weight", "sleep", "knee_pain", "calories_in", "calories_out", "spending")

TREND_UNITS = {
    "weight": "kg",
    "sleep": "hours",
    "knee_pain": "/10",
    "calories_in": "kcal",
    "calories_out": "kcal",
    "spending": None,  # base currency; the amount itself carries no unit suffix
}


def _resolve_trend_range(start_days_ago: int | None, end_days_ago: int | None) -> tuple[str, str]:
    """Converts the 'trend' intent's start/end day-counts (same plain
    day-count discipline as day_stats_days_ago) into a real [start_date,
    end_date) range for the get_*_in_range functions -- end exclusive, same
    convention as _day_stats_payload's next_day. end_days_ago None means
    through today inclusive. start_days_ago None means "since the
    beginning" -- see TREND_BEGINNING_PLACEHOLDER."""
    today = date.fromisoformat(db.today_str())
    end_day = today - timedelta(days=max(0, int(end_days_ago))) if end_days_ago else today
    end_date = (end_day + timedelta(days=1)).isoformat()
    if start_days_ago is None:
        start_date = TREND_BEGINNING_PLACEHOLDER
    else:
        start_date = (today - timedelta(days=max(0, int(start_days_ago)))).isoformat()
    return start_date, end_date


def _trend_series(chat_id: int, metric: str, start_date: str, end_date: str) -> list[tuple[str, float]]:
    """Real (date, value) pairs for one metric in [start_date, end_date),
    sorted oldest-first by the actual date field -- NOT by id/insertion
    order. This matters for exactly the same reason get_recent_vitals/
    get_recent_meals/get_recent_workouts were fixed to sort by date: a
    historical backfill inserts old-dated rows LAST (highest id), so
    get_vitals_in_range/get_meals_in_range/get_workouts_in_range (which
    order by id) can hand back rows in an order that doesn't match their
    actual dates. Sorting explicitly here, rather than trusting row order,
    is what keeps first/last/min/max correct regardless of insertion order."""
    if metric == "weight":
        vitals = db.get_vitals_in_range(chat_id, start_date, end_date)
        pairs = [(v["vitals_date"], v["weight_kg"]) for v in vitals if v["weight_kg"] is not None]
    elif metric == "sleep":
        vitals = db.get_vitals_in_range(chat_id, start_date, end_date)
        pairs = [(v["vitals_date"], v["sleep_hours"]) for v in vitals if v["sleep_hours"] is not None]
    elif metric == "knee_pain":
        vitals = db.get_vitals_in_range(chat_id, start_date, end_date)
        pairs = [(v["vitals_date"], v["knee_pain"]) for v in vitals if v["knee_pain"] is not None]
    elif metric == "calories_in":
        meals = db.get_meals_in_range(chat_id, start_date, end_date)
        totals: dict[str, float] = {}
        for m in meals:
            totals[m["meal_date"]] = totals.get(m["meal_date"], 0) + (m["calories_estimate"] or 0)
        pairs = list(totals.items())
    elif metric == "calories_out":
        workouts = db.get_workouts_in_range(chat_id, start_date, end_date)
        totals: dict[str, float] = {}
        for w in workouts:
            totals[w["workout_date"]] = totals.get(w["workout_date"], 0) + (w["calories_burned"] or 0)
        pairs = list(totals.items())
    elif metric == "spending":
        totals = db.get_daily_totals(chat_id, start_date, end_date)
        pairs = list(totals.items())
    else:
        raise ValueError(f"unknown trend metric: {metric!r}")
    return sorted(pairs, key=lambda p: p[0])


def _trend_payload(chat_id: int, metric: str, start_date: str, end_date: str) -> dict:
    """Real, deterministically-computed first/last/min/max/change figures
    for ONE metric over a date range -- fed to ai.answer_with_trend for
    synthesis. Same "real numbers in, never guessed" discipline as
    _rundown_payload/_day_stats_payload, just extended to an arbitrary
    range on a single metric instead of a fixed window across domains."""
    series = _trend_series(chat_id, metric, start_date, end_date)
    count = len(series)
    first = {"date": series[0][0], "value": series[0][1]} if count else None
    last = {"date": series[-1][0], "value": series[-1][1]} if count else None
    min_point = min(series, key=lambda p: p[1]) if count else None
    max_point = max(series, key=lambda p: p[1]) if count else None
    return {
        "metric": metric,
        "unit": TREND_UNITS[metric],
        # The REAL earliest/latest dates data was actually found on -- not
        # the requested range, and never the TREND_BEGINNING_PLACEHOLDER.
        "start_date": first["date"] if first else None,
        "end_date": last["date"] if last else None,
        "count": count,
        "first": first,
        "last": last,
        "min": {"date": min_point[0], "value": min_point[1]} if min_point else None,
        "max": {"date": max_point[0], "value": max_point[1]} if max_point else None,
        "change": round(last["value"] - first["value"], 2) if count >= 2 else None,
    }


def _trend_fallback_text(payload: dict) -> str:
    """Raw, deterministic rendering used only if the Claude synthesis call
    itself fails -- never silent, same discipline as _rundown_fallback_text/
    _day_stats_fallback_text."""
    metric = payload["metric"]
    unit = payload["unit"] or ""
    if not payload["count"]:
        return f"Nothing logged for {metric} in that range."
    if payload["count"] == 1:
        return f"Only one {metric} reading: {payload['first']['value']}{unit} on {payload['first']['date']}."
    lines = [
        f"{metric}: {payload['first']['value']}{unit} on {payload['first']['date']} "
        f"-> {payload['last']['value']}{unit} on {payload['last']['date']}"
    ]
    change = payload["change"]
    if change is not None:
        sign = "+" if change > 0 else ""
        lines.append(f"Change: {sign}{change}{unit}")
    if payload["min"] and payload["max"]:
        lines.append(
            f"Range: {payload['min']['value']}{unit} ({payload['min']['date']}) to "
            f"{payload['max']['value']}{unit} ({payload['max']['date']})"
        )
    return "\n".join(lines)


async def _trend_reply_text(
    chat_id: int, metric: str, start_days_ago: int | None, end_days_ago: int | None
) -> str:
    """Shared entry point for the natural-language 'trend' intent -- resolves
    the day-counts to a real date range, pulls the real numbers, and hands
    them to Claude only to narrate (see _trend_payload)."""
    start_date, end_date = _resolve_trend_range(start_days_ago, end_days_ago)
    payload = _trend_payload(chat_id, metric, start_date, end_date)
    try:
        return ai.answer_with_trend(payload)
    except Exception:
        logger.exception("AI trend synthesis failed, falling back to raw breakdown")
        return _trend_fallback_text(payload)


async def trend_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/trend <metric> [N days|all] -- command-path equivalent of the
    natural-language 'trend' intent, same "real numbers in" discipline as
    /daystats vs 'day_stats'. Defaults to "all" (since the beginning) with
    no range argument."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text(
            "Usage: /trend <weight|sleep|knee_pain|calories_in|calories_out|spending> [N days|all]"
        )
        return
    metric = context.args[0].lower()
    if metric not in TREND_METRICS:
        await update.message.reply_text(
            f"Unknown metric {metric!r}. Choose one of: {', '.join(TREND_METRICS)}"
        )
        return
    start_days_ago = None
    if len(context.args) > 1 and context.args[1].lower() != "all":
        try:
            start_days_ago = max(0, int(context.args[1]))
        except ValueError:
            await update.message.reply_text(
                "Usage: /trend <weight|sleep|knee_pain|calories_in|calories_out|spending> [N days|all]"
            )
            return
    await update.message.reply_text(await _trend_reply_text(chat_id, metric, start_days_ago, None))
