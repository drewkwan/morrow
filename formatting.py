"""
Plain-text formatting helpers shared across domains: turning a DB row (a
dict) into the one-line summary used in "Logged: ...", "Recent ...:", and
correction/undo confirmations. Deliberately pure functions of a dict (no DB
reads, no side effects) except _daily_meal_totals_text, which is a thin
wrapper around a single read -- kept here anyway since every meal-related
reply uses it and it has nowhere more natural to live.
"""

from datetime import date, timedelta

import config
import db


def _money(x: float, currency: str = None) -> str:
    currency = currency or config.BASE_CURRENCY
    sign = "-" if x < 0 else ""
    return f"{sign}{currency} {abs(x):,.2f}"


def _status_text(status: dict) -> str:
    lines = [
        f"Today's target: {_money(status['daily_target'])}",
        f"Rolled-over balance: {_money(status['balance'])}",
        f"Spent today: {_money(status['spent_today'])}",
        f"Available today: {_money(status['available_today'])}",
    ]
    if status["pending_claimable"] > 0:
        lines.append(f"Pending claimables: {_money(status['pending_claimable'])}")
    if status["current_streak"] > 0:
        lines.append(f"Streak: {status['current_streak']} day(s) within budget (best: {status['best_streak']})")
    return "\n".join(lines)


def _expense_line(row: dict) -> str:
    claim_tag = " [claimable]" if row.get("is_claimable") else ""
    claimed_tag = " (claimed)" if row.get("is_claimed") else ""
    return (f"#{row['id']} {_money(row['amount'], row.get('currency'))} -- "
            f"{row['description']} [{row['category']}]{claim_tag}{claimed_tag} ({row['expense_date']})")


def _calorie_range(row: dict) -> str:
    low, high, est = row.get("calories_low"), row.get("calories_high"), row.get("calories_estimate")
    if est is None:
        return "unknown kcal"
    if low is not None and high is not None and (low, high) != (est, est):
        return f"~{low:.0f}-{high:.0f} kcal (central ~{est:.0f})"
    return f"~{est:.0f} kcal"


def _meal_line(row: dict) -> str:
    items = ", ".join(row.get("items") or []) or "unspecified"
    type_tag = f" [{row['meal_type']}]" if row.get("meal_type") else ""
    water_tag = f", {row['water_ml']:.0f}ml water" if row.get("water_ml") else ""
    return f"#{row['id']} {items}{type_tag} -- {_calorie_range(row)}{water_tag} ({row['meal_date']})"


def _workout_line(row: dict) -> str:
    bits = []
    if row.get("duration_min"):
        bits.append(f"{row['duration_min']:.0f} min")
    if row.get("distance_km"):
        bits.append(f"{row['distance_km']:.1f} km")
    if row.get("calories_burned"):
        bits.append(f"{row['calories_burned']:.0f} kcal burned")
    detail = f" ({', '.join(bits)})" if bits else ""
    notes = f" -- {row['notes']}" if row.get("notes") else ""
    return f"#{row['id']} {row.get('activity') or 'workout'}{detail}{notes} ({row['workout_date']})"


def _set_str(s: dict) -> str:
    reps, load = s.get("reps"), s.get("load")
    if load and reps is not None:
        return f"{load} x{reps}"
    if load:
        return str(load)
    if reps is not None:
        return str(reps)
    return "?"


def _lift_line(row: dict) -> str:
    sets_str = ", ".join(_set_str(s) for s in (row.get("sets") or [])) or "no sets recorded"
    location_tag = f" @ {row['location']}" if row.get("location") else ""
    effort_tag = f" ({row['effort']})" if row.get("effort") else ""
    notes_tag = f" -- {row['context_notes']}" if row.get("context_notes") else ""
    return f"#{row['id']} {row['exercise']}{location_tag}: {sets_str}{effort_tag}{notes_tag} ({row['lift_date']})"


def _vitals_line(row: dict) -> str:
    bits = []
    if row.get("weight_kg") is not None:
        bits.append(f"{row['weight_kg']:.1f}kg")
    if row.get("sleep_hours") is not None:
        bits.append(f"{row['sleep_hours']:.1f}h sleep")
    if row.get("knee_pain") is not None:
        bits.append(f"knee {row['knee_pain']:.0f}/10")
    summary = ", ".join(bits) or "check-in"
    notes = f" -- {row['notes']}" if row.get("notes") else ""
    return f"#{row['id']} {summary}{notes} ({row['vitals_date']})"


def _task_line(row: dict) -> str:
    due = f" (due {row['due_at']})" if row.get("due_at") else ""
    done_tag = " [done]" if row.get("done") else ""
    recurrence_tag = f" [repeats {row['recurrence_frequency']}]" if row.get("recurrence_frequency") else ""
    notes = f" -- {row['notes']}" if row.get("notes") else ""
    return f"#{row['id']} {row['title']}{due}{recurrence_tag}{notes}{done_tag}"


def _reminder_line(row: dict) -> str:
    done_tag = " [done today]" if row.get("last_done_date") == db.today_str() else ""
    return f"#{row['id']} {row['description']}{done_tag}"


def _event_line(row: dict) -> str:
    when = row["event_date"]
    if row.get("event_time"):
        when += f" {row['event_time']}"
    notes = f" -- {row['notes']}" if row.get("notes") else ""
    return f"#{row['id']} {row['title']} (scheduled {when}){notes}"


# How each recurrence frequency reads inline -- "/mo"-style for the fixed-
# interval ones, "every N ..." for the rest, so a line reads naturally
# regardless of cadence (e.g. "$37.98 every 3 months" not "$37.98 /quarterly").
_FREQUENCY_LABELS = {
    "weekly": "/wk", "biweekly": "every 2 weeks", "monthly": "/mo",
    "quarterly": "every 3 months", "annual": "/yr",
}


def _subscription_line(row: dict) -> str:
    category_tag = f" [{row['category']}]" if row.get("category") else ""
    card_tag = f" on {row['card']}" if row.get("card") else ""
    claim_tag = " [claimable]" if row.get("is_claimable") else ""
    notes_tag = f" -- {row['notes']}" if row.get("notes") else ""
    freq_label = _FREQUENCY_LABELS.get(row.get("frequency"), f" ({row.get('frequency')})")
    return (f"#{row['id']} {row['name']} -- {_money(row['amount'], row.get('currency'))} {freq_label} "
            f"(renews {row['next_renewal_date']}){category_tag}{card_tag}{claim_tag}{notes_tag}")


def _income_line(row: dict) -> str:
    desc_tag = f" -- {row['description']}" if row.get("description") else ""
    breakdown_bits = []
    if row.get("gross_amount") is not None:
        breakdown_bits.append(f"gross {_money(row['gross_amount'], row.get('currency'))}")
    if row.get("cpf_amount") is not None:
        breakdown_bits.append(f"CPF {_money(row['cpf_amount'], row.get('currency'))}")
    if row.get("stock_amount") is not None:
        breakdown_bits.append(f"stock {_money(row['stock_amount'], row.get('currency'))}")
    breakdown = f" ({', '.join(breakdown_bits)})" if breakdown_bits else ""
    return (f"#{row['id']} {row['source']}: {_money(row['net_amount'], row.get('currency'))} net"
            f"{breakdown}{desc_tag} ({row['income_date']})")


def _deduction_line(row: dict) -> str:
    return f"#{row['id']} {row['label']} -- {_money(row['amount'], row.get('currency'))} ({row['deduction_date']})"


def _net_worth_text(net_worth: dict) -> str:
    lines = [
        f"Total income: {_money(net_worth['total_income'])}",
        f"Total deductions: {_money(net_worth['total_deductions'])}",
        f"Total spend: {_money(net_worth['total_spend'])}",
        f"Net worth: {_money(net_worth['net_worth'])}",
    ]
    return "\n".join(lines)


def _memory_line(row: dict) -> str:
    cat = f" [{row['category']}]" if row.get("category") else ""
    return f"{row['label']}{cat}: {row['content']}"


def _daily_meal_totals_text(chat_id: int, day: str | None = None) -> str:
    """day defaults to today. Pass it explicitly when the meal being
    confirmed was logged onto a different (backdated) day -- e.g. a photo
    whose caption named a past date (see ai.py's logged_days_ago) -- so the
    total shown is the total for the day actually affected, not a
    same-labelled "today's running total" that's quietly about the wrong
    day."""
    day = day or db.today_str()
    totals = db.get_daily_meal_totals(chat_id, day)
    label = "Today's running total" if day == db.today_str() else f"Running total for {day}"
    line = f"{label}: ~{totals['calories']:.0f} kcal"
    if totals["water_ml"]:
        line += f", {totals['water_ml']:.0f}ml water"
    return line


def _weekly_workout_summary_text(chat_id: int, day: str | None = None) -> str:
    """Real trailing-7-day (inclusive of day) workout count/total burned,
    appended to a workout confirmation so logging just ONE workout also
    shows how the week's shaping up -- same "real numbers, deterministic,
    no AI call" discipline as _daily_meal_totals_text/_daily_calorie_balance_text.
    day is passed explicitly for a backdated workout (see
    _daily_meal_totals_text's docstring for the same reasoning), so the
    window is centered on the day actually logged onto, not necessarily
    today."""
    day = day or db.today_str()
    window_start = (date.fromisoformat(day) - timedelta(days=6)).isoformat()
    window_end = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    workouts = db.get_workouts_in_range(chat_id, window_start, window_end)
    # The just-logged workout is always included (the window covers `day`),
    # so this is never actually empty -- 1 means it's the only one so far.
    if len(workouts) == 1:
        return "First workout logged this week."
    total_burned = sum(w["calories_burned"] or 0 for w in workouts)
    burned_tag = f", ~{total_burned:.0f} kcal burned" if total_burned else ""
    return f"This week: {len(workouts)} workouts logged{burned_tag}"


def _daily_calorie_balance_text(chat_id: int, day: str | None = None) -> str:
    """Calories in vs calories out for a given day (defaults to today) --
    shown alongside a logged workout that reports calories_burned (e.g. from
    a fitness app screenshot, see ai.extract_from_photo), so burning
    calories is actually useful information rather than a number logged in
    isolation. Meal calories logged that day is "in", that day's summed
    workout calories_burned is "out"; net can go either way depending on
    which is bigger. day is passed explicitly for a backdated workout (see
    _daily_meal_totals_text's docstring for the same reasoning) so the
    balance shown matches the day the workout was actually logged onto."""
    day = day or db.today_str()
    calories_in = db.get_daily_meal_totals(chat_id, day)["calories"]
    calories_out = db.get_daily_workout_totals(chat_id, day)["calories_burned"]
    net = calories_in - calories_out
    sign = "-" if net < 0 else ""
    label = "Today" if day == db.today_str() else day
    return (f"{label}: ~{calories_in:.0f} kcal in, ~{calories_out:.0f} kcal burned "
            f"(net {sign}{abs(net):.0f} kcal)")
