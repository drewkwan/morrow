"""
The free-text chat surface: one message in, one intent-routed reply out.
This is the single busiest call path in the bot -- ai.parse_message
classifies + extracts in one call, then this module dispatches to whichever
domain module actually owns the effect (finance/nutrition/fitness/vitals/
tasks/memory/correction/rundown), never duplicating their logic. Also home
to the global error handler, which is the last line of defense against a
silent failure reaching the user.
"""

import logging

from telegram import Update
from telegram.ext import ContextTypes

import ai
import db
import fx
from access import _reject_if_not_allowed
from correction import CORRECTION_UNDO_PHRASES, LAST_CORRECTION_KEY, _handle_correction, _revert_last_correction
from finance import _balance_text, _recent_text
from fitness import _force_log_workout_and_reply, _reply_workout_logged
from formatting import _money, _status_text
from income import _log_deductions_and_reply, _log_incomes_and_reply
from lifts import _log_lifts_and_reply, _recent_lifts_for_ai, _recent_lifts_for_narration
from memory import _memory_for_ai, _memory_text
from nutrition import _log_meals_and_reply, _target_date_from_days_ago
from events import _add_events_and_reply, _events_text
from reminders import _add_reminder_and_reply, _reminders_text
from replies import PENDING_DUPLICATE_WORKOUT_KEY, PENDING_KEY, _reply, _send_alert_if_needed
from rundown import TREND_METRICS, _day_stats_payload, _day_stats_reply_text, _rundown_reply_text, _trend_reply_text
from subscriptions import _log_subscriptions_and_reply
from tasks import _log_tasks_and_reply, _recent_tasks_for_ai, _tasks_text
from vitals import _log_vitals_and_reply

logger = logging.getLogger(__name__)

RECENT_EXPENSES_FOR_AI = 8  # how much history the model gets to resolve "that", "the duplicate", etc.
RECENT_MESSAGES_FOR_AI = 30  # rolling conversation window -- see db.py's module docstring on messages vs memory


def _recent_for_ai(chat_id: int) -> list:
    rows = db.get_recent_expenses(chat_id, limit=RECENT_EXPENSES_FOR_AI)
    return [
        {
            "id": r["id"], "amount": r["amount"], "currency": r["currency"],
            "description": r["description"], "category": r["category"],
            "expense_date": r["expense_date"], "is_claimable": bool(r["is_claimable"]),
        }
        for r in rows
    ]


def _recent_meals_for_ai(chat_id: int) -> list:
    rows = db.get_recent_meals(chat_id, limit=RECENT_EXPENSES_FOR_AI)
    return [
        {"id": r["id"], "items": r["items"], "meal_type": r["meal_type"],
         "calories_estimate": r["calories_estimate"], "meal_date": r["meal_date"]}
        for r in rows
    ]


def _recent_workouts_for_ai(chat_id: int) -> list:
    rows = db.get_recent_workouts(chat_id, limit=RECENT_EXPENSES_FOR_AI)
    return [
        {"id": r["id"], "activity": r["activity"], "duration_min": r["duration_min"],
         "workout_date": r["workout_date"]}
        for r in rows
    ]


def _recent_vitals_for_ai(chat_id: int) -> list:
    rows = db.get_recent_vitals(chat_id, limit=RECENT_EXPENSES_FOR_AI)
    return [
        {"id": r["id"], "weight_kg": r["weight_kg"], "sleep_hours": r["sleep_hours"],
         "knee_pain": r["knee_pain"], "vitals_date": r["vitals_date"]}
        for r in rows
    ]


def _recent_messages_for_ai(chat_id: int) -> list:
    rows = db.get_recent_messages(chat_id, limit=RECENT_MESSAGES_FOR_AI)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def _casual_reply_text(chat_id: int, message: str, recent_messages: list, memory_list: list,
                              fallback: str | None) -> str:
    """Real conversational reply for the 'casual' intent -- see
    ai.answer_casually's docstring for why this is its own dedicated call
    rather than reusing parse_message's casual_reply field. today_snapshot
    reuses the exact same real, deterministically-computed figures day_stats
    already trusts (db.get_status + rundown._day_stats_payload for today),
    plus db.get_net_worth (a genuinely separate "real money" total, not the
    daily spending target/balance) and db.get_month_to_date_total (the same
    real calendar-month sum /summary's "month" view uses) -- one source of
    "how's today going"/"what's my net worth"/"what's my total spend this
    month", not a second copy of any of those that could drift. See
    PARSE_SYSTEM_PROMPT's "trend" paragraph for why a plain total/sum
    question is classified "casual" (grounded here) rather than "trend"
    (a trajectory/first-last-min-max answer, the wrong shape for a total).
    recent_lifts (lifts._recent_lifts_for_narration) similarly grounds any
    gym-routine question in real logged rows instead of letting the model
    freely narrate from memory prose -- see ai.answer_casually's docstring
    for the real observed bug this replaces. Falls back to parse_message's
    own casual_reply field (or a generic line) if the dedicated call itself
    fails, same discipline as rundown/day_stats falling back to
    deterministic text on an AI-call exception."""
    today_snapshot = {
        "balance": db.get_status(chat_id),
        "today": _day_stats_payload(chat_id, db.today_str()),
        "net_worth": db.get_net_worth(chat_id),
        "month_to_date": db.get_month_to_date_total(chat_id),
    }
    recent_lifts = _recent_lifts_for_narration(chat_id)
    try:
        return ai.answer_casually(message, recent_messages, memory_list, today_snapshot, recent_lifts)
    except Exception:
        logger.exception("answer_casually failed, falling back to parse_message's casual_reply")
        return fallback or (
            "Not sure what to do with that. Use /log, /claim, /logmeal, /logworkout, /logvitals, /balance, "
            "/summary, /recent, or /help."
        )


def _recent_events_for_ai(chat_id: int) -> list:
    """Upcoming (today forward) events, soonest first -- what an event
    correction (delete only -- see events.py's design) may target. Without
    this, the model has no way to know a bare id might refer to an event
    rather than a task -- a real observed bug (see ai.parse_message's
    docstring)."""
    rows = db.get_upcoming_events(chat_id, limit=20)
    return [{"id": r["id"], "event_title": r["title"], "event_date": r["event_date"],
             "event_time": r["event_time"]} for r in rows]


def _recent_subscriptions_for_ai(chat_id: int) -> list:
    """Active subscriptions, soonest-renewal order (see db.get_subscriptions)
    -- what a subscription correction (edit_subscription/delete) may
    target. next_renewal_date is this domain's one date-shaped field,
    already included below -- there's no separate recent-by-date list the
    way logged domains have."""
    rows = db.get_subscriptions(chat_id)
    return [{"id": r["id"], "name": r["name"], "amount": r["amount"], "currency": r["currency"],
             "frequency": r["frequency"], "next_renewal_date": r["next_renewal_date"],
             "category": r["category"], "card": r["card"], "notes": r["notes"],
             "is_claimable": r["is_claimable"]} for r in rows]


def _recent_income_for_ai(chat_id: int) -> list:
    rows = db.get_recent_income(chat_id, limit=RECENT_EXPENSES_FOR_AI)
    return [{"id": r["id"], "source": r["source"], "description": r["description"],
             "net_amount": r["net_amount"], "currency": r["currency"], "income_date": r["income_date"]}
            for r in rows]


def _recent_deductions_for_ai(chat_id: int) -> list:
    rows = db.get_recent_deductions(chat_id, limit=RECENT_EXPENSES_FOR_AI)
    return [{"id": r["id"], "label": r["label"], "amount": r["amount"], "currency": r["currency"],
             "deduction_date": r["deduction_date"]} for r in rows]


def _recent_reminders_for_ai(chat_id: int) -> list:
    """Every standing daily reminder (see db.get_active_reminders) -- what a
    reminder correction (mark_done/delete) may target. No separate
    "recent" cutoff the way logged domains have -- there's no date to sort
    by, and the list is small enough (a handful of daily habits, not
    hundreds of logged rows) that there's no need to truncate it."""
    rows = db.get_active_reminders(chat_id)
    return [{"id": r["id"], "description": r["description"], "last_done_date": r["last_done_date"]}
            for r in rows]


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    # A pending duplicate-workout question (see fitness._ask_about_duplicate_workout)
    # is resolved directly here, deterministically, rather than falling through to
    # the general ai.parse_message pipeline below -- that pipeline's "log_workout"
    # shape has nowhere to carry calories_burned (see ai.py's PARSE_SYSTEM_PROMPT),
    # so re-parsing this reply as free text would silently drop the exact number
    # the photo already gave us. We already have the fully-parsed photo data
    # (stashed by fitness.py); all that's needed here is yes or no.
    dup_pending = context.chat_data.pop(PENDING_DUPLICATE_WORKOUT_KEY, None)
    if dup_pending:
        db.add_message(chat_id, "user", text)
        affirmative = any(
            w in text.lower()
            for w in ("yes", "separate", "different", "another", "anyway", "log it", "do log", "keep both")
        )
        if affirmative:
            await _force_log_workout_and_reply(update, chat_id, dup_pending["data"], dup_pending["workout_date"])
        else:
            await _reply(update, chat_id, "Got it -- skipped, since it looked like the same workout logged twice.")
        return

    pending = context.chat_data.get(PENDING_KEY)
    db.add_message(chat_id, "user", text)

    # Only treat a bare "no"/"wrong"/"undo" as reverting a correction when
    # we're not mid-clarification -- otherwise it's very likely a genuine
    # answer to whatever question was just asked (e.g. "is this claimable?").
    if not pending and text.lower().rstrip(".!") in CORRECTION_UNDO_PHRASES:
        if await _revert_last_correction(update, context):
            return
        # nothing to revert -- fall through and let it be parsed normally

    # A correction snapshot only survives for the single reply immediately
    # following it; anything else means it's no longer relevant.
    context.chat_data.pop(LAST_CORRECTION_KEY, None)

    recent_expenses = _recent_for_ai(chat_id)
    recent_ids = {r["id"] for r in recent_expenses}
    recent_meals = _recent_meals_for_ai(chat_id)
    recent_meal_ids = {r["id"] for r in recent_meals}
    recent_workouts = _recent_workouts_for_ai(chat_id)
    recent_workout_ids = {r["id"] for r in recent_workouts}
    recent_vitals = _recent_vitals_for_ai(chat_id)
    recent_vitals_ids = {r["id"] for r in recent_vitals}
    recent_tasks = _recent_tasks_for_ai(chat_id)
    recent_task_ids = {r["id"] for r in recent_tasks}
    recent_events = _recent_events_for_ai(chat_id)
    recent_event_ids = {r["id"] for r in recent_events}
    recent_lifts = _recent_lifts_for_ai(chat_id)
    recent_lift_ids = {r["id"] for r in recent_lifts}
    recent_subscriptions = _recent_subscriptions_for_ai(chat_id)
    recent_subscription_ids = {r["id"] for r in recent_subscriptions}
    recent_income = _recent_income_for_ai(chat_id)
    recent_income_ids = {r["id"] for r in recent_income}
    recent_deductions = _recent_deductions_for_ai(chat_id)
    recent_deduction_ids = {r["id"] for r in recent_deductions}
    recent_reminders = _recent_reminders_for_ai(chat_id)
    recent_reminder_ids = {r["id"] for r in recent_reminders}
    # The message just added above is deliberately included here -- the
    # model should see its own current turn as part of the running thread,
    # not just what came before it.
    recent_messages = _recent_messages_for_ai(chat_id)
    memory_list = _memory_for_ai(chat_id)

    if pending:
        # We asked a clarifying question; treat this message as the answer.
        merged_text = f"{pending['original']}\n(Additional info: {text})"
        parsed = ai.parse_message(merged_text, recent_expenses, recent_meals, recent_workouts, recent_vitals,
                                   recent_tasks, recent_messages, memory_list, recent_events, recent_lifts,
                                   recent_subscriptions, recent_income, recent_deductions, recent_reminders)
    else:
        merged_text = text
        parsed = ai.parse_message(text, recent_expenses, recent_meals, recent_workouts, recent_vitals,
                                   recent_tasks, recent_messages, memory_list, recent_events, recent_lifts,
                                   recent_subscriptions, recent_income, recent_deductions, recent_reminders)

    intent = parsed.get("intent")

    if intent == "clarification":
        # Carry forward the ACCUMULATED text, not just this latest fragment --
        # otherwise a second (or third) round of clarification silently drops
        # everything learned in earlier rounds (e.g. an amount mentioned two
        # messages ago), which was a real bug: the context would shrink with
        # every back-and-forth instead of growing.
        context.chat_data[PENDING_KEY] = {"original": merged_text}
        await _reply(update, chat_id, parsed.get("clarification_question") or "Could you clarify that?")
        return

    if intent == "correction":
        context.chat_data.pop(PENDING_KEY, None)
        # Deliberately logged BEFORE dispatch, not just on failure -- the
        # generic "I'm not sure which X you mean" reply gives no visibility
        # into whether the AI actually got the domain/id/action right (a
        # real diagnostic gap: a user reported "17 done" failing to mark a
        # to-do done with no way to tell, from the reply alone, whether
        # parse_message returned the wrong id, the wrong action, or the
        # right values against a recent-id set that didn't contain them).
        logger.info(
            "correction parsed: chat_id=%s text=%r target_domain=%r target_expense_id=%r "
            "correction_action=%r recent_task_ids=%s recent_event_ids=%s recent_lift_ids=%s",
            chat_id, merged_text, parsed.get("target_domain"), parsed.get("target_expense_id"),
            parsed.get("correction_action"), sorted(recent_task_ids), sorted(recent_event_ids),
            sorted(recent_lift_ids),
        )
        await _handle_correction(update, context, parsed, recent_ids, recent_meal_ids,
                                  recent_workout_ids, recent_vitals_ids, recent_task_ids, recent_event_ids,
                                  recent_lift_ids, recent_subscription_ids, recent_income_ids,
                                  recent_deduction_ids, recent_reminder_ids)
        return

    if intent == "show_balance":
        # Answered directly with real numbers -- the exact same code path as
        # /balance -- rather than just telling the user to go type /balance.
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, _balance_text(chat_id))
        return

    if intent == "show_recent":
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, _recent_text(chat_id))
        return

    if intent == "rundown":
        # Cross-domain synthesis (section 04 of the plan) -- real 7-day
        # figures computed in code, handed to Claude only to narrate, same
        # "never let the model guess a number" discipline as show_balance.
        # narrate=False -- this text is ALREADY a full narration
        # (ai.answer_with_rundown), so a second companion-voice pass
        # (see replies._reply) would be redundant and risks the second
        # pass quietly drifting from the first pass's real numbers.
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, await _rundown_reply_text(chat_id), narrate=False)
        return

    if intent == "day_stats":
        # Real, freshly re-read DB figures for exactly one named day, handed
        # to Claude only to narrate -- see rundown._day_stats_payload's
        # docstring for the real bug this replaces: a single-day calorie
        # question used to be "casual", which had the model re-derive the
        # sum/subtraction itself from raw context on every ask.
        # narrate=False -- ai.answer_with_day_stats already narrated this
        # from the real payload; a second pass would be redundant and
        # risks drifting from the first pass's real numbers (see the
        # 'rundown' branch above for the same reasoning).
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, await _day_stats_reply_text(chat_id, parsed.get("day_stats_days_ago")),
                     narrate=False)
        return

    if intent == "trend":
        # Real, deterministically-computed first/last/min/max/change for one
        # metric over a real date range, handed to Claude only to narrate --
        # see rundown._trend_payload's docstring for the real bug this
        # replaces: "summarise my weight progression since the beginning"
        # used to be misread as a today-scoped question.
        # narrate=False -- ai.answer_with_trend already narrated this from
        # the real payload, same reasoning as the 'day_stats' branch above.
        context.chat_data.pop(PENDING_KEY, None)
        trend_metric = parsed.get("trend_metric")
        if trend_metric not in TREND_METRICS:
            # The model classified this as "trend" but didn't (or couldn't)
            # settle on one of the covered metrics -- ask rather than guess.
            await _reply(
                update, chat_id,
                "Which one -- weight, sleep, knee pain, calories in, calories out, or spending?",
                narrate=False,
            )
            return
        await _reply(
            update, chat_id,
            await _trend_reply_text(
                chat_id, trend_metric, parsed.get("trend_start_days_ago"), parsed.get("trend_end_days_ago")
            ),
            narrate=False,
        )
        return

    if intent == "log_meal":
        context.chat_data.pop(PENDING_KEY, None)
        meals = parsed.get("meals") or []
        if not meals:
            await _reply(update, chat_id, "I didn't catch what you ate -- try describing it again.")
            return
        await _log_meals_and_reply(update, chat_id, meals)
        return

    if intent == "log_workout":
        context.chat_data.pop(PENDING_KEY, None)
        # See ai.py's logged_days_ago rule: "last night I also went for a run"
        # backdates the same way a photo's caption already could -- without
        # this it always landed on today, no matter what the message said.
        workout_date = _target_date_from_days_ago(parsed.get("logged_days_ago"))
        workout_id = db.add_workout(chat_id, parsed.get("activity"), parsed.get("duration_min"),
                                     parsed.get("distance_km"), parsed.get("workout_notes"),
                                     workout_date=workout_date)
        row = db.get_workout(chat_id, workout_id)
        # _reply_workout_logged, not a bare inline reply -- one implementation
        # of "what a logged workout's confirmation says" shared with /logworkout
        # and photo logging (see finance._balance_text's docstring for the same
        # reasoning), so the weekly-summary line (and the calorie-balance line,
        # on the rare natural-language message that does report calories_burned)
        # shows up here too instead of only on the command/photo paths.
        await _reply_workout_logged(update, chat_id, row, workout_date)
        return

    if intent == "log_lift":
        context.chat_data.pop(PENDING_KEY, None)
        lifts = parsed.get("lifts") or []
        if not lifts:
            await _reply(update, chat_id, "I didn't catch the exercise -- try describing it again.")
            return
        await _log_lifts_and_reply(update, chat_id, lifts)
        return

    if intent == "log_vitals":
        context.chat_data.pop(PENDING_KEY, None)
        vitals_date = _target_date_from_days_ago(parsed.get("logged_days_ago"))
        await _log_vitals_and_reply(update, chat_id, parsed, vitals_date=vitals_date)
        return

    if intent == "log_income":
        context.chat_data.pop(PENDING_KEY, None)
        income_items = parsed.get("income") or []
        income_items = [it for it in income_items if it.get("amount") is not None]
        if not income_items:
            await _reply(update, chat_id, "I didn't catch an amount -- try e.g. 'got a $500 bonus today'.")
            return
        await _log_incomes_and_reply(update, chat_id, income_items)
        return

    if intent == "log_deduction":
        context.chat_data.pop(PENDING_KEY, None)
        deduction_items = parsed.get("deductions") or []
        deduction_items = [it for it in deduction_items if it.get("amount") is not None]
        if not deduction_items:
            await _reply(update, chat_id, "I didn't catch an amount -- try e.g. 'paid $800 income tax today'.")
            return
        await _log_deductions_and_reply(update, chat_id, deduction_items)
        return

    if intent == "log_subscription":
        context.chat_data.pop(PENDING_KEY, None)
        subscription_items = parsed.get("subscriptions") or []
        # Only amount is rejected up front -- a missing/unclear frequency or
        # renewal date no longer sinks an item (see
        # subscriptions._next_renewal_date_from_days's fallback), so a bad
        # line in a bulk paste no longer needs its own per-item skip the way
        # the old billing_day-required design did.
        subscription_items = [it for it in subscription_items if it.get("amount") is not None]
        if not subscription_items:
            await _reply(
                update, chat_id,
                "I didn't catch a subscription to add -- try e.g. 'Netflix 15.98 on the 25th'."
            )
            return
        await _log_subscriptions_and_reply(update, chat_id, subscription_items)
        return

    if intent == "log_task":
        context.chat_data.pop(PENDING_KEY, None)
        tasks = parsed.get("tasks") or []
        if not tasks:
            await _reply(update, chat_id, "I didn't catch what to add -- try describing the to-do again.")
            return
        await _log_tasks_and_reply(update, chat_id, tasks)
        return

    if intent == "show_tasks":
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, _tasks_text(chat_id))
        return

    if intent == "add_reminder":
        context.chat_data.pop(PENDING_KEY, None)
        description = parsed.get("reminder_description")
        if not description:
            await _reply(update, chat_id, "What should I remind you about every day?")
            return
        await _add_reminder_and_reply(update, chat_id, description)
        return

    if intent == "show_reminders":
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, _reminders_text(chat_id))
        return

    if intent == "add_event":
        context.chat_data.pop(PENDING_KEY, None)
        events = parsed.get("events") or []
        if not events:
            await _reply(update, chat_id, "I didn't catch what to schedule -- try describing it again.")
            return
        await _add_events_and_reply(update, chat_id, events)
        return

    if intent == "show_events":
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, _events_text(chat_id))
        return

    if intent == "remember":
        context.chat_data.pop(PENDING_KEY, None)
        label = parsed.get("memory_label")
        content = parsed.get("memory_content")
        if not label or not content:
            await _reply(update, chat_id, "What should I remember, and what should I call it?")
            return
        db.set_memory(chat_id, label, content, parsed.get("memory_category"))
        await _reply(update, chat_id, f"Got it -- I'll remember \"{label}\": {content}")
        return

    if intent == "forget":
        context.chat_data.pop(PENDING_KEY, None)
        label = parsed.get("memory_label")
        deleted = db.delete_memory_by_label(chat_id, label) if label else None
        if not deleted:
            await _reply(update, chat_id, "I couldn't find that in what I remember -- run /memory to see the list.")
            return
        context.chat_data[LAST_CORRECTION_KEY] = {"domain": "memory", "action": "delete", "row": deleted}
        await _reply(update, chat_id, f"Forgot \"{deleted['label']}\". Reply 'undo' if that's wrong.")
        return

    if intent == "show_memory":
        # Same discipline as show_balance/show_recent -- the real saved
        # list, not the model's best guess at what it remembers.
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(update, chat_id, _memory_text(chat_id))
        return

    if intent == "casual":
        # Genuinely open-ended conversation -- a dedicated call with full
        # room to actually engage, not parse_message's own casual_reply
        # field (kept only as this call's own fallback -- see
        # _casual_reply_text's docstring).
        # narrate=False -- ai.answer_casually already IS the companion
        # voice (that's its whole purpose); a second narrate_reply pass on
        # top of it would be redundant.
        context.chat_data.pop(PENDING_KEY, None)
        reply = await _casual_reply_text(chat_id, merged_text, recent_messages, memory_list,
                                          parsed.get("casual_reply"))
        await _reply(update, chat_id, reply, narrate=False)
        return

    if intent != "log_expense":
        # A real classification gap -- the model didn't tag this cleanly as
        # ANY known intent, not even "casual". This isn't a conversation to
        # engage with, it's a "I don't know what you want" case, so point
        # at the real command surface instead of trying to converse about it.
        context.chat_data.pop(PENDING_KEY, None)
        await _reply(
            update, chat_id,
            "Not sure what to do with that. Use /log, /claim, /logmeal, /logworkout, /logvitals, /balance, "
            "/summary, /recent, or /help."
        )
        return

    context.chat_data.pop(PENDING_KEY, None)

    items = parsed.get("expenses") or []
    items = [it for it in items if it.get("amount") is not None]
    if not items:
        await _reply(update, chat_id, "I still didn't catch an amount -- try e.g. 'spent 12 on lunch'.")
        return

    logged_lines = []
    any_personal = False
    any_backdated = False
    for item in items:
        amount = float(item["amount"])
        currency = fx.normalize_currency(item.get("currency"))
        description = item.get("description") or "expense"
        category = item.get("category") or ai.categorize(description)
        is_claimable = bool(item.get("is_claimable"))
        # See ai.py's logged_days_ago rule -- "yesterday I paid 12 for lunch"
        # used to always land on today regardless of what the message said.
        expense_date = _target_date_from_days_ago(item.get("logged_days_ago"))
        db.add_expense(chat_id, amount, currency, description, category, is_claimable=is_claimable,
                       expense_date=expense_date)
        tag = " [claimable]" if is_claimable else ""
        date_tag = f" ({expense_date})" if expense_date else ""
        logged_lines.append(f"{_money(amount, currency)} -- {description} [{category}]{tag}{date_tag}")
        any_personal = any_personal or not is_claimable
        any_backdated = any_backdated or bool(expense_date)

    status = db.get_status(chat_id)
    header = "Logged:" if len(logged_lines) == 1 else f"Logged {len(logged_lines)} expenses:"
    body = "\n".join(logged_lines) if len(logged_lines) == 1 else "\n".join(f"- {ln}" for ln in logged_lines)
    # A backdated expense doesn't change today's own balance/spent figures --
    # showing _status_text right under it would misleadingly read as if it
    # did, so it's skipped whenever at least one item in this message landed
    # on an earlier day (rare enough that dropping it for the whole message,
    # rather than only the backdated item, isn't worth the extra complexity).
    status_line = f"\n\n{_status_text(status)}" if not any_backdated else ""
    await _reply(update, chat_id, f"{header}\n{body}{status_line}")
    if any_personal:
        await _send_alert_if_needed(update, chat_id)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Safety net: any exception a handler doesn't catch itself lands here.
    Without this, python-telegram-bot just logs it and the user gets dead
    silence -- this makes sure they always get *some* reply."""
    logger.error("Unhandled exception while processing update: %s", update, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Something went wrong on my end processing that -- try again in a moment."
            )
        except Exception:
            logger.exception("Failed to notify user about the error")
