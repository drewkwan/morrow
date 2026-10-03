"""
Recurring subscriptions (Netflix, gym, insurance, etc.) that auto-log
THEMSELVES as a real expense on their renewal date -- see db.py's
"subscriptions" section docstring for the full schema design
(frequency/next_renewal_date/card/notes/is_claimable, replacing the old
monthly-only billing_day design). This module is config management
(/addsubscription, /subscriptions, /removesubscription) plus two daily/
weekly jobs: subscriptions_tick (auto-logs a cycle's expense on its real
renewal day) and subscriptions_digest_tick (a separate weekly heads-up of
what's coming up -- Andrew's explicit choice to keep renewal reminders out
of the daily morning briefing, see subscriptions_digest_tick's own
docstring).
"""

import logging
from datetime import date, timedelta

from telegram import Update
from telegram.ext import ContextTypes

import config
import db
from access import _reject_if_not_allowed
from formatting import _money, _subscription_line
from replies import _reply

logger = logging.getLogger(__name__)

DEFAULT_SUBSCRIPTION_CATEGORY = "Bills & Utilities"

# "What's this subscription really costing per month" -- used only for
# /subscriptions' summary total, so quarterly/annual/weekly/biweekly
# subscriptions can be compared on one common footing instead of just
# listing raw per-cycle amounts (this is part of what Andrew asked for --
# "useful for future analytics"). 52/12 and 26/12 weeks-per-month rather
# than a flat x4/x2 -- a year has slightly more than 52/4 weeks, and this
# keeps the monthly-equivalent total from quietly under-counting weekly/
# biweekly subscriptions.
_MONTHLY_MULTIPLIER = {
    "weekly": 52 / 12,
    "biweekly": 26 / 12,
    "monthly": 1,
    "quarterly": 1 / 3,
    "annual": 1 / 12,
}


def _next_renewal_date_from_days(renews_in_days, frequency: str) -> str:
    """Deterministic day-count-to-date conversion -- same discipline as
    tasks._due_at_from_fields/income._income_date_from_days_ago: the model
    only ever extracts a day-count (how many days from today this next
    renews), never a calendar date itself (see ai.py's log_subscription
    rules). Falls back to "the next occurrence of `frequency` starting from
    today" (via db.advance_date_by_frequency) when the model couldn't
    extract a day-count at all -- e.g. "I just signed up for X, it's
    monthly" with no renewal date mentioned -- a reasonable default rather
    than refusing to log the subscription."""
    today = date.fromisoformat(db.today_str())
    if isinstance(renews_in_days, int) and renews_in_days >= 0:
        return (today + timedelta(days=renews_in_days)).isoformat()
    return db.advance_date_by_frequency(today, frequency).isoformat()


async def _log_subscriptions_and_reply(update: Update, chat_id: int, items: list):
    """Entry point for the natural-language log_subscription intent --
    covers both a single "hey I just signed up for X" message and the
    bulk-paste workflow (Andrew pasting his full list of real subscriptions
    in one message and having every row inserted in one shot). Same
    multi-item discipline as _log_tasks_and_reply/income._log_incomes_and_
    reply. frequency defaults to "monthly" if missing/not one of
    db.RECURRENCE_FREQUENCIES (mirrors db.add_subscription's own
    fallback); next_renewal_date is always computed deterministically via
    _next_renewal_date_from_days, never trusted from the model directly."""
    lines = []
    for item in items:
        name = item.get("name") or "subscription"
        amount = float(item["amount"])
        currency = item.get("currency")
        frequency = item.get("frequency") if item.get("frequency") in db.RECURRENCE_FREQUENCIES else "monthly"
        next_renewal_date = _next_renewal_date_from_days(item.get("renews_in_days"), frequency)
        category = item.get("category") or DEFAULT_SUBSCRIPTION_CATEGORY
        card = item.get("card")
        notes = item.get("notes")
        is_claimable = bool(item.get("is_claimable"))
        subscription_id = db.add_subscription(
            chat_id, name, amount, currency, frequency, next_renewal_date,
            category=category, card=card, notes=notes, is_claimable=is_claimable,
        )
        row = db.get_subscription(chat_id, subscription_id)
        lines.append(_subscription_line(row))

    header = "Added:" if len(lines) == 1 else f"Added {len(lines)} subscriptions:"
    body = "\n".join(lines) if len(lines) == 1 else "\n".join(f"- {ln}" for ln in lines)
    await _reply(update, chat_id, f"{header}\n{body}")


async def addsubscription_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/addsubscription <name> <amount> [currency] <frequency> <next_renewal YYYY-MM-DD> [category]
    -- a structured command, not natural language, since this is one-time
    config (a name, amount, cadence, and renewal date) rather than
    something reported in the moment the way a real expense/meal/workout
    is. Deliberately doesn't take card/notes/is_claimable here -- those are
    secondary detail best set via natural-language correction
    ("that Anytime Fitness one is claimable") or the bulk NL path
    (_log_subscriptions_and_reply) instead of a longer, harder-to-type
    command line. name is a single token (no spaces), same limitation the
    old billing_day-based version had -- args[0] is the name, args[1] the
    amount, same simple positional parsing."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    args = context.args
    usage = (
        "Usage: /addsubscription <name> <amount> [currency] <frequency> <next_renewal YYYY-MM-DD> [category]\n"
        f"frequency is one of: {', '.join(db.RECURRENCE_FREQUENCIES)}"
    )
    if len(args) < 4:
        await update.message.reply_text(usage)
        return
    name = args[0]
    try:
        amount = float(args[1])
    except ValueError:
        await update.message.reply_text(f"That doesn't look like an amount. {usage}")
        return
    rest = args[2:]
    currency = None
    # Same light heuristic as the other /add* commands -- the one
    # 3-letter-looking alpha token among the remaining args is the currency.
    if rest and rest[0].isalpha() and len(rest[0]) == 3:
        currency = rest[0].upper()
        rest = rest[1:]
    if not rest:
        await update.message.reply_text(usage)
        return
    frequency = rest[0].lower()
    if frequency not in db.RECURRENCE_FREQUENCIES:
        await update.message.reply_text(f"Frequency must be one of: {', '.join(db.RECURRENCE_FREQUENCIES)}. {usage}")
        return
    rest = rest[1:]
    if not rest:
        await update.message.reply_text(usage)
        return
    try:
        next_renewal_date = date.fromisoformat(rest[0]).isoformat()
    except ValueError:
        await update.message.reply_text(f"That doesn't look like a date (want YYYY-MM-DD). {usage}")
        return
    category = " ".join(rest[1:]) or DEFAULT_SUBSCRIPTION_CATEGORY
    subscription_id = db.add_subscription(
        chat_id, name, amount, currency, frequency, next_renewal_date, category=category,
    )
    row = db.get_subscription(chat_id, subscription_id)
    await _reply(update, chat_id, f"Added: {_subscription_line(row)}")


async def subscriptions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    rows = db.get_subscriptions(chat_id, active_only=True)
    if not rows:
        await update.message.reply_text("No active subscriptions. Add one with /addsubscription.")
        return
    lines = "\n".join(_subscription_line(r) for r in rows)
    monthly_total = sum(r["amount_base"] * _MONTHLY_MULTIPLIER.get(r["frequency"], 1) for r in rows)
    await update.message.reply_text(
        f"Active subscriptions:\n{lines}\n\n~{_money(monthly_total)}/mo equivalent across all cadences"
    )


async def removesubscription_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /removesubscription <id> -- see /subscriptions for ids")
        return
    try:
        subscription_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /removesubscription <id> -- see /subscriptions for ids")
        return
    row = db.delete_subscription(chat_id, subscription_id)
    if row is None:
        await update.message.reply_text("I don't have a subscription with that id -- check /subscriptions.")
        return
    await update.message.reply_text(f"Removed: {_subscription_line(row)}")


async def subscriptions_tick(context: ContextTypes.DEFAULT_TYPE):
    """Runs once a day (see app.py's job_queue registration,
    config.RECURRING_FINANCE_HOUR/MINUTE) and auto-posts any subscription
    whose next_renewal_date is today as a real expense -- mirrors
    income.income_tick exactly, just debiting (db.add_expense) instead of
    crediting. Idempotency is next_renewal_date itself: db.advance_
    subscription moves it past today immediately after posting, so a
    same-day retry (a bot restart, a job re-fire) no longer matches
    `next_renewal_date == today` on its next pass -- unlike the old
    last_logged_month guard, this needs no separate "have I already done
    this cycle" check. is_claimable flows straight through to the posted
    expense (see db.py's subscriptions section docstring), so a
    subscription you expense to your employer shows up in /claimed the
    normal way. is_subscription=True is also always passed -- a real bug
    this fixed: an auto-posted renewal used to count against the daily
    spending target and the within-budget streak exactly like a
    discretionary purchase, so a Netflix bill could silently break a
    streak the user never actually overspent on (see
    db._spent_on/expenses.is_subscription's docstrings). It still counts
    everywhere real spend matters (month-to-date total, category
    insights, net worth, /recent) -- only the daily target/streak exclude
    it."""
    today = date.fromisoformat(db.today_str())
    today_str = today.isoformat()
    for chat_id in db.get_all_chat_ids():
        for sub in db.get_subscriptions(chat_id, active_only=True):
            if sub["next_renewal_date"] != today_str:
                continue
            db.add_expense(
                chat_id, sub["amount"], sub["currency"], sub["name"],
                sub.get("category") or DEFAULT_SUBSCRIPTION_CATEGORY,
                is_claimable=bool(sub.get("is_claimable")),
                is_subscription=True,
            )
            new_next_renewal_date = db.advance_date_by_frequency(today, sub["frequency"]).isoformat()
            db.advance_subscription(sub["id"], today_str, new_next_renewal_date)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"Auto-posted subscription: {_money(sub['amount'], sub['currency'])} -- {sub['name']} "
                        f"(next renewal {new_next_renewal_date})"
                    ),
                )
            except Exception:
                logger.exception("Failed to notify chat %s of auto-posted subscription", chat_id)


async def subscriptions_digest_tick(context: ContextTypes.DEFAULT_TYPE):
    """Runs weekly (see app.py's job_queue registration, config.
    SUBSCRIPTION_DIGEST_* constants) and sends a standalone heads-up of
    subscriptions renewing within the next
    config.SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS days. Kept deliberately
    SEPARATE from the daily morning briefing -- Andrew's explicit choice
    (AskUserQuestion: "Separate weekly digest"), since a renewal heads-up
    read every single day would be noise, but is exactly the right cadence
    once a week. Purely informational -- doesn't touch next_renewal_date or
    last_logged_date; subscriptions_tick (above) is what actually
    auto-posts/advances a cycle on its real renewal day, independent of
    whether this digest ever ran. A chat with nothing renewing in the
    window gets no message at all, rather than a "nothing's due" ping
    every single week."""
    today = date.fromisoformat(db.today_str())
    window_end = today + timedelta(days=config.SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS)
    for chat_id in db.get_all_chat_ids():
        upcoming = [
            sub for sub in db.get_subscriptions(chat_id, active_only=True)
            if today <= date.fromisoformat(sub["next_renewal_date"]) <= window_end
        ]
        if not upcoming:
            continue
        body = "\n".join(_subscription_line(sub) for sub in upcoming)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Renewing in the next {config.SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS} days:\n{body}",
            )
        except Exception:
            logger.exception("Failed to send subscriptions digest to chat %s", chat_id)
