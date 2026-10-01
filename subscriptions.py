"""
Recurring monthly subscriptions (Netflix, gym, etc.) that auto-log
themselves as a real expense on their billing day every month -- see
db.py's "subscriptions" section docstring for the full design. This module
is pure config management (/addsubscription, /subscriptions,
/removesubscription) plus subscriptions_tick, the daily job that actually
does the auto-logging. The logged expense itself is an ordinary row in
`expenses` -- correctable, visible everywhere an expense already is --
there's nothing subscription-specific left to maintain once a cycle's
charge has landed.
"""

import logging
from datetime import date

from telegram import Update
from telegram.ext import ContextTypes

import db
from access import _reject_if_not_allowed
from formatting import _money, _subscription_line
from replies import _reply

logger = logging.getLogger(__name__)

DEFAULT_SUBSCRIPTION_CATEGORY = "Bills & Utilities"


async def _log_subscriptions_and_reply(update: Update, chat_id: int, items: list):
    """Entry point for the natural-language 'log_subscription' intent -- can
    register several subscriptions in ONE message, the real point of this
    path (see ai.py's log_subscription field docs): registering a whole
    starter list of recurring charges without a /addsubscription round trip
    per item (e.g. pasting in every subscription at once rather than typing
    one command per line). Each item needs a valid billing_day (1-31) to
    actually land -- an item missing one is skipped rather than guessed at
    or silently dropped, and reported back by name, so a big pasted list
    with one bad line doesn't lose the rest of it."""
    added_lines = []
    skipped = []
    for item in items:
        name = (item.get("name") or "").strip()
        amount = item.get("amount")
        billing_day = item.get("billing_day")
        if not name or amount is None:
            skipped.append(f"{name or 'unnamed'} (missing name or amount)")
            continue
        if not isinstance(billing_day, int) or not (1 <= billing_day <= 31):
            skipped.append(f"{name} (needs a billing day 1-31)")
            continue
        category = item.get("category") or DEFAULT_SUBSCRIPTION_CATEGORY
        subscription_id = db.add_subscription(
            chat_id, name, float(amount), item.get("currency"), billing_day, category=category
        )
        row = db.get_subscription(chat_id, subscription_id)
        added_lines.append(_subscription_line(row))

    parts = []
    if added_lines:
        header = "Added subscription:" if len(added_lines) == 1 else f"Added {len(added_lines)} subscriptions:"
        body = added_lines[0] if len(added_lines) == 1 else "\n".join(f"- {ln}" for ln in added_lines)
        parts.append(f"{header}\n{body}")
    if skipped:
        parts.append("Couldn't add (missing info): " + ", ".join(skipped))
    if not parts:
        parts.append("Didn't catch any subscriptions to add -- try naming each one's amount and billing day.")
    await _reply(update, chat_id, "\n\n".join(parts))


async def addsubscription_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/addsubscription <name> <amount> [currency] <billing_day> [category]
    -- a structured command, not natural language, since this is one-time
    config (a name, amount, and day of the month) rather than something
    reported in the moment the way a real expense/meal/workout is."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    args = context.args
    usage = "Usage: /addsubscription <name> <amount> [currency] <billing_day 1-31> [category]"
    if len(args) < 3:
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
    # A currency code is the one 3-letter-looking alpha token among the
    # remaining args -- the same light heuristic /log already works fine
    # without, kept deliberately simple here since this command's args are
    # few and well-ordered (name, amount, [currency], day, [category]).
    if rest and rest[0].isalpha() and len(rest[0]) == 3:
        currency = rest[0].upper()
        rest = rest[1:]
    if not rest:
        await update.message.reply_text(usage)
        return
    try:
        billing_day = int(rest[0])
    except ValueError:
        await update.message.reply_text(f"That doesn't look like a day of the month. {usage}")
        return
    if not (1 <= billing_day <= 31):
        await update.message.reply_text("Billing day must be between 1 and 31.")
        return
    category = " ".join(rest[1:]) or DEFAULT_SUBSCRIPTION_CATEGORY
    subscription_id = db.add_subscription(chat_id, name, amount, currency, billing_day, category=category)
    row = db.get_subscription(chat_id, subscription_id)
    await _reply(
        update, chat_id,
        f"Added subscription: {_subscription_line(row)}\nIt'll auto-log as an expense every month on the "
        f"{billing_day}."
    )


async def subscriptions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    rows = db.get_subscriptions(chat_id)
    if not rows:
        await update.message.reply_text("No active subscriptions. Add one with /addsubscription.")
        return
    total = sum(r["amount_base"] for r in rows)
    lines = [_subscription_line(r) for r in rows]
    await update.message.reply_text(
        "Subscriptions:\n" + "\n".join(lines) + f"\n\nTotal: {_money(total)}/mo"
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
    await update.message.reply_text(f"Removed subscription: {row['name']}. It won't auto-log anymore.")


async def subscriptions_tick(context: ContextTypes.DEFAULT_TYPE):
    """Runs once a day (see app.py's job_queue registration) and auto-logs
    an expense for every active subscription whose billing day is today --
    see db.day_matches_billing_day's docstring for how a billing_day that
    doesn't exist in every month (e.g. 31) is handled. Idempotent per
    calendar month via last_logged_month, so a bot restart or a job
    re-running the same day can never double-charge."""
    today = date.fromisoformat(db.today_str())
    month_str = today.strftime("%Y-%m")
    for chat_id in db.get_all_chat_ids():
        for sub in db.get_subscriptions(chat_id, active_only=True):
            if sub.get("last_logged_month") == month_str:
                continue
            if not db.day_matches_billing_day(today, sub["billing_day"]):
                continue
            db.add_expense(
                chat_id, sub["amount"], sub["currency"], sub["name"], sub.get("category"),
                is_claimable=False,
            )
            db.mark_subscription_logged(sub["id"], month_str)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"Auto-logged subscription: {_money(sub['amount'], sub['currency'])} -- {sub['name']}",
                )
            except Exception:
                logger.exception("Failed to notify chat %s of auto-logged subscription", chat_id)
