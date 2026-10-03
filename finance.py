"""
Expense tracking: /log, /claim, /claimed, /balance, /recent, /undo,
/delete, /edit, /settarget, /adjustbalance. The original domain this bot was
built around -- everything else follows its conventions, not the other way
around.
"""

from telegram import Update
from telegram.ext import ContextTypes

import ai
import db
import fx
from access import _reject_if_not_allowed
from correction import LAST_CORRECTION_KEY
from formatting import _expense_line, _money, _status_text
from replies import _send_alert_if_needed


async def settarget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /settarget 100")
        return
    try:
        amount = float(context.args[0])
    except ValueError:
        await update.message.reply_text("That doesn't look like a number. Try: /settarget 100")
        return
    db.set_daily_target(chat_id, amount)
    status = db.get_status(chat_id)
    await update.message.reply_text(
        f"Daily target set to {_money(amount)}.\n\n{_status_text(status)}"
    )


async def adjustbalance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The one supported way to manually correct rolled-over balance/deficit
    drift -- see db.adjust_balance's docstring for when this is actually
    needed (lost expense history from before some date, so the automatic
    day-by-day rollover has nothing to derive that period's deficit from).
    A delta, not an absolute value -- /adjustbalance -1135.89 ADDS a
    1135.89 deficit to whatever balance already is, it doesn't set balance
    to -1135.89."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text(
            "Usage: /adjustbalance <delta> -- e.g. /adjustbalance -1135.89 to add that deficit, "
            "or /adjustbalance 50 to add a credit. This ADDS to your current rolled-over balance, "
            "it doesn't replace it."
        )
        return
    try:
        delta = float(context.args[0])
    except ValueError:
        await update.message.reply_text("That doesn't look like a number. Try: /adjustbalance -1135.89")
        return
    result = db.adjust_balance(chat_id, delta)
    context.chat_data[LAST_CORRECTION_KEY] = {
        "domain": "balance", "action": "adjust_balance", "old_balance": result["old_balance"],
    }
    status = db.get_status(chat_id)
    sign = "+" if delta >= 0 else ""
    await update.message.reply_text(
        f"Balance adjusted by {sign}{_money(delta)}: {_money(result['old_balance'])} -> "
        f"{_money(result['new_balance'])}.\n\n{_status_text(status)}\n\nReply 'undo' if that's wrong."
    )


def _split_amount_currency_description(args):
    """Parses ['20', 'USD', 'taxi', 'ride'] -> (20.0, 'USD', 'taxi ride')
    or ['12.50', 'lunch'] -> (12.5, None, 'lunch')."""
    if not args:
        return None, None, ""
    try:
        amount = float(args[0])
    except ValueError:
        return None, None, ""
    rest = args[1:]
    currency = None
    if rest:
        maybe_currency = fx.normalize_currency(rest[0])
        if maybe_currency:
            currency = maybe_currency
            rest = rest[1:]
    description = " ".join(rest)
    return amount, currency, description


async def log_expense(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    await _handle_explicit_log(update, context, is_claimable=False)


async def claim_expense(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    await _handle_explicit_log(update, context, is_claimable=True)


async def _handle_explicit_log(update: Update, context: ContextTypes.DEFAULT_TYPE, is_claimable: bool):
    chat_id = update.effective_chat.id
    cmd = "/claim" if is_claimable else "/log"
    amount, currency, description = _split_amount_currency_description(context.args)
    if amount is None:
        await update.message.reply_text(f"Usage: {cmd} 12.50 [CURRENCY] [description]")
        return
    description = description or ("claimable expense" if is_claimable else "expense")
    category = ai.categorize(description)
    expense_id = db.add_expense(chat_id, amount, currency, description, category, is_claimable=is_claimable)
    # See handlers.py's log_expense branch for why this is a re-fetch, not
    # a flag threaded through the inputs above -- fx_fallback is decided
    # inside add_expense itself (fx.to_base_checked).
    row = db.get_expense(chat_id, expense_id)
    fx_tag = (
        " [couldn't fetch today's exchange rate just now -- logged at 1:1, let me know and I'll fix it]"
        if row and row.get("fx_fallback") else ""
    )

    if is_claimable:
        status = db.get_status(chat_id)
        await update.message.reply_text(
            f"Logged claimable: {_money(amount, currency)} -- {description} [{category}]{fx_tag}\n"
            f"Pending claimables: {_money(status['pending_claimable'])}\n"
            "(This doesn't touch your daily allowance.)"
        )
    else:
        status = db.get_status(chat_id)
        await update.message.reply_text(
            f"Logged: {_money(amount, currency)} -- {description} [{category}]{fx_tag}\n\n{_status_text(status)}"
        )
        await _send_alert_if_needed(update, chat_id)


async def claimed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    count, total = db.clear_claimables(chat_id)
    if count == 0:
        await update.message.reply_text("No pending claimables to clear.")
    else:
        await update.message.reply_text(
            f"Cleared {count} claimable expense(s) totaling {_money(total)}. Nice."
        )


def _month_to_date_text(chat_id: int) -> str:
    mtd = db.get_month_to_date_total(chat_id)
    return f"Month to date ({mtd['days_elapsed']} day(s)): {_money(mtd['total'])} spent since {mtd['month_start']}"


def _balance_text(chat_id: int) -> str:
    """Shared by the /balance command and the natural-language 'show me my
    balance' intent, so both paths are guaranteed to say the same thing --
    a free-text balance query is not a second, separately-maintained
    implementation."""
    status = db.get_status(chat_id)
    text = _status_text(status) + "\n\n" + _month_to_date_text(chat_id)
    pending = db.get_pending_claimables(chat_id)
    if pending:
        lines = [f"  #{p['id']} {_money(p['amount'], p['currency'])} -- {p['description']} [{p['category']}]"
                  for p in pending]
        text += "\n\nPending claimables:\n" + "\n".join(lines)
    return text


def _recent_text(chat_id: int, limit: int = 10) -> str:
    """Shared by /recent and the natural-language 'show me my recent
    expenses' intent -- see _balance_text's docstring for why."""
    rows = db.get_recent_expenses(chat_id, limit=limit)
    if not rows:
        return "No expenses logged yet."
    lines = [_expense_line(r) for r in rows]
    return "Recent expenses:\n" + "\n".join(lines)


async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    await update.message.reply_text(_balance_text(update.effective_chat.id))


async def recent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    limit = 10
    if context.args:
        try:
            limit = max(1, min(50, int(context.args[0])))
        except ValueError:
            pass
    await update.message.reply_text(_recent_text(chat_id, limit=limit))


async def undo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    row = db.delete_most_recent(chat_id)
    if row is None:
        await update.message.reply_text("Nothing to undo.")
        return
    status = db.get_status(chat_id)
    await update.message.reply_text(
        f"Removed: {_money(row['amount'], row['currency'])} -- {row['description']} [{row['category']}]\n\n"
        f"{_status_text(status)}"
    )


async def delete_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /delete <id> (see /recent for IDs)")
        return
    try:
        expense_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /delete <id> (see /recent for IDs)")
        return
    row = db.delete_expense(chat_id, expense_id)
    if row is None:
        await update.message.reply_text(f"No expense #{expense_id} found.")
        return
    status = db.get_status(chat_id)
    await update.message.reply_text(
        f"Deleted: {_money(row['amount'], row['currency'])} -- {row['description']} [{row['category']}]\n\n"
        f"{_status_text(status)}"
    )


async def edit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /edit <id> <new amount> [new description...] (see /recent for IDs)")
        return
    try:
        expense_id = int(context.args[0])
        new_amount = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Usage: /edit <id> <new amount> [new description...]")
        return
    new_description = " ".join(context.args[2:]) or None
    new_category = ai.categorize(new_description) if new_description else None

    row = db.edit_expense(chat_id, expense_id, new_amount=new_amount,
                           new_description=new_description, new_category=new_category)
    if row is None:
        await update.message.reply_text(f"No expense #{expense_id} found.")
        return
    status = db.get_status(chat_id)
    await update.message.reply_text(
        f"Updated #{expense_id}: {_money(row['amount'], row['currency'])} -- "
        f"{row['description']} [{row['category']}]\n\n{_status_text(status)}"
    )
