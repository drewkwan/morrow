"""
Income and deductions: real money arriving (salary, bonuses, other income)
or leaving the account without being a discretionary purchase (tax, a CPF
top-up). Deliberately separate from the expense/balance machinery -- see
db.py's "income"/"deductions"/"net worth" section docstrings for the full
design. The existing daily spending target/streak (`users.balance`, /balance,
/rundown's "balance" section) stays completely untouched by any of this --
net worth is a genuinely different number, answering "how much money do I
actually have" rather than "am I within my discretionary spending target
today".

Recurring salary is config (income_config, one row per chat, set via
/setincome) plus income_tick, the daily job that auto-posts it on payday --
mirrors subscriptions.py's subscriptions_tick exactly, just crediting
instead of debiting. Ad hoc income (a bonus, freelance work, an off-schedule
salary) and deductions (tax, a CPF top-up) are logged directly, either by
natural language (the "log_income"/"log_deduction" intents -- see ai.py) or
the /addincome and /adddeduction commands.
"""

import logging
from datetime import date, timedelta

from telegram import Update
from telegram.ext import ContextTypes

import db
from access import _reject_if_not_allowed
from formatting import _deduction_line, _income_line, _money, _net_worth_text
from replies import _reply

logger = logging.getLogger(__name__)

INCOME_SOURCES = ("salary", "bonus", "other")


# ---------- ad hoc income (natural language + /addincome) ----------

async def _log_incomes_and_reply(update: Update, chat_id: int, items: list):
    """Entry point for the natural-language 'log_income' intent, which can
    name several distinct amounts in one message -- same multi-item
    discipline as _log_lifts_and_reply/_log_tasks_and_reply. Each item's
    "amount" is always the real NET take-home figure (see ai.py's
    log_income field docs for why), with gross_amount/cpf_amount/
    stock_amount as optional breakdown detail."""
    lines = []
    for item in items:
        net_amount = float(item["amount"])
        currency = item.get("currency")
        source = item.get("source") if item.get("source") in INCOME_SOURCES else "other"
        income_date = _income_date_from_days_ago(item.get("logged_days_ago"))
        income_id = db.add_income(
            chat_id, source, net_amount, currency, description=item.get("description"),
            gross_amount=item.get("gross_amount"), cpf_amount=item.get("cpf_amount"),
            stock_amount=item.get("stock_amount"), income_date=income_date,
        )
        row = db.get_income(chat_id, income_id)
        lines.append(_income_line(row))

    header = "Logged:" if len(lines) == 1 else f"Logged {len(lines)} income entries:"
    body = "\n".join(lines) if len(lines) == 1 else "\n".join(f"- {ln}" for ln in lines)
    await _reply(update, chat_id, f"{header}\n{body}")


async def addincome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/addincome <amount> [currency] <description...> -- the structured
    fallback for a one-off income entry; natural language ("got a $500
    bonus today") goes through the same db.add_income via the log_income
    intent instead (see _log_incomes_and_reply)."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    args = context.args
    usage = "Usage: /addincome <amount> [currency] <description>"
    if not args:
        await update.message.reply_text(usage)
        return
    try:
        amount = float(args[0])
    except ValueError:
        await update.message.reply_text(f"That doesn't look like an amount. {usage}")
        return
    rest = args[1:]
    currency = None
    if rest and rest[0].isalpha() and len(rest[0]) == 3:
        currency = rest[0].upper()
        rest = rest[1:]
    description = " ".join(rest) or None
    income_id = db.add_income(chat_id, "other", amount, currency, description=description)
    row = db.get_income(chat_id, income_id)
    await _reply(update, chat_id, f"Logged: {_income_line(row)}")


async def recentincome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    rows = db.get_recent_income(chat_id, limit=10)
    if not rows:
        await update.message.reply_text("No income logged yet.")
        return
    await update.message.reply_text("Recent income:\n" + "\n".join(_income_line(r) for r in rows))


async def removeincome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /removeincome <id> -- see /recentincome for ids")
        return
    try:
        income_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /removeincome <id> -- see /recentincome for ids")
        return
    row = db.delete_income(chat_id, income_id)
    if row is None:
        await update.message.reply_text("I don't have an income entry with that id -- check /recentincome.")
        return
    await update.message.reply_text(f"Removed: {_income_line(row)}")


# ---------- deductions (natural language + /adddeduction) ----------

async def _log_deductions_and_reply(update: Update, chat_id: int, items: list):
    """Same multi-item discipline as _log_incomes_and_reply, for the
    natural-language 'log_deduction' intent."""
    lines = []
    for item in items:
        amount = float(item["amount"])
        currency = item.get("currency")
        label = item.get("label") or "deduction"
        deduction_date = _income_date_from_days_ago(item.get("logged_days_ago"))
        deduction_id = db.add_deduction(chat_id, label, amount, currency, deduction_date=deduction_date)
        row = db.get_deduction(chat_id, deduction_id)
        lines.append(_deduction_line(row))

    header = "Logged:" if len(lines) == 1 else f"Logged {len(lines)} deductions:"
    body = "\n".join(lines) if len(lines) == 1 else "\n".join(f"- {ln}" for ln in lines)
    await _reply(update, chat_id, f"{header}\n{body}")


async def adddeduction_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    args = context.args
    usage = "Usage: /adddeduction <amount> [currency] <label>"
    if not args:
        await update.message.reply_text(usage)
        return
    try:
        amount = float(args[0])
    except ValueError:
        await update.message.reply_text(f"That doesn't look like an amount. {usage}")
        return
    rest = args[1:]
    currency = None
    if rest and rest[0].isalpha() and len(rest[0]) == 3:
        currency = rest[0].upper()
        rest = rest[1:]
    label = " ".join(rest) or "deduction"
    deduction_id = db.add_deduction(chat_id, label, amount, currency)
    row = db.get_deduction(chat_id, deduction_id)
    await _reply(update, chat_id, f"Logged: {_deduction_line(row)}")


async def recentdeductions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    rows = db.get_recent_deductions(chat_id, limit=10)
    if not rows:
        await update.message.reply_text("No deductions logged yet.")
        return
    await update.message.reply_text("Recent deductions:\n" + "\n".join(_deduction_line(r) for r in rows))


async def removededuction_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /removededuction <id> -- see /recentdeductions for ids")
        return
    try:
        deduction_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Usage: /removededuction <id> -- see /recentdeductions for ids")
        return
    row = db.delete_deduction(chat_id, deduction_id)
    if row is None:
        await update.message.reply_text("I don't have a deduction with that id -- check /recentdeductions.")
        return
    await update.message.reply_text(f"Removed: {_deduction_line(row)}")


# ---------- recurring salary config ----------

async def setincome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/setincome <gross_amount> [currency] <pay_day 1-31> [cpf_rate%] [stock_rate%]
    -- a structured command, not natural language, same reasoning as
    subscriptions.addsubscription_cmd: this is one-time config (gross pay,
    pay day, CPF/stock percentages), not something reported in the moment.
    Always replaces the whole config -- see db.set_income_config's
    docstring for why a raise/job change means giving everything together."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    args = context.args
    usage = "Usage: /setincome <gross_amount> [currency] <pay_day 1-31> [cpf_rate%] [stock_rate%]"
    if len(args) < 2:
        await update.message.reply_text(usage)
        return
    try:
        gross_amount = float(args[0])
    except ValueError:
        await update.message.reply_text(f"That doesn't look like an amount. {usage}")
        return
    rest = args[1:]
    currency = None
    if rest and rest[0].isalpha() and len(rest[0]) == 3:
        currency = rest[0].upper()
        rest = rest[1:]
    if not rest:
        await update.message.reply_text(usage)
        return
    try:
        pay_day = int(rest[0])
    except ValueError:
        await update.message.reply_text(f"That doesn't look like a day of the month. {usage}")
        return
    if not (1 <= pay_day <= 31):
        await update.message.reply_text("Pay day must be between 1 and 31.")
        return
    rest = rest[1:]
    try:
        cpf_rate = _parse_rate(rest[0]) if len(rest) > 0 else 0.0
        stock_rate = _parse_rate(rest[1]) if len(rest) > 1 else 0.0
    except ValueError:
        await update.message.reply_text(f"CPF/stock rates should look like '20%' or '0.2'. {usage}")
        return
    db.set_income_config(chat_id, gross_amount, currency, pay_day, cpf_rate=cpf_rate, stock_rate=stock_rate)
    config_row = db.get_income_config(chat_id)
    await _reply(update, chat_id, f"Income config set:\n{_income_config_text(config_row)}")


def _parse_rate(raw: str) -> float:
    """'20%' -> 0.2, '0.2' -> 0.2, '20' -> 0.2 (bare numbers over 1 are
    assumed to be a percentage, not a fraction greater than one, since a
    100%+ CPF/stock rate would be nonsensical)."""
    raw = raw.strip().rstrip("%")
    value = float(raw)
    return value / 100 if value > 1 else value


def _income_config_text(config_row: dict) -> str:
    return (
        f"Gross: {_money(config_row['gross_amount'], config_row['currency'])}/mo on the {config_row['pay_day']}\n"
        f"CPF rate: {config_row['cpf_rate'] * 100:.1f}%\n"
        f"Stock rate: {config_row['stock_rate'] * 100:.1f}%\n"
        f"Net (estimated): {_money(_compute_net_salary(config_row), config_row['currency'])}/mo"
    )


def _compute_net_salary(config_row: dict) -> float:
    gross = config_row["gross_amount"]
    return gross * (1 - config_row["cpf_rate"] - config_row["stock_rate"])


async def incomeconfig_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    config_row = db.get_income_config(chat_id)
    if config_row is None:
        await update.message.reply_text("No recurring salary set up. Use /setincome to add one.")
        return
    await update.message.reply_text(_income_config_text(config_row))


async def clearincome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    db.clear_income_config(chat_id)
    await update.message.reply_text("Recurring salary config cleared. It won't auto-post anymore.")


async def income_tick(context: ContextTypes.DEFAULT_TYPE):
    """Runs once a day (see app.py's job_queue registration) and auto-posts
    the configured salary for every chat whose pay day is today -- mirrors
    subscriptions.subscriptions_tick exactly, just crediting (db.add_income)
    instead of debiting (db.add_expense). CPF/stock are computed from the
    config's fixed rates against gross_amount -- see _compute_net_salary --
    never a value typed in each time, since Andrew's actual CPF/stock
    contributions are fixed percentages (confirmed), not something that
    varies paycheck to paycheck."""
    today = date.fromisoformat(db.today_str())
    month_str = today.strftime("%Y-%m")
    for chat_id in db.get_all_chat_ids():
        config_row = db.get_income_config(chat_id)
        if config_row is None or not config_row.get("active"):
            continue
        if config_row.get("last_logged_month") == month_str:
            continue
        if not db.day_matches_billing_day(today, config_row["pay_day"]):
            continue
        gross = config_row["gross_amount"]
        cpf_amount = round(gross * config_row["cpf_rate"], 2)
        stock_amount = round(gross * config_row["stock_rate"], 2)
        net_amount = round(gross - cpf_amount - stock_amount, 2)
        db.add_income(
            chat_id, "salary", net_amount, config_row["currency"], description="salary",
            gross_amount=gross, cpf_amount=cpf_amount, stock_amount=stock_amount,
        )
        db.mark_income_config_logged(chat_id, month_str)
        currency = config_row["currency"]
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"Auto-posted salary: {_money(net_amount, currency)} net "
                    f"(gross {_money(gross, currency)}, CPF {_money(cpf_amount, currency)}, "
                    f"stock {_money(stock_amount, currency)})"
                ),
            )
        except Exception:
            logger.exception("Failed to notify chat %s of auto-posted salary", chat_id)


# ---------- net worth ----------

async def networth_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    await update.message.reply_text(_net_worth_text(db.get_net_worth(chat_id)))


def _income_date_from_days_ago(days_ago) -> str | None:
    """Same deterministic day-count-to-date conversion as nutrition.
    _target_date_from_days_ago -- a separate copy (rather than importing
    that one directly) only because importing nutrition here would be a
    needless cross-domain dependency for one tiny helper; the logic itself
    must stay identical, so change both together if it ever changes."""
    if not days_ago:
        return None
    days_ago = max(0, min(14, int(days_ago)))
    if days_ago == 0:
        return None
    return (date.fromisoformat(db.today_str()) - timedelta(days=days_ago)).isoformat()
