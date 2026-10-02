"""
Daily vitals check-ins: /logvitals, /recentvitals. Only fields actually
mentioned are set -- a partial check-in is never padded with a guess.
"""

from telegram import Update
from telegram.ext import ContextTypes

import ai
import db
from access import _reject_if_not_allowed
from formatting import _vitals_line
from replies import _reply


def _vitals_trend_text(chat_id: int, row: dict) -> str | None:
    """Compares this check-in to the most recent PRIOR one (weight/sleep
    deltas) -- real DB read, no AI, same discipline as
    lifts._last_lift_text. Only reports a delta for a field BOTH check-ins
    actually have -- a partial check-in (e.g. weight only) never gets a
    misleading comparison against a field it didn't report, and a
    first-ever check-in (nothing prior) returns None rather than an empty
    line. get_recent_vitals is newest-first, so the just-inserted row
    itself is normally the first result -- skip it by id and take the next."""
    for prev in db.get_recent_vitals(chat_id, limit=10):
        if prev["id"] == row["id"]:
            continue
        bits = []
        if row.get("weight_kg") is not None and prev.get("weight_kg") is not None:
            delta = row["weight_kg"] - prev["weight_kg"]
            sign = "+" if delta > 0 else ""
            bits.append(f"weight {sign}{delta:.1f}kg vs last ({prev['vitals_date']})")
        if row.get("sleep_hours") is not None and prev.get("sleep_hours") is not None:
            delta = row["sleep_hours"] - prev["sleep_hours"]
            sign = "+" if delta > 0 else ""
            bits.append(f"sleep {sign}{delta:.1f}h vs last")
        if not bits:
            return None
        return "Since last check-in: " + ", ".join(bits)
    return None


async def _log_vitals_fragment(chat_id: int, data: dict, vitals_date: str | None = None) -> str:
    """The pure write-and-describe half of logging a vitals check-in --
    writes the row and returns its confirmation text WITHOUT sending it, so
    handlers.py's multi-domain dispatch can combine it with a meal and/or a
    task fragment logged from the SAME message into one reply instead of
    several separate Telegram messages (see ai.py's log_meal COMBINING RULE
    for why a message can report vitals + a meal + a to-do all at once).
    _log_vitals_and_reply below is the thin wrapper every other caller
    (currently just /logvitals) still uses.

    vitals_date lets the natural-language log_vitals intent backdate a
    check-in the same way meals/workouts/expenses now can -- see ai.py's
    logged_days_ago rule and handlers.py's use of
    nutrition._target_date_from_days_ago for how it's computed. _vitals_line
    already always shows the date, so a backdated entry is visible without
    any extra tagging here. Also shows a real trend against the previous
    check-in when there's a comparable one (see _vitals_trend_text) -- a
    bare "logged" confirmation was the whole reply before, even though the
    point of tracking weight/sleep over time is seeing it move."""
    vitals_id = db.add_vitals(chat_id, data.get("weight_kg"), data.get("sleep_hours"),
                               data.get("knee_pain"), data.get("notes") or data.get("vitals_notes"),
                               vitals_date=vitals_date)
    row = db.get_vitals(chat_id, vitals_id)
    reply = f"Logged: {_vitals_line(row)}"
    trend = _vitals_trend_text(chat_id, row)
    if trend:
        reply += f"\n{trend}"
    return reply


async def _log_vitals_and_reply(update: Update, chat_id: int, data: dict, vitals_date: str | None = None):
    """Thin wrapper around _log_vitals_fragment for callers (currently just
    /logvitals) that want the confirmation sent directly rather than
    combined with other domains -- see _log_vitals_fragment's docstring."""
    await _reply(update, chat_id, await _log_vitals_fragment(chat_id, data, vitals_date=vitals_date))


async def logvitals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    description = " ".join(context.args)
    if not description:
        await update.message.reply_text("Usage: /logvitals weight 76.6, slept 5.5 hours, knee 2/10")
        return
    data = ai.extract_vitals(description)
    await _log_vitals_and_reply(update, chat_id, data)


async def recentvitals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    limit = 10
    if context.args:
        try:
            limit = max(1, min(50, int(context.args[0])))
        except ValueError:
            pass
    rows = db.get_recent_vitals(chat_id, limit=limit)
    if not rows:
        await update.message.reply_text("No check-ins logged yet.")
        return
    await update.message.reply_text("Recent check-ins:\n" + "\n".join(_vitals_line(r) for r in rows))
