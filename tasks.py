"""
To-dos: /addtask, /tasks, /done. A due date is only ever computed
deterministically in Python from a model-supplied day-count (+ optional
clock time) -- never invented or computed by the model itself, the same
discipline as correction.py's days_ago handling.
"""

from datetime import date, timedelta

from telegram import Update
from telegram.ext import ContextTypes

import ai
import db
from access import _reject_if_not_allowed
from correction import LAST_CORRECTION_KEY
from formatting import _task_line
from replies import _reply


def _tasks_text(chat_id: int) -> str:
    """Shared by /tasks and the natural-language 'what's on my list' intent
    -- see finance._balance_text's docstring for why. Open (not-done) tasks
    only, soonest due first -- db.get_open_tasks, not get_recent_tasks
    (that one feeds the AI's correction-target list in creation order
    instead)."""
    rows = db.get_open_tasks(chat_id)
    if not rows:
        return "Nothing on your to-do list right now."
    return "Open to-dos:\n" + "\n".join(_task_line(r) for r in rows)


def _due_at_from_fields(due_in_days, due_time) -> str | None:
    """Deterministic due-date computation in Python -- the model only ever
    extracts a day-count and an optional clock time, never a calendar date
    (see ai.py's log_task rules and days_ago's identical discipline for
    corrections)."""
    if not isinstance(due_in_days, int) or due_in_days < 0:
        return None
    today = date.fromisoformat(db.today_str())
    due_date = today + timedelta(days=due_in_days)
    return f"{due_date.isoformat()} {due_time}" if due_time else due_date.isoformat()


def _recent_tasks_for_ai(chat_id: int) -> list:
    """Open (not-done) tasks, soonest due first -- what a task correction
    (mark_done/delete/edit) may target. Deliberately db.get_open_tasks, not
    get_recent_tasks -- a done or long-since-created task shouldn't be a
    valid correction target for a fresh 'mark that done' message.
    recurrence_frequency is included so the AI can tell a correction like
    "stop reminding me about this" apart from a plain delete -- see ai.py's
    edit_task guidance."""
    rows = db.get_open_tasks(chat_id, limit=20)
    return [
        {"id": r["id"], "title": r["title"], "due_at": r["due_at"], "recurrence_frequency": r["recurrence_frequency"]}
        for r in rows
    ]


def _recurrence_from_fields(recurrence) -> str | None:
    """A task is recurring only when the model names one of
    db.RECURRENCE_FREQUENCIES explicitly -- any other value (missing,
    unrecognized, or the model guessing at a cadence that wasn't actually
    said) falls back to a plain one-off to-do rather than silently making
    something recur that wasn't meant to."""
    return recurrence if recurrence in db.RECURRENCE_FREQUENCIES else None


async def _add_task_from_fields(chat_id: int, data: dict) -> str:
    """Pure DB write, shared by every task-logging path -- data may come
    from either parse_message's natural-language schema
    (task_title/task_due_in_days/task_due_time/task_notes/task_recurrence)
    or extract_task's own field names (title/due_in_days/due_time/notes/
    recurrence) -- merged the same way vitals._log_vitals_fragment merges
    vitals_notes/notes, so /addtask and the natural-language path share
    this one code path. Returns the _task_line summary for the just-created
    row, with no "Added:"/"Added N:" wrapping -- see the callers below for
    how that's applied depending on whether it's the only item or one of
    several in the same message."""
    title = data.get("task_title") or data.get("title") or "to-do"
    due_in_days = data.get("task_due_in_days")
    if due_in_days is None:
        due_in_days = data.get("due_in_days")
    due_time = data.get("task_due_time") or data.get("due_time")
    notes = data.get("task_notes") or data.get("notes")
    recurrence_frequency = _recurrence_from_fields(data.get("task_recurrence") or data.get("recurrence"))
    due_at = _due_at_from_fields(due_in_days, due_time)
    task_id = db.add_task(chat_id, title, due_at, notes, recurrence_frequency)
    row = db.get_task(chat_id, task_id)
    return _task_line(row)


async def _log_task_fragment(chat_id: int, data: dict) -> str:
    """The pure write-and-describe half of logging a single to-do -- writes
    the row and returns its confirmation text WITHOUT sending it, so
    handlers.py's multi-domain dispatch can combine it with a meal and/or a
    vitals fragment logged from the SAME message into one reply instead of
    several separate Telegram messages (see ai.py's log_meal COMBINING RULE
    for why a message can report a to-do + a meal + vitals all at once).
    _log_task_and_reply below is the thin wrapper every other caller
    (currently just /addtask) still uses."""
    return f"Added: {await _add_task_from_fields(chat_id, data)}"


async def _log_task_and_reply(update: Update, chat_id: int, data: dict):
    """Thin wrapper around _log_task_fragment for callers (currently just
    /addtask) that want the confirmation sent directly rather than combined
    with other domains -- see _log_task_fragment's docstring."""
    await _reply(update, chat_id, await _log_task_fragment(chat_id, data))


async def _log_tasks_fragment(chat_id: int, tasks: list) -> str:
    """The pure write-and-describe half of the natural-language log_task
    intent, which can name many separate to-dos in one message (e.g. a
    numbered list of 13 things to do) -- this used to be the real bug:
    parse_message's log_task fields were singular (task_title/
    task_due_in_days/task_due_time/task_notes, one slot per message), so a
    message listing several to-dos collapsed to a single, often title-less
    "to-do" entry instead of logging each one. ai.py now always returns a
    list ("tasks"), mirroring log_expense's and log_meal's existing
    multi-item discipline. Returns the confirmation text WITHOUT sending
    it -- see _log_task_fragment's docstring for why -- _log_tasks_and_reply
    below is the thin wrapper every other caller still uses.

    A single task reuses _log_task_fragment's exact wording/behavior
    unchanged; more than one gets ONE combined fragment -- each to-do on
    its own line -- rather than a separate message per item."""
    if len(tasks) == 1:
        return await _log_task_fragment(chat_id, tasks[0])

    lines = [await _add_task_from_fields(chat_id, t) for t in tasks]
    body = "\n".join(lines)
    return f"Added {len(lines)} to-dos:\n{body}"


async def _log_tasks_and_reply(update: Update, chat_id: int, tasks: list):
    """Thin wrapper around _log_tasks_fragment for callers that want the
    confirmation sent directly rather than combined with other domains --
    see _log_tasks_fragment's docstring."""
    await _reply(update, chat_id, await _log_tasks_fragment(chat_id, tasks))


async def addtask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    description = " ".join(context.args)
    if not description:
        await update.message.reply_text("Usage: /addtask call the dentist tomorrow 5pm")
        return
    data = ai.extract_task(description)
    await _log_task_and_reply(update, chat_id, data)


async def tasks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    await update.message.reply_text(_tasks_text(chat_id))


async def done_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /done <id> (see /tasks for IDs)")
        return
    try:
        task_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("That doesn't look like a task ID. Try: /done 3")
        return
    updated = db.mark_task_done(chat_id, task_id)
    if updated is None:
        await update.message.reply_text("Couldn't find that to-do -- run /tasks to check the ID.")
        return
    context.chat_data[LAST_CORRECTION_KEY] = {"domain": "task", "action": "mark_done", "expense_id": task_id}
    await update.message.reply_text(f"Marked done: {_task_line(updated)}. Reply 'undo' if that's wrong.")
