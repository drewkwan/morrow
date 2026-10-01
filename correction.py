"""
Corrections and undo, for every domain. Two shapes coexist here on purpose:

- Expenses support a full field-level correction surface (date, currency,
  amount, description, category, delete) -- the original, most-used case.
- Meal/workout/vitals/task corrections are intentionally narrower. Each
  domain declares its own "actions" allowlist in _DOMAIN_OPS instead of
  sharing one global set, specifically so a domain-specific restriction
  (e.g. tasks support mark_done/edit_due/delete but not edit_date -- a
  forward-looking due date can't reuse days_ago's backward-only
  "today - N days" math, so it gets its own due_in_days/due_time fields
  instead, the same forward-looking shape log_task already uses) is
  enforced here in code, not just in the prompt. _handle_simple_domain_correction
  is the one implementation all four of those domains share.
- "balance" is a fifth, even narrower shape: a single per-chat running
  number, not a row with an id -- see _handle_balance_adjustment.

Every confirmation message here is built from real values just read back
from the database -- never from AI-generated text -- so the bot can never
claim to have made a change it didn't actually make. context.chat_data
holds a snapshot of the pre-change state for the single following message,
so a bare "undo" (_revert_last_correction) can reverse exactly that one
change by replaying the snapshot, never by re-guessing.
"""

from datetime import date, timedelta

from telegram import Update
from telegram.ext import ContextTypes

import config
import db
import fx
from formatting import (
    _deduction_line,
    _event_line,
    _income_line,
    _lift_line,
    _meal_line,
    _memory_line,
    _money,
    _reminder_line,
    _subscription_line,
    _task_line,
    _vitals_line,
    _workout_line,
)
from replies import _reply

LAST_CORRECTION_KEY = "last_correction"

# "Nothing about the due date was mentioned in this correction" -- distinct
# from Python None, which edit_task's due-date handling now uses as a real
# value meaning "clear it" (see db._UNSET's own docstring for the same
# "None is a real value, so the not-touched-at-all default can't also be
# None" reasoning). Only used locally within the edit_task branch below.
_DUE_UNCHANGED = object()

# A short reply matching one of these, sent as the very next message after a
# correction, reverts it directly -- deterministic and exact-match only (not
# a substring check), so an expense description that happens to contain the
# word "wrong" can't accidentally trigger it.
CORRECTION_UNDO_PHRASES = {
    "undo", "undo that", "undo it", "revert", "no", "nope",
    "wrong", "that's wrong", "thats wrong", "no that's wrong",
}

CORRECTION_ACTIONS = {
    "edit_date", "edit_currency", "edit_amount", "edit_description", "edit_category", "delete",
}

# Registry for the non-expense domains, which all share a narrower
# correction surface than expenses (see _handle_simple_domain_correction).
# Each domain declares its own "actions" allowlist -- meal/workout/vitals
# support edit_date + delete; task supports mark_done + edit_task + delete
# instead of edit_date. edit_task is deliberately one flexible action
# rather than one per field (edit_title/edit_due/edit_notes) -- a to-do has
# few enough fields that a single correction commonly touches more than one
# at once (e.g. "push #11 to tomorrow, I need Shardul's address" reschedules
# AND adds a note), and its due date is forward-looking (due_in_days/
# due_time, the same shape log_task already extracts for a new to-do) since
# it can't reuse edit_date's backward-only days_ago math (see ai.py's
# PARSE_SYSTEM_PROMPT correction rules).
TASK_DOMAIN_ACTIONS = {"mark_done", "edit_task", "delete"}
# meal/workout/lift/vitals each get the same flexible-edit bucket -- edit_meal/
# edit_workout/edit_lift/edit_vitals (flexible field corrections, mirroring edit_task) exist
# specifically so a photo- or text-logged entry that came out wrong (an item that wasn't
# really eaten, a workout's calories_burned read wrong off a screenshot, a lift's sets
# mis-typed, a mis-typed weight/sleep/knee number) can be fixed in place without forcing a
# delete-and-relog round trip (see ai.py's PHOTO_CLASSIFY_SYSTEM_PROMPT for the meal case,
# and a real observed bad interaction for workout/lift: "correct the calories out to 2862" /
# "update the v bar pulls to 10x8x1, 12x8x2" both had no supported action to land on,
# forcing the model into either a nonsense date-clarification question or a dead end --
# vitals had the exact same gap until edit_vitals closed it).
MEAL_DOMAIN_ACTIONS = {"edit_date", "edit_meal", "delete"}
WORKOUT_DOMAIN_ACTIONS = {"edit_date", "edit_workout", "delete"}
LIFT_DOMAIN_ACTIONS = {"edit_date", "edit_lift", "delete"}
VITALS_DOMAIN_ACTIONS = {"edit_date", "edit_vitals", "delete"}
# event supports "reschedule" and "delete" -- an event has no "done" state
# (see events.py's design), so those are the only two things a correction
# can do to one. "reschedule" is its own action, not "edit_date", because an
# event's date is FORWARD-looking (new_event_in_days, "N days from today" --
# the same shape add_event/rescheduleevent_cmd already use), never backward
# like edit_date's days_ago math everywhere else -- the two can't share a
# field without one of them meaning the wrong direction. This exists
# specifically because of a real, actively harmful bug: before "reschedule"
# existed, "correct both to 2026-09-23" / "push day should be 2026-09-23"
# had NO matching action for events (only "delete" was ever available), and
# the model chose "delete" -- silently destroying the event instead of
# moving it, rather than asking for clarification the way the prompt at the
# time actually told it to. A prompt instruction alone wasn't enough to
# prevent a destructive guess; giving the model a real, correct action to
# reach instead removes the guess entirely (see ai.py's correction_action
# rules for the matching CRITICAL warning).
EVENT_DOMAIN_ACTIONS = {"reschedule", "delete"}
# subscription has no "edit_date" -- its one date-shaped field is billing_day,
# which is part of the flexible edit_subscription action (name/amount/
# currency/billing_day/category can all change in one correction, same
# "give me the whole picture" shape as edit_task/edit_meal), not a backward-
# looking days_ago move like a logged entry's date.
SUBSCRIPTION_DOMAIN_ACTIONS = {"edit_subscription", "delete"}
# income and deduction entries DO have a real logged date (when the money
# actually arrived/left), so both get edit_date (backward-looking days_ago,
# same as expenses/meals/workouts/vitals) alongside their own flexible edit
# action for the non-date fields.
INCOME_DOMAIN_ACTIONS = {"edit_date", "edit_income", "delete"}
DEDUCTION_DOMAIN_ACTIONS = {"edit_date", "edit_deduction", "delete"}

_DOMAIN_OPS = {
    "meal": {"noun": "meal", "recent_cmd": "/recentmeals", "date_field": "meal_date",
              "actions": MEAL_DOMAIN_ACTIONS,
              "actions_desc": "moving the date, correcting what you actually ate, or deleting one",
              "get": db.get_meal, "edit_date": db.edit_meal_date, "delete": db.delete_meal,
              "restore": db.restore_deleted_meal, "line": _meal_line, "edit_meal": db.edit_meal},
    "workout": {"noun": "workout", "recent_cmd": "/recentworkouts", "date_field": "workout_date",
                 "actions": WORKOUT_DOMAIN_ACTIONS,
                 "actions_desc": "moving the date, correcting a field (activity/duration/distance/calories "
                                  "burned/notes), or deleting one",
                 "get": db.get_workout, "edit_date": db.edit_workout_date, "delete": db.delete_workout,
                 "restore": db.restore_deleted_workout, "line": _workout_line, "edit_workout": db.edit_workout},
    "lift": {"noun": "lift", "recent_cmd": "/recentlifts", "date_field": "lift_date",
              "actions": LIFT_DOMAIN_ACTIONS,
              "actions_desc": "moving the date, correcting the sets/location/effort/notes, or deleting one",
              "get": db.get_lift, "edit_date": db.edit_lift_date, "delete": db.delete_lift,
              "restore": db.restore_deleted_lift, "line": _lift_line, "edit_lift": db.edit_lift},
    "vitals": {"noun": "check-in", "recent_cmd": "/recentvitals", "date_field": "vitals_date",
                "actions": VITALS_DOMAIN_ACTIONS,
                "actions_desc": "moving the date, correcting a field (weight/sleep/knee pain/notes), or "
                                 "deleting one",
                "get": db.get_vitals, "edit_date": db.edit_vitals_date, "delete": db.delete_vitals,
                "restore": db.restore_deleted_vitals, "line": _vitals_line, "edit_vitals": db.edit_vitals},
    "task": {"noun": "to-do", "recent_cmd": "/tasks",
              "actions": TASK_DOMAIN_ACTIONS,
              "actions_desc": "only marking one done, editing its title/due date/notes, or deleting one",
              "get": db.get_task, "delete": db.delete_task, "restore": db.restore_deleted_task,
              "line": _task_line, "mark_done": db.mark_task_done, "unmark_done": db.unmark_task_done,
              "edit_task": db.edit_task},
    "event": {"noun": "event", "recent_cmd": "/events", "date_field": "event_date",
               "actions": EVENT_DOMAIN_ACTIONS,
               "actions_desc": "moving it to a new day, or deleting one (an event has no \"done\" state -- "
                                "\"X is done\" or \"that already happened\" both just mean remove it)",
               "get": db.get_event, "delete": db.delete_event, "restore": db.restore_deleted_event,
               "line": _event_line, "reschedule": db.edit_event_date},
    "subscription": {"noun": "subscription", "recent_cmd": "/subscriptions",
                       "actions": SUBSCRIPTION_DOMAIN_ACTIONS,
                       "actions_desc": "correcting the name/amount/currency/billing day/category, or deleting one",
                       "get": db.get_subscription, "delete": db.delete_subscription,
                       "restore": db.restore_deleted_subscription, "line": _subscription_line,
                       "edit_subscription": db.edit_subscription},
    "income": {"noun": "income entry", "recent_cmd": "/recentincome", "date_field": "income_date",
                "actions": INCOME_DOMAIN_ACTIONS,
                "actions_desc": "moving the date, correcting the source/description/amount/currency, or "
                                 "deleting one",
                "get": db.get_income, "edit_date": db.edit_income_date, "delete": db.delete_income,
                "restore": db.restore_deleted_income, "line": _income_line, "edit_income": db.edit_income},
    "deduction": {"noun": "deduction", "recent_cmd": "/recentdeductions", "date_field": "deduction_date",
                   "actions": DEDUCTION_DOMAIN_ACTIONS,
                   "actions_desc": "moving the date, correcting the label/amount/currency, or deleting one",
                   "get": db.get_deduction, "edit_date": db.edit_deduction_date, "delete": db.delete_deduction,
                   "restore": db.restore_deleted_deduction, "line": _deduction_line,
                   "edit_deduction": db.edit_deduction},
}


async def _handle_simple_domain_correction(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                            parsed: dict, domain: str, recent_ids: set):
    """Corrections for meal/workout/vitals/task targets are intentionally
    narrower than expenses right now -- each domain's own "actions"
    allowlist (see _DOMAIN_OPS) covers its flagship case (moving a date or
    deleting a duplicate for meal/workout/vitals; marking done or deleting
    for task) without needing a field-level /editmeal-style command yet."""
    ops = _DOMAIN_OPS[domain]
    chat_id = update.effective_chat.id
    target_id = parsed.get("target_expense_id")
    action = parsed.get("correction_action")
    noun, recent_cmd = ops["noun"], ops["recent_cmd"]

    # Two different failure modes, deliberately reported differently -- the
    # id genuinely wasn't found/matched (the AI's domain/id guess was wrong,
    # or the item has since been removed) vs. the id WAS found but this kind
    # of edit isn't supported for this domain (a real, deliberate scope cut,
    # not a mistake). Conflating them into one "not sure which X you mean"
    # message was itself a real observed bug: it reads as if the item was
    # never found at all, even when it was matched correctly and the only
    # actual problem was the unsupported action.
    if target_id not in recent_ids:
        await _reply(
            update, chat_id,
            f"I'm not sure which {noun} you mean -- run {recent_cmd} to see recent entries."
        )
        return
    if action not in ops["actions"]:
        await _reply(
            update, chat_id,
            f"Found that {noun}, but that kind of edit isn't supported yet -- {ops['actions_desc']} is."
        )
        return

    row = ops["get"](chat_id, target_id)
    if row is None:
        await _reply(update, chat_id, f"Couldn't find that {noun} anymore -- run {recent_cmd} to check.")
        return

    if action == "edit_date":
        days_ago = parsed.get("days_ago")
        if not isinstance(days_ago, int) or not (0 <= days_ago <= 14):
            await _reply(
                update, chat_id,
                f"Which day did you mean for that {noun}? (e.g. today, yesterday, or '3 days ago')"
            )
            return
        today = date.fromisoformat(db.today_str())
        new_date = today - timedelta(days=days_ago)
        old_date = row[ops["date_field"]]
        updated = ops["edit_date"](chat_id, target_id, new_date.isoformat())
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_date", "expense_id": target_id, "old_date": old_date,
        }
        await _reply(
            update, chat_id,
            f"Updated -- that {noun} is now dated {updated[ops['date_field']]} (was {old_date}). "
            "Reply 'undo' if that's wrong."
        )
        return

    if action == "reschedule":
        # Events only -- a forward-looking date move (new_event_in_days,
        # "N days from today"), NOT edit_date's backward-only days_ago math.
        # This is the fix for a real, actively harmful bug: "correct both to
        # 2026-09-23" / "push day should be 2026-09-23" used to have no
        # matching action for events at all (only "delete" existed), and the
        # model chose "delete" -- silently destroying the event instead of
        # moving it. See EVENT_DOMAIN_ACTIONS' own comment for the full story.
        new_event_in_days = parsed.get("new_event_in_days")
        if not isinstance(new_event_in_days, int) or new_event_in_days < 0:
            await _reply(
                update, chat_id,
                f"Which day should that {noun} move to? (e.g. today, tomorrow, or '3 days from now')"
            )
            return
        today = date.fromisoformat(db.today_str())
        new_date = today + timedelta(days=new_event_in_days)
        old_date = row[ops["date_field"]]
        updated = ops["reschedule"](chat_id, target_id, new_date.isoformat())
        # Snapshot shape matches what _revert_last_correction's own
        # domain=="event" branch already expects (event_id/old_date) -- that
        # branch has existed since /rescheduleevent shipped, this just gives
        # natural language a way to reach it too.
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "reschedule", "event_id": target_id, "old_date": old_date,
        }
        await _reply(
            update, chat_id,
            f"Rescheduled: {ops['line'](updated)} (was {old_date}). Reply 'undo' if that's wrong."
        )
        return

    if action == "edit_task":
        # One flexible action rather than one per field -- title, due date,
        # and notes can each change independently, in whatever combination
        # the message actually implies (e.g. "push #11 to tomorrow, I need
        # Shardul's address" reschedules AND adds a note in one correction).
        # A due date is forward-looking (due_in_days/due_time), unlike
        # edit_date's backward-only days_ago -- "in 3 days" can't be
        # expressed as "N days ago".
        #
        # The due-date side is a real THREE-state choice, not two: no
        # change / set a new date / clear it back to none. due_in_days only
        # ever expresses "set a new date" (see ai.py's field doc -- the code
        # below requires it to be a non-negative int), so there used to be
        # no way at all for a correction to represent "remove the due date,
        # keep the to-do open" -- a real reported bug where Morrow confirmed
        # a removal that structurally could never have happened, because
        # nothing here could ever produce a bare None for new_due_at (it
        # would just fall out of `edits` entirely, leaving db.edit_task's
        # _UNSET default -- "don't touch it" -- in place). remove_due_date
        # is the explicit third state; _DUE_UNCHANGED (distinct from Python
        # None, which is now a real "clear it" value) tracks "nothing about
        # the due date was actually requested".
        new_title = parsed.get("new_description")
        new_notes = parsed.get("new_task_notes")
        due_in_days = parsed.get("due_in_days")
        remove_due_date = bool(parsed.get("remove_due_date"))
        new_due_at = _DUE_UNCHANGED
        if isinstance(due_in_days, int) and due_in_days >= 0:
            due_time = parsed.get("due_time")
            today = date.fromisoformat(db.today_str())
            due_date = today + timedelta(days=due_in_days)
            new_due_at = f"{due_date.isoformat()} {due_time}" if due_time else due_date.isoformat()
        elif remove_due_date:
            new_due_at = None  # a real value: explicitly clear the due date

        if new_title is None and new_notes is None and new_due_at is _DUE_UNCHANGED:
            await _reply(
                update, chat_id,
                f"What should I change about that {noun} -- the title, the due date, or a note?"
            )
            return

        # Only pass fields actually changing -- db.edit_task's _UNSET default
        # leaves everything else untouched (see its docstring for why that's
        # not the same as passing None, which explicitly clears a field).
        edits = {}
        if new_title is not None:
            edits["new_title"] = new_title
        if new_due_at is not _DUE_UNCHANGED:
            edits["new_due_at"] = new_due_at  # may genuinely be None -- that's the clear case
        if new_notes is not None:
            edits["new_notes"] = new_notes

        old_title, old_due_at, old_notes = row["title"], row["due_at"], row["notes"]
        updated = ops["edit_task"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_task", "expense_id": target_id,
            "old_title": old_title, "old_due_at": old_due_at, "old_notes": old_notes,
        }
        bits = []
        if new_title is not None:
            bits.append(f"now titled \"{updated['title']}\"")
        if new_due_at is not _DUE_UNCHANGED:
            bits.append(f"due {updated['due_at']}" if updated["due_at"] else "due date removed")
        if new_notes is not None:
            bits.append("notes updated")
        await _reply(update, chat_id, f"Updated -- {', '.join(bits)}. Reply 'undo' if that's wrong.")
        return

    if action == "edit_meal":
        # Corrects what was actually in an already-logged meal (an item a
        # photo hallucinated in, a portion size off) without deleting and
        # relogging from scratch -- see ai.py's edit_meal prompt rules. The
        # model always sends the FULL corrected item list plus a fresh
        # calorie re-estimate for it, the same "give me the whole picture,
        # not a diff" shape as edit_task's title/due/notes.
        new_items = parsed.get("new_meal_items")
        new_type = parsed.get("new_meal_type")
        new_low = parsed.get("new_meal_calories_low")
        new_high = parsed.get("new_meal_calories_high")
        new_estimate = parsed.get("new_meal_calories_estimate")
        new_water_ml = parsed.get("new_meal_water_ml")

        if new_items is None and new_type is None and new_water_ml is None:
            await _reply(
                update, chat_id,
                f"What should I fix about that {noun} -- an item you didn't actually have, one you forgot to "
                "add, or the portion size?"
            )
            return

        edits = {}
        if new_items is not None:
            edits["new_items"] = new_items
            # Calories are meant to always travel with a changed item list
            # (see ai.py's rule) -- but never trust that blindly; only apply
            # them if all three actually came back set, otherwise leave the
            # old estimate in place rather than writing a broken partial one.
            if new_low is not None and new_high is not None and new_estimate is not None:
                edits["new_calories_low"] = new_low
                edits["new_calories_high"] = new_high
                edits["new_calories_estimate"] = new_estimate
        if new_type is not None:
            edits["new_meal_type"] = new_type
        if new_water_ml is not None:
            edits["new_water_ml"] = new_water_ml

        old_meal_type, old_items = row["meal_type"], row["items"]
        old_low, old_high, old_estimate = row["calories_low"], row["calories_high"], row["calories_estimate"]
        old_water_ml = row["water_ml"]
        updated = ops["edit_meal"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_meal", "expense_id": target_id,
            "old_meal_type": old_meal_type, "old_items": old_items, "old_calories_low": old_low,
            "old_calories_high": old_high, "old_calories_estimate": old_estimate, "old_water_ml": old_water_ml,
        }
        await _reply(update, chat_id, f"Updated -- {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "edit_workout":
        # Corrects a field on an already-logged workout -- most commonly
        # calories_burned read wrong off a fitness-app screenshot, or a
        # more complete/authoritative total (e.g. including BMR) the user
        # later saw -- without deleting and relogging from scratch. See
        # ai.py's edit_workout prompt rules.
        new_activity = parsed.get("new_workout_activity")
        new_duration = parsed.get("new_workout_duration_min")
        new_distance = parsed.get("new_workout_distance_km")
        new_calories = parsed.get("new_workout_calories_burned")
        new_notes = parsed.get("new_workout_notes")

        if all(v is None for v in (new_activity, new_duration, new_distance, new_calories, new_notes)):
            await _reply(
                update, chat_id,
                f"What should I fix about that {noun} -- the activity, duration, distance, calories burned, "
                "or notes?"
            )
            return

        edits = {}
        if new_activity is not None:
            edits["new_activity"] = new_activity
        if new_duration is not None:
            edits["new_duration_min"] = new_duration
        if new_distance is not None:
            edits["new_distance_km"] = new_distance
        if new_calories is not None:
            edits["new_calories_burned"] = new_calories
        if new_notes is not None:
            edits["new_notes"] = new_notes

        old_activity, old_duration = row["activity"], row["duration_min"]
        old_distance, old_calories, old_notes = row["distance_km"], row["calories_burned"], row["notes"]
        updated = ops["edit_workout"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_workout", "expense_id": target_id,
            "old_activity": old_activity, "old_duration_min": old_duration, "old_distance_km": old_distance,
            "old_calories_burned": old_calories, "old_notes": old_notes,
        }
        await _reply(update, chat_id, f"Updated -- {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "edit_lift":
        # Corrects a field on an already-logged lift -- most commonly the
        # sets themselves (a mis-typed rep count, a set left out or added
        # after the fact) -- without deleting and relogging the exercise
        # from scratch. See ai.py's edit_lift prompt rules.
        new_exercise = parsed.get("new_lift_exercise")
        new_location = parsed.get("new_lift_location")
        new_sets = parsed.get("new_lift_sets")
        new_effort = parsed.get("new_lift_effort")
        new_context_notes = parsed.get("new_lift_context_notes")

        if all(v is None for v in (new_exercise, new_location, new_sets, new_effort, new_context_notes)):
            await _reply(
                update, chat_id,
                f"What should I fix about that {noun} -- the sets, location, effort, or notes?"
            )
            return

        edits = {}
        if new_exercise is not None:
            edits["new_exercise"] = new_exercise
        if new_location is not None:
            edits["new_location"] = new_location
        if new_sets is not None:
            edits["new_sets"] = new_sets
        if new_effort is not None:
            edits["new_effort"] = new_effort
        if new_context_notes is not None:
            edits["new_context_notes"] = new_context_notes

        old_exercise, old_location, old_sets = row["exercise"], row["location"], row["sets"]
        old_effort, old_context_notes = row["effort"], row["context_notes"]
        updated = ops["edit_lift"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_lift", "expense_id": target_id,
            "old_exercise": old_exercise, "old_location": old_location, "old_sets": old_sets,
            "old_effort": old_effort, "old_context_notes": old_context_notes,
        }
        await _reply(update, chat_id, f"Updated -- {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "edit_vitals":
        # Corrects a field on an already-logged check-in -- most commonly a
        # mis-typed weight/sleep/knee number -- without deleting and
        # relogging from scratch. See ai.py's edit_vitals prompt rules.
        new_weight = parsed.get("new_vitals_weight_kg")
        new_sleep = parsed.get("new_vitals_sleep_hours")
        new_knee = parsed.get("new_vitals_knee_pain")
        new_notes = parsed.get("new_vitals_notes")

        if all(v is None for v in (new_weight, new_sleep, new_knee, new_notes)):
            await _reply(
                update, chat_id,
                f"What should I fix about that {noun} -- the weight, sleep, knee pain, or notes?"
            )
            return

        edits = {}
        if new_weight is not None:
            edits["new_weight_kg"] = new_weight
        if new_sleep is not None:
            edits["new_sleep_hours"] = new_sleep
        if new_knee is not None:
            edits["new_knee_pain"] = new_knee
        if new_notes is not None:
            edits["new_notes"] = new_notes

        old_weight, old_sleep = row["weight_kg"], row["sleep_hours"]
        old_knee, old_notes = row["knee_pain"], row["notes"]
        updated = ops["edit_vitals"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_vitals", "expense_id": target_id,
            "old_weight_kg": old_weight, "old_sleep_hours": old_sleep,
            "old_knee_pain": old_knee, "old_notes": old_notes,
        }
        await _reply(update, chat_id, f"Updated -- {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "edit_subscription":
        # Corrects a subscription's own config -- a price change, a billing-
        # day move, a rename -- in place, without a delete-and-re-add round
        # trip. Deliberately does NOT touch any already-posted expense row
        # for a past cycle (see db.edit_subscription's docstring); only
        # future auto-posts pick up the new config. See ai.py's
        # edit_subscription prompt rules.
        new_name = parsed.get("new_subscription_name")
        new_amount = parsed.get("new_subscription_amount")
        new_currency = parsed.get("new_subscription_currency")
        new_billing_day = parsed.get("new_subscription_billing_day")
        new_category = parsed.get("new_subscription_category")

        if all(v is None for v in (new_name, new_amount, new_currency, new_billing_day, new_category)):
            await _reply(
                update, chat_id,
                f"What should I fix about that {noun} -- the name, amount, currency, billing day, or category?"
            )
            return

        edits = {}
        if new_name is not None:
            edits["new_name"] = new_name
        if new_amount is not None:
            edits["new_amount"] = new_amount
        if new_currency is not None:
            edits["new_currency"] = new_currency
        if isinstance(new_billing_day, int) and 1 <= new_billing_day <= 31:
            edits["new_billing_day"] = new_billing_day
        if new_category is not None:
            edits["new_category"] = new_category

        old_name, old_amount, old_currency = row["name"], row["amount"], row["currency"]
        old_billing_day, old_category = row["billing_day"], row["category"]
        updated = ops["edit_subscription"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_subscription", "expense_id": target_id,
            "old_name": old_name, "old_amount": old_amount, "old_currency": old_currency,
            "old_billing_day": old_billing_day, "old_category": old_category,
        }
        await _reply(update, chat_id, f"Updated -- {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "edit_income":
        # Corrects a logged income entry's source/description/net amount/
        # currency in place. See ai.py's edit_income prompt rules -- amount
        # here is always NET, never gross (same discipline as log_income).
        new_source = parsed.get("new_income_source")
        new_description = parsed.get("new_income_description")
        new_amount = parsed.get("new_income_amount")
        new_currency = parsed.get("new_income_currency")

        if all(v is None for v in (new_source, new_description, new_amount, new_currency)):
            await _reply(
                update, chat_id,
                f"What should I fix about that {noun} -- the source, description, amount, or currency?"
            )
            return

        edits = {}
        if new_source is not None:
            edits["new_source"] = new_source
        if new_description is not None:
            edits["new_description"] = new_description
        if new_amount is not None:
            edits["new_amount"] = new_amount
        if new_currency is not None:
            edits["new_currency"] = new_currency

        old_source, old_description = row["source"], row["description"]
        old_amount, old_currency = row["net_amount"], row["currency"]
        updated = ops["edit_income"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_income", "expense_id": target_id,
            "old_source": old_source, "old_description": old_description,
            "old_amount": old_amount, "old_currency": old_currency,
        }
        await _reply(update, chat_id, f"Updated -- {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "edit_deduction":
        # Corrects a logged deduction's label/amount/currency in place. See
        # ai.py's edit_deduction prompt rules.
        new_label = parsed.get("new_deduction_label")
        new_amount = parsed.get("new_deduction_amount")
        new_currency = parsed.get("new_deduction_currency")

        if all(v is None for v in (new_label, new_amount, new_currency)):
            await _reply(
                update, chat_id, f"What should I fix about that {noun} -- the label, amount, or currency?"
            )
            return

        edits = {}
        if new_label is not None:
            edits["new_label"] = new_label
        if new_amount is not None:
            edits["new_amount"] = new_amount
        if new_currency is not None:
            edits["new_currency"] = new_currency

        old_label, old_amount, old_currency = row["label"], row["amount"], row["currency"]
        updated = ops["edit_deduction"](chat_id, target_id, **edits)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "edit_deduction", "expense_id": target_id,
            "old_label": old_label, "old_amount": old_amount, "old_currency": old_currency,
        }
        await _reply(update, chat_id, f"Updated -- {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "mark_done":
        updated = ops["mark_done"](chat_id, target_id)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "domain": domain, "action": "mark_done", "expense_id": target_id,
        }
        await _reply(update, chat_id, f"Marked done: {ops['line'](updated)}. Reply 'undo' if that's wrong.")
        return

    if action == "delete":
        deleted = ops["delete"](chat_id, target_id)
        context.chat_data[LAST_CORRECTION_KEY] = {"domain": domain, "action": "delete", "row": deleted}
        await _reply(update, chat_id, f"Deleted: {ops['line'](deleted)}. Reply 'undo' if that's wrong.")
        return


async def _handle_balance_adjustment(update: Update, context: ContextTypes.DEFAULT_TYPE, parsed: dict):
    """"Adjust my rolled-over balance/deficit by X" -- the one natural-
    language path onto db.adjust_balance's manual escape hatch (see its
    docstring). Unlike every other correction here, there's no recent-item
    list to match against -- balance is a single per-chat value, not a row
    -- so this only needs the delta itself, taken from the user's own words,
    never estimated or invented (same discipline as edit_amount)."""
    chat_id = update.effective_chat.id
    delta = parsed.get("new_amount")
    if not isinstance(delta, (int, float)) or delta == 0:
        await _reply(
            update, chat_id,
            "By how much should I adjust your rolled-over balance? (e.g. \"-1135.89\" to add that deficit, "
            "or a positive number to add a credit)"
        )
        return
    result = db.adjust_balance(chat_id, float(delta))
    context.chat_data[LAST_CORRECTION_KEY] = {
        "domain": "balance", "action": "adjust_balance", "old_balance": result["old_balance"],
    }
    sign = "+" if delta >= 0 else ""
    await _reply(
        update, chat_id,
        f"Balance adjusted by {sign}{_money(delta)}: {_money(result['old_balance'])} -> "
        f"{_money(result['new_balance'])}. Reply 'undo' if that's wrong."
    )


async def _handle_income_config_adjustment(update: Update, context: ContextTypes.DEFAULT_TYPE, parsed: dict):
    """"I got a raise" / "I changed jobs, new salary is X, cpf 20%" -- the
    natural-language path onto db.edit_income_config's partial-update escape
    hatch. Unlike every other correction here, there's no recent-item list to
    match against -- income_config is a single per-chat row, not a row with
    an id -- same shape as _handle_balance_adjustment. Deliberately a
    SEPARATE intent from log_income: a raise changes what gets auto-posted
    EVERY future payday, it isn't itself a one-off payment that landed today
    (see ai.py's log_income vs edit_income_config classification rule for the
    real, previously-ambiguous distinction)."""
    chat_id = update.effective_chat.id
    existing = db.get_income_config(chat_id)
    if existing is None:
        await _reply(
            update, chat_id,
            "You don't have a salary set up yet -- run /setincome <gross_amount> [currency] <pay_day> "
            "[cpf_rate%] [stock_rate%] first."
        )
        return

    new_gross = parsed.get("new_income_config_gross_amount")
    new_currency = parsed.get("new_income_config_currency")
    new_pay_day = parsed.get("new_income_config_pay_day")
    new_cpf_rate = parsed.get("new_income_config_cpf_rate")
    new_stock_rate = parsed.get("new_income_config_stock_rate")

    if all(v is None for v in (new_gross, new_currency, new_pay_day, new_cpf_rate, new_stock_rate)):
        await _reply(
            update, chat_id,
            "What's the new gross salary? (You can also mention a new pay day, CPF rate, or stock rate in "
            "the same message.)"
        )
        return

    edits = {}
    if new_gross is not None:
        edits["new_gross_amount"] = new_gross
    if new_currency is not None:
        edits["new_currency"] = new_currency
    if isinstance(new_pay_day, int) and 1 <= new_pay_day <= 31:
        edits["new_pay_day"] = new_pay_day
    if new_cpf_rate is not None:
        edits["new_cpf_rate"] = new_cpf_rate
    if new_stock_rate is not None:
        edits["new_stock_rate"] = new_stock_rate

    old_gross, old_currency = existing["gross_amount"], existing["currency"]
    old_pay_day, old_cpf_rate, old_stock_rate = existing["pay_day"], existing["cpf_rate"], existing["stock_rate"]
    updated = db.edit_income_config(chat_id, **edits)
    context.chat_data[LAST_CORRECTION_KEY] = {
        "domain": "income_config", "action": "edit_income_config", "old_gross_amount": old_gross,
        "old_currency": old_currency, "old_pay_day": old_pay_day, "old_cpf_rate": old_cpf_rate,
        "old_stock_rate": old_stock_rate,
    }
    await _reply(
        update, chat_id,
        f"Updated -- gross salary is now {_money(updated['gross_amount'], updated['currency'])}/mo, "
        f"paid on day {updated['pay_day']}, CPF {updated['cpf_rate']:.0%}, stock {updated['stock_rate']:.0%} "
        f"(was {_money(old_gross, old_currency)}/mo, day {old_pay_day}, CPF {old_cpf_rate:.0%}, "
        f"stock {old_stock_rate:.0%}). Reply 'undo' if that's wrong."
    )


async def _handle_correction(update: Update, context: ContextTypes.DEFAULT_TYPE, parsed: dict,
                              recent_ids: set, recent_meal_ids: set = frozenset(),
                              recent_workout_ids: set = frozenset(), recent_vitals_ids: set = frozenset(),
                              recent_task_ids: set = frozenset(), recent_event_ids: set = frozenset(),
                              recent_lift_ids: set = frozenset(), recent_subscription_ids: set = frozenset(),
                              recent_income_ids: set = frozenset(), recent_deduction_ids: set = frozenset()):
    """Applies a correction the AI identified against one of the chat's
    recent expenses/meals/workouts/lifts/vitals/tasks/events. Every
    confirmation message here is built from real values just read back from
    the database -- never from AI-generated text -- so the bot can never
    claim to have made a change it didn't actually make (the exact failure
    mode that prompted this feature)."""
    chat_id = update.effective_chat.id
    target_id = parsed.get("target_expense_id")
    action = parsed.get("correction_action")
    domain = parsed.get("target_domain") or "expense"

    if domain == "meal":
        await _handle_simple_domain_correction(update, context, parsed, "meal", recent_meal_ids)
        return
    if domain == "workout":
        await _handle_simple_domain_correction(update, context, parsed, "workout", recent_workout_ids)
        return
    if domain == "lift":
        await _handle_simple_domain_correction(update, context, parsed, "lift", recent_lift_ids)
        return
    if domain == "vitals":
        await _handle_simple_domain_correction(update, context, parsed, "vitals", recent_vitals_ids)
        return
    if domain == "task":
        await _handle_simple_domain_correction(update, context, parsed, "task", recent_task_ids)
        return
    if domain == "event":
        await _handle_simple_domain_correction(update, context, parsed, "event", recent_event_ids)
        return

    if domain == "balance":
        await _handle_balance_adjustment(update, context, parsed)
        return

    if domain == "subscription":
        await _handle_simple_domain_correction(update, context, parsed, "subscription", recent_subscription_ids)
        return
    if domain == "income":
        await _handle_simple_domain_correction(update, context, parsed, "income", recent_income_ids)
        return
    if domain == "deduction":
        await _handle_simple_domain_correction(update, context, parsed, "deduction", recent_deduction_ids)
        return
    if domain == "income_config":
        await _handle_income_config_adjustment(update, context, parsed)
        return

    if target_id not in recent_ids or action not in CORRECTION_ACTIONS:
        await _reply(
            update, chat_id,
            "I'm not sure which expense you mean -- run /recent to see IDs, then use /edit <id> or /delete <id>."
        )
        return

    row = db.get_expense(chat_id, target_id)
    if row is None:
        await _reply(update, chat_id, "Couldn't find that expense anymore -- run /recent to check.")
        return

    if action == "edit_date":
        days_ago = parsed.get("days_ago")
        if not isinstance(days_ago, int) or not (0 <= days_ago <= 14):
            await _reply(
                update, chat_id,
                f"Which day did you mean for {_money(row['amount'], row['currency'])} -- "
                f"{row['description']}? (e.g. today, yesterday, or '3 days ago')"
            )
            return
        today = date.fromisoformat(db.today_str())
        new_date = today - timedelta(days=days_ago)
        old_date = row["expense_date"]
        updated = db.edit_expense_date(chat_id, target_id, new_date.isoformat())
        context.chat_data[LAST_CORRECTION_KEY] = {
            "action": "edit_date", "expense_id": target_id, "old_date": old_date,
        }
        await _reply(
            update, chat_id,
            f"Updated -- {_money(updated['amount'], updated['currency'])} \"{updated['description']}\" is "
            f"now dated {updated['expense_date']} (was {old_date}). Reply 'undo' if that's wrong."
        )
        return

    if action == "edit_currency":
        new_currency = fx.normalize_currency(parsed.get("new_currency"))
        if not new_currency:
            await _reply(
                update, chat_id,
                f"What currency should {_money(row['amount'], row['currency'])} -- "
                f"{row['description']} actually be?"
            )
            return
        old_currency = row["currency"]
        old_display = _money(row["amount"], old_currency)
        updated = db.edit_expense(chat_id, target_id, new_currency=new_currency)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "action": "edit_currency", "expense_id": target_id, "old_currency": old_currency,
        }
        await _reply(
            update, chat_id,
            f"Updated -- \"{updated['description']}\" is now {_money(updated['amount'], updated['currency'])} "
            f"(was {old_display}). Reply 'undo' if that's wrong."
        )
        return

    if action == "edit_amount":
        new_amount = parsed.get("new_amount")
        if not isinstance(new_amount, (int, float)) or new_amount <= 0:
            await _reply(update, chat_id, f"What should the amount for \"{row['description']}\" actually be?")
            return
        old_display = _money(row["amount"], row["currency"])
        updated = db.edit_expense(chat_id, target_id, new_amount=float(new_amount))
        context.chat_data[LAST_CORRECTION_KEY] = {
            "action": "edit_amount", "expense_id": target_id, "old_amount": row["amount"],
        }
        await _reply(
            update, chat_id,
            f"Updated -- \"{updated['description']}\" is now {_money(updated['amount'], updated['currency'])} "
            f"(was {old_display}). Reply 'undo' if that's wrong."
        )
        return

    if action == "edit_description":
        new_description = parsed.get("new_description")
        if not new_description:
            await _reply(update, chat_id, "What should the description say instead?")
            return
        old_description = row["description"]
        updated = db.edit_expense(chat_id, target_id, new_description=new_description)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "action": "edit_description", "expense_id": target_id, "old_description": old_description,
        }
        await _reply(
            update, chat_id,
            f"Updated -- description is now \"{updated['description']}\" (was \"{old_description}\"). "
            "Reply 'undo' if that's wrong."
        )
        return

    if action == "edit_category":
        new_category = parsed.get("new_category")
        if new_category not in config.CATEGORIES:
            await _reply(
                update, chat_id,
                f"What category should \"{row['description']}\" actually be? "
                f"({', '.join(config.CATEGORIES)})"
            )
            return
        old_category = row["category"]
        updated = db.edit_expense(chat_id, target_id, new_category=new_category)
        context.chat_data[LAST_CORRECTION_KEY] = {
            "action": "edit_category", "expense_id": target_id, "old_category": old_category,
        }
        await _reply(
            update, chat_id,
            f"Updated -- \"{updated['description']}\" is now [{updated['category']}] (was [{old_category}]). "
            "Reply 'undo' if that's wrong."
        )
        return

    if action == "delete":
        deleted = db.delete_expense(chat_id, target_id)
        context.chat_data[LAST_CORRECTION_KEY] = {"action": "delete", "row": deleted}
        await _reply(
            update, chat_id,
            f"Deleted: {_money(deleted['amount'], deleted['currency'])} -- {deleted['description']} "
            f"[{deleted['category']}] ({deleted['expense_date']}). Reply 'undo' if that's wrong."
        )
        return


async def _revert_last_correction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Returns True if there was something to revert (and replies about it),
    False if there was no pending correction snapshot at all."""
    snap = context.chat_data.pop(LAST_CORRECTION_KEY, None)
    if not snap:
        return False
    chat_id = update.effective_chat.id
    action = snap["action"]
    domain = snap.get("domain", "expense")

    if domain == "memory":
        # Memory undo is simpler than the other domains -- always a
        # delete-then-restore by label, never a date edit -- so it gets its
        # own branch instead of forcing it into the expense_id-shaped
        # _DOMAIN_OPS registry.
        if action == "delete":
            restored = db.restore_deleted_memory(chat_id, snap["row"])
            await _reply(update, chat_id, f"Restored: {_memory_line(restored)}")
        return True

    if domain == "balance":
        db.set_balance(chat_id, snap["old_balance"])
        await _reply(update, chat_id, f"Reverted -- balance is back to {_money(snap['old_balance'])}.")
        return True

    if domain == "income_config":
        # Mirrors "balance" above -- a single per-chat row, not an id'd one,
        # so there's no _DOMAIN_OPS registry entry to reuse. Always passes
        # every field back explicitly (same discipline as edit_task/edit_meal
        # revert below): these are the real prior values, not "leave
        # untouched" -- db.edit_income_config's _UNSET default is only for
        # the forward edit, never for undo.
        row = db.edit_income_config(
            chat_id, new_gross_amount=snap["old_gross_amount"], new_currency=snap["old_currency"],
            new_pay_day=snap["old_pay_day"], new_cpf_rate=snap["old_cpf_rate"],
            new_stock_rate=snap["old_stock_rate"],
        )
        await _reply(
            update, chat_id,
            f"Reverted -- gross salary is back to {_money(row['gross_amount'], row['currency'])}/mo."
        )
        return True

    if domain == "reminder":
        # Slash-command-only right now (/donereminder, /removereminder --
        # see reminders.py's module docstring for why), so this is its own
        # small branch rather than living in _DOMAIN_OPS, the same reason
        # "memory" above gets one instead of forcing itself into that
        # id-set-matching, AI-correction-driven registry.
        if action == "delete":
            restored = db.restore_deleted_reminder(chat_id, snap["row"])
            await _reply(update, chat_id, f"Restored: {_reminder_line(restored)}")
        elif action == "mark_done":
            row = db.unmark_reminder_done_today(chat_id, snap["reminder_id"])
            await _reply(update, chat_id, f"Reverted -- back to not done today: {_reminder_line(row)}")
        return True

    if domain == "event":
        # Also slash-command-only right now (/rescheduleevent, /removeevent
        # -- see events.py's module docstring for why), so its own small
        # branch, the same reasoning as "reminder"/"memory" above.
        if action == "delete":
            restored = db.restore_deleted_event(chat_id, snap["row"])
            await _reply(update, chat_id, f"Restored: {_event_line(restored)}")
        elif action == "reschedule":
            row = db.edit_event_date(chat_id, snap["event_id"], snap["old_date"])
            await _reply(update, chat_id, f"Reverted -- back to {row['event_date']}: {_event_line(row)}")
        return True

    if domain in _DOMAIN_OPS:
        # meal/workout/vitals/task all share the same narrow revert shape --
        # see _handle_simple_domain_correction for why their correction
        # surface is smaller than expenses'.
        ops = _DOMAIN_OPS[domain]
        if action == "delete":
            restored = ops["restore"](chat_id, snap["row"])
            await _reply(update, chat_id, f"Restored: {ops['line'](restored)}")
        elif action == "edit_date":
            row = ops["edit_date"](chat_id, snap["expense_id"], snap["old_date"])
            await _reply(update, chat_id, f"Reverted -- {ops['noun']} is back to {row[ops['date_field']]}.")
        elif action == "mark_done":
            row = ops["unmark_done"](chat_id, snap["expense_id"])
            await _reply(update, chat_id, f"Reverted -- back to open: {ops['line'](row)}")
        elif action == "edit_task":
            # Always pass all three explicitly, even where a value is None
            # (e.g. it had no due date before) -- db.edit_task's _UNSET
            # default is for "don't touch", not for these real prior values,
            # so a real None here correctly clears the field back to no-date/
            # no-notes rather than leaving whatever the edit just set.
            row = ops["edit_task"](chat_id, snap["expense_id"], new_title=snap["old_title"],
                                    new_due_at=snap["old_due_at"], new_notes=snap["old_notes"])
            await _reply(update, chat_id, f"Reverted -- back to \"{row['title']}\".")
        elif action == "edit_meal":
            # Same "pass every field back explicitly" discipline as edit_task
            # above -- these are the real prior values, not "leave untouched".
            row = ops["edit_meal"](
                chat_id, snap["expense_id"], new_meal_type=snap["old_meal_type"], new_items=snap["old_items"],
                new_calories_low=snap["old_calories_low"], new_calories_high=snap["old_calories_high"],
                new_calories_estimate=snap["old_calories_estimate"], new_water_ml=snap["old_water_ml"],
            )
            await _reply(update, chat_id, f"Reverted -- back to {ops['line'](row)}.")
        elif action == "edit_workout":
            row = ops["edit_workout"](
                chat_id, snap["expense_id"], new_activity=snap["old_activity"],
                new_duration_min=snap["old_duration_min"], new_distance_km=snap["old_distance_km"],
                new_calories_burned=snap["old_calories_burned"], new_notes=snap["old_notes"],
            )
            await _reply(update, chat_id, f"Reverted -- back to {ops['line'](row)}.")
        elif action == "edit_lift":
            row = ops["edit_lift"](
                chat_id, snap["expense_id"], new_exercise=snap["old_exercise"],
                new_location=snap["old_location"], new_sets=snap["old_sets"],
                new_effort=snap["old_effort"], new_context_notes=snap["old_context_notes"],
            )
            await _reply(update, chat_id, f"Reverted -- back to {ops['line'](row)}.")
        elif action == "edit_vitals":
            row = ops["edit_vitals"](
                chat_id, snap["expense_id"], new_weight_kg=snap["old_weight_kg"],
                new_sleep_hours=snap["old_sleep_hours"], new_knee_pain=snap["old_knee_pain"],
                new_notes=snap["old_notes"],
            )
            await _reply(update, chat_id, f"Reverted -- back to {ops['line'](row)}.")
        elif action == "edit_subscription":
            row = ops["edit_subscription"](
                chat_id, snap["expense_id"], new_name=snap["old_name"], new_amount=snap["old_amount"],
                new_currency=snap["old_currency"], new_billing_day=snap["old_billing_day"],
                new_category=snap["old_category"],
            )
            await _reply(update, chat_id, f"Reverted -- back to {ops['line'](row)}.")
        elif action == "edit_income":
            row = ops["edit_income"](
                chat_id, snap["expense_id"], new_source=snap["old_source"],
                new_description=snap["old_description"], new_amount=snap["old_amount"],
                new_currency=snap["old_currency"],
            )
            await _reply(update, chat_id, f"Reverted -- back to {ops['line'](row)}.")
        elif action == "edit_deduction":
            row = ops["edit_deduction"](
                chat_id, snap["expense_id"], new_label=snap["old_label"], new_amount=snap["old_amount"],
                new_currency=snap["old_currency"],
            )
            await _reply(update, chat_id, f"Reverted -- back to {ops['line'](row)}.")
        return True

    if action == "delete":
        restored = db.restore_deleted_expense(chat_id, snap["row"])
        await _reply(
            update, chat_id,
            f"Restored: {_money(restored['amount'], restored['currency'])} -- {restored['description']} "
            f"[{restored['category']}] ({restored['expense_date']})"
        )
    elif action == "edit_date":
        row = db.edit_expense_date(chat_id, snap["expense_id"], snap["old_date"])
        await _reply(update, chat_id, f"Reverted -- date is back to {row['expense_date']}.")
    elif action == "edit_currency":
        row = db.edit_expense(chat_id, snap["expense_id"], new_currency=snap["old_currency"])
        await _reply(update, chat_id, f"Reverted -- currency is back to {_money(row['amount'], row['currency'])}.")
    elif action == "edit_amount":
        row = db.edit_expense(chat_id, snap["expense_id"], new_amount=snap["old_amount"])
        await _reply(update, chat_id, f"Reverted -- amount is back to {_money(row['amount'], row['currency'])}.")
    elif action == "edit_description":
        row = db.edit_expense(chat_id, snap["expense_id"], new_description=snap["old_description"])
        await _reply(update, chat_id, f"Reverted -- description is back to \"{row['description']}\".")
    elif action == "edit_category":
        row = db.edit_expense(chat_id, snap["expense_id"], new_category=snap["old_category"])
        await _reply(update, chat_id, f"Reverted -- category is back to [{row['category']}].")
    return True
