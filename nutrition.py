"""
Meal logging: /logmeal, photo logging, /recentmeals. Meals are logged as
items, not meal slots -- see the module docstring in ai.py for why.
"""

import asyncio
from datetime import date, timedelta

from telegram import Update
from telegram.ext import ContextTypes

import ai
import db
from access import _reject_if_not_allowed
from fitness import _log_workout_and_reply
from formatting import _calorie_range, _daily_meal_totals_text, _meal_line
from replies import PENDING_KEY, _reply


def _target_date_from_days_ago(days_ago) -> str | None:
    """Converts ai.extract_from_photo's logged_days_ago (a plain day-count,
    never a date -- see ai.py's PHOTO_CLASSIFY_SYSTEM_PROMPT) into a real
    ISO date deterministically, the same discipline correction.py and
    tasks.py already use for their own day-count/due-in-days fields.
    Returns None (log onto today, the existing default) when the caption
    didn't imply a specific past day, or when it implied today itself.
    Defensively re-clamps to the same 0-14 range the prompt asks the model
    to respect, in case it doesn't."""
    if not days_ago:
        return None
    days_ago = max(0, min(14, int(days_ago)))
    if days_ago == 0:
        return None
    # db.today_str() -- NOT date.today() -- for the same reason every other
    # date computation in this app goes through it: date.today() is the
    # server's/OS date (UTC on Railway), which lags Asia/Singapore's actual
    # calendar date by up to 8 hours after local midnight. Backdating a
    # photo logged in that window from the wrong "today" would land it a
    # day off from what the caption actually meant.
    return (date.fromisoformat(db.today_str()) - timedelta(days=days_ago)).isoformat()


async def _log_meal_fragment(chat_id: int, data: dict, meal_date: str | None = None) -> str:
    """The pure write-and-describe half of logging a single meal -- writes
    the row and returns its confirmation text WITHOUT sending it, so
    handlers.py's multi-domain dispatch can combine it with a vitals and/or
    a task fragment logged from the SAME message into one reply instead of
    several separate Telegram messages (see ai.py's log_meal COMBINING RULE
    for why a message can report a meal + vitals + a to-do all at once).
    _log_meal_and_reply below is the thin wrapper every other caller
    (/logmeal, photo logging) still uses.

    meal_date lets a caller that already computed a real backdated date
    (see _target_date_from_days_ago) log directly onto the right day
    instead of defaulting to today and needing a follow-up correction."""
    meal_id = db.add_meal(
        chat_id, data.get("meal_type"), data.get("items") or data.get("meal_items"),
        data.get("calories_low"), data.get("calories_high"), data.get("calories_estimate"),
        water_ml=data.get("water_ml"), meal_date=meal_date,
    )
    row = db.get_meal(chat_id, meal_id)
    items = ", ".join(row["items"]) or "meal"
    water_line = f"\nWater: +{row['water_ml']:.0f}ml" if row.get("water_ml") else ""
    return f"Logged: {items} -- {_calorie_range(row)}{water_line}\n\n{_daily_meal_totals_text(chat_id, meal_date)}"


async def _log_meal_and_reply(update: Update, chat_id: int, data: dict, meal_date: str | None = None):
    """Thin wrapper around _log_meal_fragment for callers (/logmeal, photo
    logging) that want the confirmation sent directly rather than combined
    with other domains -- see _log_meal_fragment's docstring."""
    await _reply(update, chat_id, await _log_meal_fragment(chat_id, data, meal_date=meal_date))


async def _log_meals_fragment(chat_id: int, meals: list) -> str:
    """The pure write-and-describe half of the natural-language log_meal
    intent, which can name more than one meal in a single message (e.g.
    "for breakfast: toast and coffee. For lunch: noodles and a latte") --
    this used to be the real bug: parse_message only had room for ONE
    meal's fields, so a second meal mentioned in the same message was
    silently dropped rather than logged. ai.py now always returns a list
    ("meals"), mirroring log_expense's existing multi-item discipline.
    Returns the confirmation text WITHOUT sending it, so handlers.py's
    multi-domain dispatch can combine it with a vitals and/or a task
    fragment logged from the SAME message into one reply (see ai.py's
    log_meal COMBINING RULE) -- _log_meals_and_reply below is the thin
    wrapper every other caller still uses.

    Each item carries its own "logged_days_ago" (see ai.py's logged_days_ago
    rule) -- a real bug this fixes: "last night I also had a cup of tea"
    (sent well after midnight) used to land on today regardless, since
    nothing here ever asked the model whether it meant an earlier day; the
    user had to notice and correct it by hand. Converted to a real date the
    same deterministic way handle_photo's logged_days_ago already is (see
    _target_date_from_days_ago) -- per item, since one message can
    legitimately describe more than one day's meals at once.

    A single meal reuses _log_meal_fragment's exact wording/behavior
    unchanged (same reply shape existing callers/tests expect); more than
    one meal gets ONE combined fragment -- each meal on its own line --
    plus a running total per distinct day actually touched (almost always
    just one), rather than a separate message per meal."""
    if len(meals) == 1:
        m = meals[0]
        return await _log_meal_fragment(chat_id, m, meal_date=_target_date_from_days_ago(m.get("logged_days_ago")))

    entries = []  # (line, meal_date) so date tags can be added after we know whether any actually differ
    dates_used = []
    for m in meals:
        meal_date = _target_date_from_days_ago(m.get("logged_days_ago"))
        meal_id = db.add_meal(
            chat_id, m.get("meal_type"), m.get("items") or m.get("meal_items"),
            m.get("calories_low"), m.get("calories_high"), m.get("calories_estimate"),
            water_ml=m.get("water_ml"), meal_date=meal_date,
        )
        row = db.get_meal(chat_id, meal_id)
        items = ", ".join(row["items"]) or "meal"
        water_tag = f", +{row['water_ml']:.0f}ml water" if row.get("water_ml") else ""
        entries.append((f"{items} -- {_calorie_range(row)}{water_tag}", row["meal_date"]))
        if row["meal_date"] not in dates_used:
            dates_used.append(row["meal_date"])

    # Only tag each line with its date when the items in THIS message actually
    # landed on different days -- the common case (all today, or all one
    # backdated day together) already says so via the totals line below, and
    # repeating an identical date on every line would just be noise.
    mixed_days = len(dates_used) > 1
    lines = [f"{ln} ({d})" if mixed_days else ln for ln, d in entries]
    body = "\n".join(f"- {ln}" for ln in lines)
    totals = "\n".join(_daily_meal_totals_text(chat_id, d) for d in dates_used)
    return f"Logged {len(lines)} meals:\n{body}\n\n{totals}"


async def _log_meals_and_reply(update: Update, chat_id: int, meals: list):
    """Thin wrapper around _log_meals_fragment for callers that want the
    confirmation sent directly rather than combined with other domains --
    see _log_meals_fragment's docstring."""
    await _reply(update, chat_id, await _log_meals_fragment(chat_id, meals))


async def logmeal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    description = " ".join(context.args)
    if not description:
        await update.message.reply_text("Usage: /logmeal chicken rice and an iced tea")
        return
    data = ai.extract_meal(description)
    await _log_meal_and_reply(update, chat_id, data)


async def _ask_about_caption_extra_item(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                                         photo_meal: dict, caption_extra: str, caption: str | None):
    """A photo's caption named a food that ISN'T what's actually in the
    picture (ai.extract_from_photo's caption_extra_item -- see its docstring)
    -- a real bad interaction: a photo of nachos captioned "I also had a
    small bowl of black bean pork broth" got logged as ONE entry combining
    both foods' calories into a single wrong, over-estimated total, and the
    user had to undo it and redo it manually. Rather than commit to a guess
    about which food(s) are actually meant, this asks -- and hands the
    answer to the SAME clarification loop handle_text already uses for a
    natural-language follow-up (see replies.PENDING_KEY): the user's next
    message gets merged with this description and re-parsed by
    ai.parse_message, which already knows how to log one or several meals
    from free text, so no separate resolution logic is needed here."""
    items = ", ".join(photo_meal.get("items") or []) or "something"
    cal = _calorie_range(photo_meal)
    original = (
        f"[Photo logging] A photo was sent showing: {items} ({cal}). Its caption also named a separate food: "
        f"\"{caption_extra}\" (caption in full: \"{caption}\"). These are two different foods -- log whichever "
        "one(s) the user actually means below, never combined into one entry."
    )
    context.chat_data[PENDING_KEY] = {"original": original}
    await _reply(
        update, chat_id,
        f"I see {items} in the photo (~{cal}) but your caption also mentions \"{caption_extra}\" -- did you "
        "want both logged separately, just one of them, or something else? Tell me and I'll log it."
    )


async def _handle_extracted_photo_data(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                                        data: dict, caption: str | None):
    """The actual kind-dispatch shared by a single photo and a whole
    buffered album (see handle_photo/_finish_album) -- one classify-and-
    extract result in, one logged entry (or one clarifying question) out,
    regardless of how many photos it came from."""
    kind = data.get("kind")
    target_date = _target_date_from_days_ago(data.get("logged_days_ago"))
    if kind == "meal":
        caption_extra = data.get("caption_extra_item")
        if caption_extra:
            await _ask_about_caption_extra_item(update, context, chat_id, data, caption_extra, caption)
            return
        await _log_meal_and_reply(update, chat_id, data, meal_date=target_date)
    elif kind == "workout":
        await _log_workout_and_reply(update, context, chat_id, data, workout_date=target_date)
    else:
        await _reply(
            update, chat_id,
            "I couldn't tell that was a food photo or a workout/fitness stats screen, so I didn't log anything. "
            "Tell me what it shows (a meal, or calories burned/a workout) and I'll log that instead."
        )


# media_group_id -> {"photos": [bytes, ...], "caption": str | None, "task": asyncio.Task}. Telegram delivers
# an "album" (several photos sent together in one message) as one Update PER PHOTO, each just a few hundred ms
# apart, sharing the same media_group_id, and usually with the caption attached to only ONE of them -- there's
# no single Update that represents "the whole album" to hand to a normal handler. Module-level (not
# chat_data) since it's purely transient bookkeeping across a handful of Updates a second or two apart, never
# meant to survive a restart.
_PENDING_ALBUMS: dict[str, dict] = {}

# How long to wait after the MOST RECENTLY received photo in an album before assuming it's complete and
# processing it -- each new photo in the group cancels and restarts this wait (see handle_photo below), so a
# fast album finishes almost immediately after its last photo arrives, not after a fixed total delay. 1.5s
# comfortably covers the typical few-hundred-ms gap between photos in the same Telegram album with room to
# spare. A module constant (not a magic number inline) so a test can shrink it to 0.
ALBUM_DEBOUNCE_SECONDS = 1.5


async def _finish_album(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, media_group_id: str):
    try:
        await asyncio.sleep(ALBUM_DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return  # a newer photo in this same album already restarted the wait -- that call will finish it
    entry = _PENDING_ALBUMS.pop(media_group_id, None)
    if not entry:
        return  # already finished (shouldn't happen -- only one task per group is ever left uncancelled)
    data = ai.extract_from_photos(entry["photos"], entry["caption"])
    await _handle_extracted_photo_data(update, context, chat_id, data, entry["caption"])


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A photo, sent with or without a caption, logs directly -- no command
    needed. What it logs AS depends on what the photo actually shows
    (ai.extract_from_photo classifies it first, rather than assuming every
    photo is a meal):
    - real food/drink -> logged as a meal, same as before -- UNLESS the
      caption named a different, separate food (caption_extra_item is set),
      in which case this asks which food(s) to actually log rather than
      guessing (see _ask_about_caption_extra_item).
    - a fitness app/wearable's calorie-burned or workout-stats screen ->
      logged as a workout instead (this is the fix for a real bug: this
      case used to either get logged as a fake, nonsense meal built from the
      caption, or -- an earlier, narrower fix -- get rejected outright as
      "not food" even though it plainly was useful data, just not a meal).
      If it looks like a duplicate of an already-logged workout (two photos
      of the same underlying fitness-app data), this asks before logging a
      second time instead of silently doubling the day's total (see
      fitness._find_duplicate_workout).
    - anything else (a receipt, an unrelated photo, ...) -> not logged;
      asks what the user actually wants to do with it instead of guessing.

    Both the "meal" and "workout" cases log onto the day the caption
    actually implies (see ai's logged_days_ago / _target_date_from_days_ago)
    rather than always today -- a real bad interaction: "these were my
    stats for 15 September" got logged as today anyway, needing a manual
    correction afterwards.

    Several photos sent together in ONE Telegram message (an "album",
    identified by a shared media_group_id) are buffered here and classified
    together in a single ai.extract_from_photos call, rather than each
    triggering its own independent handle_photo/extract_from_photo run --
    see _PENDING_ALBUMS and ai.extract_from_photos' own docstring for the
    real bug this fixes: two screenshots of the same day's fitness-app
    activity, analyzed one at a time with no way to know about each other,
    got logged as two separate workouts (one of them even landing on the
    wrong date, since only one of the two photos carried the caption) --
    silently doubling that day's calories-burned total. A photo sent alone
    (no media_group_id -- by far the common case) is entirely unaffected
    and still logs immediately, exactly as before."""
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    caption = (update.message.caption or "").strip() or None
    photo = update.message.photo[-1]  # highest-resolution size Telegram offers
    tg_file = await photo.get_file()
    image_bytes = bytes(await tg_file.download_as_bytearray())

    media_group_id = getattr(update.message, "media_group_id", None)
    if media_group_id is None:
        data = ai.extract_from_photo(image_bytes, caption)
        await _handle_extracted_photo_data(update, context, chat_id, data, caption)
        return

    entry = _PENDING_ALBUMS.get(media_group_id)
    if entry is None:
        entry = {"photos": [], "caption": None}
        _PENDING_ALBUMS[media_group_id] = entry
    else:
        entry["task"].cancel()  # a new photo in the group arrived -- restart the debounce wait
    entry["photos"].append(image_bytes)
    if caption:
        # Telegram attaches the caption to only one photo in the album (not
        # necessarily the first one delivered) -- keep whichever one has it.
        entry["caption"] = caption
    entry["task"] = asyncio.create_task(_finish_album(update, context, chat_id, media_group_id))


async def recentmeals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    chat_id = update.effective_chat.id
    limit = 10
    if context.args:
        try:
            limit = max(1, min(50, int(context.args[0])))
        except ValueError:
            pass
    rows = db.get_recent_meals(chat_id, limit=limit)
    if not rows:
        await update.message.reply_text("No meals logged yet.")
        return
    lines = [_meal_line(r) for r in rows]
    await update.message.reply_text(
        "Recent meals:\n" + "\n".join(lines) + f"\n\n{_daily_meal_totals_text(chat_id)}"
    )
