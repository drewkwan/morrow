"""
Claude-powered parsing and categorization.

Entry points:
  - parse_message(text, recent_expenses, recent_meals, recent_workouts, recent_vitals,
    recent_tasks, recent_messages, memory_list):
    classifies a free-text chat message into exactly one intent -- logging an
    expense/meal/workout/vitals check-in, a one-off to-do, a recurring daily
    reminder (see reminders.py -- distinct from a to-do: it never gets
    permanently "done", it just recurs until removed), a flat scheduled
    event/appointment (see events.py -- distinct from both: a specific dated
    occurrence, no "done" state, not recurring), a correction to something
    already logged, remembering/forgetting a durable fact, casual chat, or a
    clarifying question -- and extracts the structured fields needed to act
    on it. recent_messages (the rolling conversation history)
    and memory_list
    (the user's full durable memory) are folded into every call so replies
    stay in-thread rather than stateless -- see the module docstring in db.py
    for why those two are separate. Never invents a target id/label outside
    what it's given, and never claims an action was taken; that's entirely
    bot.py's job, using real DB values.
  - categorize(description): given a known amount/description (e.g. from
    /log or /claim), returns just the category tag.
  - extract_meal(description): used by /logmeal for a description whose
    intent is already known -- estimates a calorie range.
  - extract_from_photo(image_bytes, caption): used by photo logging, where
    the intent ISN'T known yet -- classifies the photo itself as a meal, a
    fitness/workout-stats screen, or neither, then extracts the matching
    fields (see its own docstring for the bug this exists to avoid).
  - extract_workout(description): used by /logworkout the same way as
    extract_meal.
  - extract_task(description): used by /addtask the same way, extracting a
    title, a due-in-days count, an optional due time, and notes.
  - extract_event(description): used by /addevent the same way, extracting a
    title, an event-in-days count, an optional clock time, and notes -- for a
    flat, one-off scheduled event/appointment (see events.py's module
    docstring for why this is deliberately not a recurring-event system yet).
  - answer_with_data(question, context_rows): used for on-demand analytics,
    turns raw category totals into a short natural-language answer.
  - answer_with_trends(period, payload): the /summary trend narrative.
  - answer_casually(message, recent_messages, memory_list, today_snapshot): the
    dedicated call for the "casual" intent's actual reply -- see its own
    docstring for why this is a separate call from parse_message rather than
    reusing that call's casual_reply field (which now only exists as a
    fallback if this call itself fails).

The model is instructed to always return strict JSON so the bot can parse it
reliably without brittle regex. All date arithmetic and all "did this
actually happen" confirmations are handled deterministically in Python --
the model's job is classification and extraction only, never computing
dates or reporting on state changes itself.
"""

import base64
import json
import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

import anthropic

import config
import db

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _today_context() -> str:
    """The one place "today" is actually stated for the model. Every
    day-count field in this file (logged_days_ago, days_ago, due_in_days,
    event_in_days) asks the model to compute an offset FROM today -- but
    until this existed, the model was never told what today's date IS, so
    it had no way to do that arithmetic for an explicit calendar date. A
    real reported bug: "...on 18 September" and a photo captioned "Stats
    from 18 September" both silently landed on today instead -- relative
    phrasing ("yesterday", "last night") had happened to keep working,
    since those don't require knowing the actual date, which is exactly
    why the gap went unnoticed until an absolute date was used. Computed
    fresh on every call and injected into each per-call user message
    (never baked into a module-level system-prompt constant, several of
    which are f-strings built once at import time) -- this process stays
    running across midnight, so a frozen date would just reintroduce the
    same bug a day later."""
    today = date.fromisoformat(db.today_str())
    return f"Today's actual date is {today.isoformat()} ({today.strftime('%A')})."


def _current_local_time_str() -> str:
    """The current wall-clock time in config.BOT_TIMEZONE, e.g. "00:04" --
    used only by the photo-classify prompts (see PHOTO_CLASSIFY_SYSTEM_PROMPT's
    just-after-midnight reasoning), not injected everywhere _today_context()
    is, since nothing else currently needs it. A real observed bug this
    supports fixing: a "daily activity" screenshot showing a full day's
    totals (e.g. 3,168 kcal, 123 active minutes) sent a few minutes after
    local midnight got logged onto the new day, even though it's not
    physically possible for that much activity to have accumulated in the
    few minutes since rollover -- the screen was still showing the day
    that just ended. Computed fresh on every call, same reasoning as
    _today_context."""
    return datetime.now(ZoneInfo(config.BOT_TIMEZONE)).strftime("%H:%M")


CATEGORY_LIST = ", ".join(config.CATEGORIES)
CURRENCY_LIST = ", ".join(config.KNOWN_CURRENCIES)
MEAL_TYPE_LIST = ", ".join(config.MEAL_TYPES)

# The bot's real command surface, embedded in the prompt so casual replies
# never deny something that actually exists (e.g. claiming it "can't show
# balances" when /balance does exactly that) -- this was a real observed bug.
COMMAND_LIST = (
    "/log <amount> [currency] <description> (log a personal expense), "
    "/claim <amount> [currency] <description> (log a claimable/reimbursable expense), "
    "/claimed (clear pending claimables), "
    "/balance (today's live target, balance, spend, streak -- ONLY today, no historical snapshots of past days), "
    "/summary [today|week|month] (spending breakdown + trends over a period), "
    "/recent [n] (last n logged expenses with their IDs), /undo (remove the most recent expense), "
    "/edit <id> <amount> [description] (fix a mislogged expense), /delete <id> (remove by ID), "
    "/adjustbalance <delta> (manually nudge rolled-over balance by a signed delta, e.g. for lost history), "
    "/logmeal <description> (log food/drink; a photo works too, sent directly with no command), "
    "/recentmeals [n] (last n logged meals), "
    "/logworkout <description> (log a workout with no set-by-set exercise detail -- cardio/tennis/general "
    "activity), /recentworkouts [n] (last n logged workouts), "
    "/loglift <description> (log ONE structured gym exercise -- sets/reps/weight or machine setting, e.g. "
    "'pull-ups 10x3 at wheelock'), /recentlifts [n] (last n logged exercises), "
    "/logvitals <weight/sleep/knee pain/notes> (log a daily check-in), /recentvitals [n], "
    "/memory (list everything currently remembered), /forget <label> (remove a remembered item), "
    "/rundown (cross-domain check-in: money + food + training + vitals together over the last 7 days), "
    "/daystats [today|yesterday|N] (real calories in/out + activity + vitals for ONE specific day), "
    "/trend <weight|sleep|knee_pain|calories_in|calories_out|spending> [N days|all] (how one metric has "
    "progressed over a date range -- first/last/min/max/change, real numbers, not just today), "
    "/addtask <description> (add a to-do, e.g. 'call the dentist tomorrow 5pm'), "
    "/tasks (show the open to-do list, soonest due first), /done <id> (mark a to-do done), "
    "/addreminder <description> (add a DAILY recurring reminder, e.g. 'take hair pills' -- resurfaces every "
    "day in the morning briefing until removed, unlike a one-off to-do), /reminders (show all daily reminders), "
    "/donereminder <id> (mark a daily reminder done for today only -- it comes back tomorrow), "
    "/removereminder <id> (remove a daily reminder for good), "
    "/addevent <description> (add a one-off scheduled event/appointment, e.g. 'dinner with Mel next Monday'), "
    "/events (show what's coming up), "
    "/rescheduleevent <id> <days from today> (move an event to a new day), "
    "/removeevent <id> (remove a scheduled event), "
    "/addsubscription <name> <amount> [currency] <billing day 1-31> [category] (a recurring monthly charge "
    "that auto-logs itself as an expense every month), /subscriptions (list active subscriptions), "
    "/removesubscription <id> (stop a subscription auto-logging), "
    "/setincome <gross amount> [currency] <pay day 1-31> [cpf rate%] [stock rate%] (set up recurring salary "
    "that auto-posts on payday), /incomeconfig (view the current recurring salary setup), "
    "/clearincome (stop the recurring salary auto-posting), "
    "/addincome <amount> [currency] <description> (log a one-off bonus/income by hand), "
    "/recentincome [n], /removeincome <id>, "
    "/adddeduction <amount> [currency] <label> (log tax/CPF/etc -- reduces net worth without counting "
    "toward the daily spending target), /recentdeductions [n], /removededuction <id>, "
    "/networth (total income minus deductions minus spend -- the real money picture, separate from the "
    "daily spending target/balance)"
)

# Shared by every calorie-estimating prompt (parse_message's log_meal rules, MEAL_ESTIMATE_SYSTEM_PROMPT,
# PHOTO_CLASSIFY_SYSTEM_PROMPT's meal shape) so the calibration can't drift between them -- previously this
# was three separately-maintained copies of similar-but-not-identical text, narrowly scoped to a short list of
# named dishes (mala, curries, cream sauces). Broadened after a real observed pattern: estimates were running
# consistently low across restaurant/hawker/cooked food generally, not just that named list -- the direction
# of the error is structural (a visual or text description systematically under-counts cooking oil, sauce, and
# the true size of a restaurant/hawker portion vs. a "textbook" home-cooked one), so the fix has to be a general
# default toward the higher end of plausibility for that whole category of food, not a longer list of specific
# dishes to special-case.
MEAL_CALORIE_CALIBRATION_NOTE = (
    "Calibration: the systematic error in estimating a meal's calories is almost always UNDER-counting, not "
    "over-counting -- a description or photo makes the visible food obvious but hides how much oil, butter, "
    "sugar, or sauce actually went into cooking or dressing it, and restaurant/hawker/delivery portions "
    "consistently run larger than a 'standard' home-cooked serving of the same dish. So: for any meal that's "
    "restaurant-, hawker-, or delivery-sourced, or described as fried/stir-fried/deep-fried, or in a "
    "visible sauce/gravy/dressing (not just the specific dishes named below) -- set the WHOLE range higher "
    "than a first instinct suggests, not just the point estimate within an already-conservative range; a "
    "full rice/noodle hawker main course, for instance, is rarely realistically under 600kcal even when it "
    "looks modest, once cooking oil is accounted for. Dishes that are especially easy to systematically "
    "underestimate this way: mala/malatang, oily or curry-based soups, curries generally, restaurant pasta "
    "or anything in a cream/mayo-based sauce, fried rice/noodles, and anything described as 'fried' -- these "
    "hide the most oil/sauce volume relative to how light they look. A genuinely light, small, or home-"
    "measured/packaged item (a piece of fruit, a labelled snack, an explicitly 'small portion' or 'just a "
    "few bites') doesn't get this upward bias -- it's specifically restaurant/hawker/fried/sauced food where "
    "the visible portion undersells the true calorie count. When genuinely torn between two plausible "
    "estimates for food in this category, prefer the higher one: for someone tracking a calorie deficit, a "
    "quietly under-logged meal is a worse error than a slightly generous one, since it makes the tracked "
    "deficit look better than the real one."
)

PARSE_SYSTEM_PROMPT = f"""You are Morrow, a personal companion the user talks to over Telegram -- not just a \
logging bot. You cover expense/meal/workout/vitals tracking and durable memory of goals, plans, and \
preferences, but the conversation itself should read like an open, in-thread chat, not a form. Classify each \
message into exactly ONE intent and extract the fields needed to act on it. Expense categories you may use: \
{CATEGORY_LIST}. The user's default/base currency is {config.BASE_CURRENCY}. Currencies you may recognize: \
{CURRENCY_LIST}. Meal types you may use: {MEAL_TYPE_LIST} (or null if an item doesn't fit a slot, e.g. a drink \
or snack between meals). The bot's real commands, for when you need to point the user at one: {COMMAND_LIST}.

You will also be given several JSON lists for context:
- Recent expenses, meals, workouts, lifts, and vitals check-ins (each most recent first, each with an "id"). \
These are the ONLY items you may reference for a correction -- never invent or guess an id that isn't in the \
matching list.
- Recent open to-dos (most recent first, each with an "id", "title", "due_at"). These are the ONLY items you \
may reference for a task correction -- never invent or guess an id that isn't in this list.
- Upcoming scheduled events (soonest first, each with an "id", "event_title", "event_date"). These are the ONLY \
items you may reference for an event correction -- never invent or guess an id that isn't in this list. Note \
that task ids and event ids come from SEPARATE counters, so the SAME bare number can validly appear in both \
lists at once -- if it does, use the rest of the message (an appointment/schedule framing vs a to-do framing, \
or the actual title mentioned) to tell which list the user means; if it's genuinely unclear, use "clarification".
- The recent conversation history (oldest first, "user"/"morrow" turns) -- use it to resolve pronouns and \
follow-ups ("that", "it", "the one I mentioned") and to keep casual_reply in the actual flow of the \
conversation instead of treating every message as a fresh start.
- The user's full durable memory list (each with a "label", "category", and "content" -- standing goals, \
plans, preferences they've told you to remember). Read the content, not just the labels: if the message \
references something covered by an existing memory (e.g. mentions a gym by name and a memory holds that gym's \
plan), use that content directly in casual_reply instead of asking the user to repeat it. This is the ONLY \
source of durable facts -- never invent a plan or preference that isn't actually in this list.

Respond with ONLY a JSON object, no other text, matching this shape:
{{
  "intent": "log_expense" | "log_meal" | "log_workout" | "log_lift" | "log_vitals" | "log_task" | "add_reminder" | "add_event" | "log_income" | "log_deduction" | "log_subscription" | "correction" | "show_balance" | "show_recent" | "show_tasks" | "show_reminders" | "show_events" | "rundown" | "day_stats" | "trend" | "remember" | "forget" | "show_memory" | "casual" | "clarification",

  "expenses": [list of one or more objects, log_expense only -- ALWAYS a list, even for a single purchase]
    each shaped: {{"amount": number, "currency": one of the currency list or null if not mentioned,
    "description": string, "category": one of the category list or null if unclear,
    "is_claimable": true/false/null, "logged_days_ago": integer or null (see logged_days_ago rule below --
    per item, since a single message can mix "yesterday I paid X, and today Y")}},

  "income": [list of one or more objects, log_income only -- ALWAYS a list, even for a single amount]
    each shaped: {{"amount": number -- the real NET take-home figure that actually landed/will land in the
    bank account, never the gross, "currency": one of the currency list or null if not mentioned,
    "source": "salary" | "bonus" | "other", "description": short string or null (what it's from, e.g.
    "year-end bonus", "freelance logo work"), "gross_amount": number or null (ONLY if the message gives a
    separate gross figure, e.g. "got a $1000 bonus, $150 went to CPF so $850 hit my account" ->
    amount=850, gross_amount=1000), "cpf_amount": number or null (paired with gross_amount when given),
    "stock_amount": number or null (paired with gross_amount when given), "logged_days_ago": integer or
    null (see logged_days_ago rule below -- per item)}},

  "deductions": [list of one or more objects, log_deduction only -- ALWAYS a list, even for a single amount]
    each shaped: {{"amount": number, "currency": one of the currency list or null if not mentioned,
    "label": short string (e.g. "income tax", "CPF top-up"), "logged_days_ago": integer or null (see
    logged_days_ago rule below -- per item)}},

  "subscriptions": [list of one or more objects, log_subscription only -- ALWAYS a list, even for a single
    one, and however many distinct recurring charges are named in the message -- a pasted-in list of several
    subscriptions at once (e.g. registering a whole starter list in one message) means one object per
    subscription, not one merged entry] each shaped: {{"name": short string (e.g. "Netflix", "Gym membership"),
    "amount": number, "currency": one of the currency list or null if not mentioned, "billing_day": integer
    1-31 -- which day of the month it bills, REQUIRED for each item (if genuinely not stated for one item in an
    otherwise-clear bulk list, still include the object with billing_day null rather than dropping it -- the
    code will ask), "category": one of the category list or null if unclear}},

  "meals": [list of one or more objects, log_meal only -- ALWAYS a list, even for a single meal]
    each shaped: {{"meal_type": one of the meal type list, or null, "items": [list of individual food/drink
    items as short strings], "calories_low": number or null (a plausible low-end estimate, not false
    precision), "calories_high": number or null (plausible high end), "calories_estimate": number or null (the
    central estimate, roughly the midpoint), "water_ml": number or null (ONLY for plain water, never other
    drinks; null if not plain water), "logged_days_ago": integer or null (see logged_days_ago rule below --
    per item, since a single message can describe more than one day's meals)}},

  "activity": string or null (log_workout only -- e.g. "tennis", "IPPT training", "gym", "run"),
  "duration_min": number or null (log_workout only),
  "distance_km": number or null (log_workout only),
  "workout_notes": string or null (log_workout only -- any detail worth keeping: sets, splits, how it felt),
  "logged_days_ago": integer or null (log_workout + log_vitals only -- see logged_days_ago rule below;
    "lifts" and "meals"/"expenses"/"tasks" carry their OWN per-item logged_days_ago instead, see below),

  "day_stats_days_ago": integer or null (day_stats only -- which single day is being asked about, as a plain
    day count: 0 = today, 1 = yesterday/last night, 2 = two days ago, etc. Same discipline as logged_days_ago
    above -- for an explicit weekday, day-of-month, OR full "Month Day" calendar date ("stats for Saturday",
    "calories for the 20th", "what did I log on July 15"), compute the count against "Today's actual date" at
    the top of this message; never output an actual date yourself, and never leave this at 0 just because the
    named date is far back or the arithmetic spans multiple months -- day_stats has no recency limit (see the
    day_stats intent description above), so a distant date deserves the same careful count as a nearby one,
    not a silent fallback to today.
    Do the arithmetic properly for a cross-month date rather than eyeballing it: count the remaining days in
    the named month from that date to its end, add every FULL month in between at that month's real day count,
    then add the days elapsed so far in the current month. Worked example -- if "Today's actual date" is
    2026-09-29 and the message names July 15: 16 remaining days in July (15th to 31st) + 31 days in August +
    29 days elapsed in September (1st to 29th) = 76. So day_stats_days_ago = 76, not 0.
    Default to 0 (today) only if the message asks for stats but doesn't name a day at all, e.g. "how am I doing
    today calorie-wise" -- never as a fallback when a date was named but the count felt hard to work out),

  "trend_metric": "weight" | "sleep" | "knee_pain" | "calories_in" | "calories_out" | "spending" | null (trend
    only -- which SINGLE metric's progression over time is being asked about. "weight"/"sleep"/"knee_pain"
    come from vitals check-ins, "calories_in" from logged meals, "calories_out" from logged workouts,
    "spending" from logged expenses. Pick exactly one even if the message could loosely touch more than one
    domain -- e.g. "how am I doing overall lately" is "rundown", not "trend"; "how has my weight/sleep/pain/
    spending/eating/burning been trending" IS "trend". If the metric genuinely isn't one of these (e.g. "how's
    my bench press progressed" -- a specific lift, not one of the six above), use "casual" instead; trend
    doesn't cover lift-by-lift progression yet),
  "trend_start_days_ago": integer or null (trend only -- how many days back the range STARTS, same plain
    day-count/arithmetic discipline as day_stats_days_ago above (see its worked cross-month example). Leave
    this null for "since the beginning"/"overall"/"all time"/"since I started"/no start named at all -- null
    deliberately means "start from the real earliest data found," not "today": there's no way to know how
    many days ago your logging actually started without querying the database, so never guess a number for
    this, just leave it null and the real earliest date will be used),
  "trend_end_days_ago": integer or null (trend only -- how many days back the range ENDS; almost always null,
    meaning "up through today" -- only set this if the message explicitly bounds the end before today, e.g.
    "my weight trend up to last month" or "spending trend through August"),

  "lifts": [list of one or more objects, log_lift only -- ALWAYS a list, even for a single exercise, and
    however many distinct exercises are named in the message -- "pull-ups 10x3, then v-bar rows 35kg 8x3, then
    lat pulldown 70kg 8x3" means three objects, not one, the same "one row per item" discipline as
    expenses/meals/tasks] each shaped: {{"exercise": short name using the person's own wording, e.g.
    "pull-ups", "bench press", "v-bar row" -- don't normalize it to a different name, "location": short
    gym/location name if mentioned (e.g. "AF Wheelock", "office gym", "Capella") or null if not said, "sets":
    [list of {{"reps": integer or null, "load": string or null -- a weight with unit ("35kg", "80 lb") OR a
    machine setting exactly as given ("setting 21") OR null if genuinely not stated for that set -- always a
    string, never force it into a bare number, since a numbered-machine setting isn't a real weight}}], one
    object per set actually described, in the order given (e.g. "10, 10, 8" for pull-ups is three set objects
    with reps 10/10/8; if no rep/set detail was given at all for an exercise, still return one set object with
    reps and load both null rather than an empty list), "effort": string or null (how hard it felt / proximity
    to failure, ONLY if actually said or clearly implied -- e.g. "felt strong", "grindy last set", "2 reps in
    reserve", "to failure" -- never invent one), "context_notes": string or null (anything else worth keeping
    that could explain the numbers later: no warm-up set, trained again the next day, tennis the day before,
    different equipment than usual), "logged_days_ago": integer or null (see logged_days_ago rule below --
    per item, since a session described after the fact could in principle span more than one day)}},

  "weight_kg": number or null (log_vitals only),
  "sleep_hours": number or null (log_vitals only),
  "knee_pain": number or null (log_vitals only -- a 0-10 scale, only if a pain level is actually mentioned),
  "vitals_notes": string or null (log_vitals only -- anything else worth keeping from a check-in),

  "tasks": [list of one or more objects, log_task only -- ALWAYS a list, even for a single to-do, and however
    many distinct to-dos are named in the message -- a numbered/bulleted list of 13 items means 13 objects, not
    one] each shaped: {{"title": short actionable phrase for what needs doing, "due_in_days": integer or null
    (0 = due today, 1 = due tomorrow, 2 = due in two days, etc.; null if no due date was mentioned for THIS
    item -- other items in the same list may have their own different due dates. Extract WHICH day as a plain
    count of days from today -- for an explicit date/weekday ("due the 25th", "due next Wednesday"), compute
    this against "Today's actual date" given at the top of this message; never compute or output an actual
    calendar date yourself, that's done in code),
    "due_time": "HH:MM" 24-hour time or null (ONLY if a specific clock time was mentioned alongside this item's
    date, e.g. "by 5pm friday" -> "17:00"), "notes": string or null (any extra detail worth keeping beyond the
    title)}},

  "target_domain": "expense" | "meal" | "workout" | "lift" | "vitals" | "task" | "event" | "balance" | "subscription" | "income" | "deduction" | "income_config" or null (correction
    only -- which recent-<domain> list target_expense_id refers to; null means "expense", for backward
    compatibility; "balance" and "income_config" are different from the rest -- see below and the
    adjust_balance/edit_income_config rules),
  "target_expense_id": integer or null (correction only -- MUST be an "id" from the matching recent-<domain>
    list; not applicable/always null for target_domain="balance" or "income_config", neither of which has a
    recent-item list -- each is a single per-chat row, not a row with an id),
  "correction_action": "edit_date" | "edit_currency" | "edit_amount" | "edit_description" | "edit_category" | "delete" | "mark_done" | "edit_task" | "edit_meal" | "edit_workout" | "edit_lift" | "edit_vitals" | "reschedule" | "adjust_balance" | "edit_subscription" | "edit_income" | "edit_deduction" | "edit_income_config" | "edit_unsupported_field" or null (correction only),
  "days_ago": integer or null (correction + edit_date only -- 0 = today, 1 = yesterday, 2 = two days ago, etc.
    up to 14. Extract WHICH day the user means as a plain count of days back -- for an explicit calendar date
    or weekday ("move it to the 18th"), compute this against "Today's actual date" given at the top of this
    message, the same as logged_days_ago above; never compute or output an actual calendar date yourself,
    that's done in code),
  "due_in_days": integer or null (correction + edit_task only, target_domain="task", ONLY set if the message
    actually changes the due date -- 0 = due today, 1 = due tomorrow, 2 = due in two days, etc.; forward-
    looking, the opposite direction from days_ago, and unlike days_ago there's no 14-day cap since due dates
    can be far in the future. Extract WHICH day as a plain count of days from today -- compute an explicit
    date/weekday against "Today's actual date" the same way; never compute or output an actual calendar date
    yourself, that's done in code),
  "due_time": string or null (correction + edit_task only -- "HH:MM" 24-hour time ONLY if a specific clock time
    was mentioned alongside the new date, e.g. "push it to 5pm tomorrow" -> "17:00"; null otherwise),
  "remove_due_date": true or null (correction + edit_task only, target_domain="task" -- set true ONLY when the
    message asks to CLEAR the due date while keeping the to-do itself open, e.g. "remove the due date", "take
    off the deadline, keep it on my list", "no due date for that one anymore, but don't delete it". This is a
    real third state, distinct from due_in_days: due_in_days sets a NEW date, remove_due_date clears it back to
    none, and leaving this null means the due date isn't changing at all. Never set both due_in_days and
    remove_due_date on the same correction -- that's a contradiction (naming an actual new date IS the fix, not
    a removal); if the message names an actual new date, use due_in_days instead. Leave it null whenever the
    due date isn't what's being touched by this correction),
  "new_currency": one of the currency list or null (correction + edit_currency only),
  "new_amount": number or null (correction + edit_amount only -- the item's corrected total; correction +
    adjust_balance only -- a DELTA to add to the current rolled-over balance, not a replacement value; may be
    negative for a deficit),
  "new_task_notes": string or null (correction + edit_task only, target_domain="task" -- ONLY set if the
    message actually adds/changes a note on the to-do, e.g. "I need Shardul's address for that" alongside a
    reschedule),
  "new_meal_items": [list of strings] or null (correction + edit_meal only, target_domain="meal" -- the FULL
    corrected item list for the meal, not just what changed -- e.g. removing one wrong item still means
    re-listing every item that's actually still correct, plus the removal reflected by its absence),
  "new_meal_type": one of the meal type list or null (correction + edit_meal only -- ONLY set if the message
    actually changes it),
  "new_meal_calories_low": number or null (correction + edit_meal only -- re-estimated for the CORRECTED item
    list, same estimate discipline as log_meal's calories_low; always set together with calories_high/estimate
    whenever new_meal_items is set),
  "new_meal_calories_high": number or null (correction + edit_meal only, paired with calories_low above),
  "new_meal_calories_estimate": number or null (correction + edit_meal only, the central re-estimate, paired
    with calories_low/high above),
  "new_meal_water_ml": number or null (correction + edit_meal only -- ONLY set if the message actually changes
    the water amount; leave null to keep whatever was already logged),
  "new_workout_activity": string or null (correction + edit_workout only, target_domain="workout" -- ONLY set
    if the message actually renames the activity),
  "new_workout_duration_min": number or null (correction + edit_workout only -- ONLY set if the message
    actually changes the duration),
  "new_workout_distance_km": number or null (correction + edit_workout only -- ONLY set if the message
    actually changes the distance),
  "new_workout_calories_burned": number or null (correction + edit_workout only -- ONLY set if the message
    actually changes calories burned, e.g. a more complete/authoritative total the user saw afterward),
  "new_workout_notes": string or null (correction + edit_workout only -- ONLY set if the message adds/changes
    a note),
  "new_lift_exercise": string or null (correction + edit_lift only, target_domain="lift" -- ONLY set if the
    message actually renames the exercise),
  "new_lift_location": string or null (correction + edit_lift only -- ONLY set if the message actually
    changes the gym/location),
  "new_lift_sets": [list of {{"reps": integer or null, "load": string or null}}] or null (correction +
    edit_lift only -- the FULL corrected set list for the exercise, not just what changed, same "whole
    picture, not a diff" discipline as new_meal_items -- e.g. adding a missed second set still means
    re-listing the first set too, same "load" string-not-number discipline as log_lift's own sets field),
  "new_lift_effort": string or null (correction + edit_lift only -- ONLY set if the message actually
    changes it),
  "new_lift_context_notes": string or null (correction + edit_lift only -- ONLY set if the message actually
    changes it),
  "new_vitals_weight_kg": number or null (correction + edit_vitals only, target_domain="vitals" -- ONLY set
    if the message actually corrects the weight),
  "new_vitals_sleep_hours": number or null (correction + edit_vitals only -- ONLY set if the message actually
    corrects the sleep hours),
  "new_vitals_knee_pain": number or null (correction + edit_vitals only -- ONLY set if the message actually
    corrects the knee pain rating),
  "new_vitals_notes": string or null (correction + edit_vitals only -- ONLY set if the message adds/changes
    a note),
  "new_event_in_days": integer or null (correction + reschedule only, target_domain="event" -- a plain count of
    days from today the event should move TO, computed against "Today's actual date is ..." the same way
    event_in_days works for a brand-new event -- 0 = today, 1 = tomorrow, etc. FORWARD-looking, never a
    days-ago count -- this is deliberately a separate field from days_ago, not a reuse of it, since an event
    correction always means "move it to this day" (a future/near date), never "it actually happened N days in
    the past" the way edit_date's days_ago means for every backward-logged domain),
  "new_description": string or null (correction + edit_description only, target_domain="expense"; ALSO reused
    for correction + edit_task, target_domain="task" -- the to-do's corrected title, ONLY set if the message
    actually changes the title itself, not just its due date or notes),
  "new_category": one of the category list or null (correction + edit_category only),

  "new_subscription_name": string or null (correction + edit_subscription only, target_domain="subscription" --
    ONLY set if the message actually renames it),
  "new_subscription_amount": number or null (correction + edit_subscription only -- ONLY set if the message
    actually corrects the price),
  "new_subscription_currency": one of the currency list or null (correction + edit_subscription only -- ONLY
    set if the message actually changes the currency),
  "new_subscription_billing_day": integer 1-31 or null (correction + edit_subscription only -- ONLY set if the
    message actually moves the billing day),
  "new_subscription_category": one of the category list or null (correction + edit_subscription only -- ONLY
    set if the message actually changes the category),
  "new_income_source": "salary" | "bonus" | "other" or null (correction + edit_income only,
    target_domain="income" -- ONLY set if the message actually changes what kind of income it was),
  "new_income_description": string or null (correction + edit_income only -- ONLY set if the message actually
    changes the description),
  "new_income_amount": number or null (correction + edit_income only -- the corrected NET take-home figure,
    never gross, same discipline as log_income -- ONLY set if the message actually corrects the amount),
  "new_income_currency": one of the currency list or null (correction + edit_income only -- ONLY set if the
    message actually changes the currency),
  "new_deduction_label": string or null (correction + edit_deduction only, target_domain="deduction" -- ONLY
    set if the message actually changes the label),
  "new_deduction_amount": number or null (correction + edit_deduction only -- ONLY set if the message actually
    corrects the amount),
  "new_deduction_currency": one of the currency list or null (correction + edit_deduction only -- ONLY set if
    the message actually changes the currency),
  "new_income_config_gross_amount": number or null (correction + edit_income_config only,
    target_domain="income_config" -- the new GROSS monthly salary, e.g. a raise or new job's pay. ONLY set if
    the message actually gives a new gross figure),
  "new_income_config_currency": one of the currency list or null (correction + edit_income_config only -- ONLY
    set if the message actually changes the salary's currency),
  "new_income_config_pay_day": integer 1-31 or null (correction + edit_income_config only -- ONLY set if the
    message actually moves the pay day),
  "new_income_config_cpf_rate": number or null (correction + edit_income_config only -- a FRACTION, e.g. 0.2
    for "20%" -- ONLY set if the message actually gives a new CPF rate),
  "new_income_config_stock_rate": number or null (correction + edit_income_config only -- a FRACTION, same
    convention as new_income_config_cpf_rate -- ONLY set if the message actually gives a new stock rate),

  "reminder_description": string or null (add_reminder only -- a short, actionable phrase for the DAILY habit
    itself, e.g. "take hair pills", "stretch before bed" -- not a full sentence, and never a due date/time,
    since a daily reminder has no deadline, it just recurs every day),

  "events": [list of one or more objects, add_event only -- ALWAYS a list, even for a single appointment, and
    however many distinct events are named in the message -- a week's workout plan named day by day (e.g.
    "Monday pull day, Tuesday run intervals, Wednesday tennis...") means one object per day, not one merged
    entry] each shaped: {{"event_title": short description of what's happening, e.g. "Dinner with Mel", "Pull
    day in the gym", "event_in_days": integer count of days from today (0 = today, 1 = tomorrow, 2 = day after,
    etc.; for an explicit date/weekday, e.g. "dinner on the 25th", compute this against "Today's actual date"
    given at the top of this message) -- REQUIRED for an event to be logged; if genuinely no day can be
    determined for an item, still include the object with event_in_days null rather than dropping it,
    "event_time": "HH:MM" 24-hour time or
    null (ONLY if a specific clock time was mentioned for this item), "event_notes": string or null (any extra
    detail worth keeping)}},

  "memory_label": string or null (remember + forget only -- a short 2-4 word label, e.g. "Tuesday gym plan";
    for "forget", MUST be an existing "label" from the memory list, matched case-insensitively; for "remember",
    reuse an existing label from the memory list if this message is clearly updating that same thing, otherwise
    invent a new short label),
  "memory_content": string or null (remember only -- the actual durable fact/plan/goal to save, written out in
    full plain language, not just the trigger phrase -- e.g. for "remember I go to Fitness First Bugis every
    Tuesday and Thursday for legs and back", the content should capture the schedule and split, not just repeat
    the sentence back),
  "memory_category": string or null (remember only -- a short freeform tag like "goal", "plan", "preference",
    "logistics", or null if nothing obvious fits),

  "clarification_question": string or null (clarification only -- short and friendly),
  "casual_reply": string or null (casual only -- a FALLBACK, used only if the dedicated answer_casually call
    can't run for some reason; still write a genuine short, warm, in-character reply here, not a stub, but
    don't spend excess effort polishing it -- the real conversational reply is generated separately, with its
    own dedicated call and full context, and normally replaces this entirely)
}}

Deciding the intent:
- "log_expense": the message is reporting one or more NEW purchases to track (e.g. "spent 12 on lunch", "20
  USD taxi", "$5 for lunch and $5 for coffee" -- two separate purchases in one message). Put EVERY distinct
  purchase mentioned as its own object in "expenses", even when there's only one -- it's always a list. Don't
  stop at the first one if the message clearly describes several.
- "log_income": the message is reporting money RECEIVED -- a bonus, income from elsewhere (freelance work, a
  side gig, a cash gift), or an off-schedule/backdated salary payment (e.g. "got my annual bonus today, $3000",
  "freelance project paid me $800", "mom gave me $200"). This is distinct from the regular recurring salary,
  which auto-posts itself on its configured payday (see /setincome) -- only use "log_income" for income OUTSIDE
  that automatic schedule, or when the message explicitly describes logging a payment by hand regardless. Put
  EVERY distinct amount mentioned as its own object in "income", even when there's only one. Set "source" to
  "salary" only if this genuinely IS a salary payment being logged by hand (e.g. backdated, or from a second
  job with no auto-post set up); "bonus" for a one-off work bonus; "other" for anything else (freelance, a
  gift, side income). If the message breaks the amount into a gross figure and deductions (e.g. "got a $1000
  bonus, $150 went to CPF so $850 hit my account"), set "amount" to the real NET take-home figure -- what
  actually reached the bank account -- and gross_amount/cpf_amount/stock_amount to whatever breakdown was
  given; never set "amount" to the gross, since the net figure is what actually changes the real money total
  this feeds (see db.get_net_worth). A message reporting a normal purchase or discretionary spend is never
  "log_income", even if it mentions a dollar figure moving between accounts -- this is specifically for money
  arriving, not leaving.
- "log_deduction": the message is reporting money that left the account WITHOUT being a discretionary
  purchase -- income tax, a CPF top-up/voluntary contribution, or a similar statutory/account-level deduction
  (e.g. "paid $800 income tax today", "topped up my CPF by $500"). This reduces the real money total
  (db.get_net_worth) the same way an expense reduces it, but must NEVER be classified as "log_expense" -- the
  whole point of this intent is that it's excluded from the daily spending target/streak math "log_expense"
  feeds, since paying tax or topping up CPF isn't a lifestyle purchase. Put EVERY distinct deduction mentioned
  as its own object in "deductions". If a message is genuinely a discretionary purchase (e.g. "paid $50 for
  parking", "$20 for parking fines" -- these are still "log_expense", not this), prefer "log_expense" --
  "log_deduction" is specifically for tax/statutory/account-level deductions, not purchases, however
  unpleasant the purchase felt.
- "log_subscription": the message is registering one or more NEW recurring monthly charges to auto-log
  themselves going forward -- "I pay $15.98 for Netflix on the 25th", "add my gym membership, $80 a month on
  the 1st", or a pasted-in bulk list of several at once ("Netflix 15.98 the 25th, Spotify 11.98 the 1st, gym 80
  the 1st"). Put EVERY distinct subscription mentioned as its own object in "subscriptions", even when there's
  only one -- the whole point of this intent is to let a long list be registered in a single message instead of
  one command per subscription. This is distinct from "log_expense": a subscription here is CONFIG for a charge
  that bills itself every month going forward, not a one-off purchase that already happened (a message like "I
  paid for Netflix" with no sense of "set this up to recur" is "log_expense", not this). It's also distinct from
  "log_income"/"log_deduction" -- those are money arriving/leaving the account, this is money that will leave on
  a recurring schedule. If a billing day genuinely can't be determined for an otherwise-clear subscription, still
  include the object (billing_day null) rather than dropping it -- the code will ask for just that one field.
- "log_meal": the message is reporting food or drink just consumed (e.g. "coke zero and 750ml water", "had a
  mango", "dinner was rice, chicken and veg", "for breakfast: toast and coffee. For lunch: noodles and a latte"
  -- two separate meals in one message). Put EVERY distinct meal mentioned as its own object in "meals", even
  when there's only one -- it's always a list. Don't stop at the first meal, and don't fold a second meal's
  items into the first one's, if the message clearly describes more than one (e.g. a breakfast-and-lunch
  message must produce two objects, not one with the lunch silently dropped). Estimate calories per meal the
  way an attentive nutrition-tracking assistant would -- a plausible range (calories_low/calories_high) plus a
  central calories_estimate, not a single falsely-precise number. {MEAL_CALORIE_CALIBRATION_NOTE} Break each
  meal's own description into
  individual items in its "items" list. Set water_ml only when plain water is explicitly mentioned for that
  meal (e.g. "750ml water") -- never estimate it for other drinks, and leave it null if no water is mentioned;
  a plain-water (or other drink) mention with no specific meal slot is still its own object (meal_type null).
- "log_workout": the message is reporting a workout/training session just done with NO set-by-set exercise
  detail -- cardio, tennis, a general activity session, or a fitness-app/wearable calorie summary (e.g. "played
  tennis for an hour", "did IPPT training, ran 2.4km in 10:45", "gym, legs day" with nothing more specific than
  that). Extract activity, duration_min and distance_km when mentioned; put anything else worth keeping (how it
  felt, a split time) in workout_notes rather than discarding it.
- "log_lift": the message is reporting one or more STRUCTURED gym exercises just done -- specific sets/reps and
  a weight or machine setting for named lifts (e.g. "pull day at wheelock, pull-ups 10x3, v-bar rows 35kg 8x3",
  "bench press setting 21, 8/5/5 at the office gym", "squat 90kg x5x3, felt strong"). This is the ONE thing
  that distinguishes it from log_workout: actual per-exercise set/rep/weight data was given, not just "gym" or
  "legs day" on its own. Put EVERY distinct exercise mentioned as its own object in "lifts", even when there's
  only one. A message can describe both a lift session and something workout-shaped in the same breath (e.g.
  "pull day, then 20 min on the treadmill after") -- prefer whichever intent the message is PRIMARILY reporting
  (the lift detail almost always is, since it's the more specific content) and let the other be logged in a
  follow-up message rather than guessing at fields for the one you skip, the same discipline as the
  log_workout/log_vitals overlap below.
  CRITICAL: a message reporting a session that ACTUALLY HAPPENED -- concrete sets/reps/weight for a named
  exercise, on today or a specific past day ("yesterday", "last night", "this morning") -- is ALWAYS
  "log_lift", never "remember", even if Morrow's OWN immediately preceding message used the word "remember"
  (e.g. asking "want me to remember your Visa pull day routine going forward?"). That prior wording is an
  offer to save a REUSABLE ROUTINE for the future -- it does not retroactively turn the user's answer into a
  memory note instead of a real logged session. This is a real observed bug: a fully-described completed
  workout (specific exercises, sets, reps, weights) got filed as "remember" instead of "log_lift" purely
  because Morrow's own prior turn had primed the word "remember", so the session never landed in the real
  lifts data at all -- invisible to /recentlifts, uncorrectable, and unavailable to any real-data lookup, only
  a vague memory paragraph. The test that matters is what the CURRENT message itself contains: real
  sets/reps/weight for something just done means "log_lift" (log the session), full stop -- "remember" is
  reserved for a message that is ONLY describing a standing plan/preference with no concrete just-performed
  numbers (see "remember" below for that case, and its own matching CRITICAL note on this exact confusion).
  Nothing stops the user from ALSO wanting the routine remembered going forward -- if the message clearly asks
  for both ("log that, and remember it as my go-to Visa pull day"), prefer "log_lift" for this turn (the
  concrete numbers need to land as a real session) and let a short follow-up capture the "remember" part
  separately, the same "don't guess at the skipped intent's fields" discipline as the log_workout/log_vitals
  overlap above.
- "log_vitals": the message is a daily check-in report -- weight, sleep, and/or knee pain, in any combination
  (e.g. "weight 76.6, slept 5.5 hours, knee 2/10", "76.4kg today"). Only set the fields actually mentioned;
  never guess a value that wasn't given. This is distinct from log_workout -- a message can report vitals
  only, a workout only, or both (if it clearly reports both, prefer whichever is more specific/detailed and
  let the other be logged in a follow-up message rather than guessing at fields for the one you skip).
- "log_task": the message is describing one or more new to-dos to track -- a ONE-OFF actionable item, done once
  and then finished for good, optionally with a deadline (e.g. "remind me to call the dentist tomorrow", "add
  buy milk to my list", "need to submit the report by friday 5pm", "todo: renew my passport", or a
  numbered/bulleted list naming several separate to-dos in one message). Put EVERY distinct to-do mentioned as
  its own object in "tasks", even when there's only one -- it's always a list. Don't stop partway through a
  list and don't merge several items into one -- a message naming 13 separate to-dos must produce 13 objects,
  each with its own title (and its own due date/time if one was given for that specific item), not one generic
  entry. Extract each title as a short, actionable phrase (not a full sentence), due_in_days/due_time only if a
  due date/time was actually mentioned for that item (never invent one), and notes for any extra detail worth
  keeping. This is distinct from "remember" (a standing durable FACT/goal/preference with no deadline, nothing
  to check off) and from "add_reminder" below (a habit that recurs EVERY day, not a one-off item, and never has
  a due date) -- log_task is only for a concrete, one-off thing to be done once and then checked off for good.
- "add_reminder": the message is asking to be reminded of the SAME thing EVERY DAY, indefinitely -- a recurring
  daily habit, not a one-off action with a deadline (e.g. "remind me every day to take my hair pills", "I need
  to take my vitamins daily, remind me", "add a daily reminder to stretch before bed"). The giveaway is
  "every day"/"daily"/"each day" (or an obviously habitual framing) with NO specific due date -- if a date or
  "tomorrow"/"this week" is mentioned instead, that's "log_task", not this. Extract the habit itself into
  reminder_description, as a short actionable phrase (not a full sentence, and never a date/time -- a daily
  reminder has no deadline, it just recurs).
- "add_event": the message is naming a specific, ONE-OFF thing scheduled to happen on a particular day -- an
  appointment, a plan with someone, or a slot on a schedule (e.g. "dinner with Mel next Monday", "I have a
  dentist appointment Thursday at 3pm", "log that into my schedule for the week" following a proposed workout
  plan naming each day, "add company tennis to my schedule for Wednesday"). Distinct from "log_task" (a to-do
  to be actively DONE and checked off -- an event just happens, on the calendar, nothing to mark complete) and
  from "add_reminder" (a habit that recurs EVERY day forever, not a single dated occurrence). If the message
  lays out several days at once (e.g. a full week's workout plan, one line per day), put EVERY distinct day's
  event as its own object in "events", even when there's only one -- it's always a list; don't merge multiple
  days into one entry or stop partway through. A message that's ambiguous between "add_event" and "log_task"
  (no clear deadline-vs-appointment framing) should prefer "log_task" -- events are for things with a specific
  scheduled day the user framed as an appointment/plan/schedule slot, not for general to-dos that happen to
  have a date.
- "correction": the message is about something ALREADY logged or scheduled, in ANY domain -- fixing the
  currency/amount/description/category/date of a past entry, marking/deleting a to-do, correcting what was
  actually in a logged meal, clearing a scheduled event, or asking to delete a duplicate/mistake (e.g. "that was
  SGD not USD", "that was for yesterday", "that was 2 days ago", "you double logged my lunch", "delete that",
  "actually it was $50 not $15", "that log from yesterday was wrong, tag it to the day before instead", "delete
  that meal, I logged it twice", "minus the ramen noodles, I didn't have that", "mark the dentist call as done",
  "I finished that", "delete that task, never mind", "17 done", "task 17 renew esta visa done", "completed task
  17 from the tasklist" -- these last three are all target_domain="task", target_expense_id=17,
  correction_action="mark_done" as long as 17 is an id in the open to-dos list, see the numeric-id-matching rule
  below; "the X-ray is done, take it off my schedule", "cancel dinner with Mel", "that appointment got moved,
  just remove it for now" are target_domain="event", correction_action="delete" as long as the id/title matches
  the upcoming-events list -- see target_domain="event"'s own paragraph below for why "done" means DELETE there,
  not mark_done; but "correct both to 2026-09-23", "push day should be tomorrow", "actually dinner with Mel is
  Thursday not Wednesday" are target_domain="event", correction_action="reschedule" (NOT "delete" -- the event
  is moving to a different day, not going away -- see the CRITICAL warning below on this exact distinction)) --
  also "that Netflix subscription is actually $17 now", "move my gym membership billing to the 5th", "cancel my
  Spotify subscription" (target_domain="subscription"); "that bonus was actually $600, not $500", "that income
  entry should be tagged as a bonus, not salary", "delete that income entry, I logged it twice"
  (target_domain="income"); "that CPF top-up was $600 not $500", "wrong, that tax payment was yesterday"
  (target_domain="deduction") -- OR about the
  rolled-over balance/deficit itself rather than any one
  logged item (target_domain="balance", see its own paragraph below), OR about the recurring salary CONFIG
  itself -- "I got a raise, now making $7000", "I changed jobs, new salary is $8000 and CPF is 20% now" --
  rather than any one logged payment (target_domain="income_config", see its own paragraph below and the
  CRITICAL distinction from "log_income" there). First decide target_domain from context
  (an amount/currency strongly implies "expense"; food/calories implies "meal"; a workout activity implies
  "workout"; a to-do title/deadline implies "task"; an appointment/schedule framing implies "event"; a
  subscription/recurring-charge name implies "subscription"; a bonus/one-off payment already logged implies
  "income"; a tax/CPF-top-up already logged implies "deduction"; the words
  "balance", "rolled-over", or "deficit" with no specific item being referenced implies "balance"; a raise, new
  job, or salary/pay-day/CPF-rate change with no specific logged payment being referenced implies
  "income_config" -- when
  genuinely ambiguous between domains, prefer whichever domain has an item matching the description/date, and if
  more than one domain plausibly matches, use "clarification" instead). IMPORTANT exception for edit_date and
  its event-domain equivalent, reschedule (an event's date field is "reschedule", never "edit_date" -- see
  target_domain="event"'s own paragraph below -- but this exception applies to it the same way): a bare
  date-only correction with no other domain-identifying content at all (e.g. just "it's for 18 September",
  "that was last night", "for yesterday", no amount/food/workout words) is almost always fixing whatever the
  assistant's OWN immediately preceding reply in the conversation history just logged -- default target_domain
  to THAT domain and target THAT item's id, not a domain picked by searching recent lists for an item that
  already happens to have the mentioned date. That "matching date" search is for finding WHICH item the user
  means when several plausibly qualify (e.g. "the one from Monday" with two Monday expenses); it inverts badly
  when applied to the NEW date being given, where the whole point is that the item currently has the WRONG
  date -- an item that already carries the NEW date the user just said is exactly the item that does NOT need
  this correction. Falling back to a match against an unrelated domain's already-correctly-dated item instead
  of the domain that was just logged into is a real observed bug: it silently no-ops (the date doesn't change
  because it already matched) and leaves the actual wrong-dated item -- the one the user was clearly reacting
  to -- untouched.
  This is different from a message that names the CURRENT (soon-to-be-replaced) date directly, e.g. "Saturday
  morning is October 3 not October 5" naming an already-logged October 5 -- that IS a safe, strong, preferred
  match signal, not the no-op trap above: searching that domain's recent list for the item that currently
  carries the OLD value named ("not October 5") correctly identifies exactly the wrong item that needs fixing,
  precisely the opposite direction from matching on the NEW value. Prefer this whenever the message names an
  old/current value being corrected away from, and only fall back to "default to the assistant's last logged
  item" when no old value or other distinguishing detail is actually nameable.
  A further real observed bug this covers: when the assistant's immediately preceding reply logged MULTIPLE
  items at once in the same domain (e.g. several separate "add_event" legs from one itinerary message -- a
  flight-out date, a return date, a landing date, each its own row per the add_event rule above), "default to
  the assistant's last logged item" is not enough by itself, since there is no single "that item" -- picking
  an arbitrary one from the batch (the most recently inserted row, or a guess) instead of the specific one the
  message actually means is exactly how a correction like this can end up with a target_expense_id that isn't
  really the right item, or isn't in the recent list at all. In that case, match the message's own content (an
  old date it names, a day-of-week that corresponds to one of those items' own event_date, a distinguishing
  word like "the flight home" or "the SF leg") against the SPECIFIC item within that just-logged batch it
  actually refers to, the same as the general per-domain matching rule below -- treat the whole batch as the
  candidate set to search, not just its last row. Only use "clarification" and ask which one if the message
  truly gives nothing to disambiguate between them.
  Once target_domain
  is settled, identify the ONE matching item in
  that domain's recent list -- match on
  whatever the message gives you: amount/description, OR just a date/day reference alone (e.g. "yesterday's
  log", "the one from Monday") is enough on its own if exactly one recent item in that domain has that date,
  even with no amount or description mentioned. A bare number the user gives you (with or without a "#", "task",
  or "item" in front of it -- e.g. "17 done", "task 17", "#17", "mark 11 done", "complete 9", "task 17 renew
  esta visa done") is ALSO enough on its own, with no other corroboration needed, AS LONG AS it matches an "id"
  actually present in that domain's recent list -- to-dos and events in particular are always shown to the user
  with their id as a "#N" prefix (see /tasks and /events output), so a bare number referencing one is the NORMAL
  way a user names it, not a weak or ambiguous signal that needs a title/date match too (see the recent-lists
  note above on resolving a bare number that matches BOTH a task id and an event id). Set target_expense_id to
  its "id". Only
  "expense" targets
  support edit_currency/edit_amount/edit_description/edit_category -- for "vitals" targets, "edit_date",
  "edit_vitals" (a flexible weight/sleep/knee-pain/notes correction -- see below), and "delete" are
  supported; for "meal" targets, "edit_date", "edit_meal" (a flexible items/
  calories correction -- see below), and "delete" are supported; for "workout" targets, "edit_date",
  "edit_workout" (a flexible activity/duration/distance/calories-burned/notes correction -- see below), and
  "delete" are supported; for "lift" targets, "edit_date", "edit_lift" (a flexible sets/location/effort/notes
  correction -- see below), and "delete" are supported; for "task" targets, "mark_done", "edit_task" (a
  flexible title/due-date/notes edit -- see below), and "delete" are supported; for "event" targets,
  "reschedule" (move it to a new day -- see target_domain="event"'s own paragraph below) and "delete" are
  supported; for "subscription" targets, "edit_subscription" (a flexible name/amount/currency/billing-day/
  category correction -- see below) and "delete" are supported (no "edit_date" -- a subscription has no logged
  date, just a billing_day, which is part of edit_subscription); for "income" and "deduction" targets,
  "edit_date", their own flexible "edit_income"/"edit_deduction" correction (see below), and "delete" are
  supported.

  CRITICAL: never default to "edit_date" just because it's the first-listed or most familiar action -- only
  use "edit_date" when the message is actually about WHEN something happened (a day/date), never as a
  fallback guess for a request that's actually about some other field. A real observed bug this fixes:
  "correct the calories out to 2862" (a workout's calories_burned, before edit_workout existed to handle it
  properly) got misclassified as correction_action="edit_date" with no day mentioned, producing a nonsense
  "which day did you mean?" question completely unrelated to what was actually asked. If the user wants a
  field fixed that genuinely isn't in that domain's supported-actions list above (e.g. an event's title/notes
  have no dedicated edit action yet -- only "reschedule" and "delete" exist for events), set
  correction_action="edit_unsupported_field" instead of guessing at the closest-sounding valid action -- this produces an accurate "that kind of edit isn't supported yet" reply naming what IS
  supported, instead of a confusing off-topic question. If nothing in the matching list clearly matches, or
  more than one plausibly does, do NOT guess -- use "clarification" instead and ask the user to specify. If a
  single message describes MORE THAN ONE correction at once, do NOT fall back to "casual" just because it's
  compound -- pick whichever one is clearest/most specific and resolve that one as a normal "correction"; the
  user will follow up separately about the other one if your reply doesn't cover it.

  CRITICAL, separately from the above: never use "delete" as a stand-in for "the date/day is wrong and needs
  to change" -- a message asking to correct, fix, change, move, or reschedule WHEN something is happening
  means the item should be MOVED, not REMOVED, even if the domain's only other obvious-looking action is
  "delete". This is a real, actively harmful bug that already happened: "correct both to 2026-09-23" and "push
  day should be 2026-09-23" (two events that had landed on the wrong day) were misread as correction_action=
  "delete" and silently destroyed two real scheduled events instead of moving them -- the exact opposite of
  what was asked, and not reversible by the user without noticing and typing "undo" in time. For "event"
  targets specifically, "reschedule" (see below) is EXACTLY this case and must be used instead of "delete"
  whenever the message names or implies a different day the event should move to. If you are ever choosing
  between "delete" and some other action for a message that mentions a date, day, or "when" at all, treat that
  as a strong signal AWAY from "delete" -- re-read the message for whether it means "get rid of this" (delete
  is right) or "this is happening on a different day than logged" (it is not).
  target_domain="balance" (correction_action is always "adjust_balance") is its own case, unlike every other
  domain above: it has no recent-item list and no target_expense_id to match, because the rolled-over balance
  is one running number per chat, not a row. Use it when the user wants to correct the balance/deficit ITSELF
  -- typically because expense history from before some date is missing (lost/reset data), so the automatic
  day-by-day rollover has nothing to derive that period's real deficit from (e.g. "add this to my rolled over
  deficit -1135.89", "adjust my balance by -50", "my balance is off by 200, I actually overspent"). Put the
  amount to add in new_amount as a signed DELTA (negative to add a deficit, positive to add a credit) -- never
  an absolute replacement value, and never invent a number the user didn't give you. This is entirely distinct
  from an "edit_amount" correction on one expense (which fixes that one item's own amount) -- don't confuse the
  two. If the user's intent is ambiguous between "log a new expense" and "adjust the balance itself", ask via
  "clarification" rather than guessing (this mirrors an existing real interaction: "adjust my rolled-over
  balance" without a specific number needs the amount, not a guess).
  correction_action="edit_task" (target_domain="task" only) is a SINGLE flexible action covering an existing
  to-do's title, due date, and notes -- set whichever of new_description (the title), due_in_days/due_time (a
  NEW due date), remove_due_date (CLEARING the due date instead), and new_task_notes (a note) the message
  actually implies changing, in any combination, and leave the rest null. Examples: "push #11 to tomorrow" sets
  only due_in_days; "actually it's calling the vet, not the dentist" sets only new_description; "push #11 to
  tomorrow, I need Shardul's address" sets BOTH due_in_days AND new_task_notes in the same correction -- don't
  split an obviously-compound edit like that into two separate turns, or silently drop the note just because
  the due date was the more obvious change; "take the due date off #22, keep it open, I'll do it later" sets
  ONLY remove_due_date=true -- do NOT invent a due_in_days for this (there's no new date to compute), and do
  NOT use "delete" (the to-do itself isn't going away, just its date). A due date here is deliberately
  forward-looking, unlike edit_date's backward-only days_ago (which can't express "in 3 days") --
  due_in_days/due_time are the SAME forward day-count fields log_task uses for a brand-new to-do, just applied
  to an existing one. At least one of the four fields must actually be changing; if the message is about a
  to-do but it's unclear WHAT should change, use "clarification" and ask.
  correction_action="edit_meal" (target_domain="meal" only) corrects what was actually eaten/drunk in an
  already-logged meal, WITHOUT deleting and relogging it from scratch -- the case this exists for is a photo- or
  text-logged meal that came out wrong (e.g. an item that wasn't really eaten got included, a portion size was
  off, something eaten was left out). Examples: "minus the ramen noodles, I didn't have that" (an item removed),
  "I also had a side salad I forgot to mention" (an item added), "that was a large, not a medium" (a portion
  correction). Always set new_meal_items to the FULL corrected item list -- use the meal's current items (from
  the recent-meals list you were given) as the starting point, then add/remove/adjust exactly what the message
  says, and re-estimate new_meal_calories_low/new_meal_calories_high/new_meal_calories_estimate fresh for that
  corrected list, the same "plausible range, not false precision" discipline as log_meal -- always set all
  three together, never left partially null, whenever new_meal_items is set. Only set new_meal_type or
  new_meal_water_ml if the message specifically changes those too; leave them null otherwise (meaning
  unchanged). If the message is about a meal but it's unclear what actually changed, use "clarification" and
  ask what to fix, rather than guessing at what was really eaten.
  correction_action="edit_workout" (target_domain="workout" only) corrects a field on an already-logged
  workout, WITHOUT deleting and relogging it from scratch -- the case this exists for is a photo-read
  calories_burned that turns out wrong (e.g. it only captured one workout's burn, not the day's full total
  including BMR, and the user later saw the real number), or any other workout field (activity name, duration,
  distance, notes) needing a fix. Examples: "correct the calories out to 2862", "actually that run was 5.2km
  not 5", "that was cycling, not running". Set ONLY the field(s) the message actually changes
  (new_workout_activity/new_workout_duration_min/new_workout_distance_km/new_workout_calories_burned/
  new_workout_notes) -- leave the rest null, meaning unchanged; a message correcting just the calories burned
  sets ONLY new_workout_calories_burned, not the others. If the message is about a workout but it's unclear
  what actually changed, use "clarification" and ask what to fix.
  correction_action="edit_lift" (target_domain="lift" only) corrects a field on an already-logged lift,
  WITHOUT deleting and relogging the exercise from scratch -- the case this exists for is a mis-typed or
  incomplete set (a rep count wrong, a set left out entirely, an extra set that wasn't actually done).
  Examples: "the v bar rows were actually 10x8x1 and 12x8x2" (the corrected full set list), "I actually did 3
  sets of pull-ups not 2" (a set added), "drop the last set of curls, my form broke down and I don't want it
  logged" (a set removed). When new_lift_sets is set, it must be the FULL corrected set list, the same "whole
  picture, not a diff" discipline as new_meal_items -- use the lift's current sets (from the recent-lifts list
  you were given) as the starting point, then add/remove/adjust exactly what the message says; "load" stays a
  string exactly like log_lift's own sets field (a machine setting is just as valid as a real weight). Only set
  new_lift_exercise/new_lift_location/new_lift_effort/new_lift_context_notes if the message specifically
  changes those too; leave them null otherwise. If the message is about a lift but it's unclear what actually
  changed, use "clarification" and ask what to fix.
  correction_action="edit_vitals" (target_domain="vitals" only) corrects a field on an already-logged
  check-in, WITHOUT deleting and relogging it from scratch -- the case this exists for is a mis-typed
  weight/sleep/knee-pain number, or a note that needs fixing. Examples: "that was 76.0kg not 76.6" (a weight
  correction), "I actually slept 7 hours, not 5.5" (a sleep correction), "knee was more like a 4 today" (a
  knee-pain correction). Set ONLY the field(s) the message actually changes (new_vitals_weight_kg/
  new_vitals_sleep_hours/new_vitals_knee_pain/new_vitals_notes) -- leave the rest null, meaning unchanged; a
  message correcting just the weight sets ONLY new_vitals_weight_kg, not the others. If the message is about
  a check-in but it's unclear what actually changed, use "clarification" and ask what to fix.
  correction_action="edit_subscription" (target_domain="subscription" only) corrects the subscription's own
  CONFIG -- a price change, a billing-day move, a rename, a category change -- in place, WITHOUT touching any
  month that's already auto-posted as a real expense (that's a normal "expense" correction instead, if needed).
  Examples: "Netflix went up to $17.98" (new_subscription_amount), "move my gym billing to the 5th"
  (new_subscription_billing_day), "rename that to Netflix Premium" (new_subscription_name). Set ONLY the
  field(s) the message actually changes; leave the rest null. If the message is about a subscription but it's
  unclear what actually changed, use "clarification" and ask what to fix.
  correction_action="edit_income" (target_domain="income" only) corrects an already-logged income entry's
  source/description/amount/currency in place. Examples: "that bonus was actually $600, not $500"
  (new_income_amount), "that should be tagged as a bonus, not salary" (new_income_source). new_income_amount is
  always the corrected NET figure, same discipline as log_income. Set ONLY the field(s) the message actually
  changes; leave the rest null. If the message is about a logged income entry but it's unclear what actually
  changed, use "clarification" and ask what to fix.
  correction_action="edit_deduction" (target_domain="deduction" only) corrects an already-logged deduction's
  label/amount/currency in place. Examples: "that CPF top-up was $600 not $500" (new_deduction_amount), "that
  should be labeled as income tax" (new_deduction_label). Set ONLY the field(s) the message actually changes;
  leave the rest null. If the message is about a logged deduction but it's unclear what actually changed, use
  "clarification" and ask what to fix.
  correction_action="edit_income_config" (target_domain="income_config" only) is its own case, like
  target_domain="balance" above: it has no recent-item list and no target_expense_id to match, because the
  recurring salary setup is one config row per chat, not a row with an id. Use it when the user is describing a
  change to what gets auto-posted EVERY future payday -- a raise, a new job, a pay-day move, a changed CPF/stock
  rate (e.g. "I got a raise, now making $7000", "I changed jobs, new salary is $8000, pay day the 1st, CPF
  20%"). CRITICAL distinction from "log_income" (an intent, not a correction): "log_income" is for a payment
  that already landed ONE TIME ("got a $500 bonus today") -- it logs a new row and never touches income_config.
  "correction"/"edit_income_config" is for a change to the ONGOING recurring setup itself -- it never adds an
  income row, it only changes what future auto-posts will use. A message like "I got a raise" with a new number
  is about the ongoing salary, not a one-off payment that landed today, even though both involve the word
  "got" and a dollar figure -- the test is whether the message is reporting a payment that already arrived
  (log_income) or describing a change to a standing, repeating arrangement (edit_income_config). Set whichever
  of new_income_config_gross_amount/new_income_config_currency/new_income_config_pay_day/
  new_income_config_cpf_rate/new_income_config_stock_rate the message actually implies changing (CPF/stock
  rates as fractions, e.g. 0.2 for "20%"), and leave the rest null -- a raise with no mention of pay day or
  rates sets ONLY new_income_config_gross_amount. If the message is clearly about the recurring salary setup
  but gives no new number at all, use "clarification" and ask what the new gross salary is.
  target_domain="event" supports exactly two actions: "delete" and "reschedule" -- an event has no "done"
  state to set (see events.py's design: it's a flat, dated occurrence, not a to-do), so correction_action is
  NEVER "mark_done" for it. Phrasing like "the X-ray is done", "that appointment already happened", "cancel
  dinner with Mel", "that's not happening anymore, take it off my schedule" all mean REMOVE it -- use "delete".
  Phrasing like "correct that to the 23rd", "push day should be tomorrow, not today", "actually that's
  Thursday, not Wednesday", "move dinner with Mel to next Tuesday", or anything else naming or implying a
  DIFFERENT day the event should be on, means MOVE it -- use "reschedule" with new_event_in_days set to that
  day's offset from today (see new_event_in_days' own field doc above; never leave it null when a day is
  actually implied -- if the day genuinely isn't clear, use "clarification" and ask which day, never guess and
  never fall back to "delete"). See the CRITICAL warning above for the real, destructive bug this distinction
  exists to prevent -- when in doubt between these two, re-read for whether the event is going away entirely
  or just moving to a different day. Match target_expense_id against the upcoming-events list the same way as
  any other domain (title/date, or a bare id).
- "show_balance": the message is asking to see the current balance/target/streak right now (e.g. "show me my
  balance", "what's my balance", "how much do I have left today", "how am I doing today"). This is answered
  directly and immediately with real numbers -- it is NOT a "casual" reply pointing at the /balance command,
  because that command does exactly this and there's no reason to make the user type it separately. Only use
  this for TODAY's live balance; a request for a specific past day's balance (which isn't tracked historically)
  should be "casual", explaining that limitation. This intent is expense-only -- a broad "how am I doing"
  covering more than money is "rundown" instead (below), not "show_balance".
- "show_recent": the message is asking to see recently logged expenses (e.g. "show me today's log", "what
  have I logged recently", "show me my expenses"). Same reasoning as show_balance -- answer directly rather
  than pointing at /recent. Expense-only, same caveat as show_balance for meals/workouts.
- "show_tasks": the message is asking to see the to-do list / what's outstanding (e.g. "what's on my list",
  "what do I need to do", "show me my tasks", "what's still open"). Answered directly with the real open-tasks
  list, same discipline as show_balance/show_recent -- not a casual_reply guessing at what's on it.
- "show_reminders": the message is asking to see the standing DAILY reminders list specifically (e.g. "what are
  my daily reminders", "show me my reminders", "what do I have set to remind me every day"). Distinct from
  "show_tasks" (one-off to-dos) -- if the message is ambiguous between the two, prefer "show_tasks" unless the
  word "daily"/"reminder"/"every day" is actually used. Answered directly with the real reminders list, same
  discipline as show_tasks.
- "show_events": the message is asking to see the schedule / what's coming up (e.g. "what's on my schedule",
  "what do I have coming up", "am I free next Monday", "what's my workout plan for this week look like now").
  Distinct from "show_tasks" (to-dos to actively do) and "show_reminders" (daily habits) -- this is specifically
  about dated appointments/events already on the calendar. Answered directly with the real upcoming-events
  list, same discipline as show_tasks/show_reminders.
- "rundown": the message is asking for a broad status update spanning MORE THAN ONE domain, over MULTIPLE
  days -- money, food, training, and vitals together, trending over roughly the last week -- not a single
  domain's number and not one specific day (e.g. "how am I doing", "how's my week been", "give me a rundown",
  "how am I doing overall", "what's going on with me lately"). This pulls real 7-day figures across all four
  domains and synthesizes them in code, the same "real numbers in, never guessed" discipline as
  show_balance/show_recent -- so use it whenever the ask is genuinely cross-domain AND open-ended/multi-day
  about how things are going overall. If the message names or clearly implies ONE specific day instead (see
  "day_stats" below), use "day_stats" even if it's also asking across more than one domain (e.g. food +
  training together) -- "rundown" is specifically for the no-particular-day, trending-over-time case. A
  question clearly about just one domain within a single already-known item (a specific meal, a single
  workout, a vitals reading) is NOT a rundown either -- that's "casual", answered in-line from context, or
  points at the matching /recent-style command.
- "day_stats": the message is asking what happened, what was logged, what was eaten, or what the
  calorie/activity/vitals picture was for ONE specific day -- today, yesterday/last night, or a named
  weekday/calendar date, HOWEVER LONG AGO (e.g. "stats from yesterday", "what did I eat today", "show me the
  stats for Saturday", "calories for the 20th", "how'd I do last night food-wise", "what did I log on July
  15", "what happened on the 3rd of August", a bare "show me the stats again" or "*stats" following an
  earlier stats question in this conversation -- reuse whichever day was just being discussed). A generic
  "what did I log on <date>" is day_stats too, not just explicit calorie/activity wording -- it's asking what
  happened that day across meals/workouts/vitals, which is exactly what day_stats answers. This is answered
  from a REAL, freshly re-read database total for that exact day (meals in, workouts burned, net, vitals) --
  never estimated, re-summed, or "corrected" based on what was said earlier in the conversation, the same
  "real numbers in, never guessed" discipline as show_balance/rundown. There's no recency limit on this --
  the database keeps every dated row indefinitely (including backfilled history from before this chat), so a
  date from weeks or months back is just as answerable as yesterday; always compute day_stats_days_ago for
  the actual date asked rather than assuming old dates aren't covered. This exists specifically to replace a
  real observed failure mode: a single-day calorie question used to be answered as "casual", which had the
  model re-add up raw recent meals/workouts from context by itself every time it was asked, producing a
  different (sometimes sign-flipped, sometimes outright fabricated) answer on each successive ask -- and,
  separately, a specific-old-date question falling to "casual" produced a false claim that the bot doesn't
  store historical data at all, when it does. Prefer "day_stats" over "casual" for ANY question about a
  specific day's food/activity/calorie/vitals totals, no matter how long ago, even a vague follow-up like
  "show me again" -- never let that fall through to casual and get freehand-recalculated or falsely denied.
  Only fall back to "casual" for a single-day question that ISN'T about totals at all (e.g. "what did that
  dry mala taste like" isn't answerable from data and is just conversation).
- "trend": the message asks how a SINGLE metric has moved/progressed/changed over a RANGE of time longer than
  one day -- weight, sleep, knee pain, calories in, calories out, or spending (e.g. "summarise my weight
  progression since the beginning", "how's my sleep been trending this month", "has my knee pain gotten
  better or worse lately", "how has my spending trended since August", "am I eating more or less than I used
  to", "how's my weight looked over the last 3 months"). This is different from "day_stats" (one specific day)
  and "rundown" (the fixed default last-7-days cross-domain snapshot) -- "trend" is for an arbitrary,
  often-open-ended range on ONE metric, answered from REAL first/last/min/max/change figures freshly computed
  from the database, never estimated or recalled from earlier in the conversation, the same "real numbers in,
  never guessed" discipline as day_stats/rundown. This is the fix for a real observed failure: "summarise my
  weight progression since the beginning" used to be misread as a today-scoped question and answered with only
  today's vitals, ignoring the actual ask (the whole history). Any phrasing asking for a trajectory, pattern,
  progression, or "since X" / "over time" / "lately" / "has it gotten better/worse" on one of the six metrics
  above is "trend", not "casual" and not "day_stats". If the message names a specific lift (e.g. "how's my
  bench progressed") or otherwise doesn't map to one of the six covered metrics, use "casual" instead --
  lift-by-lift progression isn't covered by trend yet. If it's genuinely asking for everything across domains
  recently (not one metric, not a real range) that's still "rundown".
- "remember": the message explicitly asks you to remember, save, or note something durable for later -- a
  standing plan, a goal, a preference, a recurring fact (e.g. "remember I go to Fitness First Bugis Tue/Thu for
  legs and back", "my goal is 75kg by December", "remember I'm allergic to shellfish", "note that I prefer
  metric"). Also use this if the user is clearly correcting/updating something already in the memory list (in
  which case reuse that label rather than creating a near-duplicate -- see memory_label above). Don't use this
  for one-off facts that don't need to persist (e.g. "I'm heading to the gym now" is just casual/context, not
  something to save as a standing memory) -- only save things phrased as, or that clearly function as, standing
  information worth recalling in a future conversation.
  CRITICAL, same real bug as log_lift's own matching note above: never classify a message as "remember" just
  because Morrow's OWN preceding reply used the word "remember" in a question (e.g. "want me to remember your
  Visa pull day routine going forward?") -- a "yes" or a detailed answer to that question describing an actual
  session that was just done, with real sets/reps/weight, is "log_lift" (or "log_workout"/"log_vitals" for
  those domains), not "remember". Only classify as "remember" when the message itself is stating a standing
  routine/plan/preference in the abstract, with no concrete just-performed numbers attached to a specific
  session -- e.g. "remember that my Visa pull day is pull-ups, rows, and lat pulldowns" (a template for future
  sessions, nothing to log today) is "remember", but "yesterday I did pull-ups 10x3, v-bar rows 35kg 8x3 at
  Visa" (a real session that happened) is "log_lift" even if it's ALSO establishing what a "Visa pull day"
  means going forward. When genuinely both apply, prefer logging the real session this turn (see log_lift's
  note above).
- "forget": the message asks you to forget, delete, or remove something previously remembered (e.g. "forget
  the Bugis gym plan", "that goal doesn't apply anymore, drop it"). Match it to exactly one label in the memory
  list the same way a correction matches a recent item -- if nothing clearly matches, or more than one
  plausibly does, use "clarification" instead and ask which one.
- "show_memory": the message is asking to see everything currently remembered, as a real list (e.g. "what do
  you remember about me", "show me my saved stuff", "what have I told you to remember"). This is answered
  directly with the real memory list, the same discipline as show_balance/show_recent -- not a casual_reply
  guessing at what's saved. Don't use this when the user is instead asking about ONE specific thing they
  expect you to already know in context (e.g. "what's my gym plan again?") -- that's "casual", answered
  in-line using the memory content already given to you, the same way a person would just answer instead of
  pulling up a list.
- "casual": the message isn't about logging, correcting, checking balance/recent expenses, or remembering
  something durable (e.g. "hi", "thanks", small talk, catching up, venting, asking for advice, or a specific
  question you can answer directly from conversation history or memory content, like "what's my gym plan
  again?"). Write a warm "casual_reply" as the person's companion, not a command menu and not a clipped
  one-line acknowledgment -- let it read like an actual thoughtful reply from someone who's actually
  listening, the way a good back-and-forth with a person (or a genuinely attentive ChatGPT thread) reads,
  not like a bot economizing on words. A couple of sentences is a fine default; go longer whenever the
  moment actually calls for it -- they're thinking something through, asking for advice, venting, or
  catching up -- rather than reflexively trimming every reply down to the shortest possible acknowledgment.
  Don't pad for its own sake (a genuine "thanks" still just gets a genuine short reply), but when there's
  something real to engage with, engage with it -- react to the specific thing they said, add a thought or
  a follow-up where one actually fits, instead of just closing the loop. Thoughtful beats terse.
  Match the user's own register instead of defaulting to neutral-polite -- when they're casual or sweary, talk
  back the same way; that reads as an actual companion present in the conversation, not a bot performing
  politeness at them. Have a real opinion when the data in front of you actually supports one, and say it
  plainly instead of hedging into a question -- "that's a solid week" or "that's higher than your usual
  Tuesday" lands better than fishing for what they want to hear. It's fine to push back if something in their
  recent history or what they just said doesn't add up -- be specific about why, not just performatively
  contrarian -- but when they correct you or make a fair point back, concede it directly and immediately
  ("yeah, fair, I overcorrected there") rather than digging in or quietly dropping it. Reach into the
  recent-history/memory context you're given and actually use it unprompted when it's relevant -- naming a
  specific past instance ("that's about the same as last Tuesday's lunch") reads as someone paying attention,
  not a lookup. When they're venting about a setback, a plateau, or a bad stretch, offer real perspective
  instead of just sitting with it -- separate what actually changed from what only feels like it did (losing
  a week of momentum isn't the same as losing the underlying progress) rather than letting one bad stretch
  collapse into the other.
  Telegram renders **bold**, `backticks`, a fenced ``` block for monospace column alignment, and plain unicode
  arrows (up/down/right) for a trend -- reach for these ONLY when the reply is genuinely a comparison (e.g.
  "how did this week's workouts stack up against last week") and plain prose would actually lose the shape of
  it; an ordinary casual_reply stays plain conversational sentences, not a decorated report. Draw on
  conversation history and memory content naturally rather than re-asking for information you
  already have. If they ask about a capability the bot has (a summary, undoing something), point them at the
  real command instead of saying you can't help -- never deny something on the real command list above. If they
  ask for something the bot genuinely can't do (e.g. a specific past day's balance -- /balance only ever
  reflects today), say that plainly and suggest the closest real alternative (e.g. /summary for a spending
  trend over a period) instead of inventing a capability that doesn't exist. NEVER, in a casual_reply, claim OR
  PROMISE that you performed, edited, deleted, logged, remembered, or will look into/fix/note/save anything --
  not "I've removed it", not "I'll take care of that", not "noted, I'll remember that" -- casual_reply only
  talks and has no way to follow up later, so any phrasing implying action past, present, or future is
  misleading. Any actual data change must go through "log_expense"/"log_meal"/"log_workout"/"log_vitals"/
  "correction"/"remember"/"forget" instead, in the same turn -- never deferred to "casual" with a promise.
- "clarification": something important is missing or ambiguous to safely act on -- a log_expense with no
  amount, a log_meal that's too vague to estimate at all, a correction with an unclear target/domain, or a
  "forget" with an unclear target label. Ask ONE short, specific question.

logged_days_ago rule (log_expense, log_meal, log_workout, log_vitals, log_income, log_deduction -- whenever
something is being logged
NOW for something that happened on an EARLIER day, not corrected after the fact): 0 = today/tonight (same as
leaving it null -- today is the default), 1 = yesterday/last night, 2 = two days ago, etc., up to 14. Set it
whenever the message itself names or clearly implies a day other than today -- both a RELATIVE phrase
("last night I also had...", "yesterday's lunch was...", "this morning I did...") and an EXPLICIT calendar
date or weekday ("on 18 September", "on the 18th", "on Monday") -- compute the actual day-count for the
explicit case using "Today's actual date" given at the very top of this message; never guess or leave it null
just because the message named a date instead of a relative word. A real bug this fixes: "on 18 September" and
a photo captioned "Stats from 18 September" both got logged onto today anyway -- relative phrases like
"yesterday" happened to still work (no date arithmetic needed to know that's 1 day back), which is exactly why
the gap wasn't obvious: nothing in this prompt ever stated what today's actual date IS, so an explicit calendar
date had nothing to compute the offset against and silently fell back to today. Extract WHICH day as a plain
count of days back, exactly like correction's own "days_ago" below; never compute or output an actual calendar
date yourself, that's done in code. Leave it null (not 0) when the message doesn't say or imply anything about
timing -- most messages don't, and today is already the right default without it.

Rules for log_expense fields (apply per item in "expenses"):
- A bare currency SYMBOL with no letters (e.g. "$", "£") is ambiguous on its own -- default it to the base
  currency ({config.BASE_CURRENCY}) rather than assuming USD, unless the message also spells out an actual
  currency name/code (e.g. "USD", "US dollars", "20 dollars US"). Only set "currency" at all when a currency
  is explicitly named or unambiguously symbolized (e.g. "20 USD", "€15", "50 baht" -> THB); otherwise leave it
  null so it defaults to base currency. Don't ask a clarifying question about currency.
- Every item in "expenses" needs at least an amount. If the message is clearly a log attempt but any item's
  amount is missing or genuinely unclear, use "clarification" for the WHOLE message instead -- never emit a
  partial "expenses" list with some items logged and one silently skipped or guessed at.
- Only ask about category if it's genuinely ambiguous (e.g. "spent 50 at Target" could be groceries,
  shopping, or household). Obvious cases (e.g. "uber", "coffee", "netflix") should NOT need clarification.
- "Gifts & Occasions" is for money given or spent BECAUSE of a specific one-off social occasion -- wedding ang
  pows/hongbaos, birthday presents, baby showers, funeral wreaths, festive gifting. Prefer this over "Shopping"
  or "Other" whenever the occasion itself is the reason for the purchase, even if the item bought would
  normally be Food/Shopping/etc. "Hobbies & Collectibles" is for a recurring personal hobby/collecting habit --
  trading cards (Pokemon, sports, etc.), model kits, vinyl records, video games, craft supplies. Prefer this
  over "Entertainment" or "Shopping" for hobby-specific purchases, so the category actually distinguishes
  "irregular life event" spend from "recurring hobby" spend rather than burying both in a generic bucket.
- Only ask about is_claimable if the message gives no hint either way AND the amount is large enough that it
  plausibly could be a reimbursable/business expense (e.g. over 50 in base currency). Small everyday
  purchases default is_claimable to false without asking.
- Keep clarification_question short and conversational, and only ask about ONE thing at a time (prioritize:
  amount > claimable > category > correction target).

Rules for log_meal fields (apply per item in "meals"):
- Give your best reasonable estimate even from a short description -- never refuse to estimate just because
  detail is limited; a wider low/high range is the right response to uncertainty, not a clarifying question.
  Only use "clarification" for log_meal if the message is too vague to identify what was even eaten/drunk at
  all (e.g. just "logged food").
- calories_low/calories_high/calories_estimate should always be set together for each meal -- never leave
  calories_estimate null while giving a range, or vice versa.
- Don't merge distinct meals into one object just because they're in the same message, and don't stop after
  the first one -- see the "log_meal" intent rule above: every meal named gets its own object in "meals".

Rules for log_lift fields (apply per item in "lifts"):
- exercise stays in the person's own words -- "pull-ups", "v-bar row", "bench" -- never normalized or expanded
  to a different or more formal name.
- location is free text and only set if actually named; a message that just says "gym" with no specific
  location leaves it null rather than guessing which one.
- Each set actually described becomes its own object in "sets", in the order given -- "10, 10, 8" is three
  objects; "35kg x8x3" (a fixed weight repeated for 3 sets) is three objects all with load "35kg" and reps 8.
  load is always a string, holding either a real weight with its unit or a numbered-machine setting exactly as
  given -- never convert one into the other or invent a unit that wasn't stated.
- effort and context_notes are only set when the message actually says or clearly implies something about how
  it felt or what might explain the numbers -- never invented to fill the field.
- Don't merge distinct exercises into one object just because they're in the same message, and don't stop
  after the first one -- see the "log_lift" intent rule above: every exercise named gets its own object in
  "lifts".

Rules for log_task fields (apply per item in "tasks"):
- title should be a short, actionable phrase capturing what needs doing (e.g. "call the dentist", "submit
  the report"), not a full restated sentence.
- due_in_days is a plain count of days from today (0 = today, 1 = tomorrow, 2 = day after, etc.) -- unlike
  days_ago there's no cap, since due dates can be far in the future. Leave it null if no due date was
  mentioned for that item; never invent one.
- due_time is only set if a specific clock time was mentioned alongside that item's date (e.g. "by 5pm friday"
  -> "17:00"); leave it null otherwise, even if a due date was given.
- A numbered or bulleted list of to-dos in one message (e.g. 13 lines, one to-do per line) must become 13
  objects in "tasks" -- one per line -- never collapsed into a single item or truncated partway through the
  list. Each line's own wording (a name, a timeframe like "tonight"/"this week") stays with that line's object
  only; don't let one line's due date leak onto another's.

Rules for add_event fields (apply per item in "events"):
- event_title should be short and specific (e.g. "Dinner with Mel", "Pull day in the gym"), not a full sentence.
- event_in_days is a plain count of days from today (0 = today, 1 = tomorrow, 2 = day after, etc.) -- the same
  forward day-count discipline as log_task's due_in_days, and just as deliberately never an actual calendar
  date (that's computed in code). This is the one field that actually matters for an event -- if it genuinely
  can't be determined for an item, still include the object with event_in_days null (the bot will ask, or skip
  just that one item in a multi-item batch) rather than silently dropping the item.
- event_time is only set if a specific clock time was mentioned for that item; leave it null otherwise.
- A week's workout plan (or any multi-day schedule) named day by day in one message must become one object per
  day in "events" -- never collapsed into a single entry, and never let one day's activity or time leak onto
  another's.

Rules for remember/forget fields:
- memory_content should read as a standalone fact -- someone reading only that content later, with no other
  context, should understand it. Write it out plainly rather than echoing the user's shorthand.
- Prefer reusing an existing label over creating a near-duplicate. If the message is plausibly an update to
  something already remembered (same gym, same goal, same topic), treat it as "remember" with that label, not
  a brand-new one.
- For "forget", memory_label MUST come from the memory list you were given -- never guess a label that isn't
  there.
"""


def parse_message(text: str, recent_expenses: list | None = None, recent_meals: list | None = None,
                   recent_workouts: list | None = None, recent_vitals: list | None = None,
                   recent_tasks: list | None = None, recent_messages: list | None = None,
                   memory_list: list | None = None, recent_events: list | None = None,
                   recent_lifts: list | None = None, recent_subscriptions: list | None = None,
                   recent_income: list | None = None, recent_deductions: list | None = None) -> dict:
    """recent_expenses: list of {id, amount, currency, description, category,
    expense_date, is_claimable} dicts, most recent first -- typically the
    last ~8 for this chat. recent_meals / recent_workouts / recent_vitals:
    same idea, each domain's own recent items (see db.get_recent_meals /
    get_recent_workouts / get_recent_vitals). recent_tasks: list of
    {id, title, due_at} dicts for open (not-done) to-dos (see
    db.get_open_tasks) -- what a task correction (mark_done/delete) may
    target. recent_messages: list of {role, content} dicts, oldest first
    (see db.get_recent_messages) -- the rolling conversation history,
    short-term memory. memory_list: list of {label, category, content}
    dicts (see db.get_memory_list) -- durable facts/goals/plans, read in
    full so the model can use them without a separate retrieval step.
    recent_events: list of {id, event_title, event_date, event_time} dicts,
    soonest first (see db.get_upcoming_events) -- what an event correction
    (delete only -- see events.py's design) may target. Deliberately added
    as the LAST parameter with a default, appended after every other domain,
    rather than inserted alongside recent_tasks -- this is a real call-site
    change (see handlers.py), unlike reminders/events' add/show intents,
    which needed no new context list at all; the one-off cost of updating
    every fake_parse_message test double was worth paying here because
    without this list the model has no way to even know an id like "5"
    might refer to an event rather than a task, which was a real observed
    bug (a bare "5 is done" for an event confidently misfired against the
    task domain instead, with a confusing wrong-domain error reply).
    recent_lifts: list of {id, exercise, location, sets, lift_date} dicts,
    most recent first (see db.get_recent_lifts) -- what a lift correction
    (edit_date/delete) may target; also appended as the LAST parameter, same
    reasoning as recent_events above.
    recent_subscriptions: list of {id, name, amount, currency, billing_day,
    category} dicts (see db.get_subscriptions) -- what a subscription
    correction (edit_subscription/delete) may target. recent_income: list of
    {id, source, description, net_amount, currency, income_date} dicts, most
    recent first (see db.get_recent_income) -- what an income correction
    (edit_date/edit_income/delete) may target. recent_deductions: same idea
    for deductions (see db.get_recent_deductions). All three appended as the
    LAST parameters, same "real call-site change, worth paying the one-off
    test-double update cost" reasoning as recent_events/recent_lifts above.
    Never raises -- if the Claude call itself fails (auth, rate limit,
    network blip, etc.), falls back to a clarification response so the bot
    always replies to the user instead of going silent.
    """
    recent_expenses = recent_expenses or []
    recent_meals = recent_meals or []
    recent_workouts = recent_workouts or []
    recent_vitals = recent_vitals or []
    recent_tasks = recent_tasks or []
    recent_messages = recent_messages or []
    memory_list = memory_list or []
    recent_events = recent_events or []
    recent_lifts = recent_lifts or []
    recent_subscriptions = recent_subscriptions or []
    recent_income = recent_income or []
    recent_deductions = recent_deductions or []
    user_content = (
        f"{_today_context()}\n\n"
        f"Recent expenses (most recent first, only reference an id from here for target_domain=expense):\n"
        f"{json.dumps(recent_expenses)}\n\n"
        f"Recent meals (most recent first, only reference an id from here for target_domain=meal):\n"
        f"{json.dumps(recent_meals)}\n\n"
        f"Recent workouts (most recent first, only reference an id from here for target_domain=workout):\n"
        f"{json.dumps(recent_workouts)}\n\n"
        f"Recent lifts (most recent first, only reference an id from here for target_domain=lift):\n"
        f"{json.dumps(recent_lifts)}\n\n"
        f"Recent vitals check-ins (most recent first, only reference an id from here for target_domain=vitals):\n"
        f"{json.dumps(recent_vitals)}\n\n"
        f"Open to-dos (soonest due first, only reference an id from here for target_domain=task):\n"
        f"{json.dumps(recent_tasks)}\n\n"
        f"Upcoming scheduled events (soonest first, only reference an id from here for target_domain=event):\n"
        f"{json.dumps(recent_events)}\n\n"
        f"Active subscriptions (only reference an id from here for target_domain=subscription):\n"
        f"{json.dumps(recent_subscriptions)}\n\n"
        f"Recent income entries (most recent first, only reference an id from here for target_domain=income):\n"
        f"{json.dumps(recent_income)}\n\n"
        f"Recent deductions (most recent first, only reference an id from here for target_domain=deduction):\n"
        f"{json.dumps(recent_deductions)}\n\n"
        f"Recent conversation history (oldest first):\n"
        f"{json.dumps(recent_messages)}\n\n"
        f"Durable memory (a label, category, and content per item -- only source of remembered facts, only "
        f"place a 'forget' label may come from):\n"
        f"{json.dumps(memory_list)}\n\n"
        f"Message: {text}"
    )
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            # 450 was enough for a single expense/meal/task, but expenses/meals/tasks are
            # now always lists -- a message naming a dozen-plus separate to-dos (a real
            # observed failure: 13 to-dos in one message only produced one, title-less
            # entry) needs real room to emit that many objects without truncating into
            # invalid JSON and silently falling back to a single generic item.
            max_tokens=1500,
            system=PARSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = resp.content[0].text.strip()
        return _safe_json(raw)
    except Exception:
        logger.exception("parse_message: Claude call failed, falling back to a clarification reply")
        return _clarify_fallback(
            "Sorry, I'm having trouble reaching my brain right now -- mind trying again in a "
            "moment, or use /log 12.50 lunch instead?"
        )


def categorize(description: str) -> str:
    """Never raises -- defaults to 'Other' if the Claude call fails."""
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=20,
            system=(
                f"Reply with exactly one word from this list, nothing else: {CATEGORY_LIST}. "
                "Pick the best fit for the purchase described."
            ),
            messages=[{"role": "user", "content": description or "unknown purchase"}],
        )
        text = resp.content[0].text.strip()
        return text if text in config.CATEGORIES else "Other"
    except Exception:
        logger.exception("categorize: Claude call failed, defaulting to 'Other'")
        return "Other"


MEAL_ESTIMATE_SYSTEM_PROMPT = f"""You estimate calories for a described meal, the same way an attentive \
nutrition-tracking assistant would: a plausible range, not false precision. Use the description to refine \
quantities. Reply with ONLY a JSON object: {{"meal_type": one of [{MEAL_TYPE_LIST}] or null if it doesn't fit a \
slot (e.g. a drink or snack between meals), "items": [list of individual food/drink items as short strings], \
"calories_low": number, "calories_high": number, "calories_estimate": number (the central estimate, roughly the \
midpoint), "water_ml": number or null (ONLY for plain water -- never other drinks; null if no water is \
mentioned)}}. Give your best reasonable estimate even with limited detail -- never omit calories_low/high/estimate. \
{MEAL_CALORIE_CALIBRATION_NOTE}"""

WORKOUT_EXTRACT_SYSTEM_PROMPT = """You extract structured fields from a described workout. Reply with ONLY a \
JSON object: {"activity": short activity name e.g. "tennis", "IPPT training", "gym", "run", "duration_min": \
number or null if not mentioned, "distance_km": number or null, "calories_burned": number or null if not \
mentioned, "notes": a short note capturing any detail worth keeping (splits, sets, how it felt) or null}. Give \
your best reasonable interpretation even from a short description."""

PHOTO_CLASSIFY_SYSTEM_PROMPT = f"""You look at a photo sent to a health-tracking bot and figure out what it \
actually shows, then extract structured data for it. Reply with ONLY a JSON object, matching exactly ONE of \
these three shapes:

1. An actual food or drink photo -- a plate, bowl, glass, or container of something to eat or drink, judged by \
visible portion sizes (refine with the caption if one is given):
{{"kind": "meal", "meal_type": one of [{MEAL_TYPE_LIST}] or null if it doesn't fit a slot, "items": [list of \
individual food/drink items as short strings], "calories_low": number, "calories_high": number, \
"calories_estimate": number, "water_ml": number or null (ONLY for plain water), "caption_extra_item": short \
string or null, "logged_days_ago": integer or null}}
"items"/calories/water_ml here must describe ONLY the food actually visible in the photo -- never fold in \
something the caption mentions that isn't shown. "caption_extra_item" is the escape hatch for that: set it ONLY \
when the caption clearly names a DIFFERENT, separate food that ISN'T what's shown in the photo (e.g. a caption \
"I also had a small bowl of black bean pork broth" sent with a photo that actually shows nachos -- two distinct \
foods, not one dish to average together). Leave it null when the caption is just adding detail about the SAME \
food shown (portion size, ingredients, what it's called), when there's no caption, or when you genuinely can't \
tell. This exists so the bot can ask which food(s) the user actually wants logged instead of silently merging \
two different foods into one wrong, over-estimated entry.
When the caption explicitly and specifically names the dish (e.g. "medium curry gyu don from Sukiya", not just \
"lunch" or "food"), treat that naming as the AUTHORITATIVE description of what's on the plate -- "items" should \
list what the caption actually says was eaten, refined by what the photo shows (portion size, visible \
garnishes), not padded out with extra components you infer are commonly served alongside that dish but can't \
actually confirm are in THIS photo. A restaurant dish's typical sides/plating are easy to imagine from menu \
knowledge and easy to mis-see in a photo -- guessing one in is a worse error than leaving it out, since the \
user explicitly told you what they ate and an uninvited extra item silently inflates their calorie log. If \
you're not confident an item is really there, leave it out rather than including it "to be safe".

2. A fitness/workout stats screen -- a summary from a fitness app or wearable showing calories burned, active \
minutes, steps, distance, a named workout, and similar (this is NOT a photo of food, even if calories are on \
screen):
{{"kind": "workout", "activity": short name (e.g. "run", "cycling"; use "daily activity" for a general \
activity/calorie summary rather than one named workout), "duration_min": number or null, "distance_km": number \
or null, "calories_burned": number or null, "notes": short string for any other detail worth keeping (steps, \
heart rate, active minutes) or null, "logged_days_ago": integer or null}}
On an Apple Health/Fitness-style activity-rings screen specifically, at least two DIFFERENT calorie figures are \
typically visible, and they are not interchangeable. The "Move" ring's own headline number (e.g. "772/700KCAL", \
printed large and in color right under the word "Move") is that ring's active/exercise calories against its \
own personal goal -- it EXCLUDES resting/BMR burn, so it is NOT the day's total energy expenditure. Separately, \
usually printed in small plain text just below the Move ring's chart, is a line like "TOTAL 2,762 KCAL" -- THIS \
is the real total calories burned for the day (active + resting), and calories_burned must be set from THIS \
figure whenever it's visible on screen, even though it's much smaller and less visually prominent than the \
Move ring's own bold number. A real observed bug this fixes: calories_burned kept getting set to the Move \
ring's headline figure (the biggest, boldest number on the whole screen) while a "TOTAL" line sitting right \
below it, showing the real day's total, was overlooked entirely -- actively scan for a distinct "TOTAL ... \
KCAL" line near the Move chart and prefer it over the ring's own number. Only fall back to the Move ring's \
figure if no such TOTAL line is visible anywhere on screen at all.

3. Anything else -- a receipt, a document, an unrelated photo, or an image with no food or fitness data in it \
at all:
{{"kind": "unclear"}}

Use the caption for extra detail if one is given. For kind "meal" always give calories_low/high/estimate, never \
omit them. {MEAL_CALORIE_CALIBRATION_NOTE} For kind "workout" extract whatever fields ARE actually visible on screen -- \
leave the rest null rather than guessing at numbers that aren't shown. Never invent a meal or workout that \
isn't actually shown in the photo -- that produces a nonsense logged entry, which is worse than asking.

"logged_days_ago" (both "meal" and "workout") is how many days ago the food/activity actually happened, as a \
plain integer count -- 0 = today, 1 = yesterday, 2 = two days ago, etc. -- ONLY set this when the caption \
clearly states or implies a specific past day, whether a RELATIVE phrase ("this was yesterday's lunch", "from \
Tuesday's workout") or an EXPLICIT calendar date ("these were my stats for 15 September", "stats from the \
18th"); for the explicit case, compute the actual day-count using "Today's actual date", stated at the very \
start of the message alongside the photo(s) -- never guess or leave it null just because a date was named \
instead of a relative word. Leave it null when the caption doesn't mention a day at all, since the photo was \
just sent and defaults to today. Extract ONLY the day-count -- NEVER compute or output an \
actual calendar date yourself; the bot converts your count into a real date deterministically, the same \
discipline used for every other date field in this app. Cap it at 14 (two weeks) -- if the caption implies \
something older than that, leave logged_days_ago null instead of guessing a huge number.

One exception to "leave it null with no caption": for a "workout" daily-activity/fitness-app summary screen \
specifically (not a single named workout), check the current local time given alongside "Today's actual \
date" -- if it's shortly after local midnight (roughly within the first hour or so) and the screen shows a \
substantial day's worth of totals (meaningful active minutes, calories burned, or distance -- not just a \
couple of stray steps), that data physically cannot have accumulated in the few minutes since the calendar \
flipped, even with no caption saying so -- it's still showing the day that just ended. In that specific \
situation, set logged_days_ago to 1 rather than leaving it null. This does NOT apply to a real named workout \
(e.g. "tennis") or to a photo sent well after midnight, where defaulting to today is still correct -- only to \
a cumulative daily-totals screen whose numbers are the tell."""


def _meal_fallback(seed_text: str | None) -> dict:
    return {
        "meal_type": None,
        "items": [seed_text] if seed_text else [],
        "calories_low": None,
        "calories_high": None,
        "calories_estimate": None,
        "water_ml": None,
    }


def extract_meal(description: str) -> dict:
    """Used by /logmeal for a known description -- same estimate style as
    parse_message's log_meal fields. Never raises -- falls back to an
    all-null estimate (item kept as the raw text) rather than blocking the
    log if the Claude call fails."""
    fallback = _meal_fallback(description)
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=250,
            system=MEAL_ESTIMATE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": description or "unknown meal"}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_json_or_none(raw)
        return data if data else fallback
    except Exception:
        logger.exception("extract_meal: Claude call failed, falling back to a null estimate")
        return fallback


def extract_from_photo(image_bytes: bytes, caption: str | None = None) -> dict:
    """Vision-based classify-and-extract for any photo sent to the bot (see
    handle_photo), optionally with a caption for extra detail. A photo can
    show three different things, and each needs a different downstream
    action, so the model reports which one it actually is rather than
    handle_photo just assuming every photo is a meal:
    - real food/drink -> {"kind": "meal", ...} logged as a meal. A meal's
      "caption_extra_item" is set only when the caption clearly names a food
      that ISN'T what's in the photo (e.g. photo shows nachos, caption adds
      "I also had some broth") -- a real bug this guards against: that
      caption's separate food used to get folded straight into the photo's
      item list, producing one wrong entry with both foods' calories mashed
      together. handle_photo checks for this and asks which food(s) the
      user actually wants logged instead of committing to that guess.
    - a fitness app/wearable's calorie-burned or workout-stats screen ->
      {"kind": "workout", ...}, calories_burned included so it can be shown
      against today's calories eaten
    - anything else (a receipt, a random photo, ...) -> {"kind": "unclear"}

    Both "meal" and "workout" also carry "logged_days_ago" -- a plain day
    count (never an actual date, same discipline as every other date field
    in this app), set only when the caption names a specific past day (e.g.
    "these were my stats for 15 September") so handle_photo can log it on
    the right date immediately instead of defaulting to today and needing a
    follow-up correction.

    This replaces an earlier, narrower version that only ever asked "is this
    food, yes or no" -- a real bug from that version: someone's screenshot of
    their FITNESS app's daily calorie-burned stats got rejected outright
    ("not food") instead of being recognized for what it actually was and
    logged as a workout. Never raises, and falls back to {"kind": "unclear"}
    rather than guessing wildly if the vision call fails or the model's reply
    doesn't parse into one of the three known shapes. Telegram's "photo"
    message type always transcodes to JPEG, so media_type is fixed.

    Only ever called with ONE image -- see extract_from_photos for more than
    one sent together in a single message (a Telegram "album")."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    user_content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
        {"type": "text", "text": f"{_today_context()} It's currently {_current_local_time_str()}. "
                                  + (caption or "What does this photo show?")},
    ]
    return _classify_photo_content(user_content, "extract_from_photo")


# Telegram delivers an "album" (several photos sent together in one message)
# as one Update per photo, each carrying the same media_group_id but usually
# only ONE of them carrying the caption -- see nutrition.handle_photo's
# album-buffering for how those get collected into a single call here. This
# note is appended only for that multi-image call, not extract_from_photo's
# single-image one, since it doesn't apply there.
MULTI_PHOTO_NOTE = """

IMPORTANT: more than one photo was sent together in a single message, so treat them as describing ONE \
underlying meal or workout, not several -- almost always several screenshots of the SAME fitness app/health \
screen (e.g. one screen's activity-rings detail plus another screen that separately restates part of the \
same day's totals, like just the Move ring's own contribution to a total shown in full elsewhere), or several \
angles/parts of the SAME plate of food, not two unrelated days or two unrelated meals. A real bug this fixes: \
two screenshots of the same day's activity data, analyzed one at a time, got logged as two separate workouts \
-- each read the numbers actually in front of it correctly, but couldn't tell it was looking at data another \
photo in the same message already covered, so the same calories-burned total effectively got counted twice.

Read every image together and synthesize ONE combined entry from whichever numbers are the most complete and \
authoritative across all of them -- e.g. if one image shows a day's TOTAL calories burned and another shows a \
narrower sub-total that's already part of that total, use the total; never add the two together or report \
both as if they were separate. Only return "kind": "unclear" if the photos genuinely don't add up to one \
coherent meal or workout at all. In the rare case they truly show two unrelated days or two unrelated meals, \
extract whichever one the caption is actually about (or the more complete one if the caption doesn't say) --
only one combined entry can be logged from this call."""


def extract_from_photos(images: list[bytes], caption: str | None = None) -> dict:
    """Same classify-and-extract as extract_from_photo, for when more than
    one photo arrives in a single Telegram message (an "album" -- see
    nutrition.handle_photo's module docstring for the buffering that
    collects them before this is ever called). ALL images are sent to
    Claude in ONE multimodal call so it can reason about them together --
    see MULTI_PHOTO_NOTE for exactly why that matters and the real bug it
    fixes; a single-image album (by far the common case) never reaches this
    function at all, extract_from_photo handles it unchanged."""
    user_content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                      "data": base64.b64encode(img).decode("ascii")}}
        for img in images
    ]
    user_content.append({
        "type": "text",
        "text": (f"{_today_context()} It's currently {_current_local_time_str()}. "
                 f"({len(images)} photos sent together in one message.) "
                 + (caption or "What do these show?")),
    })
    return _classify_photo_content(user_content, "extract_from_photos", extra_system_note=MULTI_PHOTO_NOTE)


def _classify_photo_content(user_content: list, caller_name: str, extra_system_note: str = "") -> dict:
    """Shared Claude-call/parse/fallback plumbing for extract_from_photo and
    extract_from_photos -- one image or several, the request/response shape
    and failure handling are identical; only the content blocks differ."""
    fallback = {"kind": "unclear"}
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=300,
            system=PHOTO_CLASSIFY_SYSTEM_PROMPT + extra_system_note,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_json_or_none(raw)
        if not data or data.get("kind") not in ("meal", "workout", "unclear"):
            return fallback
        return data
    except Exception:
        logger.exception("%s: Claude vision call failed, falling back to 'unclear'", caller_name)
        return fallback


def extract_workout(description: str) -> dict:
    """Used by /logworkout for a known description. Never raises -- falls
    back to the raw text as both activity and notes, with no duration/
    distance, rather than blocking the log if the Claude call fails."""
    fallback = {"activity": description or "workout", "duration_min": None,
                "distance_km": None, "notes": description}
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=150,
            system=WORKOUT_EXTRACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": description or "workout"}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_json_or_none(raw)
        return data if data else fallback
    except Exception:
        logger.exception("extract_workout: Claude call failed, falling back to the raw description")
        return fallback


LIFT_EXTRACT_SYSTEM_PROMPT = """You extract one structured gym exercise from a short description -- sets, \
reps, and weight or machine setting, for a specific lift (not a whole session; one call is always one exercise). \
Reply with ONLY a JSON object: {"exercise": short name using the person's own wording (e.g. "pull-ups", "bench \
press", "v-bar row"), "location": short gym/location name if mentioned (e.g. "AF Wheelock", "office gym", \
"Capella") or null if not said, "sets": [list of {"reps": integer or null, "load": string or null -- a weight \
with unit ("35kg", "80 lb") OR a machine setting exactly as given ("setting 21") OR null if genuinely not \
stated for that set -- always a string, never force it into a bare number}], one object per set actually \
described in the order given (e.g. "10, 10, 8" is three set objects with reps 10/10/8), "effort": string or \
null (how hard it felt / proximity to failure, ONLY if actually said, e.g. "felt strong", "grindy last set", \
"2 reps in reserve", "to failure" -- never invent one), "context_notes": string or null (anything else worth \
keeping that could explain the numbers later, e.g. no warm-up set, trained again the next day)}. Give your best \
reasonable interpretation even from a short description; never omit "sets" -- if no rep/set detail was given at \
all, still return one set object with reps and load both null rather than an empty list."""


def extract_lift(description: str) -> dict:
    """Used by /loglift for a known description -- always ONE exercise per
    call (mirrors extract_workout/extract_task's singular shape; several
    exercises in one message is a natural-language-only capability, via
    parse_message's "lifts" list, the same split as log_task's singular
    /addtask vs its multi-item natural-language path). Never raises --
    falls back to the raw text as the exercise name with one null set,
    rather than blocking the log if the Claude call fails."""
    fallback = {"exercise": description or "exercise", "location": None,
                "sets": [{"reps": None, "load": None}], "effort": None, "context_notes": None}
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=250,
            system=LIFT_EXTRACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": description or "exercise"}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_json_or_none(raw)
        return data if data else fallback
    except Exception:
        logger.exception("extract_lift: Claude call failed, falling back to the raw description")
        return fallback


VITALS_EXTRACT_SYSTEM_PROMPT = """You extract a daily check-in report from a short message. Reply with ONLY \
a JSON object: {"weight_kg": number or null, "sleep_hours": number or null, "knee_pain": number or null (a \
0-10 scale, only if mentioned), "notes": a short string capturing anything else worth keeping, or null}. Only \
set fields actually mentioned or clearly implied -- never guess a value that wasn't given."""


def extract_vitals(description: str) -> dict:
    """Used by /logvitals for a known description. Never raises -- falls
    back to an all-null reading with the raw text kept as notes, rather than
    blocking the log if the Claude call fails."""
    fallback = {"weight_kg": None, "sleep_hours": None, "knee_pain": None, "notes": description}
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=150,
            system=VITALS_EXTRACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": description or "check-in"}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_json_or_none(raw)
        return data if data else fallback
    except Exception:
        logger.exception("extract_vitals: Claude call failed, falling back to the raw description")
        return fallback


TASK_EXTRACT_SYSTEM_PROMPT = """You extract a to-do/reminder from a short description. Reply with ONLY a \
JSON object: {"title": short actionable phrase capturing what needs doing, "due_in_days": integer count of \
days from today (0 = today, 1 = tomorrow, 2 = day after, etc.) or null if no due date was mentioned, \
"due_time": "HH:MM" 24-hour time ONLY if a specific clock time was mentioned alongside the date (e.g. "by 5pm \
friday" -> "17:00"), else null, "notes": a short string capturing anything else worth keeping, or null}. Never \
invent a due date or time that wasn't mentioned. The message you're given starts with "Today's actual date is \
..." -- use it to compute due_in_days for an explicit calendar date or weekday ("due the 25th", "by next \
Wednesday"), not just a relative phrase like "tomorrow"; never guess or leave it null just because the date \
was named explicitly rather than said relatively."""


def extract_task(description: str) -> dict:
    """Used by /addtask for a known description -- same division of labor as
    extract_workout/extract_vitals. Never raises -- falls back to the raw
    text as the title with no due date, rather than blocking the add if the
    Claude call fails."""
    fallback = {"title": description or "to-do", "due_in_days": None, "due_time": None, "notes": None}
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=150,
            system=TASK_EXTRACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{_today_context()} {description or 'to-do'}"}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_json_or_none(raw)
        return data if data else fallback
    except Exception:
        logger.exception("extract_task: Claude call failed, falling back to the raw description")
        return fallback


EVENT_EXTRACT_SYSTEM_PROMPT = """You extract a scheduled one-off event/appointment from a short description. \
Reply with ONLY a JSON object: {"title": short specific description of what's happening (e.g. "Dinner with \
Mel", "Dentist appointment"), "event_in_days": integer count of days from today (0 = today, 1 = tomorrow, 2 = \
day after, etc.) or null if no day could be determined, "event_time": "HH:MM" 24-hour time ONLY if a specific \
clock time was mentioned (e.g. "at 3pm" -> "15:00"), else null, "notes": a short string capturing anything else \
worth keeping, or null}. Never invent a day or time that wasn't mentioned. The message you're given starts \
with "Today's actual date is ..." -- use it to compute event_in_days for an explicit calendar date or weekday \
("dinner on the 25th", "next Wednesday"), not just a relative phrase like "tomorrow"; never guess or leave it \
null just because the date was named explicitly rather than said relatively."""


def extract_event(description: str) -> dict:
    """Used by /addevent for a known description -- same division of labor as
    extract_task. Never raises -- falls back to the raw text as the title
    with no day (the caller then asks what day it's on) rather than
    blocking the add if the Claude call fails."""
    fallback = {"title": description or "event", "event_in_days": None, "event_time": None, "notes": None}
    try:
        client = _get_client()
        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=150,
            system=EVENT_EXTRACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{_today_context()} {description or 'event'}"}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_json_or_none(raw)
        return data if data else fallback
    except Exception:
        logger.exception("extract_event: Claude call failed, falling back to the raw description")
        return fallback


# Shared by every prompt below that writes a freeform narrative for Telegram (this one,
# TRENDS_SYSTEM_PROMPT, and answer_with_rundown's own prompt -- casual_reply gets the same
# convention in its own words above, since its tone is personal rather than analytical) --
# one place so the convention can't drift between them. Telegram has no real table support in
# any client, so a genuine comparison ("this week vs last week", a few categories side by side)
# can't be a markdown table -- it would just render as a wall of pipes and dashes.
TELEGRAM_FORMATTING_NOTE = (
    "Formatting for Telegram: never write a markdown table (pipes/dashes) -- it renders as a wall of "
    "punctuation, not a table. Instead use **bold** for a word or figure worth emphasizing, `backticks` to "
    "call out one short value inline, and a fenced ``` block ONLY when a comparison genuinely needs "
    "monospace column alignment (e.g. two or three rows of this-period-vs-last-period numbers) -- most "
    "replies don't need one. Plain unicode arrows (↑ for an increase, ↓ for a decrease, → for roughly flat) "
    "are welcome in place of writing \"increased\"/\"decreased\" every time. Use all of this sparingly -- it "
    "should read like a person highlighting the one thing that matters, not a decorated report. When you "
    "want a line break, write an actual line break in your output -- never the two literal characters "
    "backslash-n as a stand-in for one, even if you're echoing or closely paraphrasing something from the "
    "conversation history above that happens to show that literal sequence (that's just how a real newline "
    "looks once it's been serialized into the JSON you were handed, not a literal string the reply should "
    "reproduce)."
)


def answer_with_data(question: str, rows: list) -> str:
    """rows: list of {category, total, n} dicts. Returns a short natural-language summary."""
    client = _get_client()
    data_str = json.dumps(rows)
    resp = client.messages.create(
        model=config.CLAUDE_NARRATION_MODEL,
        max_tokens=400,
        system=(
            "You are a personal finance assistant. You're given category spending totals as JSON "
            "and a question from the user. Answer concisely, suitable for a Telegram message -- a couple "
            f"of short lines plus a one-line takeaway is ideal. {TELEGRAM_FORMATTING_NOTE}"
        ),
        messages=[{"role": "user", "content": f"Data: {data_str}\n\nQuestion: {question}"}],
    )
    return resp.content[0].text.strip()


TRENDS_SYSTEM_PROMPT = """You are a thoughtful personal financial advisor writing a short, on-demand \
spending summary for Telegram -- not a category-totals report. The failure mode to avoid is the \
generic "you spent more on Food/Other, spend less" summary that's technically true but not useful, \
because it can't tell a one-off from a genuine pattern.

You're given, for the current period vs. the prior equal-length period:
- current_period / previous_period: total spend and a per-category breakdown.
- category_insights: the SAME per-category totals, but each paired with typical_per_period (the real \
average for that category over the several periods before this one -- null means no history yet, \
not zero) and vs_typical_ratio (this period's total divided by that average -- null when there's no \
baseline to compare against). This is the actual signal for "is this normal for you" -- use the ratio, \
don't estimate one yourself.
- top_transactions: the largest individual purchases this period (amount, description, category, date) \
-- often the real story behind a category total, not the aggregate itself.
- avg_spend_by_weekday, budget_adherence, month_to_date, streak: as available.

How to reason about it, concretely:
- A big vs_typical_ratio (say 3x+) is worth naming specifically -- cite the actual multiple and the real \
typical amount, not just "higher than usual."
- Before calling a spike concerning, check top_transactions for what's actually driving it. If one or two \
large purchases explain most of a category's total, say so by name (amount + description), don't just \
report the category sum.
- "Gifts & Occasions" spend is inherently one-off by nature (weddings, ang pows, baby showers) -- a spike \
there from a specific identifiable event (several similar-sized transactions, or one clearly tied to an \
occasion) is NOT something to flag as a spending problem. Say plainly that it's an unpredictable, \
one-off category and the underlying budget looks fine without it, rather than treating it as overspend.
- "Hobbies & Collectibles" spend is a recurring personal habit by nature -- a real multiple over its own \
typical_per_period (not the category's absolute size) IS worth flagging directly and specifically, \
without moralizing: state the multiple, the typical amount, and that it's worth a look if the person's \
financial situation hasn't changed, then stop -- their call, not a lecture.
- Every other category: judge by vs_typical_ratio the same way -- a big one-off transaction in an \
otherwise-typical category (e.g. one large "Shopping" purchase) reads as a one-off unless the category \
total itself is now well above typical_per_period too.
- If nothing is a genuine standout (every ratio is close to 1, or there isn't enough history yet), say so \
briefly and warmly -- don't invent a concern to fill space.

Write 3-7 short plain-text lines: the total spend and how it compares to the prior period, the standout \
worth naming (by the reasoning above -- category, ratio, and the actual transaction if that's the real \
driver), at most one more secondary note (a day-of-week pattern only if genuinely notable, or budget \
adherence), and a one-line closing take. Plain conversational lines by default, like a person who actually \
looked at the numbers, not a template or bullet list. Be matter-of-fact and specific (name real amounts and \
multiples), not alarmist, not vague, and never moralizing beyond one plain sentence when something really is \
worth a second look.

""" + TELEGRAM_FORMATTING_NOTE


def answer_with_trends(period: str, payload: dict) -> str:
    """payload holds this period's category totals (plus category_insights --
    the same totals paired with a real historical baseline/ratio -- and
    top_transactions, the largest individual purchases this period), the
    prior equal-length period's totals, average spend by day-of-week (if
    available), and budget-adherence info (streak, days under/over target).
    Returns a short, on-demand natural-language summary -- called only when
    the user asks for /summary, never pushed unprompted. See
    TRENDS_SYSTEM_PROMPT for how it's asked to reason like an advisor
    (distinguishing a one-off occasion from a genuine behavioral spike)
    rather than just reading off category totals."""
    client = _get_client()
    data_str = json.dumps(payload)
    resp = client.messages.create(
        model=config.CLAUDE_NARRATION_MODEL,
        max_tokens=600,
        system=TRENDS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Period: {period}\nData: {data_str}"}],
    )
    return resp.content[0].text.strip()


def answer_with_rundown(payload: dict) -> str:
    """payload holds the last 7 days across all four domains -- today's
    balance status, meal count/calories/water, workout count/activities,
    and vitals check-ins/latest weight/weight change/avg sleep/avg knee
    pain -- computed deterministically in Python (see bot.py's
    _rundown_payload), never estimated by the model. Returns a short
    cross-domain narrative, called only when the user asks a broad "how am
    I doing" question (the "rundown" intent), never pushed unprompted.
    Same discipline as answer_with_trends -- real numbers in, plain-language
    narrative out, thin sections skipped rather than padded."""
    client = _get_client()
    data_str = json.dumps(payload)
    resp = client.messages.create(
        model=config.CLAUDE_NARRATION_MODEL,
        max_tokens=450,
        system=(
            "You are Morrow, a personal companion, giving a short cross-domain check-in when asked "
            "something like 'how am I doing'. You're given the last 7 days of real computed figures "
            "across money (today's balance status), food (meal count, total estimated calories, water), "
            "training (workout count and activities), and vitals (check-in count, latest weight, weight "
            "change over the week, average sleep, average knee pain). Any section can be empty or null "
            "(nothing logged that domain this week) -- skip it silently rather than mentioning its "
            "absence or guessing at a number that isn't there. Write 3-6 short plain-text lines: lead "
            "with whichever domain has the most notable signal (a clear trend, a streak, a gap worth "
            "naming), touch the others briefly, and end with one matter-of-fact takeaway. Plain "
            "conversational lines by default, the way a companion would actually talk, not a report or "
            "bullet list. Matter-of-fact and useful, never alarmist or nagging, and never invent "
            f"a number that isn't in the data. {TELEGRAM_FORMATTING_NOTE}"
        ),
        messages=[{"role": "user", "content": data_str}],
    )
    return resp.content[0].text.strip()


def answer_with_day_stats(payload: dict) -> str:
    """payload holds real computed figures for exactly ONE day -- meals
    (count, the actual logged item names, total estimated calories, water),
    workouts (count, activities, total calories burned), net_calories
    (calories in minus calories out, or null if nothing at all was logged
    that day), and vitals (that day's single check-in, or null) -- computed
    deterministically in Python (see rundown._day_stats_payload), never
    estimated or re-summed by the model. Returns a short single-day
    narrative, called only for the 'day_stats' intent (a specific-day
    question, e.g. "stats from yesterday", "what did I eat Saturday",
    "calories for the 20th"). Same discipline as answer_with_rundown --
    real numbers in, plain-language narrative out -- except here getting
    the numbers right matters even more, since this replaces a path that
    used to let the model do the arithmetic itself and get it wrong."""
    client = _get_client()
    data_str = json.dumps(payload)
    resp = client.messages.create(
        model=config.CLAUDE_NARRATION_MODEL,
        max_tokens=350,
        system=(
            "You are Morrow, a personal companion, answering a question about ONE specific day (e.g. "
            "'stats from yesterday', 'what did I eat Saturday'). You're given real computed figures for "
            "that exact day: meals (count, the actual food items logged, total estimated calories, "
            "water), workouts (count, activities, total calories burned), net_calories (calories in minus "
            "calories out -- POSITIVE means a surplus/ate more than burned, NEGATIVE means a deficit/burned "
            "more than ate -- use net_calories' own sign exactly as given, never recompute or re-derive it "
            "yourself), and vitals (that day's check-in, if any). Every number here is already correct and "
            "final -- restate it plainly, never recalculate, round differently, or 'correct' it based on "
            "anything said earlier in the conversation; if the user previously disputed a number, this "
            "fresh read from the database is the real one. If a section is empty (nothing logged), say so "
            "plainly rather than omitting it silently -- for a single specific day the user is asking "
            "about, \"nothing logged\" is itself useful information, unlike the multi-day rundown where a "
            "quiet domain is just skipped. Name the actual food items eaten when there are any, not just "
            "the total. Write 2-5 short plain-text lines, matter-of-fact, the way a companion who actually "
            f"checked would answer -- not a report. {TELEGRAM_FORMATTING_NOTE}"
        ),
        messages=[{"role": "user", "content": data_str}],
    )
    return resp.content[0].text.strip()


def answer_with_trend(payload: dict) -> str:
    """payload holds real computed figures for ONE metric over a date range --
    metric (which of weight/sleep/knee_pain/calories_in/calories_out/spending),
    unit, start_date/end_date (the REAL earliest/latest dates actual data was
    found on, not any placeholder), count (how many data points), first/last
    (each {date, value} -- the earliest and latest actual readings in range),
    min/max (each {date, value}), and change (last.value - first.value, or
    null if count < 2) -- all computed deterministically in Python (see
    rundown._trend_payload), never estimated or re-derived by the model.
    Returns a short narrative, called only for the 'trend' intent (e.g.
    "summarise my weight progression since the beginning", "how's my sleep
    been trending"). Same discipline as answer_with_day_stats -- every number
    here is already correct and final, restate it plainly, never recompute
    or "correct" it based on anything said earlier in the conversation. If
    count is 0 (nothing logged for this metric in range at all), say that
    plainly rather than inventing a trend."""
    client = _get_client()
    data_str = json.dumps(payload)
    resp = client.messages.create(
        model=config.CLAUDE_NARRATION_MODEL,
        max_tokens=350,
        system=(
            "You are Morrow, a personal companion, answering a question about how ONE metric has "
            "progressed over a date range (e.g. 'summarise my weight progression since the beginning', "
            "'how's my sleep been trending'). You're given real computed figures: metric and unit, the "
            "actual start_date/end_date data was found on (these are the REAL earliest/latest dates with "
            "a reading, not the range the person literally asked for -- if they said 'since the "
            "beginning', start_date IS the real beginning, state it as a real date, don't say something "
            "vague like 'since you started'), count (how many readings), first and last (each a "
            "{date, value} -- the earliest and latest actual readings), min and max (each {date, value}), "
            "and change (last's value minus first's value -- POSITIVE means it went up, NEGATIVE means it "
            "went down; use change's own sign exactly as given, never recompute it yourself). Every number "
            "here is already correct and final -- restate it plainly, never recalculate, round "
            "differently, or 'correct' it based on anything said earlier in the conversation. If count is "
            "0, say plainly that nothing's logged for that metric in range rather than inventing a trend; "
            "if count is 1, say there's only the one reading so there's no real trend yet, and give it. "
            "Otherwise name the real start and end values and dates, the overall direction and size of "
            "the change, and mention the min/max only if they're not just the first/last (i.e. there was a "
            "real swing in between worth naming). Write 2-5 short plain-text lines, matter-of-fact, the "
            f"way a companion who actually checked would answer -- not a report. {TELEGRAM_FORMATTING_NOTE}"
        ),
        messages=[{"role": "user", "content": data_str}],
    )
    return resp.content[0].text.strip()


CASUAL_SYSTEM_PROMPT = f"""You are Morrow, the person's personal companion and tracker, having an ordinary \
back-and-forth conversation with them on Telegram -- not extracting data, not filling out a form. This is a \
dedicated call just for this: nothing here needs to come back as JSON, and nothing you write commits to any \
action, so there's no schema competing for your attention -- just write the actual reply.

Write like an actual thoughtful companion who's genuinely listening, the way a good back-and-forth with a \
person (or a genuinely attentive ChatGPT thread) reads -- not a bot economizing on words, and not a command \
menu. A couple of sentences is a fine default for something small; go longer, with real substance, whenever \
the message actually calls for it -- a real question deserves a real answer, not a deflection toward a \
command. Don't pad for its own sake (a genuine "thanks" still just gets a genuine short reply), but when \
there's something real to engage with, engage with it: react to the specific thing they said, add a thought \
or a follow-up where one actually fits, ask something back if it's natural to, instead of just closing the \
loop. Thoughtful beats terse, and a real opinion beats a hedge -- when the numbers or the conversation \
actually support one, say it plainly ("that's a solid week" or "that's higher than your usual Tuesday") \
instead of turning it into a question back at them. Match the person's own register instead of defaulting to \
neutral-polite -- when they're casual or sweary, talk back the same way; that's what makes it read as an \
actual companion present in the conversation, not a bot performing politeness at them.

You're given three things:
- The recent conversation history (oldest first, "user"/"morrow" turns) -- use it to resolve pronouns and \
follow-ups ("that", "it", "the one I mentioned") and to keep this reply in the actual flow of the \
conversation instead of treating every message as a fresh start.
- The user's full durable memory list (each with a "label", "category", and "content" -- standing goals, \
plans, preferences they've told you to remember). Read the content, not just the labels: if the message \
references something covered by an existing memory (e.g. mentions a gym by name and a memory holds that \
gym's plan), use that content directly instead of asking the user to repeat it. This is the ONLY source of \
durable facts -- never invent a plan or preference that isn't actually in this list.
- A "today_snapshot": real, deterministically-computed numbers for right now -- today's spending balance/ \
target/streak, today's meals/workouts/vitals so far (counts, actual items, calorie totals, net calories, \
the latest weight/sleep/knee-pain reading if logged today), and "net_worth" (total income minus deductions \
minus all-time non-claimable spend -- see db.get_net_worth; a genuinely DIFFERENT number from the daily \
balance/target above, since that's a discretionary-spending-allowance rollover, not a real money total). \
This is here so you can actually converse with real knowledge of how today's going ("you're already over \
target today, but barely" or "nothing logged yet today, quiet one so far") instead of talking in a vacuum, \
and so a genuine "what's my net worth" / "how much money do I actually have saved up" question gets answered \
from net_worth's real figures instead of guessed at -- use whichever part is actually relevant to what they \
said, don't force it into every reply. Every number in it is already correct and final; restate it, never \
recompute, re-estimate, or "correct" it.
- "recent_lifts": real logged gym-exercise rows (exercise, location, sets, effort, context_notes, lift_date), \
most recent first. THIS is the real source of truth for any question about what a specific gym day/routine \
actually involved ("what's my push day at Visa look like", "what did I do for pull day last time") -- ground \
exact sets/reps/weight/exercise-name answers in these rows, never in the durable-memory prose or in your own \
paraphrase of something said earlier in the conversation history. A memory entry may describe a routine in \
general terms (e.g. "pull day at Visa is pull-ups, rows, lat pulldowns") -- that's fine for naming WHICH \
exercises belong to a routine, but the exact numbers for what was actually done on a given day must come from \
recent_lifts, not be invented or half-remembered from context. This matters because giving different exact \
numbers to the same question asked twice in one conversation is a real observed bug -- if recent_lifts simply \
doesn't have the session being asked about, say so plainly (e.g. "I don't have an exact logged session for \
that yet") rather than guessing or reconstructing one from memory/conversation text.

The bot's real command surface (never deny something on this list, and never invent a capability that isn't \
on it): {COMMAND_LIST}

If they ask about a capability the bot has, point them at the real command or say you can already do it \
inline. If they ask for something the bot genuinely can't do (e.g. a specific past day's spending BALANCE -- \
/balance only ever reflects today's live number, it has no historical snapshot to show), say that plainly \
and suggest the closest real alternative instead of inventing a capability that doesn't exist.

Be careful not to over-generalize that one real limitation into a false one: a specific past day's FOOD, \
WORKOUT, or VITALS picture is NOT the same limitation as /balance -- the database keeps every dated meal/ \
workout/vitals row indefinitely (including backfilled history from before this chat existed), and /daystats \
(or just asking "what did I eat/log on <date>") pulls a real answer for any day, however long ago. If asked \
something like "what did I log on July 15" and you don't have that day's numbers in front of you in this \
call, don't claim the bot doesn't store historical data -- it does. Just say you don't have that specific \
day pulled up right now and point them at asking it as its own message (e.g. "what did I eat on July 15") \
so it goes through the real per-day lookup instead of this conversational one.

NEVER claim OR promise that you performed, edited, deleted, logged, remembered, or will look into/fix/note/ \
save anything -- not "I've removed it", not "I'll take care of that", not "noted, I'll remember that". This \
call only talks and has no way to follow up later or write anything down; any phrasing implying action past, \
present, or future is misleading. If the message actually needs a real action taken (logging something, \
correcting something, remembering something), say so plainly and tell them to just say it as its own \
message (or use the matching command) and you'll do it for real -- don't pretend to have already done it here.

{TELEGRAM_FORMATTING_NOTE}"""


def answer_casually(message: str, recent_messages: list | None = None, memory_list: list | None = None,
                     today_snapshot: dict | None = None, recent_lifts: list | None = None) -> str:
    """Dedicated call for the 'casual' intent -- genuinely open-ended
    conversation (small talk, catching up, venting, asking for advice, a
    question answerable from context/memory), pulled OUT of parse_message's
    single mega-extraction call into its own focused call, the same
    "narration gets its own call" discipline as answer_with_rundown/
    answer_with_day_stats. The reasoning: parse_message already has to
    correctly classify+extract across ~20 structured fields in one
    completion -- asking it to ALSO write a genuinely warm, engaged
    conversational reply in that same JSON blob means the reply is
    competing for the model's attention with a giant extraction schema, and
    in practice reads flatter than a call with nothing else to do. Uses
    CLAUDE_NARRATION_MODEL (see config.py) rather than CLAUDE_MODEL --
    actually conversing well benefits from a stronger model even though
    fast structured extraction doesn't need one.

    today_snapshot: {{"balance": <db.get_status(chat_id) dict>, "today":
    <rundown._day_stats_payload(chat_id, today) dict>}} -- real numbers,
    never estimated, same discipline as every other narration call here.
    recent_lifts (see lifts._recent_lifts_for_narration): real logged
    gym-exercise rows, most recent first -- what grounds a gym-routine
    question ("what's my push day at Visa look like") in real data instead
    of the model freely narrating from memory prose, which was a real
    observed bug (exact numbers drifting between successive identical
    questions in the same conversation). Never raises -- callers should
    catch and fall back to parse_message's own casual_reply field (or a
    generic line) the same way rundown/day_stats fall back to their own
    deterministic text on failure."""
    client = _get_client()
    recent_messages = recent_messages or []
    memory_list = memory_list or []
    today_snapshot = today_snapshot or {}
    recent_lifts = recent_lifts or []
    user_content = (
        f"{_today_context()}\n\n"
        f"Recent conversation history (oldest first):\n{json.dumps(recent_messages)}\n\n"
        f"Durable memory:\n{json.dumps(memory_list)}\n\n"
        f"today_snapshot:\n{json.dumps(today_snapshot)}\n\n"
        f"recent_lifts:\n{json.dumps(recent_lifts)}\n\n"
        f"Message: {message}"
    )
    resp = client.messages.create(
        model=config.CLAUDE_NARRATION_MODEL,
        max_tokens=600,
        system=CASUAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return resp.content[0].text.strip()


# ---------- narrate_reply: the companion voice for EVERY reply, not just casual ----------
# answer_casually gave open-ended conversation real personality, but everything else Morrow
# sends -- a log confirmation, a correction, a clarifying question, an "I didn't catch that" --
# stayed a flat, hard-coded string, which is most of what the bot actually says day to day.
# That's the real reason logging still read as "standard" even after answer_casually shipped:
# the companion voice only ever fired on the minority of messages classified as pure chit-chat.
# narrate_reply closes that gap by running EVERY reply (see replies._reply, the one chokepoint
# on the free-text/command reply path) through the same restyling treatment -- but as a
# RESTYLE, not a rewrite: the content (every number, id, date, fact) is already correct and
# final by the time this call runs, computed the same deterministic way it always was; this
# call's only job is how it's said, the same "real numbers in, model only narrates" discipline
# used everywhere else in this codebase, just applied to delivery instead of substance.

NARRATE_REPLY_SYSTEM_PROMPT = f"""You are Morrow, the person's personal companion and tracker, replying on \
Telegram. This call has ONE job: take an already-written, already-correct reply and say it the way an actual \
attentive companion would -- not rewrite what happened, just how it's said. Every fact, number, id (like \
"#7"), date, and name in the reply you're given is ALREADY correct and final -- your job is delivery, not \
content.

Rules, in order of importance:
1. NEVER drop, change, round, or invent a number, id, date, name, or fact that's in the original reply. If \
you're not sure how to work something in naturally, just state it plainly rather than leaving it out -- an \
omission here is a real bug (the user loses real information), not just an awkward sentence.
2. NEVER drop an actionable instruction that's in the original -- especially "Reply 'undo' if that's wrong" \
or any similar follow-up prompt, or a usage/command hint. Keep the exact meaning even if you reword it.
3. NEVER claim any action beyond what the original reply already states happened -- don't add "I've also...", \
don't imply you did something extra that isn't already stated.
4. Match the length and register to what's actually there. A single plain log ("Logged: chicken rice -- ~650 \
kcal") stays ONE short, natural line or two -- real personality, not padding it into a paragraph. Something \
with more real content already in it (a multi-item log, a same-exercise comparison, a correction, a richer \
confirmation) can be a bit more conversational, but stays concise -- this is a restyle of an existing reply, \
never a new essay.
5. Use the recent conversation history, durable memory, today_snapshot, and recent_lifts you're given the \
same way a casual reply would -- for continuity and personality (referencing something relevant, matching the \
person's own register), never to add or contradict a fact that isn't already in the original reply.
6. A reply that's a genuine question (a clarification, "what day is that on?") or an honest miss ("I didn't \
catch what you ate") should still read like an actual person asking or admitting it, not a form-validation \
error -- but stays exactly that: a real question or a real admission, never dressed up as knowing something \
it doesn't.

{TELEGRAM_FORMATTING_NOTE}"""


def narrate_reply(deterministic_text: str, recent_messages: list | None = None, memory_list: list | None = None,
                   today_snapshot: dict | None = None, recent_lifts: list | None = None) -> str:
    """Rewrites an already-correct, deterministically-computed reply (a log
    confirmation, a correction confirmation, a clarifying question, a
    show_balance/show_recent/... answer, an undo confirmation, an evening
    nudge, ...) in Morrow's actual companion voice before it's sent -- see
    the module-level comment above and replies._reply's docstring for why
    this now runs on every reply by default, not just the 'casual' intent.
    This call NEVER changes what happened, only how it's said --
    deterministic_text is the real, already-correct content; see
    NARRATE_REPLY_SYSTEM_PROMPT for the exact rules (never drop a
    number/id/fact, never drop an 'undo' offer, never claim an extra
    action). Same context shape as answer_casually
    (recent_messages/memory_list/today_snapshot/recent_lifts) so it can
    actually sound like a continuation of the same conversation, not a
    stateless rewrite. Never raises -- callers (replies._reply) fall back
    to deterministic_text unchanged on any failure, the same "never go
    silent, never say something false" discipline as every other narration
    call here."""
    client = _get_client()
    recent_messages = recent_messages or []
    memory_list = memory_list or []
    today_snapshot = today_snapshot or {}
    recent_lifts = recent_lifts or []
    user_content = (
        f"{_today_context()}\n\n"
        f"Recent conversation history (oldest first):\n{json.dumps(recent_messages)}\n\n"
        f"Durable memory:\n{json.dumps(memory_list)}\n\n"
        f"today_snapshot:\n{json.dumps(today_snapshot)}\n\n"
        f"recent_lifts:\n{json.dumps(recent_lifts)}\n\n"
        f"The reply about to be sent (already correct and final -- restyle it, never change its content):\n"
        f"{deterministic_text}"
    )
    resp = client.messages.create(
        model=config.CLAUDE_NARRATION_MODEL,
        max_tokens=400,
        system=NARRATE_REPLY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return resp.content[0].text.strip()


def _clarify_fallback(message: str) -> dict:
    return {
        "intent": "clarification",
        "expenses": None,
        "income": None,
        "deductions": None,
        "subscriptions": None,
        "meals": None,
        "activity": None,
        "duration_min": None,
        "distance_km": None,
        "workout_notes": None,
        "lifts": None,
        "weight_kg": None,
        "sleep_hours": None,
        "knee_pain": None,
        "vitals_notes": None,
        "logged_days_ago": None,
        "day_stats_days_ago": None,
        "trend_metric": None,
        "trend_start_days_ago": None,
        "trend_end_days_ago": None,
        "tasks": None,
        "reminder_description": None,
        "events": None,
        "target_domain": None,
        "target_expense_id": None,
        "correction_action": None,
        "days_ago": None,
        "new_currency": None,
        "new_amount": None,
        "new_description": None,
        "new_category": None,
        "new_meal_items": None,
        "new_meal_type": None,
        "new_meal_calories_low": None,
        "new_meal_calories_high": None,
        "new_meal_calories_estimate": None,
        "new_meal_water_ml": None,
        "new_workout_activity": None,
        "new_workout_duration_min": None,
        "new_workout_distance_km": None,
        "new_workout_calories_burned": None,
        "new_workout_notes": None,
        "new_lift_exercise": None,
        "new_lift_location": None,
        "new_lift_sets": None,
        "new_lift_effort": None,
        "new_lift_context_notes": None,
        "new_event_in_days": None,
        "memory_label": None,
        "memory_content": None,
        "memory_category": None,
        "clarification_question": message,
        "casual_reply": None,
    }


def _strip_fences(raw: str) -> str:
    # Strip accidental code fences if the model adds them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return raw


def _extract_json_object(raw: str) -> str:
    """Best-effort recovery for a response that's ALMOST pure JSON but has
    stray prose wrapped around it -- e.g. a one-line preamble ("Sure, here's
    the correction:") before the object, despite the system prompt saying
    "Reply with ONLY a JSON object". A real observed bug this guards
    against: a genuinely edge-case-y request (e.g. correcting a workout/
    lift field no domain action actually supported yet) occasionally
    nudged the model into explaining itself instead of committing to pure
    JSON -- which used to fail json.loads on the WHOLE string and fall back
    to a generic "didn't catch that" clarification, even for a completely
    unrelated LATER message in the same conversation, even though a real,
    valid JSON object was sitting right there in the response the whole
    time. Finds the first '{' and the LAST '}' and returns just that span;
    this only strips clean prose wrapped around a clean object, it doesn't
    attempt to repair genuinely broken/truncated JSON."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return raw
    return raw[start:end + 1]


def _safe_json(raw: str) -> dict:
    stripped = _strip_fences(raw)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_extract_json_object(stripped))
    except json.JSONDecodeError:
        return _clarify_fallback(
            "Sorry, I didn't quite catch that — could you rephrase it, e.g. 'spent 12 on lunch'?"
        )


def _parse_json_or_none(raw: str):
    """Used by the single-purpose extract_* helpers, which have their own
    domain-appropriate fallback shape rather than a clarification dict."""
    stripped = _strip_fences(raw)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_extract_json_object(stripped))
    except json.JSONDecodeError:
        return None
