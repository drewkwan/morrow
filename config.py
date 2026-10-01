"""
Configuration for the expense-tracking Telegram bot.
All values are read from environment variables so secrets never live in code.
See .env.example for the full list of variables you need to set.
"""

import os

# --- Required ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# --- Optional ---
# Only these Telegram user IDs may use the bot. Leave blank to allow anyone who
# has your bot's link (fine for a private bot, but locking it down is safer).
ALLOWED_CHAT_IDS = [
    int(x) for x in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if x.strip()
]

# Where the SQLite database file lives. On Railway, set this to a path inside
# a mounted volume so data survives redeploys (see README).
DB_PATH = os.environ.get("DB_PATH", "expenses.db")

# Single timezone used for "what day is it" (daily rollover, /summary today).
# This is a personal single-user-ish bot, so one global timezone keeps things
# simple. Use an IANA name, e.g. "America/Los_Angeles", "Asia/Singapore".
BOT_TIMEZONE = os.environ.get("BOT_TIMEZONE", "Asia/Singapore")

# Claude model used for categorization / natural-language expense parsing --
# i.e. every classify-and-extract call (parse_message, extract_meal,
# extract_from_photo, categorize, and friends). Haiku is fast and cheap,
# which is plenty for this kind of structured extraction.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# Claude model used for NARRATION instead -- the calls that turn real
# computed numbers (or an open-ended chat message) into an actual written
# reply: answer_with_rundown, answer_with_day_stats, answer_with_trends,
# answer_casually, and friends. Defaults to CLAUDE_MODEL (no behavior
# change out of the box), but this is deliberately a separate knob: writing
# a genuinely warm, thoughtful conversational reply is a different job from
# fast structured extraction, and benefits from a stronger model even
# though extraction doesn't need one. Set this to a Sonnet-tier model in
# your .env to make casual conversation and narrated summaries noticeably
# better without paying Sonnet's cost on every single expense/meal logged.
CLAUDE_NARRATION_MODEL = os.environ.get("CLAUDE_NARRATION_MODEL", CLAUDE_MODEL)

# Default daily target used the very first time a user interacts with the bot,
# before they run /settarget.
DEFAULT_DAILY_TARGET = float(os.environ.get("DEFAULT_DAILY_TARGET", "100"))

# All targets, balances, and totals are tracked in this currency. Expenses
# logged in a different currency are converted to this one (via fx.py) for
# budget math, while the original amount + currency is still kept on the
# expense record so you can see what you actually paid.
BASE_CURRENCY = os.environ.get("BASE_CURRENCY", "SGD")

# Currencies fx.py knows how to convert (matches what the free Frankfurter/ECB
# rates API supports). Add more here if you need a currency that's missing —
# just check it's on https://www.frankfurter.app/currencies first.
KNOWN_CURRENCIES = [
    "AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "GBP", "HKD", "HUF",
    "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD",
    "PHP", "PLN", "RON", "SEK", "SGD", "THB", "TRY", "USD", "ZAR", "EUR",
]

# Fraction of today's available budget (target + rolled-over balance) that,
# once crossed, triggers a one-time heads-up message for the day.
BUDGET_ALERT_THRESHOLD = float(os.environ.get("BUDGET_ALERT_THRESHOLD", "0.9"))

# Local time (in BOT_TIMEZONE above) the automatic morning briefing goes out
# each day -- see morning.py. /morning previews the same content on demand
# at any time, so this only controls the proactive daily push.
MORNING_BRIEFING_HOUR = int(os.environ.get("MORNING_BRIEFING_HOUR", "7"))
MORNING_BRIEFING_MINUTE = int(os.environ.get("MORNING_BRIEFING_MINUTE", "30"))

# Local time (in BOT_TIMEZONE) the recurring-finance tick runs -- auto-
# logging any subscription or recurring salary whose billing/pay day is
# today (see subscriptions.subscriptions_tick/income.income_tick). Runs
# BEFORE the morning briefing above, so a same-day auto-post is already in
# the database by the time that briefing (or /rundown, /morning) reads it.
RECURRING_FINANCE_HOUR = int(os.environ.get("RECURRING_FINANCE_HOUR", "6"))
RECURRING_FINANCE_MINUTE = int(os.environ.get("RECURRING_FINANCE_MINUTE", "30"))

# Weekly subscription-renewal digest -- see subscriptions.subscriptions_digest_tick.
# Deliberately separate from the daily morning briefing (Andrew's explicit
# choice: a renewal heads-up is useful once a week, not every single day).
# Runs Monday mornings (0 = Monday, matching PTB's JobQueue.run_daily `days`
# convention) at the same local time as the morning briefing, listing
# anything renewing in the next SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS days.
SUBSCRIPTION_DIGEST_WEEKDAY = int(os.environ.get("SUBSCRIPTION_DIGEST_WEEKDAY", "0"))
SUBSCRIPTION_DIGEST_HOUR = int(os.environ.get("SUBSCRIPTION_DIGEST_HOUR", "7"))
SUBSCRIPTION_DIGEST_MINUTE = int(os.environ.get("SUBSCRIPTION_DIGEST_MINUTE", "30"))
SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS = int(os.environ.get("SUBSCRIPTION_DIGEST_LOOKAHEAD_DAYS", "7"))

# Local time (in BOT_TIMEZONE) the evening "quiet day" nudge checks in --
# see nudges.py. Deliberately late evening, not late afternoon: the point is
# to catch a day where genuinely nothing got logged at all, not to nag
# partway through a normal day.
EVENING_NUDGE_HOUR = int(os.environ.get("EVENING_NUDGE_HOUR", "20"))
EVENING_NUDGE_MINUTE = int(os.environ.get("EVENING_NUDGE_MINUTE", "30"))

CATEGORIES = [
    "Food",
    "Groceries",
    "Transport",
    "Entertainment",
    "Shopping",
    "Bills & Utilities",
    "Health",
    "Travel",
    "Gifts & Occasions",
    "Hobbies & Collectibles",
    "Other",
]

# Meal types for /logmeal and natural-language meal logging.
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack"]
