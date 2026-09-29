"""
Entrypoint: /start, /help, the background rollover tick, and main() itself
-- building the Application and registering every handler from every domain
module. This is the only place that needs to know about all of them at
once; every other module only imports the handful it actually depends on.
"""

import datetime
import logging
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()  # must run before `import config`, which reads env vars at import time

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    Defaults,
    ExtBot,
    MessageHandler,
    ContextTypes,
    filters,
)

import config
import db
from tg_html import to_telegram_html
from access import _reject_if_not_allowed
from finance import (
    adjustbalance_cmd,
    balance,
    claim_expense,
    claimed,
    delete_cmd,
    edit_cmd,
    log_expense,
    recent,
    settarget,
    undo,
)
from events import addevent_cmd, events_cmd, removeevent_cmd, rescheduleevent_cmd
from fitness import logworkout_cmd, recentworkouts
from formatting import _money, _status_text
from handlers import handle_text, on_error
from lifts import loglift_cmd, recentlifts
from memory import forget_cmd, memory_cmd
from morning import morning_briefing_tick, morning_cmd
from nudges import evening_nudge_tick
from nutrition import handle_photo, logmeal_cmd, recentmeals
from reminders import addreminder_cmd, donereminder_cmd, reminders_cmd, removereminder_cmd
from rundown import daystats_cmd, rundown_cmd, trend_cmd
from summary import summary
from tasks import addtask_cmd, done_cmd, tasks_cmd
from vitals import logvitals_cmd, recentvitals

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class _FormattingBot(ExtBot):
    """The one actual chokepoint for every outgoing message, however it was
    sent -- update.message.reply_text(...) (used throughout every domain
    module) and context.bot.send_message(...) (used by rollover_tick and
    morning.py's proactive push) both end up calling THIS send_message
    under the hood, so running tg_html.to_telegram_html here, once, is
    equivalent to every call site remembering to do it itself, without
    actually touching any of them. See tg_html's module docstring for why
    this -- rather than each module escaping its own strings -- is the
    right place: Defaults(parse_mode=HTML) below turns HTML parsing on for
    every send, so from this point on EVERY outgoing text needs the same
    escaping/promotion treatment, not just the ones some future edit
    happens to remember."""

    async def send_message(self, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"] = to_telegram_html(kwargs["text"])
        elif len(args) >= 2:
            args = (args[0], to_telegram_html(args[1]), *args[2:])
        return await super().send_message(*args, **kwargs)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await _reject_if_not_allowed(update):
        return
    db.get_or_create_user(update.effective_chat.id)
    await update.message.reply_text(
        "Hey! I'll track your daily spending allowance, claimable expenses, meals, workouts, and vitals.\n\n"
        "Set a daily target: /settarget 100\n"
        "Log a personal expense: /log 12.50 lunch\n"
        "Log in another currency: /log 20 USD taxi\n"
        "Log a claimable one: /claim 300 groceries\n"
        "Cleared a reimbursement: /claimed\n"
        "Check where you stand: /balance\n"
        "Get a breakdown: /summary week\n"
        "See recent entries: /recent\n"
        "Fix a mistake: /undo, /edit <id> <amount>, or /delete <id>\n"
        "Manually correct rolled-over balance: /adjustbalance -1135.89\n\n"
        "Log a meal: /logmeal chicken rice and iced tea, or just send a photo of your food\n"
        "See recent meals: /recentmeals\n"
        "Log a workout: /logworkout tennis for an hour\n"
        "See recent workouts: /recentworkouts\n"
        "Log a structured lift: /loglift pull-ups 10x3 at wheelock\n"
        "See recent lifts: /recentlifts\n"
        "Log vitals: /logvitals weight 76.6, slept 5.5 hours, knee 2/10\n"
        "See recent check-ins: /recentvitals\n\n"
        "See what I remember: /memory\n"
        "Remove something remembered: /forget <label>\n\n"
        "Add a to-do: /addtask call the dentist tomorrow 5pm\n"
        "See what's open: /tasks\n"
        "Mark one done: /done <id>\n\n"
        "Add a daily reminder (recurs every day, e.g. take hair pills): /addreminder take hair pills\n"
        "See your daily reminders: /reminders\n"
        "Mark one done for today only (comes back tomorrow): /donereminder <id>\n"
        "Remove one for good: /removereminder <id>\n\n"
        "Add a scheduled event/appointment: /addevent dinner with Mel next Monday\n"
        "See what's coming up: /events\n"
        "Move one to a new day: /rescheduleevent <id> <days from today>\n"
        "Remove one: /removeevent <id>\n\n"
        "How's everything going, across money/food/training/vitals together: /rundown\n"
        "Calories/activity for one specific day: /daystats [today|yesterday|N]\n"
        "How one metric has progressed over time: /trend <weight|sleep|knee_pain|calories_in|"
        "calories_out|spending> [N days|all]\n"
        "See today's briefing (today's budget + due to-dos + a look back at yesterday) any time: /morning\n\n"
        "Or just tell me naturally, e.g. \"spent 15 on uber\", \"had a mango\", \"played tennis for an hour\", "
        "\"weight 76.6, slept 5.5 hours\", \"remember I go to Fitness First Bugis Tue/Thu\", \"remind me to call "
        "the dentist tomorrow\", \"dinner with Mel next Monday\", or \"how am I doing this week\" -- and just "
        "talk to me the rest of the time, I'll keep up with the thread."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def rollover_tick(context: ContextTypes.DEFAULT_TYPE):
    """Runs periodically so rollovers happen even if no one messages the bot
    right at midnight. Notifies each user only if a rollover actually occurred."""
    for chat_id in db.get_all_chat_ids():
        result = db.ensure_rollover(chat_id)
        if not result["rollovers"]:
            continue
        status = db.get_status(chat_id)
        lines = [f"{d}: leftover {_money(leftover)}" for d, leftover in result["rollovers"]]
        text = "New day, balance rolled over:\n" + "\n".join(lines) + f"\n\n{_status_text(status)}"
        if result["new_best_streak"]:
            text += f"\n\nNew personal best streak: {result['new_best_streak']} days within budget!"
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            logger.exception("Failed to notify chat %s of rollover", chat_id)


def main():
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set. See .env.example.")
    if not config.ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set. See .env.example.")

    db.init_db()

    # A custom bot instance (not .token(...)) so every send_message call --
    # from any domain module, not just this file -- routes through
    # _FormattingBot's override. Defaults(parse_mode=HTML) is what actually
    # turns HTML parsing on for every send that doesn't explicitly pass its
    # own parse_mode (none currently do); see _FormattingBot's docstring
    # for why the two need to travel together.
    bot = _FormattingBot(token=config.TELEGRAM_BOT_TOKEN, defaults=Defaults(parse_mode=ParseMode.HTML))
    app = Application.builder().bot(bot).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("settarget", settarget))
    app.add_handler(CommandHandler("log", log_expense))
    app.add_handler(CommandHandler("claim", claim_expense))
    app.add_handler(CommandHandler("claimed", claimed))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("rundown", rundown_cmd))
    app.add_handler(CommandHandler("daystats", daystats_cmd))
    app.add_handler(CommandHandler("trend", trend_cmd))
    app.add_handler(CommandHandler("recent", recent))
    app.add_handler(CommandHandler("undo", undo))
    app.add_handler(CommandHandler("delete", delete_cmd))
    app.add_handler(CommandHandler("edit", edit_cmd))
    app.add_handler(CommandHandler("adjustbalance", adjustbalance_cmd))
    app.add_handler(CommandHandler("logmeal", logmeal_cmd))
    app.add_handler(CommandHandler("recentmeals", recentmeals))
    app.add_handler(CommandHandler("logworkout", logworkout_cmd))
    app.add_handler(CommandHandler("recentworkouts", recentworkouts))
    app.add_handler(CommandHandler("loglift", loglift_cmd))
    app.add_handler(CommandHandler("recentlifts", recentlifts))
    app.add_handler(CommandHandler("logvitals", logvitals_cmd))
    app.add_handler(CommandHandler("recentvitals", recentvitals))
    app.add_handler(CommandHandler("memory", memory_cmd))
    app.add_handler(CommandHandler("forget", forget_cmd))
    app.add_handler(CommandHandler("addtask", addtask_cmd))
    app.add_handler(CommandHandler("tasks", tasks_cmd))
    app.add_handler(CommandHandler("done", done_cmd))
    app.add_handler(CommandHandler("addreminder", addreminder_cmd))
    app.add_handler(CommandHandler("reminders", reminders_cmd))
    app.add_handler(CommandHandler("donereminder", donereminder_cmd))
    app.add_handler(CommandHandler("removereminder", removereminder_cmd))
    app.add_handler(CommandHandler("addevent", addevent_cmd))
    app.add_handler(CommandHandler("events", events_cmd))
    app.add_handler(CommandHandler("rescheduleevent", rescheduleevent_cmd))
    app.add_handler(CommandHandler("removeevent", removeevent_cmd))
    app.add_handler(CommandHandler("morning", morning_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(on_error)

    if app.job_queue is not None:
        app.job_queue.run_repeating(rollover_tick, interval=3600, first=10)
        briefing_time = datetime.time(
            config.MORNING_BRIEFING_HOUR, config.MORNING_BRIEFING_MINUTE, tzinfo=ZoneInfo(config.BOT_TIMEZONE)
        )
        app.job_queue.run_daily(morning_briefing_tick, time=briefing_time)
        nudge_time = datetime.time(
            config.EVENING_NUDGE_HOUR, config.EVENING_NUDGE_MINUTE, tzinfo=ZoneInfo(config.BOT_TIMEZONE)
        )
        app.job_queue.run_daily(evening_nudge_tick, time=nudge_time)
    else:
        logger.warning(
            "JobQueue not available -- install with pip install python-telegram-bot[job-queue] "
            "to get automatic daily rollover notifications, the morning briefing, and the evening "
            "nudge. Rollover math still runs correctly, and /morning still works on demand, "
            "whenever a user interacts with the bot -- only the proactive daily pushes need JobQueue."
        )

    logger.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
