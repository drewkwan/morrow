"""
SQLite persistence layer for the expense bot.

Design (see README for the full explanation):

- `users`: one row per Telegram chat. Tracks the current daily target and a
  running `balance` -- the cumulative rollover of unspent (or overspent) money
  from previous days, plus streak counters and alert throttling.
- `expenses`: every logged expense. `amount`/`currency` are what you actually
  paid; `amount_base` is that converted into BASE_CURRENCY and is what all
  budget math uses, so mixed-currency spending still adds up correctly.
  `is_claimable` separates "claim this back from someone" spending from your
  personal daily allowance. `is_claimed` marks claimables that have been
  reimbursed and cleared.
- `meals`: one row per logged food/drink item or entry (not one row per meal
  slot -- matches how this is actually used: logged as things are eaten
  throughout the day, not decomposed into breakfast/lunch/dinner buckets).
  Calories are stored as a `calories_low`/`calories_high`/`calories_estimate`
  range rather than a single false-precise number, matching the estimate
  style already validated by hand in ChatGPT. `water_ml` is a separate axis
  (hydration, not calories) and is null on entries that aren't plain water.
- `workouts`: one row per logged session. `notes` stays free text rather than
  forcing IPPT times or tennis sets into rigid columns before it's clear
  what's worth tracking structurally. `calories_burned` is nullable and
  separate from `meals.calories_estimate` (calories out vs calories in) --
  a fitness app/wearable's stats screenshot is the main source for it (see
  ai.extract_from_photo), rather than something typed workouts usually
  report themselves.
- `vitals`: one row per daily check-in (weight, sleep, knee pain, or any
  subset -- all nullable, since not every check-in reports everything).
  Separate from `workouts` because it's reported on its own cadence, not
  tied to a specific session.
- `messages`: a rolling log of the conversation itself (both sides), so
  Morrow can stay in a multi-turn thread instead of only seeing structured
  log rows. Read as the last ~30 turns on every call -- this is short-term
  memory (the current conversation), not long-term.
- `memory`: durable, labeled facts that outlast any one conversation --
  goals, standing plans, preferences. One row per `label` per chat (a
  case-insensitive match on an existing label updates that row in place
  rather than creating a near-duplicate), so "remember" always edits the
  same named slot instead of piling up copies. This is what answers "pull
  up my workout plan" -- unlike `messages`, it's meant to be read in full on
  every call, not just recently.
- `tasks`: to-dos, one row per item. `due_at` is an ISO date (`YYYY-MM-DD`)
  or date+time (`YYYY-MM-DD HH:MM`) string, or null for a task with no due
  date -- text-sortable either way, so ORDER BY due_at still works with a
  mix of dated and undated rows. `done` is a simple flag rather than a
  separate completed-tasks table; a done task stays queryable for "did I
  already do X" without needing a join.

Rollover math (matches the spec exactly):
  Day 1: target=$100, spend $30 -> leftover = $100 - $30 = $70 -> balance += 70
  Day 2: available = target($100) + balance($70) = $170

`balance` is always derived from `expenses` -- there's deliberately no
"just set it to X" path in the normal flow, so it can never silently drift
from what was actually logged. adjust_balance is the one escape hatch, for
when the derivation itself can't be trusted (lost history from before a
given date), and it's a nudge (+/- delta) rather than an absolute set, so
it's always visible in the reply/undo trail rather than a value replaced
outright.
"""

import json
import sqlite3
from collections.abc import Iterator
from datetime import date, timedelta
from contextlib import contextmanager

import config
import fx


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY,
                daily_target REAL NOT NULL DEFAULT 0,
                balance REAL NOT NULL DEFAULT 0,
                last_rollover_date TEXT NOT NULL,
                last_alert_date TEXT,
                current_streak INTEGER NOT NULL DEFAULT 0,
                best_streak INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT '{config.BASE_CURRENCY}',
                amount_base REAL NOT NULL,
                description TEXT,
                category TEXT,
                is_claimable INTEGER NOT NULL DEFAULT 0,
                is_claimed INTEGER NOT NULL DEFAULT 0,
                expense_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                meal_type TEXT,
                items TEXT,
                calories_low REAL,
                calories_high REAL,
                calories_estimate REAL,
                water_ml REAL,
                meal_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                activity TEXT,
                duration_min REAL,
                distance_km REAL,
                calories_burned REAL,
                notes TEXT,
                workout_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vitals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                weight_kg REAL,
                sleep_hours REAL,
                knee_pain REAL,
                notes TEXT,
                vitals_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                category TEXT,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                due_at TEXT,
                done INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                recurrence_frequency TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                description TEXT NOT NULL,
                last_done_date TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                exercise TEXT NOT NULL,
                location TEXT,
                sets TEXT,
                effort TEXT,
                context_notes TEXT,
                lift_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                dedup_key TEXT NOT NULL,
                sent_date TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT '{config.BASE_CURRENCY}',
                amount_base REAL NOT NULL,
                category TEXT,
                frequency TEXT NOT NULL DEFAULT 'monthly',
                next_renewal_date TEXT NOT NULL,
                card TEXT,
                notes TEXT,
                is_claimable INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                last_logged_date TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS income_config (
                chat_id INTEGER PRIMARY KEY,
                gross_amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT '{config.BASE_CURRENCY}',
                pay_day INTEGER NOT NULL,
                cpf_rate REAL NOT NULL DEFAULT 0,
                stock_rate REAL NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                last_logged_month TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS income (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                description TEXT,
                gross_amount REAL,
                cpf_amount REAL,
                stock_amount REAL,
                net_amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT '{config.BASE_CURRENCY}',
                net_amount_base REAL NOT NULL,
                income_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS deductions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT '{config.BASE_CURRENCY}',
                amount_base REAL NOT NULL,
                deduction_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # Forward-compatible migration in case this is an existing db from
        # before currency/streak/alert support was added.
        _add_column_if_missing(conn, "users", "last_alert_date", "last_alert_date TEXT")
        _add_column_if_missing(conn, "users", "current_streak", "current_streak INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "users", "best_streak", "best_streak INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "expenses", "currency", f"currency TEXT NOT NULL DEFAULT '{config.BASE_CURRENCY}'")
        _add_column_if_missing(conn, "expenses", "amount_base", "amount_base REAL")
        # Backfill amount_base for any pre-existing rows (assume same as amount if it was null).
        conn.execute("UPDATE expenses SET amount_base = amount WHERE amount_base IS NULL")
        _add_column_if_missing(conn, "workouts", "calories_burned", "calories_burned REAL")
        # tasks.recurrence_frequency was added to the CREATE TABLE above for
        # recurring to-dos (see add_task's docstring), but CREATE TABLE IF
        # NOT EXISTS is a no-op against an existing database -- tasks is a
        # long-standing table with real production rows already, so without
        # this migration call, every read of a pre-existing task row (e.g.
        # tasks._recent_tasks_for_ai, which runs on EVERY handle_text call)
        # raises KeyError: 'recurrence_frequency' against a live database
        # that was never actually given the new column. Real production
        # incident this fixes -- see the regression test guarding it.
        _add_column_if_missing(conn, "tasks", "recurrence_frequency", "recurrence_frequency TEXT")
        # subscriptions got a full schema redesign (billing_day/
        # last_logged_month replaced by frequency/next_renewal_date/card/
        # notes/is_claimable/last_logged_date -- see this table's own
        # CREATE TABLE comment above) on the wrong assumption that the table
        # didn't exist in production yet. It did -- an earlier push had
        # already created it with the old columns, so (same lesson as the
        # tasks.recurrence_frequency incident right above) CREATE TABLE IF
        # NOT EXISTS silently left it in the old shape, and every new
        # subscriptions function immediately broke with "no such column"
        # against a live database with real rows. frequency and
        # is_claimable get a literal DEFAULT in the ALTER itself (SQLite
        # back-fills every existing row with it as part of adding the
        # column, same as the users.current_streak/best_streak migrations
        # above) -- next_renewal_date can't, since there's no one sensible
        # default date, so it's added nullable and backfilled explicitly
        # below, the same two-step pattern as expenses.amount_base above.
        _add_column_if_missing(conn, "subscriptions", "frequency", "frequency TEXT NOT NULL DEFAULT 'monthly'")
        _add_column_if_missing(conn, "subscriptions", "next_renewal_date", "next_renewal_date TEXT")
        _add_column_if_missing(conn, "subscriptions", "card", "card TEXT")
        _add_column_if_missing(conn, "subscriptions", "notes", "notes TEXT")
        _add_column_if_missing(conn, "subscriptions", "is_claimable", "is_claimable INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "subscriptions", "last_logged_date", "last_logged_date TEXT")
        # The old billing_day design was never actually used to log a real
        # subscription in production (confirmed against Andrew's own
        # backup db -- the table existed but held no real rows), so there's
        # no real renewal date to reconstruct for any leftover row; today
        # is a safe placeholder -- it just means subscriptions_tick would
        # treat such a row as due on its next run, rather than leaving a
        # NULL in a column every subscriptions query now expects to be set.
        conn.execute("UPDATE subscriptions SET next_renewal_date = ? WHERE next_renewal_date IS NULL", (today_str(),))
        # The six ALTER TABLE calls above add every NEW column, but they
        # can't touch the OLD ones -- and billing_day, in particular, is
        # NOT NULL with no DEFAULT in the original schema. add_subscription
        # (rightly) never sets it, so every insert through the new code
        # against a leftover production table violated that constraint
        # outright: a second real production incident,
        # "sqlite3.IntegrityError: NOT NULL constraint failed:
        # subscriptions.billing_day". SQLite has no "drop this constraint"
        # statement, so the only way to actually get rid of it is to rebuild
        # the table -- safe to call on every startup, since it's a no-op
        # once billing_day is already gone.
        _drop_subscriptions_legacy_columns(conn)


def _drop_subscriptions_legacy_columns(conn: sqlite3.Connection) -> None:
    """Rebuilds `subscriptions` via the standard SQLite rename/create/copy/
    drop sequence to actually get rid of the old billing_day/
    last_logged_month columns -- an ALTER TABLE ADD COLUMN can backfill a
    new column, but there's no ALTER TABLE statement that can relax or drop
    an existing NOT NULL constraint, which is what billing_day needs. Runs
    on every startup; a table that's already clean (never had billing_day,
    or had it dropped by a previous run) is left untouched."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(subscriptions)").fetchall()}
    if "billing_day" not in cols:
        return
    conn.execute("ALTER TABLE subscriptions RENAME TO subscriptions_pre_frequency_schema")
    conn.execute(f"""
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT NOT NULL DEFAULT '{config.BASE_CURRENCY}',
            amount_base REAL NOT NULL,
            category TEXT,
            frequency TEXT NOT NULL DEFAULT 'monthly',
            next_renewal_date TEXT NOT NULL,
            card TEXT,
            notes TEXT,
            is_claimable INTEGER NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            last_logged_date TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # frequency/next_renewal_date/is_claimable are already guaranteed non-NULL
    # on every row by the ALTER+backfill steps above, which always run first
    # -- the COALESCE calls here are just a defensive second line, not load-
    # bearing.
    conn.execute(f"""
        INSERT INTO subscriptions (
            id, chat_id, name, amount, currency, amount_base, category,
            frequency, next_renewal_date, card, notes, is_claimable, active,
            last_logged_date, created_at
        )
        SELECT
            id, chat_id, name, amount, currency, amount_base, category,
            COALESCE(frequency, 'monthly'),
            COALESCE(next_renewal_date, '{today_str()}'),
            card, notes, COALESCE(is_claimable, 0), active,
            last_logged_date, created_at
        FROM subscriptions_pre_frequency_schema
    """)
    conn.execute("DROP TABLE subscriptions_pre_frequency_schema")


def _now_local_date() -> date:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(config.BOT_TIMEZONE)).date()


def today_str() -> str:
    return _now_local_date().isoformat()


def get_or_create_user(chat_id: int) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO users (chat_id, daily_target, balance, last_rollover_date) VALUES (?, ?, 0, ?)",
                (chat_id, config.DEFAULT_DAILY_TARGET, today_str()),
            )
            row = conn.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,)).fetchone()
        return dict(row)


def set_daily_target(chat_id: int, amount: float) -> None:
    get_or_create_user(chat_id)
    with get_conn() as conn:
        conn.execute("UPDATE users SET daily_target = ? WHERE chat_id = ?", (amount, chat_id))


def _spent_on(conn: sqlite3.Connection, chat_id: int, day_str: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(amount_base), 0) AS total FROM expenses "
        "WHERE chat_id = ? AND expense_date = ? AND is_claimable = 0",
        (chat_id, day_str),
    ).fetchone()
    return row["total"]


def ensure_rollover(chat_id: int) -> dict:
    """
    Catches the user's balance (and streak) up to today. Safe to call before
    every command. If the bot was offline for multiple days, rolls forward
    one day at a time using whatever the current daily_target is (a
    reasonable approximation for missed days).

    Returns {"rollovers": [(date, leftover), ...], "new_best_streak": int|None}
    """
    user = get_or_create_user(chat_id)
    last = date.fromisoformat(user["last_rollover_date"])
    today = date.fromisoformat(today_str())
    result = {"rollovers": [], "new_best_streak": None}
    if today <= last:
        return result

    with get_conn() as conn:
        balance = user["balance"]
        target = user["daily_target"]
        streak = user["current_streak"]
        best = user["best_streak"]
        d = last
        while d < today:
            spent = _spent_on(conn, chat_id, d.isoformat())
            leftover = target - spent
            balance += leftover
            result["rollovers"].append((d.isoformat(), leftover))

            if spent <= target:
                streak += 1
                if streak > best:
                    best = streak
                    result["new_best_streak"] = best
            else:
                streak = 0

            d += timedelta(days=1)

        conn.execute(
            "UPDATE users SET balance = ?, last_rollover_date = ?, current_streak = ?, best_streak = ? "
            "WHERE chat_id = ?",
            (balance, today.isoformat(), streak, best, chat_id),
        )
    return result


def get_status(chat_id: int) -> dict:
    ensure_rollover(chat_id)
    user = get_or_create_user(chat_id)
    today = today_str()
    with get_conn() as conn:
        spent_today = _spent_on(conn, chat_id, today)
        pending_claimable = conn.execute(
            "SELECT COALESCE(SUM(amount_base), 0) AS total FROM expenses "
            "WHERE chat_id = ? AND is_claimable = 1 AND is_claimed = 0",
            (chat_id,),
        ).fetchone()["total"]
    available_today = user["daily_target"] + user["balance"] - spent_today
    return {
        "daily_target": user["daily_target"],
        "balance": user["balance"],
        "spent_today": spent_today,
        "available_today": available_today,
        "pending_claimable": pending_claimable,
        "current_streak": user["current_streak"],
        "best_streak": user["best_streak"],
    }


def adjust_balance(chat_id: int, delta: float) -> dict:
    """Manually nudges the rolling `balance` by `delta` (negative = adding a
    deficit, positive = adding a credit) -- the one supported way to correct
    balance drift that the automatic day-by-day rollover can't reconstruct on
    its own (see ensure_rollover's docstring: it derives each day's leftover
    entirely from that day's rows in `expenses`, so if expense history from
    before some date is gone -- e.g. lost to a redeploy without persistent
    storage -- those days silently roll over as if $0 was spent, which is
    the opposite of a real deficit). The amount always comes from the user
    literally, never invented or estimated by the model, the same "real
    numbers only" discipline as edit_amount. Calls ensure_rollover first so
    the adjustment lands on an up-to-date balance rather than a stale one.
    Returns {"old_balance": ..., "new_balance": ...}."""
    ensure_rollover(chat_id)
    user = get_or_create_user(chat_id)
    old_balance = user["balance"]
    new_balance = old_balance + delta
    with get_conn() as conn:
        conn.execute("UPDATE users SET balance = ? WHERE chat_id = ?", (new_balance, chat_id))
    return {"old_balance": old_balance, "new_balance": new_balance}


def set_balance(chat_id: int, balance: float) -> None:
    """Only used to revert a prior adjust_balance -- restores the exact
    pre-adjustment value from the undo snapshot rather than recomputing
    anything, the same "replay the snapshot" discipline as every other
    domain's undo."""
    with get_conn() as conn:
        conn.execute("UPDATE users SET balance = ? WHERE chat_id = ?", (balance, chat_id))


def maybe_alert(chat_id: int) -> bool:
    """Returns True (and marks it sent) the first time today that spending
    crosses config.BUDGET_ALERT_THRESHOLD of today's available budget."""
    status = get_status(chat_id)
    today = today_str()
    with get_conn() as conn:
        user = conn.execute("SELECT last_alert_date FROM users WHERE chat_id = ?", (chat_id,)).fetchone()
        if user["last_alert_date"] == today:
            return False
        available_total = status["daily_target"] + status["balance"]
        if available_total <= 0:
            return False
        ratio = status["spent_today"] / available_total
        if ratio >= config.BUDGET_ALERT_THRESHOLD:
            conn.execute("UPDATE users SET last_alert_date = ? WHERE chat_id = ?", (today, chat_id))
            return True
    return False


def add_expense(chat_id: int, amount: float, currency: str, description: str, category: str,
                 is_claimable: bool = False, expense_date: str | None = None) -> int:
    """expense_date defaults to today -- pass it explicitly only when the
    caller already computed a real backdated date deterministically (see
    add_meal's docstring for the same reasoning; never a date guessed by
    the AI itself). Rollover is always ensured for TODAY regardless -- an
    expense logged onto an earlier day doesn't change where today's own
    balance boundary sits."""
    ensure_rollover(chat_id)
    get_or_create_user(chat_id)
    currency = (currency or config.BASE_CURRENCY).upper()
    amount_base = fx.to_base(amount, currency)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO expenses (chat_id, amount, currency, amount_base, description, category, "
            "is_claimable, expense_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, amount, currency, amount_base, description, category, int(is_claimable),
             expense_date or today_str()),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def clear_claimables(chat_id: int) -> tuple[int, float]:
    """Marks all pending claimables as claimed. Returns (count, total_base)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount_base), 0) AS total FROM expenses "
            "WHERE chat_id = ? AND is_claimable = 1 AND is_claimed = 0",
            (chat_id,),
        ).fetchone()
        conn.execute(
            "UPDATE expenses SET is_claimed = 1 WHERE chat_id = ? AND is_claimable = 1 AND is_claimed = 0",
            (chat_id,),
        )
        return rows["n"], rows["total"]


def get_pending_claimables(chat_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, amount, currency, description, category, expense_date FROM expenses "
            "WHERE chat_id = ? AND is_claimable = 1 AND is_claimed = 0 ORDER BY id",
            (chat_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_expenses(chat_id: int, limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, amount, currency, description, category, is_claimable, is_claimed, expense_date "
            "FROM expenses WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_expense(chat_id: int, expense_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM expenses WHERE id = ? AND chat_id = ?", (expense_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def _adjust_balance_for_past_day(conn: sqlite3.Connection, chat_id: int, delta_base: float) -> None:
    """delta_base > 0 means less was spent than before (refund to balance)."""
    if delta_base == 0:
        return
    conn.execute("UPDATE users SET balance = balance + ? WHERE chat_id = ?", (delta_base, chat_id))


def delete_expense(chat_id: int, expense_id: int) -> dict | None:
    """Deletes an expense. If it was a past (already rolled-over) non-claimable
    day, credits the balance back so history stays consistent. Returns the
    deleted row as a dict, or None if it didn't exist / wasn't this chat's."""
    ensure_rollover(chat_id)
    row = get_expense(chat_id, expense_id)
    if row is None:
        return None
    with get_conn() as conn:
        if not row["is_claimable"] and row["expense_date"] < today_str():
            _adjust_balance_for_past_day(conn, chat_id, row["amount_base"])
        conn.execute("DELETE FROM expenses WHERE id = ? AND chat_id = ?", (expense_id, chat_id))
    return row


def delete_most_recent(chat_id: int) -> dict | None:
    """Undo: deletes the single most recent expense for this chat."""
    recent = get_recent_expenses(chat_id, limit=1)
    if not recent:
        return None
    return delete_expense(chat_id, recent[0]["id"])


def restore_deleted_expense(chat_id: int, row: dict) -> dict | None:
    """Re-inserts a previously deleted expense row exactly as it was (same
    amount/currency/amount_base/description/category/claimable-ness/date),
    applying the exact opposite balance adjustment delete_expense would have
    made. Used to support one-step 'undo' after a natural-language
    correction deletes the wrong entry -- gets a fresh row id, since SQLite
    won't recycle the old one, but every other field is preserved."""
    ensure_rollover(chat_id)
    with get_conn() as conn:
        if not row["is_claimable"] and row["expense_date"] < today_str():
            _adjust_balance_for_past_day(conn, chat_id, -row["amount_base"])
        conn.execute(
            "INSERT INTO expenses (chat_id, amount, currency, amount_base, description, category, "
            "is_claimable, is_claimed, expense_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, row["amount"], row["currency"], row["amount_base"], row["description"],
             row["category"], row["is_claimable"], row.get("is_claimed", 0), row["expense_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_expense(chat_id, new_id)


def edit_expense(chat_id: int, expense_id: int, new_amount: float | None = None,
                  new_currency: str | None = None, new_description: str | None = None,
                  new_category: str | None = None) -> dict | None:
    """Updates an expense in place. Only touches fields that are passed in.
    Recomputes amount_base if amount or currency changed, and (for
    non-claimable expenses on already-rolled-over days) adjusts the running
    balance by the difference so history stays consistent. Returns the
    updated row, or None if it didn't exist / wasn't this chat's."""
    ensure_rollover(chat_id)
    row = get_expense(chat_id, expense_id)
    if row is None:
        return None

    final_amount = new_amount if new_amount is not None else row["amount"]
    final_currency = (new_currency or row["currency"]).upper()
    recompute = new_amount is not None or new_currency is not None
    new_amount_base = fx.to_base(final_amount, final_currency) if recompute else row["amount_base"]

    final_description = new_description if new_description is not None else row["description"]
    final_category = new_category if new_category is not None else row["category"]

    with get_conn() as conn:
        if not row["is_claimable"] and row["expense_date"] < today_str():
            delta = row["amount_base"] - new_amount_base
            _adjust_balance_for_past_day(conn, chat_id, delta)
        conn.execute(
            "UPDATE expenses SET amount = ?, currency = ?, amount_base = ?, description = ?, category = ? "
            "WHERE id = ? AND chat_id = ?",
            (final_amount, final_currency, new_amount_base, final_description, final_category,
             expense_id, chat_id),
        )
    return get_expense(chat_id, expense_id)


def edit_expense_date(chat_id: int, expense_id: int, new_date_str: str) -> dict | None:
    """Moves a non-claimable expense to a different expense_date, adjusting
    the running balance so history stays consistent. new_date_str is an ISO
    date string; dates after today are clamped to today (this app doesn't
    support logging into the future).

    The balance math depends on whether each side of the move is "today"
    (live -- spent_today is computed fresh on every read, nothing stored) or
    an already-rolled-over past day (baked into the single cumulative
    `balance` number at rollover time):

      - past day -> past day: the amount is removed from one already-rolled
        day's spend and added to another already-rolled day's spend. Since
        both use the same (current) daily_target approximation and both feed
        the same cumulative `balance`, the two adjustments cancel out
        exactly -- net zero change to balance.
      - today (live) -> past day: the amount leaves today's live spend
        (automatic once expense_date changes) and now retroactively counts
        against a day whose rollover already happened, so that day's
        leftover -- and therefore balance -- decreases by amount_base.
      - past day -> today (live): the reverse -- the amount is removed from
        an already-rolled day (balance increases by amount_base) and now
        counts against today's live, not-yet-rolled spend instead.
      - same date (no-op): nothing to do.

    Claimable expenses never touch balance, so their date can move freely
    with no adjustment. Returns the updated row, or None if the expense
    doesn't exist / isn't this chat's.
    """
    ensure_rollover(chat_id)
    row = get_expense(chat_id, expense_id)
    if row is None:
        return None

    today = today_str()
    if new_date_str > today:
        new_date_str = today
    old_date_str = row["expense_date"]

    if new_date_str == old_date_str:
        return row

    with get_conn() as conn:
        if not row["is_claimable"]:
            old_is_past = old_date_str < today
            new_is_past = new_date_str < today
            if old_is_past and not new_is_past:
                # leaving an already-rolled day -> that day's leftover goes up
                _adjust_balance_for_past_day(conn, chat_id, row["amount_base"])
            elif not old_is_past and new_is_past:
                # entering an already-rolled day -> that day's leftover goes down
                _adjust_balance_for_past_day(conn, chat_id, -row["amount_base"])
            # past -> past nets to zero (see docstring); no adjustment needed
        conn.execute(
            "UPDATE expenses SET expense_date = ? WHERE id = ? AND chat_id = ?",
            (new_date_str, expense_id, chat_id),
        )
    return get_expense(chat_id, expense_id)


def get_all_chat_ids() -> list[int]:
    with get_conn() as conn:
        rows = conn.execute("SELECT chat_id FROM users").fetchall()
        return [r["chat_id"] for r in rows]


def get_category_totals(chat_id: int, start_date: str, end_date: str) -> list[dict]:
    """Category breakdown (in base currency) of non-claimable spending in
    [start_date, end_date) -- both ISO date strings, end_date exclusive."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT category, COALESCE(SUM(amount_base), 0) AS total, COUNT(*) AS n FROM expenses "
            "WHERE chat_id = ? AND is_claimable = 0 AND expense_date >= ? AND expense_date < ? "
            "GROUP BY category ORDER BY total DESC",
            (chat_id, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


def get_largest_expenses(chat_id: int, start_date: str, end_date: str, limit: int = 5) -> list[dict]:
    """The `limit` largest individual non-claimable expenses (by amount_base)
    in [start_date, end_date) -- both ISO date strings, end_date exclusive.
    Feeds the /summary insights upgrade: a category total alone can't tell
    a one-off big-ticket item from a spread of small purchases, so the
    summary needs the actual standout transactions, not just the aggregate."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, amount, currency, amount_base, description, category, expense_date FROM expenses "
            "WHERE chat_id = ? AND is_claimable = 0 AND expense_date >= ? AND expense_date < ? "
            "ORDER BY amount_base DESC LIMIT ?",
            (chat_id, start_date, end_date, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_daily_totals(chat_id: int, start_date: str, end_date: str) -> dict[str, float]:
    """Per-day non-claimable totals (base currency) in [start_date, end_date)
    -- both ISO date strings, end_date exclusive. Returns {date_str: total},
    omitting days with no spend (callers should default missing days to 0)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT expense_date, COALESCE(SUM(amount_base), 0) AS total FROM expenses "
            "WHERE chat_id = ? AND is_claimable = 0 AND expense_date >= ? AND expense_date < ? "
            "GROUP BY expense_date",
            (chat_id, start_date, end_date),
        ).fetchall()
        return {r["expense_date"]: r["total"] for r in rows}


def get_month_to_date_total(chat_id: int) -> dict:
    """Sum of non-claimable spend (base currency) from the 1st of the
    current month through today (inclusive). Computed fresh from the raw
    expense rows every call -- no stored running total, so this needed no
    schema change and naturally resets to zero on the 1st of every month
    with zero migration risk, alongside (not replacing) the existing
    rolling `balance` on the user row."""
    today = date.fromisoformat(today_str())
    month_start = today.replace(day=1)
    tomorrow = today + timedelta(days=1)
    totals = get_category_totals(chat_id, month_start.isoformat(), tomorrow.isoformat())
    return {
        "total": round(sum(r["total"] for r in totals), 2),
        "month_start": month_start.isoformat(),
        "today": today.isoformat(),
        "days_elapsed": (today - month_start).days + 1,
    }


# ---------- meals ----------
# Each row is one logged item/entry, not one row per meal slot -- see the
# module docstring. calories_estimate is the number everything else (running
# totals, /summary-style rollups) sums; low/high are kept for display only.

def _meal_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["items"] = json.loads(d["items"]) if d["items"] else []
    except (TypeError, json.JSONDecodeError):
        d["items"] = []
    return d


def add_meal(chat_id: int, meal_type: str | None, items: list[str] | None, calories_low: float | None,
             calories_high: float | None, calories_estimate: float | None, water_ml: float | None = None,
             meal_date: str | None = None) -> int:
    """items: list of strings. Never raises on bad estimate math -- a null
    calories_estimate is stored as-is rather than blocking the log (mirrors
    fx.py's "never block on an estimate/lookup failure" discipline).
    meal_date defaults to today -- pass it explicitly only when the caller
    already computed a real backdated date deterministically (e.g.
    nutrition.handle_photo, from a photo caption naming a specific past
    day) -- never pass a date guessed by the AI itself."""
    get_or_create_user(chat_id)
    items_json = json.dumps(items or [])
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO meals (chat_id, meal_type, items, calories_low, calories_high, "
            "calories_estimate, water_ml, meal_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, meal_type, items_json, calories_low, calories_high, calories_estimate,
             water_ml, meal_date or today_str()),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_recent_meals(chat_id: int, limit: int = 10) -> list[dict]:
    """Ordered by meal_date DESC (id DESC only as a same-day tiebreak), NOT
    just id DESC -- a real observed bug: a historical backfill inserts old-
    dated rows LAST, so they get the newest ids despite being the oldest
    dates. Ordering by id alone then made a July entry outrank a genuinely
    more recent live-logged one in every "recent" list (this function,
    plus the vitals.py trend line and any 'edit/undo my last X' correction
    that resolves against "the most recent" row) -- id no longer tracks
    chronological order once any backfill/import has ever run."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM meals WHERE chat_id = ? ORDER BY meal_date DESC, id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [_meal_row(r) for r in rows]


def get_meals_in_range(chat_id: int, start_date: str, end_date: str) -> list[dict]:
    """Meal rows in [start_date, end_date) -- both ISO date strings,
    end_date exclusive. Same bounds convention as get_category_totals --
    used for the rundown synthesis (bot.py's _rundown_payload), not for
    display, so it returns full rows rather than a pre-aggregated total."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM meals WHERE chat_id = ? AND meal_date >= ? AND meal_date < ? ORDER BY id",
            (chat_id, start_date, end_date),
        ).fetchall()
        return [_meal_row(r) for r in rows]


def get_meal(chat_id: int, meal_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM meals WHERE id = ? AND chat_id = ?", (meal_id, chat_id)
        ).fetchone()
        return _meal_row(row) if row else None


def get_daily_meal_totals(chat_id: int, day_str: str) -> dict:
    """Running totals for one day -- the reply pattern this mirrors (from the
    ChatGPT thread this replaces) always shows a running calorie/water total
    alongside each new item, not just the item just logged."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(calories_estimate), 0) AS calories, "
            "COALESCE(SUM(water_ml), 0) AS water_ml FROM meals "
            "WHERE chat_id = ? AND meal_date = ?",
            (chat_id, day_str),
        ).fetchone()
        return {"calories": row["calories"], "water_ml": row["water_ml"]}


def edit_meal_date(chat_id: int, meal_id: int, new_date_str: str) -> dict | None:
    """Moves a meal to a different meal_date. Unlike expenses, meals don't
    feed a rolling balance, so this is a plain field update -- no balance
    adjustment needed. Dates after today are clamped to today."""
    row = get_meal(chat_id, meal_id)
    if row is None:
        return None
    today = today_str()
    if new_date_str > today:
        new_date_str = today
    with get_conn() as conn:
        conn.execute(
            "UPDATE meals SET meal_date = ? WHERE id = ? AND chat_id = ?",
            (new_date_str, meal_id, chat_id),
        )
    return get_meal(chat_id, meal_id)


# Distinct from None -- a meal/to-do's editable fields are legitimately
# nullable, so None is a real value ("clear this field") that edit_meal/
# edit_task must be able to set on purpose (e.g. undoing an edit that added
# a due date to a task that previously had none). _UNSET is the "caller
# didn't mention this field at all" default instead, the same "only touches
# what's passed" discipline as edit_expense, but edit_expense never needs to
# null out amount/description/category, so it never ran into this. Defined
# here, before its first use (edit_meal, below) -- a default argument value
# is bound at `def` time, not call time, so it has to exist by the time
# Python evaluates this function's signature, not just by the time it runs.
_UNSET = object()


def edit_meal(chat_id: int, meal_id: int, new_meal_type=_UNSET, new_items=_UNSET, new_calories_low=_UNSET,
              new_calories_high=_UNSET, new_calories_estimate=_UNSET, new_water_ml=_UNSET) -> dict | None:
    """Corrects what was actually in an already-logged meal -- e.g. an item a
    photo (or the model) added that wasn't really eaten -- without deleting
    and relogging from scratch. Mirrors edit_task's _UNSET "only touch what's
    passed" discipline (see its docstring for why that's not the same as
    passing None), but unlike edit_task the three calorie fields are meant to
    always be set together whenever items changes (see correction.py's
    edit_meal handling) -- a corrected item list with a stale calorie
    estimate would be worse than not correcting it at all."""
    row = get_meal(chat_id, meal_id)
    if row is None:
        return None
    final_meal_type = row["meal_type"] if new_meal_type is _UNSET else new_meal_type
    final_items = row["items"] if new_items is _UNSET else new_items
    final_low = row["calories_low"] if new_calories_low is _UNSET else new_calories_low
    final_high = row["calories_high"] if new_calories_high is _UNSET else new_calories_high
    final_estimate = row["calories_estimate"] if new_calories_estimate is _UNSET else new_calories_estimate
    final_water_ml = row["water_ml"] if new_water_ml is _UNSET else new_water_ml
    with get_conn() as conn:
        conn.execute(
            "UPDATE meals SET meal_type = ?, items = ?, calories_low = ?, calories_high = ?, "
            "calories_estimate = ?, water_ml = ? WHERE id = ? AND chat_id = ?",
            (final_meal_type, json.dumps(final_items or []), final_low, final_high, final_estimate,
             final_water_ml, meal_id, chat_id),
        )
    return get_meal(chat_id, meal_id)


def delete_meal(chat_id: int, meal_id: int) -> dict | None:
    row = get_meal(chat_id, meal_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM meals WHERE id = ? AND chat_id = ?", (meal_id, chat_id))
    return row


def delete_most_recent_meal(chat_id: int) -> dict | None:
    recent = get_recent_meals(chat_id, limit=1)
    if not recent:
        return None
    return delete_meal(chat_id, recent[0]["id"])


def restore_deleted_meal(chat_id: int, row: dict) -> dict | None:
    """Re-inserts a previously deleted meal row exactly as it was. Used for
    one-step 'undo' after a natural-language correction deletes the wrong
    entry -- gets a fresh row id, every other field preserved."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO meals (chat_id, meal_type, items, calories_low, calories_high, "
            "calories_estimate, water_ml, meal_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, row["meal_type"], json.dumps(row["items"]), row["calories_low"],
             row["calories_high"], row["calories_estimate"], row["water_ml"], row["meal_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_meal(chat_id, new_id)


# ---------- workouts ----------

def add_workout(chat_id: int, activity: str, duration_min: float | None = None,
                 distance_km: float | None = None, notes: str | None = None,
                 calories_burned: float | None = None, workout_date: str | None = None) -> int:
    """workout_date defaults to today -- pass it explicitly only when the
    caller already computed a real backdated date deterministically (see
    add_meal's docstring for the same reasoning; never a date guessed by
    the AI itself)."""
    get_or_create_user(chat_id)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO workouts (chat_id, activity, duration_min, distance_km, calories_burned, notes, "
            "workout_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, activity, duration_min, distance_km, calories_burned, notes, workout_date or today_str()),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_recent_workouts(chat_id: int, limit: int = 10) -> list[dict]:
    """Ordered by workout_date DESC (id DESC only as a same-day tiebreak) --
    see get_recent_meals's docstring for why id alone isn't safe once a
    historical backfill has ever inserted old-dated rows with new ids."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM workouts WHERE chat_id = ? ORDER BY workout_date DESC, id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_workouts_in_range(chat_id: int, start_date: str, end_date: str) -> list[dict]:
    """See get_meals_in_range's docstring -- same convention, used by
    bot.py's _rundown_payload."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM workouts WHERE chat_id = ? AND workout_date >= ? AND workout_date < ? ORDER BY id",
            (chat_id, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


def get_workout(chat_id: int, workout_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM workouts WHERE id = ? AND chat_id = ?", (workout_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def edit_workout_date(chat_id: int, workout_id: int, new_date_str: str) -> dict | None:
    row = get_workout(chat_id, workout_id)
    if row is None:
        return None
    today = today_str()
    if new_date_str > today:
        new_date_str = today
    with get_conn() as conn:
        conn.execute(
            "UPDATE workouts SET workout_date = ? WHERE id = ? AND chat_id = ?",
            (new_date_str, workout_id, chat_id),
        )
    return get_workout(chat_id, workout_id)


def edit_workout(chat_id: int, workout_id: int, new_activity=_UNSET, new_duration_min=_UNSET,
                  new_distance_km=_UNSET, new_calories_burned=_UNSET, new_notes=_UNSET) -> dict | None:
    """Corrects a field on an already-logged workout (most commonly
    calories_burned, e.g. a photo's workout-summary total was read
    wrong or the user's tracker later showed a more complete number
    including BMR) without deleting and relogging from scratch. Same
    _UNSET "only touch what's passed" discipline as edit_meal/edit_task --
    see db._UNSET's docstring, defined above this in db.py."""
    row = get_workout(chat_id, workout_id)
    if row is None:
        return None
    final_activity = row["activity"] if new_activity is _UNSET else new_activity
    final_duration = row["duration_min"] if new_duration_min is _UNSET else new_duration_min
    final_distance = row["distance_km"] if new_distance_km is _UNSET else new_distance_km
    final_calories = row["calories_burned"] if new_calories_burned is _UNSET else new_calories_burned
    final_notes = row["notes"] if new_notes is _UNSET else new_notes
    with get_conn() as conn:
        conn.execute(
            "UPDATE workouts SET activity = ?, duration_min = ?, distance_km = ?, calories_burned = ?, "
            "notes = ? WHERE id = ? AND chat_id = ?",
            (final_activity, final_duration, final_distance, final_calories, final_notes, workout_id, chat_id),
        )
    return get_workout(chat_id, workout_id)


def delete_workout(chat_id: int, workout_id: int) -> dict | None:
    row = get_workout(chat_id, workout_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM workouts WHERE id = ? AND chat_id = ?", (workout_id, chat_id))
    return row


def delete_most_recent_workout(chat_id: int) -> dict | None:
    recent = get_recent_workouts(chat_id, limit=1)
    if not recent:
        return None
    return delete_workout(chat_id, recent[0]["id"])


def restore_deleted_workout(chat_id: int, row: dict) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO workouts (chat_id, activity, duration_min, distance_km, calories_burned, notes, "
            "workout_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, row["activity"], row["duration_min"], row["distance_km"], row.get("calories_burned"),
             row["notes"], row["workout_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_workout(chat_id, new_id)


def get_daily_workout_totals(chat_id: int, day_str: str) -> dict:
    """Running calories-burned total for one day -- the counterpart to
    get_daily_meal_totals, used to show calories burned against calories
    eaten (see formatting._daily_calorie_balance_text)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(calories_burned), 0) AS calories_burned FROM workouts "
            "WHERE chat_id = ? AND workout_date = ?",
            (chat_id, day_str),
        ).fetchone()
        return {"calories_burned": row["calories_burned"]}


# ---------- lifts ----------
# Structured gym-exercise logging -- deliberately separate from workouts,
# which stays a single free-text blob per session (cardio, tennis, a
# fitness-app daily-activity summary). A lift row is one EXERCISE within a
# session, not one session -- a single gym day producing "pull-ups 10x3,
# v-bar rows 35kg 8x3, lat pulldown 70kg 8x3" is three rows, the same "one
# row per item, not one per message" discipline as meals. This is Phase A
# of the coaching-engine work: just getting real per-(exercise, location)
# data flowing into structured rows instead of vanishing into a free-text
# activity string -- no progression/target logic reads this yet.

def _lift_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["sets"] = json.loads(d["sets"]) if d["sets"] else []
    except (TypeError, json.JSONDecodeError):
        d["sets"] = []
    return d


def add_lift(chat_id: int, exercise: str, location: str | None, sets: list[dict] | None,
             effort: str | None = None, context_notes: str | None = None,
             lift_date: str | None = None) -> int:
    """sets: list of {"reps": int|None, "load": str|None} dicts, one per set
    actually described, in the order given -- "load" is always a string
    (never forced into a number) since a numbered-machine setting ("setting
    21") is just as valid a load as a real weight ("35kg"). lift_date
    defaults to today -- pass it explicitly only when the caller already
    computed a real backdated date deterministically, same discipline as
    add_meal/add_workout's own docstrings."""
    get_or_create_user(chat_id)
    sets_json = json.dumps(sets or [])
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO lifts (chat_id, exercise, location, sets, effort, context_notes, lift_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, exercise, location, sets_json, effort, context_notes, lift_date or today_str()),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_recent_lifts(chat_id: int, limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM lifts WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [_lift_row(r) for r in rows]


def get_lift(chat_id: int, lift_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM lifts WHERE id = ? AND chat_id = ?", (lift_id, chat_id)
        ).fetchone()
        return _lift_row(row) if row else None


def edit_lift_date(chat_id: int, lift_id: int, new_date_str: str) -> dict | None:
    row = get_lift(chat_id, lift_id)
    if row is None:
        return None
    today = today_str()
    if new_date_str > today:
        new_date_str = today
    with get_conn() as conn:
        conn.execute(
            "UPDATE lifts SET lift_date = ? WHERE id = ? AND chat_id = ?",
            (new_date_str, lift_id, chat_id),
        )
    return get_lift(chat_id, lift_id)


def edit_lift(chat_id: int, lift_id: int, new_exercise=_UNSET, new_location=_UNSET, new_sets=_UNSET,
              new_effort=_UNSET, new_context_notes=_UNSET) -> dict | None:
    """Corrects a field on an already-logged lift -- most commonly the sets
    themselves (a mis-typed rep count, a set left out, a set that wasn't
    actually done) -- without deleting and relogging the exercise from
    scratch. Same _UNSET "only touch what's passed" discipline as
    edit_meal/edit_workout -- see db._UNSET's docstring above."""
    row = get_lift(chat_id, lift_id)
    if row is None:
        return None
    final_exercise = row["exercise"] if new_exercise is _UNSET else new_exercise
    final_location = row["location"] if new_location is _UNSET else new_location
    final_sets = row["sets"] if new_sets is _UNSET else new_sets
    final_effort = row["effort"] if new_effort is _UNSET else new_effort
    final_context_notes = row["context_notes"] if new_context_notes is _UNSET else new_context_notes
    with get_conn() as conn:
        conn.execute(
            "UPDATE lifts SET exercise = ?, location = ?, sets = ?, effort = ?, context_notes = ? "
            "WHERE id = ? AND chat_id = ?",
            (final_exercise, final_location, json.dumps(final_sets or []), final_effort, final_context_notes,
             lift_id, chat_id),
        )
    return get_lift(chat_id, lift_id)


def delete_lift(chat_id: int, lift_id: int) -> dict | None:
    row = get_lift(chat_id, lift_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM lifts WHERE id = ? AND chat_id = ?", (lift_id, chat_id))
    return row


def restore_deleted_lift(chat_id: int, row: dict) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO lifts (chat_id, exercise, location, sets, effort, context_notes, lift_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, row["exercise"], row["location"], json.dumps(row.get("sets") or []),
             row.get("effort"), row.get("context_notes"), row["lift_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_lift(chat_id, new_id)


# ---------- vitals ----------
# One row per daily check-in. All fields nullable -- a check-in commonly
# reports only some of weight/sleep/knee, plus a free-text note.

def add_vitals(chat_id: int, weight_kg: float | None = None, sleep_hours: float | None = None,
                knee_pain: float | None = None, notes: str | None = None,
                vitals_date: str | None = None) -> int:
    """vitals_date defaults to today -- pass it explicitly only when the
    caller already computed a real backdated date deterministically (see
    add_meal's docstring for the same reasoning; never a date guessed by
    the AI itself)."""
    get_or_create_user(chat_id)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO vitals (chat_id, weight_kg, sleep_hours, knee_pain, notes, vitals_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, weight_kg, sleep_hours, knee_pain, notes, vitals_date or today_str()),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_recent_vitals(chat_id: int, limit: int = 10) -> list[dict]:
    """Ordered by vitals_date DESC (id DESC only as a same-day tiebreak) --
    see get_recent_meals's docstring for why id alone isn't safe once a
    historical backfill has ever inserted old-dated rows with new ids.
    This is a real, observed bug this fixes: after a July-September
    backfill ran, _vitals_trend_text's "since last check-in" compared a
    new entry against a stale July/September-11 row instead of a
    genuinely more recent one, because id order no longer matched date
    order for the stretch where backfilled and live-logged rows overlap."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM vitals WHERE chat_id = ? ORDER BY vitals_date DESC, id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_vitals_in_range(chat_id: int, start_date: str, end_date: str) -> list[dict]:
    """See get_meals_in_range's docstring -- same convention, used by
    bot.py's _rundown_payload. Ordered oldest-first so callers can read
    weights[0]/weights[-1] as "start of window" / "latest" directly."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM vitals WHERE chat_id = ? AND vitals_date >= ? AND vitals_date < ? ORDER BY id",
            (chat_id, start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


def get_vitals(chat_id: int, vitals_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM vitals WHERE id = ? AND chat_id = ?", (vitals_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def edit_vitals_date(chat_id: int, vitals_id: int, new_date_str: str) -> dict | None:
    row = get_vitals(chat_id, vitals_id)
    if row is None:
        return None
    today = today_str()
    if new_date_str > today:
        new_date_str = today
    with get_conn() as conn:
        conn.execute(
            "UPDATE vitals SET vitals_date = ? WHERE id = ? AND chat_id = ?",
            (new_date_str, vitals_id, chat_id),
        )
    return get_vitals(chat_id, vitals_id)


def edit_vitals(chat_id: int, vitals_id: int, new_weight_kg=_UNSET, new_sleep_hours=_UNSET,
                 new_knee_pain=_UNSET, new_notes=_UNSET) -> dict | None:
    """Corrects a field on an already-logged check-in -- most commonly a
    mis-typed weight/sleep/knee number -- without deleting and relogging
    from scratch. Same _UNSET "only touch what's passed" discipline as
    edit_meal/edit_workout/edit_lift -- see db._UNSET's docstring above."""
    row = get_vitals(chat_id, vitals_id)
    if row is None:
        return None
    final_weight = row["weight_kg"] if new_weight_kg is _UNSET else new_weight_kg
    final_sleep = row["sleep_hours"] if new_sleep_hours is _UNSET else new_sleep_hours
    final_knee = row["knee_pain"] if new_knee_pain is _UNSET else new_knee_pain
    final_notes = row["notes"] if new_notes is _UNSET else new_notes
    with get_conn() as conn:
        conn.execute(
            "UPDATE vitals SET weight_kg = ?, sleep_hours = ?, knee_pain = ?, notes = ? "
            "WHERE id = ? AND chat_id = ?",
            (final_weight, final_sleep, final_knee, final_notes, vitals_id, chat_id),
        )
    return get_vitals(chat_id, vitals_id)


def delete_vitals(chat_id: int, vitals_id: int) -> dict | None:
    row = get_vitals(chat_id, vitals_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM vitals WHERE id = ? AND chat_id = ?", (vitals_id, chat_id))
    return row


def delete_most_recent_vitals(chat_id: int) -> dict | None:
    recent = get_recent_vitals(chat_id, limit=1)
    if not recent:
        return None
    return delete_vitals(chat_id, recent[0]["id"])


def restore_deleted_vitals(chat_id: int, row: dict) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO vitals (chat_id, weight_kg, sleep_hours, knee_pain, notes, vitals_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, row["weight_kg"], row["sleep_hours"], row["knee_pain"], row["notes"],
             row["vitals_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_vitals(chat_id, new_id)


# ---------- rolling conversation history ----------

def add_message(chat_id: int, role: str, content: str) -> None:
    """role is 'user' or 'morrow'. No upper bound on table growth yet --
    a personal chat's text is small enough that this isn't worth trimming
    until it's actually a problem."""
    get_or_create_user(chat_id)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content),
        )


def get_recent_messages(chat_id: int, limit: int = 30) -> list[dict]:
    """Returns the last `limit` turns in chronological order (oldest first)
    -- ready to drop straight into a prompt as conversation history, unlike
    get_recent_* elsewhere which return newest-first for display."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ---------- durable memory ----------

def set_memory(chat_id: int, label: str, content: str, category: str | None = None) -> int:
    """Upsert by (chat_id, label), case-insensitive -- 'remember the biopolis
    plan' twice edits the same row instead of creating a near-duplicate.
    Returns the memory id."""
    get_or_create_user(chat_id)
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM memory WHERE chat_id = ? AND label = ? COLLATE NOCASE",
            (chat_id, label),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE memory SET content = ?, category = COALESCE(?, category), "
                "updated_at = datetime('now') WHERE id = ?",
                (content, category, existing["id"]),
            )
            return existing["id"]
        conn.execute(
            "INSERT INTO memory (chat_id, label, category, content) VALUES (?, ?, ?, ?)",
            (chat_id, label, category, content),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_memory_list(chat_id: int, limit: int = 40) -> list[dict]:
    """Most-recently-updated first -- meant to be read in full into every
    prompt (see the messages module docstring), so this is capped the same
    way get_recent_meals etc. are, not because it's a display list."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memory WHERE chat_id = ? ORDER BY updated_at DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_memory_by_label(chat_id: int, label: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM memory WHERE chat_id = ? AND label = ? COLLATE NOCASE",
            (chat_id, label),
        ).fetchone()
        return dict(row) if row else None


def delete_memory_by_label(chat_id: int, label: str) -> dict | None:
    row = get_memory_by_label(chat_id, label)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM memory WHERE id = ? AND chat_id = ?", (row["id"], chat_id))
    return row


def restore_deleted_memory(chat_id: int, row: dict) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO memory (chat_id, label, category, content, updated_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, row["label"], row["category"], row["content"], row["updated_at"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_memory_by_label(chat_id, row["label"]) if new_id else None


# ---------- tasks ----------

def add_task(chat_id: int, title: str, due_at: str | None = None, notes: str | None = None,
             recurrence_frequency: str | None = None) -> int:
    """recurrence_frequency (one of db.RECURRENCE_FREQUENCIES, or None for a
    plain one-off to-do) makes this a RECURRING task -- see mark_task_done's
    docstring for what that changes about "done". Deliberately its own
    optional field on the existing tasks table rather than a second table:
    a recurring task is still fundamentally a to-do (title, due date,
    notes, open/not-open) with one extra piece of config, so it reuses
    every existing to-do code path (/tasks, /done, correction.py's
    edit_task) instead of duplicating them for a second domain."""
    get_or_create_user(chat_id)
    if recurrence_frequency is not None and recurrence_frequency not in RECURRENCE_FREQUENCIES:
        recurrence_frequency = None
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks (chat_id, title, due_at, notes, recurrence_frequency) VALUES (?, ?, ?, ?, ?)",
            (chat_id, title, due_at, notes, recurrence_frequency),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_task(chat_id: int, task_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ? AND chat_id = ?", (task_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def get_recent_tasks(chat_id: int, limit: int = 10) -> list[dict]:
    """Most-recently-created first, matching every other domain's
    get_recent_* -- this is what feeds the AI's recent-items list for
    corrections, not the day-to-day to-do view (see get_open_tasks)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_open_tasks(chat_id: int, limit: int = 20) -> list[dict]:
    """Not-done tasks, soonest due first (rows with no due_at sort last) --
    what /tasks and a future morning briefing actually want to show,
    as opposed to get_recent_tasks's creation-order list."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE chat_id = ? AND done = 0 "
            "ORDER BY (due_at IS NULL), due_at, id LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def edit_task_due(chat_id: int, task_id: int, new_due_at: str | None) -> dict | None:
    row = get_task(chat_id, task_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET due_at = ? WHERE id = ? AND chat_id = ?", (new_due_at, task_id, chat_id)
        )
    return get_task(chat_id, task_id)


def edit_task(chat_id: int, task_id: int, new_title=_UNSET, new_due_at=_UNSET, new_notes=_UNSET,
              new_recurrence_frequency=_UNSET) -> dict | None:
    """General to-do editor -- title, due date, notes, and recurrence can
    each change independently in one call, whichever combination the
    correction actually implies (e.g. "push #11 to tomorrow, I need
    Shardul's address" reschedules AND adds a note in one shot). Pass a
    real value (including None) for a field to change it; omit a field
    entirely to leave it untouched -- see _UNSET's docstring for why that's
    not the same as passing None. new_recurrence_frequency follows the same
    rule as new_due_at: pass one of RECURRENCE_FREQUENCIES to make/keep a
    task recurring, or None to make it (or keep it) a plain one-off --
    e.g. "stop reminding me about this every month" is new_recurrence_
    frequency=None, not _UNSET."""
    row = get_task(chat_id, task_id)
    if row is None:
        return None
    final_title = row["title"] if new_title is _UNSET else new_title
    final_due_at = row["due_at"] if new_due_at is _UNSET else new_due_at
    final_notes = row["notes"] if new_notes is _UNSET else new_notes
    if new_recurrence_frequency is _UNSET:
        final_recurrence_frequency = row["recurrence_frequency"]
    elif new_recurrence_frequency is None or new_recurrence_frequency in RECURRENCE_FREQUENCIES:
        final_recurrence_frequency = new_recurrence_frequency
    else:
        final_recurrence_frequency = row["recurrence_frequency"]
    with get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET title = ?, due_at = ?, notes = ?, recurrence_frequency = ? "
            "WHERE id = ? AND chat_id = ?",
            (final_title, final_due_at, final_notes, final_recurrence_frequency, task_id, chat_id),
        )
    return get_task(chat_id, task_id)


def mark_task_done(chat_id: int, task_id: int) -> dict | None:
    """For a plain one-off to-do, closes it permanently (done=1), same as
    always. For a RECURRING task (recurrence_frequency set -- see add_task),
    "done" means "done for THIS cycle", not gone for good: instead of
    closing it, this advances due_at to the next occurrence (via
    advance_date_by_frequency, anchored on the current due_at, or today if
    it had none) and leaves it open (done stays 0), so it resurfaces on
    /tasks again next cycle without the user re-adding it. A time-of-day
    component on the old due_at (e.g. "2026-10-10 17:00") is preserved on
    the new date rather than silently dropped."""
    row = get_task(chat_id, task_id)
    if row is None:
        return None
    if row["recurrence_frequency"]:
        due_at = row["due_at"]
        if due_at:
            date_part, _, time_part = due_at.partition(" ")
            anchor = date.fromisoformat(date_part)
        else:
            anchor = date.fromisoformat(today_str())
            time_part = None
        new_due_date = advance_date_by_frequency(anchor, row["recurrence_frequency"])
        new_due_at = f"{new_due_date.isoformat()} {time_part}" if time_part else new_due_date.isoformat()
        with get_conn() as conn:
            conn.execute("UPDATE tasks SET due_at = ? WHERE id = ? AND chat_id = ?", (new_due_at, task_id, chat_id))
        return get_task(chat_id, task_id)
    with get_conn() as conn:
        conn.execute("UPDATE tasks SET done = 1 WHERE id = ? AND chat_id = ?", (task_id, chat_id))
    return get_task(chat_id, task_id)


def unmark_task_done(chat_id: int, task_id: int, restore_due_at=_UNSET) -> dict | None:
    """Undo for mark_task_done. For a one-off task, flips done back to 0
    (it never deleted anything). For a RECURRING task's "done for this
    cycle" (which never set done=1 at all -- see mark_task_done), this
    instead reverts due_at back to what it was before that cycle was
    completed; the caller must pass the snapshot's old due_at as
    restore_due_at for a recurring task's undo to actually work (a missing
    restore_due_at on a recurring task is a caller bug, not a silent
    no-op -- but falls back to the done=0 flip rather than raising, so a
    missing snapshot degrades to "nothing visibly changes" instead of a
    crash)."""
    row = get_task(chat_id, task_id)
    if row is None:
        return None
    if row["recurrence_frequency"] and restore_due_at is not _UNSET:
        with get_conn() as conn:
            conn.execute(
                "UPDATE tasks SET due_at = ? WHERE id = ? AND chat_id = ?", (restore_due_at, task_id, chat_id)
            )
        return get_task(chat_id, task_id)
    with get_conn() as conn:
        conn.execute("UPDATE tasks SET done = 0 WHERE id = ? AND chat_id = ?", (task_id, chat_id))
    return get_task(chat_id, task_id)


def delete_task(chat_id: int, task_id: int) -> dict | None:
    row = get_task(chat_id, task_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM tasks WHERE id = ? AND chat_id = ?", (task_id, chat_id))
    return row


def restore_deleted_task(chat_id: int, row: dict) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks (chat_id, title, due_at, done, notes, recurrence_frequency) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, row["title"], row["due_at"], row["done"], row["notes"], row.get("recurrence_frequency")),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_task(chat_id, new_id)


# ---------- daily reminders ----------
#
# Deliberately its own domain, not just another "task" row: a to-do's
# "done" is permanent (see mark_task_done), but a daily reminder (e.g.
# "take hair pills") has no such state -- it recurs forever, every day,
# until the user removes it entirely. last_done_date tracks only whether
# TODAY's occurrence has been checked off; nothing has to reset it at
# midnight -- every read compares it against today_str() itself (see
# get_active_reminders and formatting._reminder_line), so tomorrow it's
# automatically "not done" again with no separate rollover job needed,
# unlike balance's daily rollover.

def add_reminder(chat_id: int, description: str) -> int:
    get_or_create_user(chat_id)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO daily_reminders (chat_id, description) VALUES (?, ?)",
            (chat_id, description),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_reminder(chat_id: int, reminder_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM daily_reminders WHERE id = ? AND chat_id = ?", (reminder_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def get_active_reminders(chat_id: int, limit: int = 40) -> list[dict]:
    """Every standing daily reminder for this chat, oldest first (the order
    they were added in -- there's no due date to sort by, unlike tasks).
    Feeds /reminders, the natural-language show_reminders intent, and the
    morning briefing -- callers filter by last_done_date themselves for
    "still pending today" (see morning._morning_briefing_payload)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_reminders WHERE chat_id = ? ORDER BY id LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_reminder_done_today(chat_id: int, reminder_id: int) -> dict | None:
    row = get_reminder(chat_id, reminder_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute(
            "UPDATE daily_reminders SET last_done_date = ? WHERE id = ? AND chat_id = ?",
            (today_str(), reminder_id, chat_id),
        )
    return get_reminder(chat_id, reminder_id)


def unmark_reminder_done_today(chat_id: int, reminder_id: int) -> dict | None:
    """Undo for mark_reminder_done_today. Clears last_done_date back to
    NULL rather than restoring whatever it happened to be before -- the
    only thing that ever mattered for display/briefing purposes is whether
    it equals TODAY (see get_active_reminders' docstring), and a mark-done
    can only be undone as the single next message anyway (see
    correction.LAST_CORRECTION_KEY), so the prior value was never today's
    date to begin with."""
    row = get_reminder(chat_id, reminder_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute(
            "UPDATE daily_reminders SET last_done_date = NULL WHERE id = ? AND chat_id = ?",
            (reminder_id, chat_id),
        )
    return get_reminder(chat_id, reminder_id)


def delete_reminder(chat_id: int, reminder_id: int) -> dict | None:
    row = get_reminder(chat_id, reminder_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM daily_reminders WHERE id = ? AND chat_id = ?", (reminder_id, chat_id))
    return row


def restore_deleted_reminder(chat_id: int, row: dict) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO daily_reminders (chat_id, description, last_done_date) VALUES (?, ?, ?)",
            (chat_id, row["description"], row["last_done_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_reminder(chat_id, new_id)


# ---------- scheduled events ----------
#
# Flat, one-off dated entries (e.g. "Dinner with Mel next Monday", or a
# week's worth of workout-plan slots) -- distinct from both to-dos (no
# due/overdue framing, no "done" state -- an event just passes by date, it's
# never checked off) and daily reminders (a single occurrence, not a
# recurring-forever habit). Deliberately NOT modeling true recurrence yet
# (e.g. "Company Tennis every Wednesday") -- see events.py's module
# docstring for the reasoning; a genuinely recurring rule needs day-of-week/
# interval matching plus the "edit one occurrence vs. all future
# occurrences" problem, neither of which exists anywhere in this codebase
# yet, and the concrete near-term driver (a week's workout plan) is really
# just several flat dated rows, not a recurring rule.

def add_event(chat_id: int, title: str, event_date: str, event_time: str | None = None,
              notes: str | None = None) -> int:
    get_or_create_user(chat_id)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO events (chat_id, title, event_date, event_time, notes) VALUES (?, ?, ?, ?, ?)",
            (chat_id, title, event_date, event_time, notes),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_event(chat_id: int, event_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ? AND chat_id = ?", (event_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def get_upcoming_events(chat_id: int, from_date: str | None = None, limit: int = 40) -> list[dict]:
    """Events from from_date (defaults to today) forward, soonest first --
    an event that's already passed has nothing left to show for, unlike a
    to-do (which stays open/overdue until explicitly done). Rows with no
    event_time sort before ones with a time on the same day, matching how a
    person would actually read a day's schedule."""
    from_date = from_date or today_str()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE chat_id = ? AND event_date >= ? "
            "ORDER BY event_date, (event_time IS NULL), event_time, id LIMIT ?",
            (chat_id, from_date, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def edit_event_date(chat_id: int, event_id: int, new_event_date: str) -> dict | None:
    """Reschedule -- the only field /rescheduleevent changes right now (see
    events.py's module docstring for why rescheduling is slash-command-only).
    A separate function rather than folding into a general edit_event,
    mirroring meal/workout/vitals' edit_*_date, since it's the one field
    that actually needs changing for this use case today."""
    row = get_event(chat_id, event_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute(
            "UPDATE events SET event_date = ? WHERE id = ? AND chat_id = ?", (new_event_date, event_id, chat_id)
        )
    return get_event(chat_id, event_id)


def delete_event(chat_id: int, event_id: int) -> dict | None:
    row = get_event(chat_id, event_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM events WHERE id = ? AND chat_id = ?", (event_id, chat_id))
    return row


def restore_deleted_event(chat_id: int, row: dict) -> dict | None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO events (chat_id, title, event_date, event_time, notes) VALUES (?, ?, ?, ?, ?)",
            (chat_id, row["title"], row["event_date"], row["event_time"], row["notes"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_event(chat_id, new_id)


# ---------- proactive insight dedup ----------
# A record that a given insight (identified by its own domain-specific
# dedup_key, e.g. "expense:category_spike:Dining") was already surfaced to
# this chat on a given day. insights.py's underlying detectors are pure
# recomputation over the same real rows every time they run -- nothing
# about the detected condition itself "resets" day to day -- so without
# this table, an unchanged observation (a spending spike that's still
# ongoing, a lift that's still stale) would repeat in every single morning
# briefing rather than being mentioned once and then left alone for a
# while. This is real persisted state, unlike context.chat_data (which only
# survives until the next message -- fine for a correction's one-shot undo
# snapshot, not for "don't repeat this for a week").

def record_insight_sent(chat_id: int, dedup_key: str, sent_date: str | None = None) -> None:
    get_or_create_user(chat_id)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sent_insights (chat_id, dedup_key, sent_date) VALUES (?, ?, ?)",
            (chat_id, dedup_key, sent_date or today_str()),
        )


def was_insight_sent_recently(chat_id: int, dedup_key: str, within_days: int = 7) -> bool:
    """True if this exact dedup_key was already sent to this chat within the
    last `within_days` days (inclusive of today) -- see record_insight_sent's
    docstring for why this table exists at all."""
    cutoff = (date.fromisoformat(today_str()) - timedelta(days=within_days - 1)).isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sent_insights WHERE chat_id = ? AND dedup_key = ? AND sent_date >= ? LIMIT 1",
            (chat_id, dedup_key, cutoff),
        ).fetchone()
        return row is not None


def day_matches_billing_day(day: date, billing_day: int) -> bool:
    """True if `day` is the effective occurrence of a monthly billing_day
    (1-31) in day's own month -- clamped to that month's real last day for
    a billing_day that doesn't exist in every month (e.g. 31 in a 30-day
    month, or in February), rather than silently skipping those months
    entirely or rolling over into the next one. Used by income.py's
    income_tick for the recurring-salary pay day, which is always a plain
    monthly day-of-month -- subscriptions now use the more general
    advance_date_by_frequency below instead (not every subscription bills
    monthly), but a salary's pay day never needed anything richer than
    this, so it keeps its own simpler helper rather than being forced
    through the frequency machinery for no real benefit."""
    from calendar import monthrange
    last_day_of_month = monthrange(day.year, day.month)[1]
    return day.day == min(int(billing_day), last_day_of_month)


# Every recurring cadence subscriptions and recurring tasks can both use --
# a shared vocabulary so advance_date_by_frequency (below) is the one
# "what's the next occurrence" implementation for both domains, rather than
# two copies that could drift apart.
RECURRENCE_FREQUENCIES = ("weekly", "biweekly", "monthly", "quarterly", "annual")


def advance_date_by_frequency(d: date, frequency: str) -> date:
    """Computes the next occurrence of a recurring date after `d`, for one
    of RECURRENCE_FREQUENCIES. "weekly"/"biweekly" are a plain fixed-day
    offset; "monthly"/"quarterly"/"annual" use real calendar month/year
    arithmetic rather than a fixed day count, clamping the day-of-month to
    the target month's real last day (e.g. the 31st of a subscription that
    renews monthly lands on the 30th in a 30-day month, the 28th/29th in
    February) -- the same clamping discipline day_matches_billing_day
    already used for salary pay days, generalized here to cover any
    calendar-based cadence, not just "every month". Shared by
    subscriptions.subscriptions_tick (advancing next_renewal_date) and
    db.mark_task_done (advancing a recurring task's due_at) -- one
    implementation for "what's the next occurrence", not two that could
    silently diverge on the leap-year/short-month edge cases."""
    if frequency not in RECURRENCE_FREQUENCIES:
        raise ValueError(f"Unknown recurrence frequency: {frequency!r}")
    if frequency == "weekly":
        return d + timedelta(days=7)
    if frequency == "biweekly":
        return d + timedelta(days=14)
    from calendar import monthrange
    months_to_add = {"monthly": 1, "quarterly": 3, "annual": 12}[frequency]
    total_month_index = (d.month - 1) + months_to_add
    target_year = d.year + total_month_index // 12
    target_month = total_month_index % 12 + 1
    last_day_of_target_month = monthrange(target_year, target_month)[1]
    target_day = min(d.day, last_day_of_target_month)
    return date(target_year, target_month, target_day)


# ---------- subscriptions ----------
#
# Recurring charges (Netflix, gym, insurance, etc.) that auto-log
# THEMSELVES as a real expense on their renewal date -- see
# subscriptions.subscriptions_tick for the daily job that drives this.
# Each subscription row here is pure config (name/amount/currency/
# frequency/next_renewal_date, plus card/notes/is_claimable for anything
# that needs it); the actual spend each cycle lands as an ordinary row in
# `expenses`, so it's visible to /balance, /rundown, /trend, and
# correctable the exact same way as any other expense -- there's
# deliberately no special-cased "undo a subscription charge" path here,
# since editing/deleting that cycle's auto-logged expense already covers
# it. is_claimable flows straight through to that expense row (see
# subscriptions_tick), so a subscription you expense to your employer
# (e.g. a gym membership) shows up in /claimed the normal way, with no
# separate claimable-subscriptions concept needed.
#
# Unlike the old billing_day (1-31, monthly-only) design, next_renewal_date
# is a real ISO date and frequency says how far to jump forward after each
# post (see advance_date_by_frequency above) -- this is what makes
# quarterly/annual/biweekly subscriptions (not just monthly ones)
# representable at all.

def add_subscription(chat_id: int, name: str, amount: float, currency: str, frequency: str,
                      next_renewal_date: str, category: str | None = None, card: str | None = None,
                      notes: str | None = None, is_claimable: bool = False) -> int:
    get_or_create_user(chat_id)
    currency = (currency or config.BASE_CURRENCY).upper()
    amount_base = fx.to_base(amount, currency)
    frequency = frequency if frequency in RECURRENCE_FREQUENCIES else "monthly"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO subscriptions (chat_id, name, amount, currency, amount_base, category, frequency, "
            "next_renewal_date, card, notes, is_claimable) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, name, amount, currency, amount_base, category, frequency, next_renewal_date, card,
             notes, bool(is_claimable)),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_subscription(chat_id: int, subscription_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM subscriptions WHERE id = ? AND chat_id = ?", (subscription_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def get_subscriptions(chat_id: int, active_only: bool = True) -> list[dict]:
    """Ordered by next_renewal_date then name -- a predictable, calendar-
    shaped list (soonest renewal first) rather than insertion order, so
    /subscriptions and the weekly renewal digest both read naturally."""
    query = "SELECT * FROM subscriptions WHERE chat_id = ?"
    params: list = [chat_id]
    if active_only:
        query += " AND active = 1"
    query += " ORDER BY next_renewal_date, name"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def delete_subscription(chat_id: int, subscription_id: int) -> dict | None:
    row = get_subscription(chat_id, subscription_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM subscriptions WHERE id = ? AND chat_id = ?", (subscription_id, chat_id))
    return row


def advance_subscription(subscription_id: int, logged_date: str, new_next_renewal_date: str) -> None:
    """Called right after subscriptions_tick posts one cycle's expense --
    records last_logged_date (the idempotency guard against a same-day job
    retry or bot restart double-charging) and advances next_renewal_date to
    the following cycle, in one update. Once this runs, next_renewal_date
    no longer equals today, so a second tick call today naturally won't
    match it again -- last_logged_date is a belt-and-suspenders backstop,
    not the only thing standing between here and a double charge."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE subscriptions SET last_logged_date = ?, next_renewal_date = ? WHERE id = ?",
            (logged_date, new_next_renewal_date, subscription_id),
        )


def edit_subscription(chat_id: int, subscription_id: int, new_name=_UNSET, new_amount=_UNSET,
                       new_currency=_UNSET, new_frequency=_UNSET, new_next_renewal_date=_UNSET,
                       new_category=_UNSET, new_card=_UNSET, new_notes=_UNSET,
                       new_is_claimable=_UNSET) -> dict | None:
    """Corrects a subscription's own config (a price change, a renewal-date
    move, a frequency change, a rename) in place -- deliberately NOT the
    same thing as correcting one cycle's already-posted charge in
    `expenses` (see this module's "no special-cased undo a subscription
    charge" note above, which is still true and still a separate concern).
    Mirrors edit_meal's _UNSET "only touch what's passed" discipline.
    last_logged_date is never touched here -- a price, date, or frequency
    correction shouldn't retroactively re-trigger (or re-skip) the cycle
    that already posted."""
    row = get_subscription(chat_id, subscription_id)
    if row is None:
        return None
    final_name = row["name"] if new_name is _UNSET else new_name
    final_amount = row["amount"] if new_amount is _UNSET else new_amount
    final_currency = row["currency"] if new_currency is _UNSET else (new_currency or config.BASE_CURRENCY).upper()
    final_category = row["category"] if new_category is _UNSET else new_category
    final_card = row["card"] if new_card is _UNSET else new_card
    final_notes = row["notes"] if new_notes is _UNSET else new_notes
    final_is_claimable = row["is_claimable"] if new_is_claimable is _UNSET else bool(new_is_claimable)
    if new_frequency is _UNSET:
        final_frequency = row["frequency"]
    else:
        final_frequency = new_frequency if new_frequency in RECURRENCE_FREQUENCIES else row["frequency"]
    final_next_renewal_date = (
        row["next_renewal_date"] if new_next_renewal_date is _UNSET else new_next_renewal_date
    )
    # Only re-convert to base currency if the amount or currency actually
    # changed -- avoids a pointless extra FX lookup on a rename/date-only
    # edit, and avoids drift from re-converting at a possibly different
    # exchange rate than when it was first added.
    if new_amount is _UNSET and new_currency is _UNSET:
        final_amount_base = row["amount_base"]
    else:
        final_amount_base = fx.to_base(final_amount, final_currency)
    with get_conn() as conn:
        conn.execute(
            "UPDATE subscriptions SET name = ?, amount = ?, currency = ?, amount_base = ?, "
            "category = ?, frequency = ?, next_renewal_date = ?, card = ?, notes = ?, is_claimable = ? "
            "WHERE id = ? AND chat_id = ?",
            (final_name, final_amount, final_currency, final_amount_base, final_category, final_frequency,
             final_next_renewal_date, final_card, final_notes, final_is_claimable, subscription_id, chat_id),
        )
    return get_subscription(chat_id, subscription_id)


def restore_deleted_subscription(chat_id: int, row: dict) -> dict | None:
    """Re-inserts a previously deleted subscription row exactly as it was,
    including its frequency/next_renewal_date/card/notes/is_claimable/
    active/last_logged_date -- used for one-step 'undo' after a natural-
    language correction deletes the wrong one. Gets a fresh row id, since
    SQLite won't recycle the old one. Preserving last_logged_date (rather
    than resetting it to None) matters: undoing an accidental delete made
    right after a cycle's auto-post must not make the subscription look
    un-logged and eligible to double-charge the same cycle on the next
    tick."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO subscriptions (chat_id, name, amount, currency, amount_base, category, frequency, "
            "next_renewal_date, card, notes, is_claimable, active, last_logged_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, row["name"], row["amount"], row["currency"], row["amount_base"], row["category"],
             row["frequency"], row["next_renewal_date"], row.get("card"), row.get("notes"),
             row.get("is_claimable", 0), row.get("active", 1), row.get("last_logged_date")),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_subscription(chat_id, new_id)


# ---------- income ----------
#
# Deliberately separate from the expense/balance machinery -- see
# get_net_worth's docstring below for why "how much money do I actually
# have" is a genuinely different question from the existing daily spending
# target/streak (`users.balance`), which stays completely untouched by any
# of this. `income_config` is a single-row-per-chat recurring-salary setup
# (mirrors `users`' one-row-per-chat shape); `income` is the real ledger --
# one row per actual payment, whether it auto-posted itself on payday or was
# logged ad hoc (a bonus, freelance income, an off-schedule salary).

def set_income_config(chat_id: int, gross_amount: float, currency: str, pay_day: int,
                       cpf_rate: float = 0.0, stock_rate: float = 0.0) -> None:
    """Upsert -- /setincome always replaces the whole config in one call
    rather than patching individual fields, since a raise or job change
    naturally means giving the new gross/pay-day/rates together, not
    editing one field at a time. last_logged_month is preserved across a
    re-set (it's simply not touched by this UPDATE) so changing your salary
    mid-month doesn't risk a double auto-post for the cycle that already
    ran this month."""
    get_or_create_user(chat_id)
    currency = (currency or config.BASE_CURRENCY).upper()
    pay_day = max(1, min(31, int(pay_day)))
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO income_config (chat_id, gross_amount, currency, pay_day, cpf_rate, stock_rate, active) "
            "VALUES (?, ?, ?, ?, ?, ?, 1) "
            "ON CONFLICT(chat_id) DO UPDATE SET gross_amount = excluded.gross_amount, "
            "currency = excluded.currency, pay_day = excluded.pay_day, cpf_rate = excluded.cpf_rate, "
            "stock_rate = excluded.stock_rate, active = 1",
            (chat_id, gross_amount, currency, pay_day, cpf_rate, stock_rate),
        )


def edit_income_config(chat_id: int, new_gross_amount=_UNSET, new_currency=_UNSET, new_pay_day=_UNSET,
                        new_cpf_rate=_UNSET, new_stock_rate=_UNSET) -> dict | None:
    """Partial update for an existing income_config row -- "I got a raise" or
    "I changed jobs, new pay day is the 1st" should only touch the field(s)
    actually mentioned, unlike /setincome's full-replace set_income_config
    (which exists for the explicit initial-setup command, where giving the
    whole config at once is the natural shape). Without this, a raise
    reported through natural language with only the new gross amount in hand
    would have to go through set_income_config and silently zero out the
    real cpf_rate/stock_rate (its defaults) -- a real, easy-to-hit data-loss
    bug this sentinel pattern (mirroring edit_meal/edit_task) exists
    specifically to prevent. Returns None if there's no config yet (the
    caller should ask the user to run /setincome first instead of silently
    creating a half-formed one). last_logged_month and active are never
    touched here, same reasoning as set_income_config."""
    row = get_income_config(chat_id)
    if row is None:
        return None
    final_gross = row["gross_amount"] if new_gross_amount is _UNSET else new_gross_amount
    final_currency = row["currency"] if new_currency is _UNSET else (new_currency or config.BASE_CURRENCY).upper()
    if new_pay_day is _UNSET:
        final_pay_day = row["pay_day"]
    else:
        final_pay_day = max(1, min(31, int(new_pay_day)))
    final_cpf_rate = row["cpf_rate"] if new_cpf_rate is _UNSET else new_cpf_rate
    final_stock_rate = row["stock_rate"] if new_stock_rate is _UNSET else new_stock_rate
    with get_conn() as conn:
        conn.execute(
            "UPDATE income_config SET gross_amount = ?, currency = ?, pay_day = ?, cpf_rate = ?, "
            "stock_rate = ? WHERE chat_id = ?",
            (final_gross, final_currency, final_pay_day, final_cpf_rate, final_stock_rate, chat_id),
        )
    return get_income_config(chat_id)


def get_income_config(chat_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM income_config WHERE chat_id = ?", (chat_id,)).fetchone()
        return dict(row) if row else None


def clear_income_config(chat_id: int) -> None:
    """Stops the recurring auto-post entirely (e.g. between jobs) -- deletes
    the config row rather than just flipping active=0, since there's no
    history worth keeping on this row (unlike a deleted expense/meal/etc.,
    which has a restore_deleted_* path) -- running /setincome again starts
    a fresh config from scratch."""
    with get_conn() as conn:
        conn.execute("DELETE FROM income_config WHERE chat_id = ?", (chat_id,))


def mark_income_config_logged(chat_id: int, month_str: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE income_config SET last_logged_month = ? WHERE chat_id = ?", (month_str, chat_id)
        )


def add_income(chat_id: int, source: str, net_amount: float, currency: str, description: str | None = None,
               gross_amount: float | None = None, cpf_amount: float | None = None,
               stock_amount: float | None = None, income_date: str | None = None) -> int:
    """net_amount is always the real take-home figure -- what actually
    landed in the bank account -- never the gross; gross_amount/cpf_amount/
    stock_amount are optional breakdown detail, kept for the record but
    deliberately NOT what get_total_income sums (see its own docstring)."""
    get_or_create_user(chat_id)
    currency = (currency or config.BASE_CURRENCY).upper()
    net_amount_base = fx.to_base(net_amount, currency)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO income (chat_id, source, description, gross_amount, cpf_amount, stock_amount, "
            "net_amount, currency, net_amount_base, income_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, source, description, gross_amount, cpf_amount, stock_amount, net_amount, currency,
             net_amount_base, income_date or today_str()),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_income(chat_id: int, income_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM income WHERE id = ? AND chat_id = ?", (income_id, chat_id)).fetchone()
        return dict(row) if row else None


def get_recent_income(chat_id: int, limit: int = 10) -> list[dict]:
    """Ordered by income_date DESC (id DESC only as a same-day tiebreak),
    NOT id DESC alone -- same real lesson as get_recent_meals/get_recent_
    workouts/get_recent_vitals (see their docstrings): an auto-posted salary
    and an ad hoc bonus logged out of date order would otherwise be able to
    outrank each other incorrectly in "recent"."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM income WHERE chat_id = ? ORDER BY income_date DESC, id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def edit_income_date(chat_id: int, income_id: int, new_date_str: str) -> dict | None:
    """Moves an income entry to a different income_date. Like meals/
    workouts/vitals, income doesn't feed a rolling balance (see this
    module's own docstring on why net_worth stays separate from `users.
    balance`), so this is a plain field update -- no balance adjustment
    needed. Dates after today are clamped to today."""
    row = get_income(chat_id, income_id)
    if row is None:
        return None
    today = today_str()
    if new_date_str > today:
        new_date_str = today
    with get_conn() as conn:
        conn.execute(
            "UPDATE income SET income_date = ? WHERE id = ? AND chat_id = ?", (new_date_str, income_id, chat_id)
        )
    return get_income(chat_id, income_id)


def edit_income(chat_id: int, income_id: int, new_source=_UNSET, new_description=_UNSET,
                 new_amount=_UNSET, new_currency=_UNSET) -> dict | None:
    """Corrects a logged income entry's source/description/net amount/
    currency in place, mirroring edit_meal's _UNSET "only touch what's
    passed" discipline. new_amount always means the NET (take-home) figure,
    same as add_income -- never gross. Deliberately doesn't touch gross_
    amount/cpf_amount/stock_amount: those are auto-posted salary's own
    breakdown detail, not something a plain-text correction realistically
    re-derives, and get_total_income never sums them anyway (see add_
    income's docstring)."""
    row = get_income(chat_id, income_id)
    if row is None:
        return None
    final_source = row["source"] if new_source is _UNSET else new_source
    final_description = row["description"] if new_description is _UNSET else new_description
    final_amount = row["net_amount"] if new_amount is _UNSET else new_amount
    final_currency = row["currency"] if new_currency is _UNSET else (new_currency or config.BASE_CURRENCY).upper()
    if new_amount is _UNSET and new_currency is _UNSET:
        final_amount_base = row["net_amount_base"]
    else:
        final_amount_base = fx.to_base(final_amount, final_currency)
    with get_conn() as conn:
        conn.execute(
            "UPDATE income SET source = ?, description = ?, net_amount = ?, currency = ?, "
            "net_amount_base = ? WHERE id = ? AND chat_id = ?",
            (final_source, final_description, final_amount, final_currency, final_amount_base,
             income_id, chat_id),
        )
    return get_income(chat_id, income_id)


def delete_income(chat_id: int, income_id: int) -> dict | None:
    row = get_income(chat_id, income_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM income WHERE id = ? AND chat_id = ?", (income_id, chat_id))
    return row


def restore_deleted_income(chat_id: int, row: dict) -> dict | None:
    """Re-inserts a previously deleted income row exactly as it was
    (including the gross/cpf/stock breakdown, if it had one) -- used for
    one-step 'undo' after a natural-language correction deletes the wrong
    entry. Gets a fresh row id, every other field preserved."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO income (chat_id, source, description, gross_amount, cpf_amount, stock_amount, "
            "net_amount, currency, net_amount_base, income_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, row["source"], row["description"], row["gross_amount"], row["cpf_amount"],
             row["stock_amount"], row["net_amount"], row["currency"], row["net_amount_base"], row["income_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_income(chat_id, new_id)


def get_total_income(chat_id: int) -> float:
    """All-time sum of net_amount_base -- the real take-home total that
    actually reached the bank account (salary net of CPF/stock, bonuses,
    other income), in BASE_CURRENCY. Feeds get_net_worth."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(net_amount_base), 0) AS total FROM income WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return row["total"]


# ---------- deductions ----------
#
# Money that genuinely left the account but ISN'T a discretionary purchase
# -- income tax, a voluntary CPF top-up, and similar. Reduces get_net_worth
# the same way spend does, but is deliberately kept out of `expenses`
# entirely so it never touches the daily spending target/streak math --
# see ai.py's "log_deduction" intent docstring for the real distinction
# from a normal expense.

def add_deduction(chat_id: int, label: str, amount: float, currency: str,
                   deduction_date: str | None = None) -> int:
    get_or_create_user(chat_id)
    currency = (currency or config.BASE_CURRENCY).upper()
    amount_base = fx.to_base(amount, currency)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO deductions (chat_id, label, amount, currency, amount_base, deduction_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, label, amount, currency, amount_base, deduction_date or today_str()),
        )
        return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def get_deduction(chat_id: int, deduction_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM deductions WHERE id = ? AND chat_id = ?", (deduction_id, chat_id)
        ).fetchone()
        return dict(row) if row else None


def get_recent_deductions(chat_id: int, limit: int = 10) -> list[dict]:
    """Ordered by deduction_date DESC, id DESC -- same date-not-id-order
    discipline as get_recent_income/get_recent_meals/etc."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM deductions WHERE chat_id = ? ORDER BY deduction_date DESC, id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def edit_deduction_date(chat_id: int, deduction_id: int, new_date_str: str) -> dict | None:
    """Moves a deduction to a different deduction_date -- a plain field
    update, no balance adjustment (deductions already stay out of `users.
    balance` entirely, see this section's own docstring). Dates after today
    are clamped to today."""
    row = get_deduction(chat_id, deduction_id)
    if row is None:
        return None
    today = today_str()
    if new_date_str > today:
        new_date_str = today
    with get_conn() as conn:
        conn.execute(
            "UPDATE deductions SET deduction_date = ? WHERE id = ? AND chat_id = ?",
            (new_date_str, deduction_id, chat_id),
        )
    return get_deduction(chat_id, deduction_id)


def edit_deduction(chat_id: int, deduction_id: int, new_label=_UNSET, new_amount=_UNSET,
                    new_currency=_UNSET) -> dict | None:
    """Corrects a logged deduction's label/amount/currency in place, mirroring
    edit_meal's _UNSET "only touch what's passed" discipline."""
    row = get_deduction(chat_id, deduction_id)
    if row is None:
        return None
    final_label = row["label"] if new_label is _UNSET else new_label
    final_amount = row["amount"] if new_amount is _UNSET else new_amount
    final_currency = row["currency"] if new_currency is _UNSET else (new_currency or config.BASE_CURRENCY).upper()
    if new_amount is _UNSET and new_currency is _UNSET:
        final_amount_base = row["amount_base"]
    else:
        final_amount_base = fx.to_base(final_amount, final_currency)
    with get_conn() as conn:
        conn.execute(
            "UPDATE deductions SET label = ?, amount = ?, currency = ?, amount_base = ? "
            "WHERE id = ? AND chat_id = ?",
            (final_label, final_amount, final_currency, final_amount_base, deduction_id, chat_id),
        )
    return get_deduction(chat_id, deduction_id)


def delete_deduction(chat_id: int, deduction_id: int) -> dict | None:
    row = get_deduction(chat_id, deduction_id)
    if row is None:
        return None
    with get_conn() as conn:
        conn.execute("DELETE FROM deductions WHERE id = ? AND chat_id = ?", (deduction_id, chat_id))
    return row


def restore_deleted_deduction(chat_id: int, row: dict) -> dict | None:
    """Re-inserts a previously deleted deduction row exactly as it was --
    used for one-step 'undo' after a natural-language correction deletes
    the wrong entry. Gets a fresh row id, every other field preserved."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO deductions (chat_id, label, amount, currency, amount_base, deduction_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, row["label"], row["amount"], row["currency"], row["amount_base"], row["deduction_date"]),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    return get_deduction(chat_id, new_id)


def get_total_deductions(chat_id: int) -> float:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount_base), 0) AS total FROM deductions WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return row["total"]


# ---------- net worth ----------

def get_total_spend(chat_id: int) -> float:
    """All-time non-claimable spend, in BASE_CURRENCY -- reuses
    get_daily_totals with a start date old enough to predate any real data
    (same placeholder-start discipline as rundown.TREND_BEGINNING_
    PLACEHOLDER), rather than writing a second SUM query that could quietly
    drift from the one /balance and /trend's "spending" metric already
    trust."""
    tomorrow = (date.fromisoformat(today_str()) + timedelta(days=1)).isoformat()
    totals = get_daily_totals(chat_id, "2000-01-01", tomorrow)
    return sum(totals.values())


def get_net_worth(chat_id: int) -> dict:
    """Real, deterministically-computed money picture: everything that's
    actually arrived (income) minus everything that's actually left
    (deductions + non-claimable spend) -- a genuinely different number from
    the existing daily `balance` (a discretionary-spending-target rollover,
    not a real money total; see this module's docstring). Deliberately
    leaves out CLAIMABLE expenses -- money fronted for someone else isn't
    really spent long-term once reimbursed, and this feature doesn't
    reconcile against /claimed yet, so folding claimables in here would
    double-count money that's coming back. A known, documented
    simplification, not an oversight."""
    total_income = get_total_income(chat_id)
    total_deductions = get_total_deductions(chat_id)
    total_spend = get_total_spend(chat_id)
    return {
        "total_income": round(total_income, 2),
        "total_deductions": round(total_deductions, 2),
        "total_spend": round(total_spend, 2),
        "net_worth": round(total_income - total_deductions - total_spend, 2),
    }
