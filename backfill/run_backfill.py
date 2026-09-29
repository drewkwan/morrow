"""
One-off historical backfill: inserts vitals/workout rows from the coaching-
thread CSV and meal rows from the food-journal CSV into Morrow's real
database, going back to 2026-07-04.

Why this is safe to run against the live database (the data-persistence
worry Andrew raised): rather than relying on a hardcoded cutoff date to
avoid colliding with what he's already logged into live Morrow since he
started using it, this script checks, per date and per domain, whether
Morrow already has a row there (db.get_vitals_in_range / get_meals_in_range
/ get_workouts_in_range) and skips inserting if so. A date with a live-
logged meal but no live-logged vitals still gets its vitals backfilled;
it's never all-or-nothing per day. That means this script can be re-run
safely (it's idempotent: a second run inserts nothing new, since everything
it would insert already exists) and doesn't require Andrew or me to pin
down the exact day he switched over -- the food journal source data itself
is unreliable/gappy right around that transition anyway (see the "14-19
Sep: Mostly tracked in user's new bot / not retained here" row).

Every inserted row is tagged "[backfilled]" (vitals/meal notes get an
explicit note; workouts get it prefixed onto notes) so they stay visually
distinguishable from what was actually logged live, in both Telegram
history and any dashboard view, and stay honest about being reconstructed
from an old ChatGPT thread rather than same-day logging.

USAGE
    Defaults to a dry run -- prints exactly what it would insert/skip and
    why, writes nothing. Review that output before trusting it.

        python backfill/run_backfill.py

    Once the dry-run output looks right, actually write the rows:

        python backfill/run_backfill.py --commit

    This has to run in the same environment as the deployed bot (wherever
    DB_PATH actually points -- almost certainly not this laptop checkout,
    since config.py's own comment says production uses a Railway-mounted
    volume). On Railway that's:

        railway run python backfill/run_backfill.py --commit

    chat_id is auto-detected from config.ALLOWED_CHAT_IDS (this is a
    personal single-user bot, so that list should have exactly one entry
    in the real deployment's env). Override with --chat-id if that's ever
    not true, e.g. testing against a local .env with a different chat_id.
"""

import argparse
import csv
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# Make the repo root (parent of this backfill/ dir) importable, whether this
# script is run as `python backfill/run_backfill.py` from the repo root or
# `python run_backfill.py` from inside backfill/ itself.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # must run before `import config` -- it reads env vars at import time

import config  # noqa: E402
import db  # noqa: E402

BACKFILL_TAG = "[backfilled]"

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_NUM_RE = re.compile(r"[\d][\d,]*(?:\.\d+)?")


def _numbers(s: str | None) -> list[float]:
    """Every number-looking token in s, commas stripped, as floats. Used
    for both single values ("~2,500 kcal" -> [2500.0]) and ranges
    ("~2,580-3,390" -> [2580.0, 3390.0])."""
    if not s:
        return []
    return [float(m.replace(",", "")) for m in _NUM_RE.findall(s)]


def parse_kcal(s: str | None) -> tuple[float | None, float | None, float | None]:
    """Returns (estimate, low, high). A single number is the estimate with
    no low/high band; two numbers are treated as a range (estimate =
    midpoint); free text with no digits at all ("Not retained", "Gap / not
    reliably tracked...") returns all-None rather than guessing."""
    nums = _numbers(s)
    if not nums:
        return None, None, None
    if len(nums) == 1:
        return nums[0], None, None
    return (nums[0] + nums[1]) / 2, nums[0], nums[1]


def parse_protein_g(s: str | None) -> float | None:
    nums = _numbers(s)
    if not nums:
        return None
    return sum(nums[:2]) / len(nums[:2])


def parse_water_ml(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"([\d.]+)\s*L\b", s)
    if not m:
        return None
    return float(m.group(1)) * 1000


def parse_date_label(label: str, year: int) -> tuple[date, date] | None:
    """"8 Jul" -> (Jul 8, Jul 8). "29-30 Aug" (also handles the en-dash the
    source doc actually uses, "29–30 Aug") -> (Aug 29, Aug 30) -- a
    genuine two-day combined entry, not a typo. Returns None for anything
    that doesn't match this shape at all."""
    label = label.strip()
    m = re.match(r"^(\d{1,2})(?:[–\-](\d{1,2}))?\s+([A-Za-z]{3,})$", label)
    if not m:
        return None
    d1, d2, mon_name = m.groups()
    mon = MONTHS.get(mon_name[:3].lower())
    if mon is None:
        return None
    try:
        start = date(year, mon, int(d1))
        end = date(year, mon, int(d2)) if d2 else start
    except ValueError:
        return None
    return start, end


def _split_items(s: str | None) -> list[str]:
    if not s:
        return []
    return [part.strip() for part in s.split(";") if part.strip()]


def _existing_dates(chat_id: int, domain: str, start: str, end_exclusive: str) -> set[str]:
    """Set of ISO date strings that already have at least one row in this
    domain within [start, end_exclusive). One existence-check call per
    domain covering the whole backfill window, rather than one per row."""
    if domain == "vitals":
        rows = db.get_vitals_in_range(chat_id, start, end_exclusive)
        return {r["vitals_date"] for r in rows}
    if domain == "meals":
        rows = db.get_meals_in_range(chat_id, start, end_exclusive)
        return {r["meal_date"] for r in rows}
    if domain == "workouts":
        rows = db.get_workouts_in_range(chat_id, start, end_exclusive)
        return {r["workout_date"] for r in rows}
    raise ValueError(domain)


def ingest_coaching(chat_id: int, csv_path: Path, commit: bool) -> dict:
    counts = {"vitals_inserted": 0, "vitals_skipped_existing": 0,
              "workouts_inserted": 0, "workouts_skipped_existing": 0}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return counts
    dates = sorted(r["date"] for r in rows)
    window_end = (date.fromisoformat(dates[-1]) + timedelta(days=1)).isoformat()
    existing_vitals = _existing_dates(chat_id, "vitals", dates[0], window_end)
    existing_workouts = _existing_dates(chat_id, "workouts", dates[0], window_end)

    for row in rows:
        d = row["date"]
        weight = float(row["weight_kg"]) if row["weight_kg"] else None
        sleep = float(row["sleep_hours"]) if row["sleep_hours"] else None
        knee = float(row["knee_pain"]) if row["knee_pain"] else None
        other_pain = row["other_pain"].strip() or None
        notes = row["notes"].strip() or None

        if weight is not None or sleep is not None or knee is not None or other_pain or notes:
            if d in existing_vitals:
                counts["vitals_skipped_existing"] += 1
                print(f"  SKIP vitals {d}: Morrow already has a vitals row that day")
            else:
                combined_notes = " | ".join(
                    filter(None, [BACKFILL_TAG, f"other pain: {other_pain}" if other_pain else None, notes])
                )
                print(f"  {'INSERT' if commit else 'WOULD INSERT'} vitals {d}: "
                      f"weight={weight} sleep={sleep} knee_pain={knee} notes={combined_notes!r}")
                if commit:
                    db.add_vitals(chat_id, weight_kg=weight, sleep_hours=sleep, knee_pain=knee,
                                   notes=combined_notes, vitals_date=d)
                counts["vitals_inserted"] += 1

        workout_summary = row["workout_summary"].strip()
        if workout_summary:
            if d in existing_workouts:
                counts["workouts_skipped_existing"] += 1
                print(f"  SKIP workout {d}: Morrow already has a workout row that day")
            else:
                print(f"  {'INSERT' if commit else 'WOULD INSERT'} workout {d}: {workout_summary!r}")
                if commit:
                    db.add_workout(chat_id, activity=workout_summary,
                                    notes=BACKFILL_TAG, workout_date=d)
                counts["workouts_inserted"] += 1
    return counts


def ingest_nutrition(chat_id: int, csv_path: Path, commit: bool, year: int) -> dict:
    counts = {"meals_inserted": 0, "meals_skipped_existing": 0, "meals_skipped_unparseable": 0}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return counts

    parsed_rows = []
    for row in rows:
        span = parse_date_label(row["date_label"], year)
        if span is None:
            print(f"  SKIP meal {row['date_label']!r}: couldn't parse a date from this label")
            counts["meals_skipped_unparseable"] += 1
            continue
        start, end = span
        parsed_rows.append((end.isoformat(), start, end, row))  # insert on the range's LAST day

    if not parsed_rows:
        return counts
    all_dates = sorted(r[0] for r in parsed_rows)
    window_end = (date.fromisoformat(all_dates[-1]) + timedelta(days=1)).isoformat()
    existing_meals = _existing_dates(chat_id, "meals", all_dates[0], window_end)

    for d, start, end, row in parsed_rows:
        estimate, low, high = parse_kcal(row["intake_kcal"])
        protein = parse_protein_g(row["protein_g"])
        water_ml = parse_water_ml(row["plain_water"])
        food_items = _split_items(row["notable_food_context"])

        # A row with no calorie number AND only one food_items entry is almost
        # always a meta-note about the day ("Gap / not reliably tracked...",
        # "User shifted to a custom bot...", "Staycation/recovery context.")
        # rather than an actual food description -- those aren't real
        # semicolon-separated dishes, just one explanatory sentence. Real
        # food context, even without a calorie estimate, is normally several
        # semicolon-separated items (see the 29-30 Aug row, which has no
        # kcal number but does list actual food and is worth keeping).
        if estimate is None and len(food_items) < 2:
            print(f"  SKIP meal {d}: no usable kcal and no real food context, looks like a "
                  f"meta-note rather than a logged day "
                  f"({row['intake_kcal']!r} / {row['notable_food_context']!r})")
            counts["meals_skipped_unparseable"] += 1
            continue

        if d in existing_meals:
            counts["meals_skipped_existing"] += 1
            print(f"  SKIP meal {d}: Morrow already has a meal row that day")
            continue

        items = [BACKFILL_TAG]
        if start != end:
            items.append(f"combined entry for {start.isoformat()}..{end.isoformat()}")
        if protein is not None:
            items.append(f"~{protein:.0f}g protein")
        if water_ml is not None:
            items.append(f"~{water_ml / 1000:.2f}L water")
        watch = row["watch_activity"].strip()
        if watch and watch not in ("—", "-", "Not retained"):
            items.append(f"activity: {watch}")
        items.extend(food_items)

        print(f"  {'INSERT' if commit else 'WOULD INSERT'} meal {d}: "
              f"calories_estimate={estimate} (low={low} high={high}) water_ml={water_ml} "
              f"items={items!r}")
        if commit:
            db.add_meal(chat_id, meal_type=None, items=items, calories_low=low, calories_high=high,
                        calories_estimate=estimate, water_ml=water_ml, meal_date=d)
        counts["meals_inserted"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", action="store_true",
                         help="Actually write rows. Without this flag, runs as a dry run: prints "
                              "every insert/skip decision but writes nothing.")
    parser.add_argument("--chat-id", type=int, default=None,
                         help="Override auto-detection from config.ALLOWED_CHAT_IDS.")
    parser.add_argument("--year", type=int, default=2026,
                         help="Year to assume for the food journal's day-month-only date labels "
                              "('8 Jul'). Default 2026, which is correct for the current backfill data.")
    args = parser.parse_args()

    chat_id = args.chat_id
    if chat_id is None:
        if len(config.ALLOWED_CHAT_IDS) == 1:
            chat_id = config.ALLOWED_CHAT_IDS[0]
        elif not config.ALLOWED_CHAT_IDS:
            print("Couldn't auto-detect a chat_id: config.ALLOWED_CHAT_IDS is empty in this "
                  "environment's .env. Pass --chat-id explicitly.", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"config.ALLOWED_CHAT_IDS has {len(config.ALLOWED_CHAT_IDS)} entries "
                  f"({config.ALLOWED_CHAT_IDS}) -- can't auto-pick one. Pass --chat-id explicitly.",
                  file=sys.stderr)
            sys.exit(1)

    mode = "COMMIT" if args.commit else "DRY RUN (nothing will be written -- pass --commit to write)"
    print(f"=== Morrow historical backfill -- {mode} ===")
    print(f"chat_id: {chat_id}")
    print(f"DB_PATH: {config.DB_PATH}\n")

    # CREATE TABLE IF NOT EXISTS for every table -- a no-op against the real,
    # already-initialized production DB. Same call bot.py/app.py already
    # make at startup; included here so the script doesn't depend on having
    # been run against a live bot process first.
    db.init_db()

    backfill_dir = Path(__file__).resolve().parent

    print("--- coaching thread (vitals + workouts) ---")
    coaching_counts = ingest_coaching(chat_id, backfill_dir / "coaching_thread_daily_log.csv", args.commit)

    print("\n--- food journal (meals) ---")
    nutrition_counts = ingest_nutrition(chat_id, backfill_dir / "nutrition_daily_log.csv", args.commit, args.year)

    print("\n=== summary ===")
    for k, v in {**coaching_counts, **nutrition_counts}.items():
        print(f"  {k}: {v}")
    if not args.commit:
        print("\nThis was a dry run -- nothing was written. Re-run with --commit once this looks right.")


if __name__ == "__main__":
    main()
