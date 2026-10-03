"""
Currency conversion for multi-currency logging.

Uses Frankfurter (https://www.frankfurter.app), a free no-API-key exchange
rate service backed by European Central Bank reference rates. Rates update
once a day, so an in-memory per-day cache avoids hammering it.
"""

import json
import logging
import time
import urllib.request
from datetime import date, datetime
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger(__name__)

_rate_cache = {}  # (from_ccy, to_ccy, iso_date) -> rate

# How many times to try the real rate lookup before giving up and falling
# back to 1:1 (see to_base_checked) -- a module constant, not a magic
# number inline, so a test can shrink the retry delay to 0 (same discipline
# as nutrition.ALBUM_DEBOUNCE_SECONDS). Frankfurter is normally reliable;
# this exists for the ordinary transient blip (a dropped connection, a
# single slow response) rather than a real, sustained outage -- one retry
# is enough to meaningfully cut how often a logged expense silently lands
# unconverted without adding much latency to the common case where the
# first attempt just works.
GET_RATE_ATTEMPTS = 2
GET_RATE_RETRY_DELAY_SECONDS = 1.0


def _now_local_date() -> date:
    """Local calendar date in config.BOT_TIMEZONE -- deliberately NOT
    date.today() (the server's/OS date, UTC on Railway). That was a real
    bug: right after local midnight in Asia/Singapore (UTC+8) but before
    UTC midnight, date.today() still returns YESTERDAY's date for up to 8
    hours, so a rate fetched in that window got cached under the wrong day
    -- silently disagreeing with every other "what day is it" computation
    in the app (db.today_str() is the canonical one; see its docstring).
    A twin of db._now_local_date, not imported from there: db.py already
    imports fx.py (for currency conversion), so fx.py importing db.py back
    would be a circular import. Kept in sync by hand -- change one, change
    the other. Exposed as its own function (like db._now_local_date) so
    tests can monkeypatch it directly instead of needing to fake the
    system clock."""
    return datetime.now(ZoneInfo(config.BOT_TIMEZONE)).date()


def get_rate(from_ccy: str, to_ccy: str) -> float:
    """Retries once (see GET_RATE_ATTEMPTS/GET_RATE_RETRY_DELAY_SECONDS)
    before raising -- added after a real incident where a single failed
    lookup silently produced an unconverted expense (see to_base_checked's
    docstring). Still raises on a sustained failure; to_base/to_base_checked
    are what fall back to 1:1, this just gives a transient blip a second
    chance first."""
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()
    if from_ccy == to_ccy:
        return 1.0

    key = (from_ccy, to_ccy, _now_local_date().isoformat())
    if key in _rate_cache:
        return _rate_cache[key]

    url = f"https://api.frankfurter.app/latest?from={from_ccy}&to={to_ccy}"
    last_error = None
    for attempt in range(1, GET_RATE_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            rate = data["rates"][to_ccy]
            _rate_cache[key] = rate
            return rate
        except Exception as e:
            last_error = e
            if attempt < GET_RATE_ATTEMPTS:
                logger.warning("FX lookup attempt %d/%d failed for %s -> %s, retrying: %s",
                                attempt, GET_RATE_ATTEMPTS, from_ccy, to_ccy, e)
                time.sleep(GET_RATE_RETRY_DELAY_SECONDS)
    raise last_error


def to_base_checked(amount: float, currency: str | None) -> tuple[float, bool]:
    """Like to_base, but also reports whether the real exchange rate was
    actually used: (amount_base, True) on a normal conversion, (amount
    unchanged, False) when every attempt failed and the 1:1 fallback kicked
    in. Added after a real production incident: a transient Frankfurter
    failure silently produced a materially wrong amount_base (a $115 USD
    lunch landed in the SGD total as if it were $115 SGD, no conversion at
    all), with nothing in the user-facing confirmation to say the real rate
    hadn't actually been used. db.add_expense uses this (not plain
    to_base) to store the fallback flag on the row itself
    (expenses.fx_fallback), so handlers.py's log_expense confirmation can
    warn the user instead of presenting a quietly-wrong number as final.
    to_base itself is unchanged, for every other caller that doesn't need
    the flag."""
    currency = (currency or config.BASE_CURRENCY).upper()
    if currency == config.BASE_CURRENCY:
        return amount, True
    try:
        rate = get_rate(currency, config.BASE_CURRENCY)
        return amount * rate, True
    except Exception:
        logger.exception("FX lookup failed for %s -> %s, using 1:1 as a fallback", currency, config.BASE_CURRENCY)
        return amount, False


def to_base(amount: float, currency: str | None) -> float:
    """Converts `amount` in `currency` into config.BASE_CURRENCY. Falls back
    to a 1:1 conversion (with a logged warning) if the rate lookup fails, so
    a network hiccup never blocks logging an expense. See to_base_checked
    if the caller needs to know whether the fallback actually happened."""
    return to_base_checked(amount, currency)[0]


def normalize_currency(raw: str | None) -> str | None:
    """Returns a valid known currency code from user input, or None if it
    doesn't match anything we recognize."""
    if not raw:
        return None
    raw = raw.strip().upper()
    return raw if raw in config.KNOWN_CURRENCIES else None
