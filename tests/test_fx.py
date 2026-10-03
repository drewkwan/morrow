"""
Tests for fx.py's per-day rate cache and its own BOT_TIMEZONE-aware clock.

fx.get_rate is monkeypatched to raise by conftest's autouse
_no_real_network fixture (see conftest.py), so these tests call the real
implementation directly via a reference captured at import time, before
any per-test monkeypatching happens -- the only way to exercise get_rate's
own caching logic without hitting the network guard meant for every OTHER
test in the suite.
"""

import json
from datetime import date

import config
import fx

# Captured now, at module import/collection time -- before any test's
# autouse fixtures run and replace the module attribute fx.get_rate with
# conftest's network-guard stub. This is the real, un-patched function.
_REAL_GET_RATE = fx.get_rate


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_get_rate_cache_key_uses_bot_timezone_date_not_server_clock(monkeypatch):
    """Regression test for a real bug: the per-day rate cache used to key
    off date.today() (the server's/OS date -- UTC on Railway), which lags
    Asia/Singapore's actual calendar date by up to 8 hours after local
    midnight -- a rate fetched in that window got cached under the wrong
    day, silently disagreeing with every other "what day is it"
    computation in the app (db.today_str(), the canonical one). get_rate
    must key off fx._now_local_date() -- a BOT_TIMEZONE-aware,
    independently monkeypatchable twin of db._now_local_date -- not the
    raw system clock, which is what this test drives directly."""
    fx._rate_cache.clear()
    calls = []

    def fake_urlopen_v1(url, timeout=8):
        calls.append(url)
        return _FakeResponse({"rates": {"SGD": 1.35}})

    monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen_v1)
    monkeypatch.setattr(fx, "_now_local_date", lambda: date(2026, 9, 15))

    assert _REAL_GET_RATE("USD", "SGD") == 1.35
    assert len(calls) == 1

    # Same local day (per _now_local_date) -- served from cache, no second
    # network call, even though we haven't touched urlopen's fake return.
    assert _REAL_GET_RATE("USD", "SGD") == 1.35
    assert len(calls) == 1

    # A new local day per _now_local_date -- nothing about the real system
    # clock changed, but this must still re-fetch rather than silently
    # reusing yesterday's cached rate.
    def fake_urlopen_v2(url, timeout=8):
        calls.append(url)
        return _FakeResponse({"rates": {"SGD": 1.40}})

    monkeypatch.setattr(fx.urllib.request, "urlopen", fake_urlopen_v2)
    monkeypatch.setattr(fx, "_now_local_date", lambda: date(2026, 9, 16))

    assert _REAL_GET_RATE("USD", "SGD") == 1.40
    assert len(calls) == 2


# ---------- get_rate: retry before giving up ----------
#
# Added after a real production incident (see to_base_checked's docstring):
# a single failed Frankfurter lookup used to fall straight through to the
# 1:1 fallback with no second attempt, even for an ordinary transient blip.

def test_get_rate_retries_once_then_succeeds(monkeypatch):
    fx._rate_cache.clear()
    monkeypatch.setattr(fx, "GET_RATE_RETRY_DELAY_SECONDS", 0)  # no real sleep in tests
    monkeypatch.setattr(fx, "_now_local_date", lambda: date(2026, 9, 17))
    calls = []

    def flaky_urlopen(url, timeout=8):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("simulated transient network failure")
        return _FakeResponse({"rates": {"SGD": 1.35}})

    monkeypatch.setattr(fx.urllib.request, "urlopen", flaky_urlopen)
    assert _REAL_GET_RATE("USD", "SGD") == 1.35
    assert len(calls) == 2  # first attempt failed, second succeeded


def test_get_rate_raises_after_every_attempt_fails(monkeypatch):
    fx._rate_cache.clear()
    monkeypatch.setattr(fx, "GET_RATE_RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(fx, "_now_local_date", lambda: date(2026, 9, 18))
    calls = []

    def always_fails(url, timeout=8):
        calls.append(url)
        raise OSError("simulated sustained outage")

    monkeypatch.setattr(fx.urllib.request, "urlopen", always_fails)
    try:
        _REAL_GET_RATE("USD", "SGD")
        assert False, "expected get_rate to raise after exhausting retries"
    except OSError:
        pass
    assert len(calls) == fx.GET_RATE_ATTEMPTS


# ---------- to_base_checked: reports whether the real rate was used ----------

def test_to_base_checked_reports_success_on_a_normal_conversion(monkeypatch):
    monkeypatch.setattr(fx, "get_rate", lambda f, t: 1.35)
    amount_base, ok = fx.to_base_checked(20, "USD")
    assert amount_base == 27.0
    assert ok is True


def test_to_base_checked_reports_failure_and_falls_back_to_1to1(monkeypatch):
    """Regression test for the real incident: amount_base must still come
    back usable (never block logging) but the caller must be able to tell
    the real rate wasn't actually used."""
    def _always_fails(f, t):
        raise RuntimeError("simulated Frankfurter outage")

    monkeypatch.setattr(fx, "get_rate", _always_fails)
    amount_base, ok = fx.to_base_checked(115, "USD")
    assert amount_base == 115  # unconverted
    assert ok is False


def test_to_base_checked_same_currency_is_always_success():
    amount_base, ok = fx.to_base_checked(50, config.BASE_CURRENCY)
    assert amount_base == 50
    assert ok is True


def test_to_base_still_returns_a_plain_float_unaffected_by_the_checked_variant(monkeypatch):
    """to_base's existing signature/behavior must be untouched for its
    other seven call sites (income, deductions, edits) that don't need
    the fallback flag."""
    monkeypatch.setattr(fx, "get_rate", lambda f, t: 1.35)
    assert fx.to_base(20, "USD") == 27.0
