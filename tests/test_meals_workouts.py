"""
Tests for the meals/workouts additions: db.py CRUD, ai.py's extraction
resilience (text, vision, and workout), and bot.py's natural-language and
photo logging paths, including cross-domain corrections.

Same discipline as the rest of the suite: no real network, no real Claude
calls (ai._get_client is always mocked), throwaway SQLite per test.
"""

import asyncio
import datetime as dt

import ai
import bot
import db
import nutrition
from conftest import CHAT
from test_ai import _FakeClient, _mock_client  # reuse the existing fake client


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------- db.py: meals ----------

def test_add_and_get_meal_roundtrip():
    meal_id = db.add_meal(CHAT, "Snack", ["coke zero", "750ml water"], 0, 0, 0, water_ml=750)
    row = db.get_meal(CHAT, meal_id)
    assert row["items"] == ["coke zero", "750ml water"]
    assert row["water_ml"] == 750
    assert row["meal_date"] == db.today_str()


def test_get_recent_meals_orders_by_date_not_insertion_order():
    """Regression test for a real bug: get_recent_meals used to order by
    "id DESC" alone, so a historical backfill (which inserts old-dated
    rows LAST, giving them the newest ids) could outrank a genuinely more
    recent meal in "recent" lists -- see db.get_recent_meals's docstring
    and the matching vitals.py bug this was first observed in."""
    db.add_meal(CHAT, "Breakfast", ["toast"], 200, 300, 250, meal_date="2026-07-01")
    db.add_meal(CHAT, "Lunch", ["salad"], 300, 400, 350, meal_date="2026-09-20")
    # Inserted last (highest id), but dated before the Sept 20 entry above.
    db.add_meal(CHAT, "Dinner", ["pasta"], 500, 700, 600, meal_date="2026-07-15")
    dates = [m["meal_date"] for m in db.get_recent_meals(CHAT)]
    assert dates == sorted(dates, reverse=True)


def test_get_recent_workouts_orders_by_date_not_insertion_order():
    db.add_workout(CHAT, "run", workout_date="2026-07-01")
    db.add_workout(CHAT, "tennis", workout_date="2026-09-20")
    db.add_workout(CHAT, "gym", workout_date="2026-07-15")  # inserted last, dated earliest of the three
    dates = [w["workout_date"] for w in db.get_recent_workouts(CHAT)]
    assert dates == sorted(dates, reverse=True)


def test_daily_meal_totals_sum_same_day_only():
    db.add_meal(CHAT, "Snack", ["coke zero"], 0, 0, 0, water_ml=750)
    db.add_meal(CHAT, "Dinner", ["rice", "chicken"], 990, 1360, 1170, water_ml=None)
    yesterday = dt.date.today() - dt.timedelta(days=1)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO meals (chat_id, meal_type, items, calories_estimate, meal_date) "
            "VALUES (?, 'Snack', '[]', 500, ?)",
            (CHAT, yesterday.isoformat()),
        )
    totals = db.get_daily_meal_totals(CHAT, db.today_str())
    assert totals["calories"] == 1170  # today's two entries only, not yesterday's 500
    assert totals["water_ml"] == 750


def test_edit_meal_date_clamps_future_to_today():
    meal_id = db.add_meal(CHAT, "Snack", ["mango"], 90, 120, 105)
    future = (dt.date.today() + dt.timedelta(days=5)).isoformat()
    row = db.edit_meal_date(CHAT, meal_id, future)
    assert row["meal_date"] == db.today_str()


def test_delete_and_restore_meal_roundtrip():
    meal_id = db.add_meal(CHAT, "Dinner", ["rice"], 200, 300, 250, water_ml=None)
    deleted = db.delete_meal(CHAT, meal_id)
    assert db.get_meal(CHAT, meal_id) is None
    restored = db.restore_deleted_meal(CHAT, deleted)
    assert restored["items"] == ["rice"]
    assert restored["calories_estimate"] == 250
    assert restored["id"] != meal_id  # fresh id, same as delete_expense's documented behavior


def test_delete_most_recent_meal():
    db.add_meal(CHAT, "Snack", ["first"], 10, 10, 10)
    db.add_meal(CHAT, "Snack", ["second"], 20, 20, 20)
    deleted = db.delete_most_recent_meal(CHAT)
    assert deleted["items"] == ["second"]
    assert [m["items"][0] for m in db.get_recent_meals(CHAT)] == ["first"]


# ---------- db.py: workouts ----------

def test_add_and_get_workout_roundtrip():
    workout_id = db.add_workout(CHAT, "tennis", duration_min=60, distance_km=None, notes="won 2 sets")
    row = db.get_workout(CHAT, workout_id)
    assert row["activity"] == "tennis"
    assert row["duration_min"] == 60
    assert row["notes"] == "won 2 sets"
    assert row["calories_burned"] is None  # not every workout reports it (e.g. a typed /logworkout)
    assert row["workout_date"] == db.today_str()


def test_add_workout_stores_calories_burned():
    """Calories burned is the main field a fitness app/wearable screenshot
    contributes (see ai.extract_from_photo) -- distinct from meals'
    calories_estimate (calories out vs calories in)."""
    workout_id = db.add_workout(CHAT, "daily activity", calories_burned=540, notes="2,300 steps")
    row = db.get_workout(CHAT, workout_id)
    assert row["calories_burned"] == 540


def test_get_daily_workout_totals_sums_same_day_only():
    db.add_workout(CHAT, "run", calories_burned=300)
    db.add_workout(CHAT, "daily activity", calories_burned=240)
    yesterday = dt.date.today() - dt.timedelta(days=1)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO workouts (chat_id, activity, calories_burned, workout_date) VALUES (?, 'run', 999, ?)",
            (CHAT, yesterday.isoformat()),
        )
    totals = db.get_daily_workout_totals(CHAT, db.today_str())
    assert totals["calories_burned"] == 540  # today's two entries only, not yesterday's 999


def test_edit_workout_date_and_delete_restore():
    workout_id = db.add_workout(CHAT, "run", distance_km=2.4, calories_burned=310)
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    moved = db.edit_workout_date(CHAT, workout_id, yesterday)
    assert moved["workout_date"] == yesterday

    deleted = db.delete_workout(CHAT, workout_id)
    assert db.get_workout(CHAT, workout_id) is None
    restored = db.restore_deleted_workout(CHAT, deleted)
    assert restored["activity"] == "run"
    assert restored["distance_km"] == 2.4
    assert restored["calories_burned"] == 310  # must carry through delete/restore, not just get dropped


def test_edit_workout_only_touches_fields_passed():
    """Regression test for a real bad interaction: a photo-read
    calories_burned that turned out to only cover one workout, not the
    day's real total including BMR, had no way to be corrected without
    deleting and relogging the whole entry."""
    workout_id = db.add_workout(CHAT, "daily activity", duration_min=108, calories_burned=934,
                                 notes="Steps: 14,524")
    updated = db.edit_workout(CHAT, workout_id, new_calories_burned=2862)
    assert updated["calories_burned"] == 2862
    assert updated["activity"] == "daily activity"  # untouched
    assert updated["duration_min"] == 108  # untouched
    assert updated["notes"] == "Steps: 14,524"  # untouched


def test_edit_workout_returns_none_for_missing_workout():
    assert db.edit_workout(CHAT, 999999, new_calories_burned=100) is None


# ---------- correction.py: edit_workout / edit_lift dispatch ----------

def test_correction_can_edit_workout_calories_burned_and_undo(monkeypatch):
    """Regression test for a real bad interaction: "correct the calories out
    to 2862" used to have no supported action at all for a workout target,
    and the model would sometimes wrongly emit correction_action="edit_date"
    instead (producing a nonsense "which day?" question) rather than a
    proper field edit."""
    db.get_or_create_user(CHAT)
    workout_id = db.add_workout(CHAT, "daily activity", calories_burned=934)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "workout", "target_expense_id": workout_id,
            "correction_action": "edit_workout", "new_workout_calories_burned": 2862,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="correct the calories out to 2862")
    _run(bot.handle_text(edit_update, context))
    assert db.get_workout(CHAT, workout_id)["calories_burned"] == 2862

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_workout(CHAT, workout_id)["calories_burned"] == 934


def test_correction_edit_workout_with_nothing_set_asks_what_to_fix(monkeypatch):
    db.get_or_create_user(CHAT)
    workout_id = db.add_workout(CHAT, "run", calories_burned=300)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "workout", "target_expense_id": workout_id,
            "correction_action": "edit_workout",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="fix that workout")
    _run(bot.handle_text(update, FakeContext()))
    assert "What should I fix" in update.message.replies[-1]


def test_correction_unsupported_field_gives_accurate_message_not_edit_date(monkeypatch):
    """Regression test for the exact observed bug: a request for a field
    edit that isn't supported for a domain must produce the accurate "not
    supported yet" message (naming what IS supported), never the nonsense
    "which day did you mean?" question that came from wrongly defaulting
    to correction_action="edit_date" with no day actually mentioned."""
    db.get_or_create_user(CHAT)
    vitals_id = db.add_vitals(CHAT, weight_kg=76.0)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "vitals", "target_expense_id": vitals_id,
            "correction_action": "edit_unsupported_field",
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="actually my sleep was 8 hours not 6")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "isn't supported yet" in reply
    assert "which day" not in reply.lower()


# ---------- ai.py: extract_meal / extract_from_photo / extract_workout ----------

def test_extract_meal_happy_path(monkeypatch):
    import json
    payload = {"meal_type": "Snack", "items": ["mango"], "calories_low": 90,
               "calories_high": 120, "calories_estimate": 105, "water_ml": None}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_meal("had a mango")
    assert result["items"] == ["mango"]
    assert result["calories_estimate"] == 105


def test_extract_meal_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, ConnectionError("network blip"))
    result = ai.extract_meal("some meal")
    assert result["calories_estimate"] is None
    assert result["items"] == ["some meal"]  # raw text preserved rather than lost


def test_extract_from_photo_sends_image_content_block(monkeypatch):
    """Regression guard mirroring test_parse_message_passes_recent_expenses_into_the_prompt:
    the image must actually be sent as an image content block, not just the caption text."""
    import json
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    payload = {"kind": "meal", "meal_type": "Dinner", "items": ["rice", "chicken"], "calories_low": 990,
               "calories_high": 1360, "calories_estimate": 1170, "water_ml": None}
    fake = _CapturingClient(json.dumps(payload))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)

    result = ai.extract_from_photo(b"fake-jpeg-bytes", caption="my usual dinner")
    content_blocks = captured["messages"][0]["content"]
    types = [b["type"] for b in content_blocks]
    assert "image" in types
    assert content_blocks[types.index("image")]["source"]["media_type"] == "image/jpeg"
    assert result["kind"] == "meal"
    assert result["calories_estimate"] == 1170


def test_extract_from_photo_tells_the_model_todays_actual_date(monkeypatch):
    """Root-cause regression guard for the real reported bug: a photo
    captioned "Stats from 18 September" needs today's real date to compute
    logged_days_ago against an explicit date -- relative phrasing alone
    doesn't need this, which is exactly why the gap went unnoticed."""
    import json
    monkeypatch.setattr(db, "today_str", lambda: "2026-09-19")
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    payload = {"kind": "meal", "meal_type": "Dinner", "items": ["rice"], "calories_low": 100,
               "calories_high": 200, "calories_estimate": 150, "water_ml": None}
    fake = _CapturingClient(json.dumps(payload))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)

    ai.extract_from_photo(b"fake-jpeg-bytes", caption="Stats from 18 September")
    content_blocks = captured["messages"][0]["content"]
    text_block = next(b["text"] for b in content_blocks if b["type"] == "text")
    assert "2026-09-19" in text_block
    assert "Saturday" in text_block


def test_extract_from_photo_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, TypeError("boom"))
    result = ai.extract_from_photo(b"bytes", caption="dinner")
    assert result == {"kind": "unclear"}


def test_extract_from_photo_recognizes_a_fitness_stats_screenshot_as_a_workout(monkeypatch):
    """Regression test for a real bad interaction: a screenshot of a
    FITNESS app's daily calorie-burned stats -- not food, but genuinely
    useful data -- got either logged as a fake meal built from the caption
    ("Here are my calorie stats..." -- unknown kcal) or, an earlier and
    still-wrong fix, rejected outright as "not food" even though it plainly
    showed real calories-burned data. The model now classifies this shape of
    photo as kind "workout" with calories_burned set, and extract_from_photo
    must surface that as-is so it gets logged as a workout, not dropped."""
    import json
    payload = {"kind": "workout", "activity": "daily activity", "duration_min": None,
               "distance_km": None, "calories_burned": 540, "notes": "2,300 steps"}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_from_photo(b"fake-screenshot-bytes", caption="my calorie stats from today")
    assert result == payload


def test_extract_from_photo_falls_back_to_unclear_on_an_unparseable_reply(monkeypatch):
    _mock_client(monkeypatch, "not valid json at all")
    result = ai.extract_from_photo(b"fake-bytes", caption=None)
    assert result == {"kind": "unclear"}


def test_extract_from_photo_surfaces_a_caption_that_names_a_different_food(monkeypatch):
    """Regression test for a real bad interaction: a photo of nachos sent
    with the caption "I also had a small bowl of black bean pork broth" got
    logged as ONE meal combining both foods -- "loaded nachos ... black bean
    pork broth" at a wildly over-estimated calorie range, because the
    caption's separate food got folded straight into the photo's item list.
    The model now reports that separate food via caption_extra_item instead
    of blending it into items/calories, and extract_from_photo must surface
    it as-is (handle_photo is what actually asks about it, tested
    separately)."""
    import json
    payload = {"kind": "meal", "meal_type": "Lunch", "items": ["loaded nachos with pulled pork"],
               "calories_low": 600, "calories_high": 800, "calories_estimate": 700, "water_ml": None,
               "caption_extra_item": "a small bowl of black bean pork broth"}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_from_photo(b"fake-nachos-bytes", caption="I also had a small bowl of black bean pork broth")
    assert result["caption_extra_item"] == "a small bowl of black bean pork broth"
    assert result["items"] == ["loaded nachos with pulled pork"]  # the broth must NOT be folded in here


def test_extract_workout_happy_path(monkeypatch):
    import json
    payload = {"activity": "tennis", "duration_min": 60, "distance_km": None, "notes": "won 2 sets"}
    _mock_client(monkeypatch, json.dumps(payload))
    result = ai.extract_workout("played tennis for an hour, won 2 sets")
    assert result["activity"] == "tennis"
    assert result["duration_min"] == 60


def test_extract_workout_never_raises_on_api_failure(monkeypatch):
    _mock_client(monkeypatch, ConnectionError("network blip"))
    result = ai.extract_workout("gym leg day")
    assert result["activity"] == "gym leg day"
    assert result["notes"] == "gym leg day"


def test_parse_message_passes_recent_meals_and_workouts_into_the_prompt(monkeypatch):
    import json
    captured = {}

    class _CapturingClient(_FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return super().create(**kwargs)

    fake = _CapturingClient(json.dumps({
        "intent": "casual", "casual_reply": "hey!",
    }))
    monkeypatch.setattr(ai, "_get_client", lambda: fake)
    recent_meals = [{"id": 7, "items": ["mango"], "meal_type": "Snack",
                      "calories_estimate": 105, "meal_date": "2026-09-03"}]
    recent_workouts = [{"id": 3, "activity": "tennis", "duration_min": 60, "workout_date": "2026-09-02"}]
    ai.parse_message("hi", [], recent_meals, recent_workouts)
    sent = captured["messages"][0]["content"]
    assert "mango" in sent
    assert "tennis" in sent


# ---------- bot.py: natural-language and photo logging ----------

def _log_meal_response(**overrides):
    meals = overrides.pop("meals", None) or [
        {"meal_type": "Snack", "items": ["mango"],
         "calories_low": 90, "calories_high": 120, "calories_estimate": 105, "water_ml": None}
    ]
    base = {
        "intent": "log_meal", "meals": meals,
        "clarification_question": None, "casual_reply": None,
    }
    base.update(overrides)
    return base


class FakeMessage:
    def __init__(self, text=None, caption=None, photo=None):
        self.text = text
        self.caption = caption
        self.photo = photo or []
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeChat:
    def __init__(self, chat_id):
        self.id = chat_id


class FakeUpdate:
    def __init__(self, chat_id, text=None, caption=None, photo=None):
        self.message = FakeMessage(text=text, caption=caption, photo=photo)
        self.effective_chat = FakeChat(chat_id)


class FakeContext:
    def __init__(self):
        self.chat_data = {}
        self.args = []


def test_natural_language_log_meal_updates_running_total(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return _log_meal_response()

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="had a mango")
    _run(bot.handle_text(update, FakeContext()))
    assert any("mango" in r for r in update.message.replies)
    assert any("running total" in r.lower() for r in update.message.replies)
    assert db.get_recent_meals(CHAT)[0]["items"] == ["mango"]


def test_natural_language_log_meal_with_two_meals_logs_both(monkeypatch):
    """Regression test for a real bad interaction: a message describing
    breakfast AND lunch only got breakfast logged, with the lunch (noodles,
    broth, water, iced latte) silently dropped -- because parse_message's
    log_meal fields used to be singular (one meal_type/items/etc slot per
    message) instead of a list like log_expense's "expenses". Both meals
    must now be logged from one message, and running-total calories must
    reflect both, not just the first."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return _log_meal_response(meals=[
            {"meal_type": "Breakfast", "items": ["toast with strawberry jam", "Old Town white coffee"],
             "calories_low": 320, "calories_high": 420, "calories_estimate": 370, "water_ml": None},
            {"meal_type": "Lunch", "items": ["dry chilli oil dumpling noodles", "tomato pork bone broth",
                                              "iced latte"],
             "calories_low": 650, "calories_high": 900, "calories_estimate": 780, "water_ml": 500},
        ])

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text=(
        "for my breakfast: toast with strawberry jam and an old town white coffee. For lunch: dry chilli oil "
        "dumpling noodles and some tomato pork bone broth, 500ml water and an iced latte"
    ))
    _run(bot.handle_text(update, FakeContext()))

    reply = update.message.replies[-1]
    assert "toast with strawberry jam" in reply
    assert "dry chilli oil dumpling noodles" in reply, "the lunch must not be dropped"
    assert "Logged 2 meals" in reply

    logged = db.get_recent_meals(CHAT)
    assert len(logged) == 2
    assert {"Breakfast", "Lunch"} == {m["meal_type"] for m in logged}
    totals = db.get_daily_meal_totals(CHAT, db.today_str())
    assert totals["calories"] == 370 + 780
    assert totals["water_ml"] == 500


def test_logmeal_command_uses_extract_meal(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "extract_meal", lambda desc: {
        "meal_type": "Dinner", "items": ["rice", "chicken"],
        "calories_low": 400, "calories_high": 600, "calories_estimate": 500, "water_ml": None,
    })
    update = FakeUpdate(CHAT)
    context = FakeContext()
    context.args = ["rice", "and", "chicken"]
    _run(bot.logmeal_cmd(update, context))
    assert any("rice, chicken" in r for r in update.message.replies)
    assert db.get_recent_meals(CHAT)[0]["calories_estimate"] == 500


class _FakeTelegramFile:
    async def download_as_bytearray(self):
        return bytearray(b"fake-jpeg-bytes")


class _FakePhotoSize:
    async def get_file(self):
        return _FakeTelegramFile()


def test_photo_message_logs_a_meal(monkeypatch):
    db.get_or_create_user(CHAT)
    captured = {}

    def fake_extract_from_photo(image_bytes, caption=None):
        captured["image_bytes"] = image_bytes
        captured["caption"] = caption
        return {"kind": "meal", "meal_type": "Dinner", "items": ["rice", "sambal sotong", "fried chicken"],
                "calories_low": 990, "calories_high": 1360, "calories_estimate": 1170, "water_ml": None}

    monkeypatch.setattr(bot.ai, "extract_from_photo", fake_extract_from_photo)
    update = FakeUpdate(CHAT, caption="dinner at home", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, FakeContext()))
    assert captured["caption"] == "dinner at home"
    assert captured["image_bytes"] == b"fake-jpeg-bytes"
    assert any("1170" in r or "990" in r for r in update.message.replies)
    assert db.get_recent_meals(CHAT)[0]["calories_estimate"] == 1170


def test_photo_of_a_fitness_stats_screenshot_logs_a_workout_against_todays_calories(monkeypatch):
    """Regression test for the real bad interaction: a screenshot of a
    FITNESS app's daily calorie-burned stats (not food) was logged as a
    fake meal titled with the caption text ("unknown kcal") -- and even
    after that was fixed to reject non-food photos outright, it was STILL
    wrong, because this genuinely useful data (calories burned) was simply
    dropped instead of being recognized and logged. handle_photo must now
    log it as a workout with calories_burned set, and the reply must show
    it against today's calories already eaten -- burning calories in
    isolation, with nothing to compare it to, isn't the point."""
    db.get_or_create_user(CHAT)
    db.add_meal(CHAT, "Lunch", ["chicken rice"], 500, 700, 600, water_ml=None)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {"kind": "workout", "activity": "daily activity",
                                            "duration_min": None, "distance_km": None,
                                            "calories_burned": 540, "notes": "2,300 steps"},
    )
    update = FakeUpdate(CHAT, caption="Here are my calorie stats from the day that's just past",
                         photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, FakeContext()))

    assert len(db.get_recent_meals(CHAT)) == 1  # just the lunch logged above -- the photo added no meal
    workouts = db.get_recent_workouts(CHAT)
    assert len(workouts) == 1
    assert workouts[0]["calories_burned"] == 540

    reply = update.message.replies[-1]
    assert "540" in reply  # calories burned
    assert "600" in reply  # today's calories in (the lunch logged above)
    assert not any(r.startswith("I couldn't tell") for r in update.message.replies)


def test_photo_of_something_unrelated_is_not_logged_at_all(monkeypatch):
    """Once extract_from_photo reports kind "unclear" (a receipt, a random
    photo, anything with no food or fitness data), handle_photo must skip
    logging entirely -- no meal row, no workout row -- and ask instead of
    guessing."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(bot.ai, "extract_from_photo", lambda image_bytes, caption=None: {"kind": "unclear"})
    update = FakeUpdate(CHAT, caption="random photo", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, FakeContext()))
    assert db.get_recent_meals(CHAT) == []
    assert db.get_recent_workouts(CHAT) == []
    assert not any("Logged" in r for r in update.message.replies)


def test_photo_with_a_caption_naming_a_different_food_asks_instead_of_guessing(monkeypatch):
    """Regression test for the real bad interaction: a photo of nachos
    captioned "I also had a small bowl of black bean pork broth" got logged
    as ONE entry combining both foods' calories into a wrong, over-estimated
    total -- the user had to /undo it and redo it by hand. Once
    extract_from_photo reports caption_extra_item, handle_photo must NOT log
    anything yet -- it should ask which food(s) are actually meant, and set
    up the pending-clarification state so the user's plain-text answer
    resolves it (see the follow-up test)."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {
            "kind": "meal", "meal_type": "Lunch", "items": ["loaded nachos with pulled pork"],
            "calories_low": 600, "calories_high": 800, "calories_estimate": 700, "water_ml": None,
            "caption_extra_item": "a small bowl of black bean pork broth",
        },
    )
    context = FakeContext()
    update = FakeUpdate(CHAT, caption="I also had a small bowl of black bean pork broth",
                         photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, context))

    assert db.get_recent_meals(CHAT) == []  # nothing committed yet
    reply = update.message.replies[-1]
    assert "nachos" in reply and "black bean pork broth" in reply
    assert not reply.startswith("Logged")
    assert bot.PENDING_KEY in context.chat_data  # the text reply that follows must be able to resolve this


def test_resolving_a_photo_caption_clarification_logs_only_what_the_user_confirms(monkeypatch):
    """Follow-up to the test above: once handle_photo asks and stores the
    pending clarification, the user's plain-text answer ("just log the
    broth") must flow through the SAME clarification loop handle_text
    already uses for a natural-language follow-up question, resolving to
    exactly what the user actually asked for -- not the photo's nachos, not
    both, just the broth -- the same real resolution as the live bug
    report."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {
            "kind": "meal", "meal_type": "Lunch", "items": ["loaded nachos with pulled pork"],
            "calories_low": 600, "calories_high": 800, "calories_estimate": 700, "water_ml": None,
            "caption_extra_item": "a small bowl of black bean pork broth",
        },
    )
    context = FakeContext()
    photo_update = FakeUpdate(CHAT, caption="I also had a small bowl of black bean pork broth",
                               photo=[_FakePhotoSize()])
    _run(bot.handle_photo(photo_update, context))

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        assert "black bean pork broth" in text and "nachos" in text  # the pending context must reach the model
        return _log_meal_response(meals=[{
            "meal_type": "Dinner", "items": ["black bean pork broth"],
            "calories_low": 80, "calories_high": 150, "calories_estimate": 115, "water_ml": None,
        }])

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    text_update = FakeUpdate(CHAT, text="Just log the black bean broth and it's for dinner")
    _run(bot.handle_text(text_update, context))

    logged = db.get_recent_meals(CHAT)
    assert len(logged) == 1
    assert logged[0]["items"] == ["black bean pork broth"]
    assert logged[0]["meal_type"] == "Dinner"


def test_natural_language_log_workout(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {"intent": "log_workout", "activity": "tennis", "duration_min": 60,
                "distance_km": None, "workout_notes": "won 2 sets",
                "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="played tennis for an hour, won 2 sets")
    _run(bot.handle_text(update, FakeContext()))
    assert any("tennis" in r for r in update.message.replies)
    assert db.get_recent_workouts(CHAT)[0]["activity"] == "tennis"


# ---------- richer log confirmations: weekly workout summary ----------

def test_weekly_workout_summary_text_says_first_workout_when_alone():
    db.add_workout(CHAT, "tennis", 60)
    assert bot._weekly_workout_summary_text(CHAT) == "First workout logged this week."


def test_weekly_workout_summary_text_counts_and_sums_calories_burned():
    db.add_workout(CHAT, "tennis", 60)
    db.add_workout(CHAT, "run", 30, calories_burned=300)
    text = bot._weekly_workout_summary_text(CHAT)
    assert "2 workouts logged" in text
    assert "300" in text


def test_weekly_workout_summary_text_excludes_workouts_outside_the_7_day_window():
    old_date = (dt.date.today() - dt.timedelta(days=10)).isoformat()
    db.add_workout(CHAT, "old run", 30, workout_date=old_date)
    db.add_workout(CHAT, "tennis", 60)
    assert bot._weekly_workout_summary_text(CHAT) == "First workout logged this week."


def test_natural_language_log_workout_shows_weekly_summary(monkeypatch):
    """The natural-language path (the one actually used day to day, not just
    /logworkout or photo logging) must also get the weekly summary -- a real
    gap: handlers.py used to build its own bare 'Logged: ...' reply inline
    instead of going through fitness._reply_workout_logged."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {"intent": "log_workout", "activity": "tennis", "duration_min": 60,
                "distance_km": None, "workout_notes": None,
                "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="played tennis for an hour")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Logged:" in reply
    assert "First workout logged this week." in reply


def test_natural_language_log_workout_backdates_with_logged_days_ago(monkeypatch):
    """Regression test for a real reported bug: 'last night I also went for
    a run' (or any natural-language log with no photo involved) always
    landed on today regardless of what the message said -- there was no
    field carrying a day cue at all for text-based logging (unlike photo
    captions, which already had logged_days_ago). The user had to notice
    and fix it with a manual correction afterwards."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {"intent": "log_workout", "activity": "run", "duration_min": 30,
                "distance_km": 5.0, "workout_notes": None, "logged_days_ago": 1,
                "clarification_question": None, "casual_reply": None}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="last night I went for a 5k run")
    _run(bot.handle_text(update, FakeContext()))

    expected_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    logged = db.get_recent_workouts(CHAT)[0]
    assert logged["workout_date"] == expected_date
    assert logged["workout_date"] != db.today_str()


def test_natural_language_log_meal_single_item_backdates_with_logged_days_ago(monkeypatch):
    """Same fix as the workout case above, for the exact interaction
    reported: 'last night I also had a cup of decaf tea with milk and
    500ml water to end' (sent well after midnight) got logged onto today."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return _log_meal_response(meals=[
            {"meal_type": None, "items": ["cup of decaf tea with milk"], "calories_low": 20, "calories_high": 50,
             "calories_estimate": 35, "water_ml": 500, "logged_days_ago": 1},
        ])

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="last night I also had a cup of decaf tea with milk and 500ml water to end")
    _run(bot.handle_text(update, FakeContext()))

    expected_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    logged = db.get_recent_meals(CHAT)[0]
    assert logged["meal_date"] == expected_date
    assert logged["meal_date"] != db.today_str()
    reply = update.message.replies[-1]
    assert expected_date in reply  # the running-total line must be labelled for that day, not "today"


def test_natural_language_log_meal_mixed_days_in_one_message_gets_per_day_totals(monkeypatch):
    """A message can legitimately describe more than one day's meals at once
    (e.g. catching up on yesterday AND mentioning something just eaten) --
    each item's own logged_days_ago must land on its own day, and since
    they're not all the same day, each line is date-tagged and BOTH days'
    totals are shown rather than one misleading "today" total."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return _log_meal_response(meals=[
            {"meal_type": "Dinner", "items": ["mango"], "calories_low": 90, "calories_high": 120,
             "calories_estimate": 105, "water_ml": None, "logged_days_ago": 1},
            {"meal_type": "Snack", "items": ["coffee"], "calories_low": 5, "calories_high": 15,
             "calories_estimate": 10, "water_ml": None, "logged_days_ago": 0},
        ])

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="yesterday I had a mango, and just now a coffee")
    _run(bot.handle_text(update, FakeContext()))

    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    today = db.today_str()
    dates_logged = {m["meal_date"] for m in db.get_recent_meals(CHAT)}
    assert dates_logged == {yesterday, today}
    reply = update.message.replies[-1]
    # Mixed days in one message -> every line is date-tagged (not just the
    # backdated one), since "today" alone would no longer be the unstated
    # default once a message spans more than one day.
    assert reply.split("\n")[1].endswith(f"({yesterday})") and "mango" in reply.split("\n")[1]
    assert reply.split("\n")[2].endswith(f"({today})") and "coffee" in reply.split("\n")[2]
    assert yesterday in reply and today in reply  # both days' totals shown, not just one


def test_correction_can_target_a_meal_by_domain(monkeypatch):
    """Cross-domain version of the existing expense date-correction test:
    a meal can be deleted by natural language too, dispatched via target_domain."""
    db.get_or_create_user(CHAT)
    meal_id = db.add_meal(CHAT, "Snack", ["duplicate mango"], 90, 120, 105)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "meal", "target_expense_id": meal_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="delete that mango, I logged it twice")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_meal(CHAT, meal_id) is None
    assert any("Deleted" in r for r in update.message.replies)


def test_undo_reverts_a_meal_deletion(monkeypatch):
    db.get_or_create_user(CHAT)
    meal_id = db.add_meal(CHAT, "Snack", ["mango"], 90, 120, 105)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "meal", "target_expense_id": meal_id,
            "correction_action": "delete", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    delete_update = FakeUpdate(CHAT, text="delete that mango")
    _run(bot.handle_text(delete_update, context))
    assert db.get_meal(CHAT, meal_id) is None

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_recent_meals(CHAT)[0]["items"] == ["mango"]


def test_correction_can_edit_a_meal_to_remove_a_hallucinated_item(monkeypatch):
    """Regression test for a real bad interaction: a photo of a curry gyu don
    added "ramen noodles" that were never actually eaten, and the only way
    to fix it was deleting the whole meal and relogging it from scratch.
    edit_meal lets the model send back the FULL corrected item list plus a
    fresh calorie re-estimate for it, applied in one shot via db.edit_meal."""
    db.get_or_create_user(CHAT)
    meal_id = db.add_meal(CHAT, "Lunch", ["curry gyu don (beef curry rice)", "ramen noodles"], 850, 1050, 950)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "meal", "target_expense_id": meal_id,
            "correction_action": "edit_meal", "days_ago": None,
            "new_meal_items": ["curry gyu don (beef curry rice)"],
            "new_meal_calories_low": 550, "new_meal_calories_high": 750, "new_meal_calories_estimate": 650,
            "new_meal_type": None, "new_meal_water_ml": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="minus the ramen noodles, I didn't have that")
    _run(bot.handle_text(update, FakeContext()))

    row = db.get_meal(CHAT, meal_id)
    assert row["items"] == ["curry gyu don (beef curry rice)"]
    assert row["calories_estimate"] == 650
    assert any("Updated" in r for r in update.message.replies)


def test_undo_reverts_a_meal_edit(monkeypatch):
    db.get_or_create_user(CHAT)
    meal_id = db.add_meal(CHAT, "Lunch", ["curry gyu don (beef curry rice)", "ramen noodles"], 850, 1050, 950)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "meal", "target_expense_id": meal_id,
            "correction_action": "edit_meal", "days_ago": None,
            "new_meal_items": ["curry gyu don (beef curry rice)"],
            "new_meal_calories_low": 550, "new_meal_calories_high": 750, "new_meal_calories_estimate": 650,
            "new_meal_type": None, "new_meal_water_ml": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    edit_update = FakeUpdate(CHAT, text="minus the ramen noodles, I didn't have that")
    _run(bot.handle_text(edit_update, context))
    assert db.get_meal(CHAT, meal_id)["items"] == ["curry gyu don (beef curry rice)"]

    undo_update = FakeUpdate(CHAT, text="undo")
    _run(bot.handle_text(undo_update, context))
    row = db.get_meal(CHAT, meal_id)
    assert row["items"] == ["curry gyu don (beef curry rice)", "ramen noodles"]
    assert row["calories_estimate"] == 950


def test_meal_correction_found_but_unsupported_action_says_so_distinctly(monkeypatch):
    """The id WAS matched correctly here -- only the action itself isn't
    supported for meals (e.g. edit_category, which only expenses support).
    This must read differently from "I'm not sure which meal you mean" --
    conflating the two was a real observed bug (see correction.py's
    _handle_simple_domain_correction docstring)."""
    db.get_or_create_user(CHAT)
    meal_id = db.add_meal(CHAT, "Lunch", ["gyu don"], 550, 750, 650)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None):
        return {
            "intent": "correction", "target_domain": "meal", "target_expense_id": meal_id,
            "correction_action": "edit_category", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, text="recategorize that")
    _run(bot.handle_text(update, FakeContext()))
    reply = update.message.replies[-1]
    assert "Found that meal" in reply
    assert "isn't supported yet" in reply
    assert db.get_meal(CHAT, meal_id)["items"] == ["gyu don"]  # untouched


# ---------- duplicate-workout detection (two photos, same underlying data) ----------

def test_photo_of_duplicate_fitness_stats_asks_instead_of_logging_twice(monkeypatch):
    """Regression test for a real bad interaction: two photos of the same
    underlying fitness-app data (identical calories_burned: 2279) got
    logged as two separate workouts, silently doubling the day's total to
    4558. A second photo reporting a calories_burned that matches an
    already-logged workout for the same day must ask instead of
    auto-logging a duplicate."""
    db.get_or_create_user(CHAT)
    db.add_workout(CHAT, "daily activity", calories_burned=2279, notes="Move goal 380/700")
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {"kind": "workout", "activity": "daily activity",
                                            "duration_min": None, "distance_km": None,
                                            "calories_burned": 2279, "notes": "Total active time 13h 22m"},
    )
    context = FakeContext()
    update = FakeUpdate(CHAT, caption="my stats", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, context))

    assert len(db.get_recent_workouts(CHAT)) == 1  # still just the one logged above
    reply = update.message.replies[-1]
    assert not reply.startswith("Logged")
    assert bot.PENDING_DUPLICATE_WORKOUT_KEY in context.chat_data


def test_confirming_a_duplicate_workout_is_actually_separate_logs_it(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_workout(CHAT, "daily activity", calories_burned=2279)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {"kind": "workout", "activity": "run",
                                            "duration_min": 30, "distance_km": 5.0,
                                            "calories_burned": 2279, "notes": None},
    )
    context = FakeContext()
    photo_update = FakeUpdate(CHAT, caption="my stats", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(photo_update, context))
    assert len(db.get_recent_workouts(CHAT)) == 1

    text_update = FakeUpdate(CHAT, text="No that's a separate workout, log it anyway")
    _run(bot.handle_text(text_update, context))

    workouts = db.get_recent_workouts(CHAT)
    assert len(workouts) == 2
    assert any(w["activity"] == "run" for w in workouts)
    assert bot.PENDING_DUPLICATE_WORKOUT_KEY not in context.chat_data


def test_declining_a_duplicate_workout_skips_it(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_workout(CHAT, "daily activity", calories_burned=2279)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {"kind": "workout", "activity": "daily activity",
                                            "duration_min": None, "distance_km": None,
                                            "calories_burned": 2279, "notes": None},
    )
    context = FakeContext()
    photo_update = FakeUpdate(CHAT, caption="my stats", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(photo_update, context))

    text_update = FakeUpdate(CHAT, text="No, same one shown again, skip it")
    _run(bot.handle_text(text_update, context))

    assert len(db.get_recent_workouts(CHAT)) == 1  # unchanged -- no duplicate created
    assert bot.PENDING_DUPLICATE_WORKOUT_KEY not in context.chat_data


def test_photo_workout_not_flagged_as_duplicate_when_calories_differ_substantially(monkeypatch):
    """A genuinely different workout that happens to also report a
    calories_burned figure must NOT be blocked -- only an actual match
    (within the 2%/5kcal tolerance) counts as a likely duplicate."""
    db.get_or_create_user(CHAT)
    db.add_workout(CHAT, "run", calories_burned=300)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {"kind": "workout", "activity": "cycling",
                                            "duration_min": None, "distance_km": None,
                                            "calories_burned": 900, "notes": None},
    )
    update = FakeUpdate(CHAT, caption="stats", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, FakeContext()))
    assert len(db.get_recent_workouts(CHAT)) == 2
    assert any(r.startswith("Logged") for r in update.message.replies)


def test_find_duplicate_workout_catches_a_match_on_the_adjacent_day():
    """Regression test for a real bad interaction: two related photos (a
    named workout + that same day's daily-activity total) landed on
    ADJACENT dates instead of the same one, because only one of the two
    carried usable date context -- same-date-only matching silently missed
    it and both got logged, doubling the day's real burn total. Checking a
    one-day-either-side window catches this."""
    import datetime as dt
    db.get_or_create_user(CHAT)
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    db.add_workout(CHAT, "daily activity", calories_burned=2548, workout_date=yesterday)
    match = bot._find_duplicate_workout(CHAT, dt.date.today().isoformat(), 2548)
    assert match is not None
    assert match["workout_date"] == yesterday


def test_find_duplicate_workout_does_not_reach_two_days_out():
    """The widened window is deliberately narrow (one day either side, not
    unbounded) -- a match two days away is far enough that it's more
    likely a real coincidence than the same mis-dated data."""
    import datetime as dt
    db.get_or_create_user(CHAT)
    two_days_ago = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    db.add_workout(CHAT, "daily activity", calories_burned=2548, workout_date=two_days_ago)
    match = bot._find_duplicate_workout(CHAT, dt.date.today().isoformat(), 2548)
    assert match is None


# ---------- backdating a photo log from its caption's date (logged_days_ago) ----------

def test_target_date_from_days_ago_caps_and_handles_none():
    assert bot._target_date_from_days_ago(None) is None
    assert bot._target_date_from_days_ago(0) is None
    assert bot._target_date_from_days_ago(999) == (dt.date.today() - dt.timedelta(days=14)).isoformat()
    assert bot._target_date_from_days_ago(1) == (dt.date.today() - dt.timedelta(days=1)).isoformat()


def test_target_date_from_days_ago_uses_bot_timezone_not_server_clock(monkeypatch):
    """Regression test for a real bug: this used to anchor on date.today()
    (the server's/OS date -- UTC on Railway) instead of db.today_str()
    (BOT_TIMEZONE-aware). Asia/Singapore is UTC+8, so a photo backdated
    right after local midnight there, during the up-to-8-hour window before
    UTC midnight also rolls over, landed a day off from what the caption
    actually meant. A fake "today" deliberately different from whatever
    the real system clock says in CI makes sure this fails if date.today()
    ever creeps back in."""
    fake_today = dt.date(2026, 9, 20)
    monkeypatch.setattr(db, "_now_local_date", lambda: fake_today)
    assert bot._target_date_from_days_ago(2) == (fake_today - dt.timedelta(days=2)).isoformat()


def test_photo_meal_with_caption_date_logs_directly_onto_that_day(monkeypatch):
    """Regression test for a real bad interaction: a photo captioned "these
    were my stats for 15 September" got logged as today anyway, needing a
    manual /undo + date correction afterwards. Once extract_from_photo
    reports logged_days_ago, handle_photo must log directly onto the
    correct backdated day instead."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {
            "kind": "meal", "meal_type": "Dinner", "items": ["rice", "chicken"],
            "calories_low": 400, "calories_high": 600, "calories_estimate": 500, "water_ml": None,
            "logged_days_ago": 2,
        },
    )
    update = FakeUpdate(CHAT, caption="this was from 2 days ago", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, FakeContext()))

    expected_date = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    logged = db.get_recent_meals(CHAT)[0]
    assert logged["meal_date"] == expected_date
    assert logged["meal_date"] != db.today_str()
    reply = update.message.replies[-1]
    assert expected_date in reply  # the running-total line must be labelled for that day, not "today"


def test_photo_workout_with_caption_date_logs_directly_onto_that_day(monkeypatch):
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(
        bot.ai, "extract_from_photo",
        lambda image_bytes, caption=None: {
            "kind": "workout", "activity": "daily activity", "duration_min": None,
            "distance_km": None, "calories_burned": 2279, "notes": None,
            "logged_days_ago": 1,
        },
    )
    update = FakeUpdate(CHAT, caption="these were my stats for yesterday", photo=[_FakePhotoSize()])
    _run(bot.handle_photo(update, FakeContext()))

    expected_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    logged = db.get_recent_workouts(CHAT)[0]
    assert logged["workout_date"] == expected_date


# ---------- nutrition.py: photo albums (several photos sent in one message) ----------

def test_album_of_two_photos_logs_only_one_workout_not_two(monkeypatch):
    """Regression test for a real reported bug: two screenshots of the same
    day's fitness-app activity, sent together in one Telegram message, got
    logged as TWO separate workouts (one of them landing on the wrong date,
    since Telegram only attaches the caption to one of the two photos) --
    silently doubling that day's calories-burned total. Both photos must
    now be buffered and handed to ai.extract_from_photos in ONE call,
    producing exactly one logged workout."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(nutrition, "ALBUM_DEBOUNCE_SECONDS", 0.05)
    calls = {"extract_from_photo": 0, "extract_from_photos": None}

    def fake_extract_from_photo(image_bytes, caption=None):
        calls["extract_from_photo"] += 1
        raise AssertionError("the single-image path must not be used for an album")

    def fake_extract_from_photos(images, caption=None):
        calls["extract_from_photos"] = {"images": list(images), "caption": caption}
        return {"kind": "workout", "activity": "daily activity", "duration_min": None,
                "distance_km": 9.3, "calories_burned": 2732, "notes": None, "logged_days_ago": 1}

    monkeypatch.setattr(bot.ai, "extract_from_photo", fake_extract_from_photo)
    monkeypatch.setattr(bot.ai, "extract_from_photos", fake_extract_from_photos)

    context = FakeContext()
    update1 = FakeUpdate(CHAT, caption="Here were my stats from yesterday", photo=[_FakePhotoSize()])
    update1.message.media_group_id = "album-1"
    update1.message.photo = [_FakePhotoSize()]
    update2 = FakeUpdate(CHAT, photo=[_FakePhotoSize()])  # part two -- no caption of its own, as Telegram sends it
    update2.message.media_group_id = "album-1"

    async def _simulate_album():
        await bot.handle_photo(update1, context)
        await bot.handle_photo(update2, context)
        # Both calls above scheduled/cancelled tasks on THIS running loop -- await
        # whichever one is still pending to drive the debounced processing to
        # completion, the same way production just lets it fire on its own.
        pending = nutrition._PENDING_ALBUMS["album-1"]["task"]
        await pending

    _run(_simulate_album())

    assert calls["extract_from_photo"] == 0
    assert calls["extract_from_photos"]["caption"] == "Here were my stats from yesterday"
    assert len(calls["extract_from_photos"]["images"]) == 2
    assert "album-1" not in nutrition._PENDING_ALBUMS, "the buffer must be cleaned up once processed"

    workouts = db.get_recent_workouts(CHAT)
    assert len(workouts) == 1, f"expected exactly one logged workout, got {len(workouts)}"
    assert workouts[0]["calories_burned"] == 2732


def test_album_caption_can_arrive_on_a_later_photo_not_the_first(monkeypatch):
    """Telegram doesn't guarantee the caption lands on the first photo
    delivered -- whichever photo in the album actually carries it must
    still be used."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(nutrition, "ALBUM_DEBOUNCE_SECONDS", 0.05)
    captured = {}

    def fake_extract_from_photos(images, caption=None):
        captured["caption"] = caption
        return {"kind": "meal", "meal_type": "Lunch", "items": ["curry gyu don"],
                "calories_low": 550, "calories_high": 750, "calories_estimate": 650, "water_ml": None}

    monkeypatch.setattr(bot.ai, "extract_from_photos", fake_extract_from_photos)

    context = FakeContext()
    update1 = FakeUpdate(CHAT, photo=[_FakePhotoSize()])  # no caption on the first photo
    update1.message.media_group_id = "album-2"
    update2 = FakeUpdate(CHAT, caption="lunch was a curry gyu don", photo=[_FakePhotoSize()])
    update2.message.media_group_id = "album-2"

    async def _simulate_album():
        await bot.handle_photo(update1, context)
        await bot.handle_photo(update2, context)
        await nutrition._PENDING_ALBUMS["album-2"]["task"]

    _run(_simulate_album())
    assert captured["caption"] == "lunch was a curry gyu don"


def test_second_photo_in_album_supersedes_the_first_pending_task(monkeypatch):
    """The first photo's debounce task must be cancelled (not left to also
    fire independently) once a second photo in the same album arrives --
    otherwise the album would still be processed twice."""
    db.get_or_create_user(CHAT)
    monkeypatch.setattr(nutrition, "ALBUM_DEBOUNCE_SECONDS", 0.05)
    monkeypatch.setattr(bot.ai, "extract_from_photos", lambda images, caption=None: {
        "kind": "meal", "meal_type": "Snack", "items": ["mango"],
        "calories_low": 90, "calories_high": 120, "calories_estimate": 105, "water_ml": None,
    })

    context = FakeContext()
    update1 = FakeUpdate(CHAT, photo=[_FakePhotoSize()])
    update1.message.media_group_id = "album-3"
    update2 = FakeUpdate(CHAT, photo=[_FakePhotoSize()])
    update2.message.media_group_id = "album-3"

    async def _simulate_album():
        await bot.handle_photo(update1, context)
        first_task = nutrition._PENDING_ALBUMS["album-3"]["task"]
        await bot.handle_photo(update2, context)
        assert first_task.cancelled() or first_task.cancelling(), "the first photo's task must be cancelled"
        await nutrition._PENDING_ALBUMS["album-3"]["task"]

    _run(_simulate_album())
    assert len(db.get_recent_meals(CHAT)) == 1
