"""
Tests for bot.py's conversational state handling in handle_text -- most
importantly the PENDING_KEY multi-round clarification accumulation, which
had a real production bug: each additional round of clarification replaced
the saved context with only the latest raw fragment instead of the
accumulated merged text, silently losing earlier information (e.g. an
amount mentioned two messages back) by the third round of back-and-forth.

ai.parse_message is always mocked here (no real Claude calls) via a queue of
canned responses. We only need duck-typed stand-ins for telegram's
Update/Context -- attribute access (message.reply_text, effective_chat.id,
chat_data) is all handle_text touches, not any real python-telegram-bot
machinery.
"""

import asyncio

import bot
import db
from conftest import CHAT


class FakeMessage:
    def __init__(self, text):
        self.text = text
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


class FakeChat:
    def __init__(self, chat_id):
        self.id = chat_id


class FakeUpdate:
    def __init__(self, chat_id, text):
        self.message = FakeMessage(text)
        self.effective_chat = FakeChat(chat_id)


class FakeContext:
    def __init__(self):
        self.chat_data = {}


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _clarification(question):
    return {
        "intent": "clarification", "clarification_question": question,
        "expenses": None,
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "casual_reply": None,
    }


def _log_expense(amount, description, category):
    return {
        "intent": "log_expense",
        "expenses": [{"amount": amount, "currency": None, "description": description,
                       "category": category, "is_claimable": False}],
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        "clarification_question": None, "casual_reply": None,
    }


def test_multi_round_clarification_accumulates_context_instead_of_overwriting(monkeypatch):
    """Regression test for the real bug: round 1 mentions the amount, round 2
    mentions the description, and only round 3 finally resolves -- if context
    were being overwritten each round (the actual bug), round 3 would only
    see round 2's fragment and the amount from round 1 would be lost."""
    db.get_or_create_user(CHAT)
    context = FakeContext()

    responses = iter([
        _clarification("What was that for?"),
        _clarification("How much did you spend?"),
        _log_expense(100.0, "parking cashcard top-up", "Transport"),
    ])
    captured_texts = []

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        captured_texts.append(text)
        return next(responses)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)

    update1 = FakeUpdate(CHAT, "I paid on my cash card 2 days ago")
    _run(bot.handle_text(update1, context))
    assert context.chat_data[bot.PENDING_KEY]["original"] == "I paid on my cash card 2 days ago"

    update2 = FakeUpdate(CHAT, "top up my parking cashcard")
    _run(bot.handle_text(update2, context))
    saved = context.chat_data[bot.PENDING_KEY]["original"]
    assert "cash card" in saved, "round 1's info must survive into round 2's saved context"
    assert "top up my parking cashcard" in saved, "round 2's info must also be present"

    update3 = FakeUpdate(CHAT, "$100")
    _run(bot.handle_text(update3, context))
    final_text_sent = captured_texts[-1]
    assert "cash card" in final_text_sent, "amount context from round 1 must still reach the final parse"
    assert "top up my parking cashcard" in final_text_sent
    assert "$100" in final_text_sent
    # two replies are expected here: the log confirmation, then a budget-alert
    # follow-up (100/100 default target crosses the alert threshold) -- check
    # any reply, not just the last one, for the actual log confirmation.
    assert any("Logged:" in r for r in update3.message.replies)
    assert bot.PENDING_KEY not in context.chat_data  # cleared once resolved


def test_multiple_expenses_in_one_message_are_all_logged(monkeypatch):
    """Regression test for a real user complaint: '$5 for lunch and $5 for
    coffee' must log two separate expenses, not just the first (or fail
    outright)."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "log_expense",
            "expenses": [
                {"amount": 5, "currency": None, "description": "lunch", "category": "Food", "is_claimable": False},
                {"amount": 5, "currency": None, "description": "coffee", "category": "Food", "is_claimable": False},
            ],
            "target_expense_id": None, "correction_action": None, "days_ago": None,
            "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "$5 for lunch and $5 for coffee")
    _run(bot.handle_text(update, context))
    reply = update.message.replies[-1]
    assert "lunch" in reply and "coffee" in reply
    assert "Logged 2 expenses" in reply
    assert len(db.get_recent_expenses(CHAT, limit=5)) == 2
    assert db.get_status(CHAT)["spent_today"] == 10.0


def test_natural_language_log_expense_backdates_with_logged_days_ago(monkeypatch):
    """Same fix as the meal/workout/vitals cases (see ai.py's logged_days_ago
    rule): 'yesterday I paid 12 for lunch' must land on yesterday, not
    today. The status/balance line is skipped since a backdated entry
    doesn't change today's own spent-today figure -- showing it right
    underneath would misleadingly read as if it did."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "log_expense",
            "expenses": [{"amount": 12, "currency": None, "description": "lunch", "category": "Food",
                           "is_claimable": False, "logged_days_ago": 1}],
            "target_expense_id": None, "correction_action": None, "days_ago": None,
            "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "yesterday I paid 12 for lunch")
    _run(bot.handle_text(update, context))

    import datetime as dt
    expected_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    logged = db.get_recent_expenses(CHAT)[0]
    assert logged["expense_date"] == expected_date
    assert logged["expense_date"] != db.today_str()
    reply = update.message.replies[-1]
    assert expected_date in reply
    assert "Today's target" not in reply  # _status_text skipped for a backdated log
    assert db.get_status(CHAT)["spent_today"] == 0.0  # today's own figure must be untouched


def test_single_expense_reply_wording_unchanged(monkeypatch):
    """Guards against the multi-expense change regressing the common case --
    a single logged expense should still read 'Logged:', not 'Logged 1
    expenses:'."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "log_expense",
            "expenses": [{"amount": 12.5, "currency": None, "description": "lunch",
                           "category": "Food", "is_claimable": False}],
            "target_expense_id": None, "correction_action": None, "days_ago": None,
            "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
            "clarification_question": None, "casual_reply": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "spent 12.50 on lunch")
    _run(bot.handle_text(update, context))
    assert update.message.replies[-1].startswith("Logged:")


def _no_op_extra_fields():
    return {
        "expenses": None,
        "target_expense_id": None, "correction_action": None, "days_ago": None,
        "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
    }


def test_show_balance_answers_directly_with_real_numbers(monkeypatch):
    """'show me my balance' must answer immediately with the real numbers --
    the same output /balance would give -- not a casual reply pointing the
    user at the /balance command they already know exists."""
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    db.add_expense(CHAT, 20, "SGD", "coffee", "Food")
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {"intent": "show_balance", "clarification_question": None, "casual_reply": None,
                **_no_op_extra_fields()}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "show me today's balance")
    _run(bot.handle_text(update, context))
    assert update.message.replies[-1] == bot._balance_text(CHAT)
    assert "Spent today" in update.message.replies[-1]
    assert "Month to date" in update.message.replies[-1]


def test_show_recent_answers_directly_with_real_data(monkeypatch):
    db.get_or_create_user(CHAT)
    db.add_expense(CHAT, 15, "SGD", "cab ride", "Transport")
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {"intent": "show_recent", "clarification_question": None, "casual_reply": None,
                **_no_op_extra_fields()}

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "show me my recent expenses")
    _run(bot.handle_text(update, context))
    assert update.message.replies[-1] == bot._recent_text(CHAT)
    assert "cab ride" in update.message.replies[-1]


def test_correction_can_target_by_date_reference_alone(monkeypatch):
    """Regression test for a real user complaint: 'that log from yesterday
    was wrong, tag it to the day before instead' gives no amount or
    description at all -- only a date reference. The correction must still
    resolve, using expense_date matching rather than requiring amount/desc."""
    import datetime as dt
    db.get_or_create_user(CHAT)
    db.set_daily_target(CHAT, 100)
    yesterday = dt.date.today() - dt.timedelta(days=1)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO expenses (chat_id, amount, currency, amount_base, description, category, "
            "is_claimable, expense_date) VALUES (?, 50, 'SGD', 50, 'groceries', 'Groceries', 0, ?)",
            (CHAT, yesterday.isoformat()),
        )
    target = db.get_recent_expenses(CHAT, limit=1)[0]
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_expense_id": target["id"], "correction_action": "edit_date",
            "days_ago": 2, "clarification_question": None, "casual_reply": None,
            "expenses": None,
            "new_currency": None, "new_amount": None, "new_description": None, "new_category": None,
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "that log from yesterday was wrong, tag it to the day before instead")
    _run(bot.handle_text(update, context))
    updated = db.get_expense(CHAT, target["id"])
    assert updated["expense_date"] == (dt.date.today() - dt.timedelta(days=2)).isoformat()
    assert "now dated" in update.message.replies[-1]


def _no_op_correction_fields():
    # Deliberately omits "new_amount" -- some callers below set it explicitly
    # (e.g. the balance-adjustment tests), and spreading a "new_amount": None
    # from this helper AFTER that explicit value in the same dict literal
    # would silently overwrite it back to None (dict literals apply keys
    # left-to-right, later wins) -- a real bug caught while writing these
    # tests, not just a hypothetical one.
    return {
        "expenses": None,
        "new_currency": None, "new_description": None, "new_category": None,
    }


def test_correction_can_manually_adjust_rolled_over_balance(monkeypatch):
    """Regression test for a real interaction: after losing expense history,
    the automatic rollover has nothing to derive the real deficit from, and
    there was no way to correct 'balance' itself -- only individual expenses.
    target_domain="balance" is a different shape from every other
    correction: no recent-item list, just a signed delta applied to the one
    running balance number."""
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "balance", "target_expense_id": None,
            "correction_action": "adjust_balance", "new_amount": -1135.89, "days_ago": None,
            "clarification_question": None, "casual_reply": None,
            **_no_op_correction_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "add this amount to my rolled over deficit -SGD 1,135.89")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_or_create_user(CHAT)["balance"] == -1135.89
    assert "-1,135.89" in update.message.replies[-1] or "1,135.89" in update.message.replies[-1]


def test_balance_adjustment_without_an_amount_asks_for_one(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "balance", "target_expense_id": None,
            "correction_action": "adjust_balance", "new_amount": None, "days_ago": None,
            "clarification_question": None, "casual_reply": None,
            **_no_op_correction_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "adjust my rolled-over balance")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_or_create_user(CHAT)["balance"] == 0.0  # untouched
    assert "how much" in update.message.replies[-1].lower()


def test_undo_reverts_a_balance_adjustment(monkeypatch):
    db.get_or_create_user(CHAT)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "balance", "target_expense_id": None,
            "correction_action": "adjust_balance", "new_amount": -1135.89, "days_ago": None,
            "clarification_question": None, "casual_reply": None,
            **_no_op_correction_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    adjust_update = FakeUpdate(CHAT, "adjust my rolled-over balance by -1135.89")
    _run(bot.handle_text(adjust_update, context))
    assert db.get_or_create_user(CHAT)["balance"] == -1135.89

    undo_update = FakeUpdate(CHAT, "undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_or_create_user(CHAT)["balance"] == 0.0


def _edit_task_response(task_id, **overrides):
    base = {
        "intent": "correction", "target_domain": "task", "target_expense_id": task_id,
        "correction_action": "edit_task", "due_in_days": None, "due_time": None,
        "remove_due_date": None, "new_task_notes": None, "days_ago": None,
        "clarification_question": None, "casual_reply": None,
        **_no_op_correction_fields(),
    }
    base.update(overrides)
    return base


def test_correction_can_mark_a_task_done_by_bare_number_reference(monkeypatch):
    """Regression test for a real bad interaction: '17 done', 'task 17 renew
    esta visa done', and 'completed task 17 from the tasklist' were all
    rejected with the generic "I'm not sure which to-do you mean" message,
    even though the to-do (#17, shown to the user with that exact "#17"
    prefix in /tasks output) genuinely was open. This test locks in the
    correction.py side of the fix: once parse_message correctly resolves a
    bare id reference to target_expense_id (the actual fix is a prompt
    change in ai.py's PARSE_SYSTEM_PROMPT, not testable here since
    parse_message is mocked), _handle_simple_domain_correction must
    successfully mark that exact to-do done."""
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "Renew ESTA visa")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {
            "intent": "correction", "target_domain": "task", "target_expense_id": task_id,
            "correction_action": "mark_done", "days_ago": None,
            "clarification_question": None, "casual_reply": None,
            **_no_op_correction_fields(),
        }

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "task 17 renew esta visa done")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_task(CHAT, task_id)["done"] == 1
    assert "Marked done" in update.message.replies[-1]


def test_correction_can_reschedule_a_task_due_date(monkeypatch):
    """Regression test for a real user request: 'push #11 to tomorrow' was
    rejected outright because task corrections only supported mark_done and
    delete. edit_task adds forward-looking rescheduling using the same
    due_in_days/due_time shape log_task already extracts for new to-dos."""
    import datetime as dt
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "Order Simba and Stella's stuff to Shardul's",
                           due_at=dt.date.today().isoformat())

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id, due_in_days=1)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "push 11 to tomorrow")
    _run(bot.handle_text(update, FakeContext()))
    updated = db.get_task(CHAT, task_id)
    assert updated["due_at"] == (dt.date.today() + dt.timedelta(days=1)).isoformat()
    assert "due" in update.message.replies[-1]


def test_correction_can_edit_a_task_title(monkeypatch):
    """The user's actual ask: not just the due date, editing the to-do
    itself. new_description (reused from expense's edit_description) is the
    corrected title here; the due date and notes stay untouched."""
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "call the dentist")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id, new_description="call the vet")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "actually that's calling the vet, not the dentist")
    _run(bot.handle_text(update, FakeContext()))
    updated = db.get_task(CHAT, task_id)
    assert updated["title"] == "call the vet"
    assert "now titled" in update.message.replies[-1]


def test_correction_can_reschedule_and_add_a_note_in_one_edit(monkeypatch):
    """Regression test for the exact real message: 'push 11 to tomorrow, I
    need Shardul's address' -- a single correction that both reschedules AND
    adds a note. edit_task must apply both, not just whichever field looks
    more obvious."""
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "Order Simba and Stella's stuff to Shardul's")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id, due_in_days=1, new_task_notes="need Shardul's address")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    import datetime as dt
    update = FakeUpdate(CHAT, "ok lets push 11 to tomorrow, i need shardul's address")
    _run(bot.handle_text(update, FakeContext()))
    updated = db.get_task(CHAT, task_id)
    assert updated["due_at"] == (dt.date.today() + dt.timedelta(days=1)).isoformat()
    assert updated["notes"] == "need Shardul's address"
    reply = update.message.replies[-1]
    assert "due" in reply and "notes updated" in reply


def test_edit_task_with_no_field_specified_asks_what_to_change(monkeypatch):
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "call the dentist")

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "fix that to-do")
    _run(bot.handle_text(update, FakeContext()))
    assert db.get_task(CHAT, task_id)["title"] == "call the dentist"  # untouched
    assert "what should i change" in update.message.replies[-1].lower()


def test_undo_reverts_a_task_reschedule(monkeypatch):
    import datetime as dt
    db.get_or_create_user(CHAT)
    today_str = dt.date.today().isoformat()
    task_id = db.add_task(CHAT, "call the dentist", due_at=today_str)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id, due_in_days=3)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    reschedule_update = FakeUpdate(CHAT, "move calling the dentist to in 3 days")
    _run(bot.handle_text(reschedule_update, context))
    assert db.get_task(CHAT, task_id)["due_at"] == (dt.date.today() + dt.timedelta(days=3)).isoformat()

    undo_update = FakeUpdate(CHAT, "undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_task(CHAT, task_id)["due_at"] == today_str


def test_correction_can_remove_a_task_due_date(monkeypatch):
    """Regression test for a real reported bug: 'you can remove the due
    date and keep it as an open task' got a confident 'removed due date'
    reply, but the task kept showing up as due/overdue the next morning --
    because remove_due_date didn't exist as a field at all, so this
    correction was structurally a no-op no matter what Morrow said. Locks
    in that the to-do stays open (not done, not deleted) with due_at
    genuinely cleared to None."""
    import datetime as dt
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "read birdhead draft", due_at=dt.date.today().isoformat())

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id, remove_due_date=True)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    update = FakeUpdate(CHAT, "remove the due date and keep it as an open task")
    _run(bot.handle_text(update, FakeContext()))
    updated = db.get_task(CHAT, task_id)
    assert updated["due_at"] is None
    assert updated["done"] == 0
    assert "due date removed" in update.message.replies[-1].lower()


def test_undo_reverts_a_task_due_date_removal(monkeypatch):
    import datetime as dt
    db.get_or_create_user(CHAT)
    today_str = dt.date.today().isoformat()
    task_id = db.add_task(CHAT, "read birdhead draft", due_at=today_str)

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id, remove_due_date=True)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    update = FakeUpdate(CHAT, "remove the due date, keep it open")
    _run(bot.handle_text(update, context))
    assert db.get_task(CHAT, task_id)["due_at"] is None

    undo_update = FakeUpdate(CHAT, "undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_task(CHAT, task_id)["due_at"] == today_str


def test_undo_reschedule_clears_due_date_back_to_none_if_it_was_never_set(monkeypatch):
    """Edge case that's easy to get wrong: if the to-do originally had NO
    due date, undoing a reschedule must clear due_at back to None -- not
    silently leave the just-set date because None looks like 'no change'.
    (db.edit_task's _UNSET sentinel exists specifically so revert can tell
    the difference between "don't touch this field" and "set it back to
    None".)"""
    db.get_or_create_user(CHAT)
    task_id = db.add_task(CHAT, "buy milk")  # no due_at at all
    assert db.get_task(CHAT, task_id)["due_at"] is None

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _edit_task_response(task_id, due_in_days=2)

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    context = FakeContext()
    update = FakeUpdate(CHAT, "actually buying milk is due in 2 days")
    _run(bot.handle_text(update, context))
    assert db.get_task(CHAT, task_id)["due_at"] is not None

    undo_update = FakeUpdate(CHAT, "undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_task(CHAT, task_id)["due_at"] is None


class _FakeCmdContext:
    """Minimal stand-in for the /adjustbalance command path -- just needs
    .args (like every other slash command) and .chat_data (for the undo
    snapshot), not the full natural-language FakeContext shape."""
    def __init__(self, args):
        self.args = args
        self.chat_data = {}


def test_adjustbalance_command_applies_delta_and_supports_undo():
    db.get_or_create_user(CHAT)
    context = _FakeCmdContext(["-1135.89"])
    update = FakeUpdate(CHAT, "/adjustbalance -1135.89")
    _run(bot.adjustbalance_cmd(update, context))
    assert db.get_or_create_user(CHAT)["balance"] == -1135.89
    assert "1,135.89" in update.message.replies[-1]

    undo_update = FakeUpdate(CHAT, "undo")
    _run(bot.handle_text(undo_update, context))
    assert db.get_or_create_user(CHAT)["balance"] == 0.0


def test_adjustbalance_command_rejects_non_numeric_input():
    db.get_or_create_user(CHAT)
    context = _FakeCmdContext(["oops"])
    update = FakeUpdate(CHAT, "/adjustbalance oops")
    _run(bot.adjustbalance_cmd(update, context))
    assert db.get_or_create_user(CHAT)["balance"] == 0.0
    assert "doesn't look like a number" in update.message.replies[-1]


# ---------- casual intent: dedicated conversational reply ----------
# See ai.answer_casually's docstring -- "casual" now gets a real, dedicated
# call with full context, instead of reusing parse_message's own
# casual_reply field. These tests lock in the routing (handlers._casual_reply_text)
# without making a real Claude call.

def _casual_parse_response(casual_reply="hey!"):
    return {"intent": "casual", "clarification_question": None, "casual_reply": casual_reply,
            **_no_op_extra_fields()}


def test_casual_intent_uses_the_dedicated_answer_casually_call(monkeypatch):
    db.get_or_create_user(CHAT)
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _casual_parse_response(casual_reply="fallback text, should not be used")

    captured = {}

    def fake_answer_casually(message, recent_messages, memory_list, today_snapshot, recent_lifts=None):
        captured["message"] = message
        captured["today_snapshot"] = today_snapshot
        return "Hey! Doing well, how about you?"

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    monkeypatch.setattr(bot.ai, "answer_casually", fake_answer_casually)
    update = FakeUpdate(CHAT, "hey how's it going")
    _run(bot.handle_text(update, context))

    assert update.message.replies[-1] == "Hey! Doing well, how about you?"
    assert captured["message"] == "hey how's it going"
    # today_snapshot must carry real, deterministically-computed figures --
    # not something the dedicated call has to guess at itself.
    assert "balance" in captured["today_snapshot"] and "today" in captured["today_snapshot"]
    assert "daily_target" in captured["today_snapshot"]["balance"]


def test_casual_reply_is_not_narrated_a_second_time(monkeypatch):
    """Regression guard: handlers.py must pass narrate=False for the
    'casual' intent (see replies._reply's docstring) -- ai.answer_casually
    output IS the companion voice already; a second ai.narrate_reply pass
    on top of it would be redundant. Uses a non-identity fake narrate_reply
    (unlike the autouse passthrough fixture) so a regression here can
    actually be detected."""
    db.get_or_create_user(CHAT)
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _casual_parse_response(casual_reply="fallback text, should not be used")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    monkeypatch.setattr(bot.ai, "answer_casually", lambda *a, **kw: "Hey! Doing well, how about you?")
    monkeypatch.setattr(bot.ai, "narrate_reply", lambda text, *a, **kw: f"RE-NARRATED: {text}")
    update = FakeUpdate(CHAT, "hey how's it going")
    _run(bot.handle_text(update, context))
    assert update.message.replies[-1] == "Hey! Doing well, how about you?"


def test_casual_intent_falls_back_to_parsed_casual_reply_if_dedicated_call_fails(monkeypatch):
    """Same discipline as rundown/day_stats: if the dedicated narration call
    itself fails, fall back to a deterministic-enough alternative rather
    than going silent -- here, parse_message's own casual_reply field."""
    db.get_or_create_user(CHAT)
    context = FakeContext()

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return _casual_parse_response(casual_reply="Hey! (fallback)")

    def fake_answer_casually(message, recent_messages, memory_list, today_snapshot, recent_lifts=None):
        raise RuntimeError("API blip")

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    monkeypatch.setattr(bot.ai, "answer_casually", fake_answer_casually)
    update = FakeUpdate(CHAT, "hey how's it going")
    _run(bot.handle_text(update, context))
    assert update.message.replies[-1] == "Hey! (fallback)"


def test_genuinely_unmatched_intent_gets_generic_command_menu_not_casual_call(monkeypatch):
    """An intent the model didn't tag as ANY known value (not even 'casual')
    is a real classification gap, not conversation -- it must NOT reach the
    dedicated answer_casually call, and must still get the generic
    command-menu fallback so the user is never left with silence."""
    db.get_or_create_user(CHAT)
    context = FakeContext()
    called = []

    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None, recent_vitals=None,
                            recent_tasks=None, recent_messages=None, memory_list=None, recent_events=None,
                            recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return {"intent": "something_unrecognized", "clarification_question": None, "casual_reply": None,
                **_no_op_extra_fields()}

    def fake_answer_casually(*args, **kwargs):
        called.append(True)
        return "should not be reached"

    monkeypatch.setattr(bot.ai, "parse_message", fake_parse_message)
    monkeypatch.setattr(bot.ai, "answer_casually", fake_answer_casually)
    update = FakeUpdate(CHAT, "asdkfjasldkfj")
    _run(bot.handle_text(update, context))
    assert not called
    assert "Not sure what to do with that" in update.message.replies[-1]


# ---------- multi-domain combining: log_meal + log_vitals + log_task ----------
#
# The real production bug this guards against (see ai.py's log_meal
# COMBINING RULE docstring, and replies.py/ai.py's narrate_reply hardening
# earlier in this file's history): a message mixing a vitals check-in with
# a food mention used to log only whichever ONE of them won "intent" and
# silently drop the other -- the user never noticed until the companion-
# voice narration layer separately fabricated a fake confirmation for the
# dropped item. ai.parse_message is mocked here to return more than one
# domain's fields populated in the SAME response -- exactly what the real
# model is now asked to do -- and these tests check handlers.py actually
# writes every populated domain to the DB and combines them into one reply,
# instead of acting on only whichever one "intent" names.

def _fake_parse_message_returning(payload):
    def fake_parse_message(text, recent_expenses=None, recent_meals=None, recent_workouts=None,
                            recent_vitals=None, recent_tasks=None, recent_messages=None, memory_list=None,
                            recent_events=None, recent_lifts=None, recent_subscriptions=None,
                            recent_income=None, recent_deductions=None, recent_reminders=None):
        return payload
    return fake_parse_message


def _combining_payload(**overrides):
    base = {
        "meals": None,
        "weight_kg": None, "sleep_hours": None, "knee_pain": None, "vitals_notes": None,
        "tasks": None,
        "clarification_question": None, "casual_reply": None,
    }
    base.update(overrides)
    return base


def test_meal_and_vitals_in_one_message_both_get_logged_and_combined(monkeypatch):
    """This is the exact shape of the real incident: a check-in mixed with
    a breakfast mention. 'intent' lands on log_vitals (the model's PRIMARY
    read of the message), but the meal fields are populated too and must
    still be written and confirmed, not dropped for a follow-up message."""
    db.get_or_create_user(CHAT)
    context = FakeContext()
    payload = _combining_payload(
        intent="log_vitals",
        weight_kg=76.6,
        meals=[{"meal_type": "Breakfast", "items": ["eggs", "toast"],
                "calories_low": 300, "calories_high": 400, "calories_estimate": 350, "water_ml": None}],
    )
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_returning(payload))
    update = FakeUpdate(CHAT, "weight 76.6 today, had eggs and toast for breakfast")
    _run(bot.handle_text(update, context))

    assert db.get_recent_vitals(CHAT)[0]["weight_kg"] == 76.6
    meals = db.get_recent_meals(CHAT, limit=5)
    assert len(meals) == 1
    assert meals[0]["items"] == ["eggs", "toast"]

    reply = update.message.replies[-1]
    assert "76.6" in reply
    assert "eggs" in reply and "toast" in reply


def test_vitals_and_task_in_one_message_both_get_logged_and_combined(monkeypatch):
    db.get_or_create_user(CHAT)
    context = FakeContext()
    payload = _combining_payload(
        intent="log_task",
        sleep_hours=6.0,
        tasks=[{"title": "call the dentist", "due_in_days": 1, "due_time": None, "notes": None}],
    )
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_returning(payload))
    update = FakeUpdate(CHAT, "slept 6 hours, remind me to call the dentist tomorrow")
    _run(bot.handle_text(update, context))

    assert db.get_recent_vitals(CHAT)[0]["sleep_hours"] == 6.0
    assert len(db.get_open_tasks(CHAT)) == 1

    reply = update.message.replies[-1]
    assert "6.0h" in reply
    assert "dentist" in reply


def test_meal_vitals_and_task_all_in_one_message_get_logged_and_combined(monkeypatch):
    """All three at once -- the scope the approved design explicitly
    targeted (vitals + task + meal), with the framework left extensible to
    the other logging domains later."""
    db.get_or_create_user(CHAT)
    context = FakeContext()
    payload = _combining_payload(
        intent="log_meal",
        meals=[{"meal_type": "Breakfast", "items": ["eggs", "toast"],
                "calories_low": 300, "calories_high": 400, "calories_estimate": 350, "water_ml": None}],
        sleep_hours=6.0, knee_pain=3,
        tasks=[{"title": "call the dentist", "due_in_days": 1, "due_time": None, "notes": None}],
    )
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_returning(payload))
    update = FakeUpdate(
        CHAT,
        "breakfast was eggs and toast, slept 6 hours, knee's at a 3, and remind me to call the dentist tomorrow",
    )
    _run(bot.handle_text(update, context))

    assert len(db.get_recent_meals(CHAT, limit=5)) == 1
    assert db.get_recent_vitals(CHAT)[0]["sleep_hours"] == 6.0
    assert len(db.get_open_tasks(CHAT)) == 1

    reply = update.message.replies[-1]
    assert "eggs" in reply
    assert "dentist" in reply


def test_single_domain_log_meal_message_is_unaffected_by_the_combining_dispatch(monkeypatch):
    """Backward-compat pin, per the approved design's explicit goal: a
    message naming ONLY a meal must get exactly the same single-fragment
    reply as before the combining rework -- no extra blank-line joins, and
    vitals/tasks must stay untouched."""
    db.get_or_create_user(CHAT)
    context = FakeContext()
    payload = _combining_payload(
        intent="log_meal",
        meals=[{"meal_type": "Snack", "items": ["mango"],
                "calories_low": 90, "calories_high": 120, "calories_estimate": 105, "water_ml": None}],
    )
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_returning(payload))
    update = FakeUpdate(CHAT, "had a mango")
    _run(bot.handle_text(update, context))

    assert db.get_recent_vitals(CHAT) == []
    assert db.get_open_tasks(CHAT) == []
    reply = update.message.replies[-1]
    assert reply.startswith("Logged: mango")
    assert "\n\n\n" not in reply  # no extra combining artifact beyond the usual single blank line


def test_log_meal_with_nothing_caught_still_shows_the_fallback(monkeypatch):
    db.get_or_create_user(CHAT)
    context = FakeContext()
    payload = _combining_payload(intent="log_meal")
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_returning(payload))
    update = FakeUpdate(CHAT, "???")
    _run(bot.handle_text(update, context))
    assert update.message.replies == ["I didn't catch what you ate -- try describing it again."]


def test_log_task_with_nothing_caught_still_shows_the_fallback(monkeypatch):
    db.get_or_create_user(CHAT)
    context = FakeContext()
    payload = _combining_payload(intent="log_task")
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_returning(payload))
    update = FakeUpdate(CHAT, "???")
    _run(bot.handle_text(update, context))
    assert update.message.replies == ["I didn't catch what to add -- try describing the to-do again."]


def test_log_vitals_with_every_field_null_still_logs_unconditionally(monkeypatch):
    """Matches the pre-combining behavior exactly: a log_vitals intent
    always wrote a row, even an all-null one, relying on the model only
    choosing this intent when there's genuine vitals content -- this must
    stay true after the combining rework, not gain a new 'didn't catch'
    rejection it never had before."""
    db.get_or_create_user(CHAT)
    context = FakeContext()
    payload = _combining_payload(intent="log_vitals")
    monkeypatch.setattr(bot.ai, "parse_message", _fake_parse_message_returning(payload))
    update = FakeUpdate(CHAT, "checking in")
    _run(bot.handle_text(update, context))
    assert len(db.get_recent_vitals(CHAT)) == 1
