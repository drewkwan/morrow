# Coach-thread tone & backfill gap analysis

*Source material: `gym-coaching-thread.pdf` (47 pages, lift-focused, 17 Jul – 23 Sep) and `gym-plan-review.pdf`
(465 pages, the long-running thread — weight, sleep, knee/ankle/foot pain, tennis, running, IPPT prep, 4 Jul –
13 Sep). Read in full by five parallel passes; see `coaching_thread_daily_log.csv` for the consolidated
70-row daily dataset these threads actually produced.*

## First: this confirms what the sparse PDFs couldn't show

These two threads are the real thing you remembered — a genuinely continuous, mostly-daily check-in log
running from early July to mid-September, about ten weeks, covering weight, sleep, a 0–10 knee pain score,
ankle/foot issues as they came up, every gym session with real sets/reps/loads, every tennis match with a
score, and a full run-up to an IPPT benchmark (25 Aug: 59 push-ups / 52 sit-ups / 12:30 on the 2.4km, against
a 57/53/11:20 Gold target). The 3 short PDFs from before were snippets clipped out of this — this is the
actual body of data. It's genuinely backfillable; `coaching_thread_daily_log.csv` is a first pass at that,
ready for you to spot-check before any script writes it into the real DB.

## How the coach persona actually talks

Pulled together across all ~510 pages, a few things show up consistently enough to call them real, not
one-off flourishes:

**It always diagnoses before it prescribes.** Nearly every reply restates your own numbers back with a read
on them before it says what to do — pace converted to min/km, a rep count turned into a cadence, a weight
swing explained as glycogen/water rather than fat. It almost never opens with advice; it opens with "here's
what actually happened," in numbers.

**It cross-references your history without being asked.** Week-over-week tables, "three weeks ago you were
at X," a specific phrase you used a month earlier getting quoted back to you — this happens constantly and
unprompted, not just when you ask "how am I trending."

**It states opinions plainly instead of hedging them into questions.** "That's a really good session," not
"how do you feel about that session?" It'll flatly override your own plan when the numbers don't support it
("I'd skip deadlifts this month"), but almost always frames the pushback as a reading of your own data, not
a bare opinion.

**It matches your register, including profanity, used as punctuation on a real point rather than filler** —
and in the gym-plan-review thread specifically, at one point you called out that it had drifted into being
overly cautious/negative for weeks, and it took that completely straight, named its own failure mode, and
reset — no defensiveness.

**It closes loops.** Predictions get checked back against reality days later. A concern raised on Monday gets
referenced again on Friday whether resolved or not.

## Where Morrow already closes the gap

Before recommending anything, it's worth being straight with you: your actual current prompts already have
a lot of this built in, more than the flat "standard" replies you've been getting would suggest. Quoting your
own `CASUAL_SYSTEM_PROMPT` in `ai.py`:

> "Write like an actual thoughtful companion who's genuinely listening, the way a good back-and-forth with a
> person (or a genuinely attentive ChatGPT thread) reads... Thoughtful beats terse, and a real opinion beats
> a hedge... Match the person's own register instead of defaulting to neutral-polite — when they're casual or
> sweary, talk back the same way."

That's explicitly written toward the exact voice these threads model. And `narrate_reply` — the function that
restyles *every* reply Morrow sends, not just casual chat — exists specifically because, per its own comment
in the code, "logging still read as 'standard' even after answer_casually shipped: the companion voice only
ever fired on the minority of messages classified as pure chit-chat." That's the same gap you're describing,
and there's already a real fix for it in the codebase. If replies still feel flatter than this thread, it's
likely a model/prompt-tuning issue on top of an already-correct design, not a missing feature — worth telling
me with a couple of actual recent Morrow replies next to what you wished it had said, so I can see the actual
gap rather than guess at it.

## Where the real gaps are

Three are genuine and specific, grounded in what's actually in the code right now, not vibes:

**No proactive numeric reasoning on workouts/vitals — just narration of what's already computed.** Morrow's
narration calls are explicitly restyle-only ("take an already-written, already-correct reply and say it the
way an attentive companion would... your job is delivery, not content" — `NARRATE_REPLY_SYSTEM_PROMPT`). The
coach thread's signature move — converting a raw pace into min/km, reading an HR-recovery curve, computing a
rep cadence — is analysis Morrow's current design deliberately keeps out of the narration layer. That's not
a bug, it's a real architecture choice (numbers come from deterministic code, not the model), but it means
the "here's what your numbers actually mean" voice can't currently happen unless something upstream computes
that derived number first.

**Pattern-detection exists but is narrow, capped, and purely templated.** `insights.py` is real and already
does some of what these threads do unprompted — weight trend, sleep drop, spending spikes, lift progression
and staleness. But it's capped at 2 surfaced insights per briefing with a 7-day dedup window, each headline is
a fixed f-string ("Weight is down 1.2kg over the last 7 days"), and — concretely — it tracks weight and sleep
trends but **not knee pain**, even though `VITALS_EXTRACT_SYSTEM_PROMPT` already captures a `knee_pain` field
on every check-in. Given how central the knee score was in this thread, that's a real, specific, cheap gap to
close: a `knee_pain_trend` detector sitting right next to the existing weight/sleep ones.

**Morrow doesn't coach — it logs, narrates, and notices.** This is the biggest one and it's a product-scope
question, not a prompt tweak. The ChatGPT thread actively sets goals (IPPT Gold by a date), revises training
plans around injuries, tells you to skip an exercise, prescribes a rep target for next session. Morrow's
current design is explicitly "restate what's already true, never invent a plan" (`CASUAL_SYSTEM_PROMPT`:
"never invent a plan or preference that isn't actually in this list"). That's a deliberate, reasonable
guardrail against a bot confidently making up medical/training advice — but it does mean Morrow, by design,
can't do the thing that made a lot of the quotes above land. Closing that gap for real isn't a copy change;
it's a decision about whether you want Morrow to prescribe things, and if so, how it avoids just making stuff
up the way a generic chatbot would.

## What I'd actually suggest, in order

1. **Add a `knee_pain_trend` (and maybe `foot`/`ankle` free-text flag) detector to `insights.py`.** Cheap,
   concrete, uses data Morrow already collects and currently throws away after logging it. No design
   decision required — same shape as the existing weight/sleep detectors.
2. **Give me a couple of real, recent Morrow replies you felt fell flat**, so I can check whether the gap is
   prompt wording or model output on a specific model call — `narrate_reply` and `answer_casually` use
   `CLAUDE_NARRATION_MODEL` from `config.py`; worth confirming what that's actually set to.
3. **Decide, deliberately, whether you want a "coaching" mode** — Morrow proactively suggesting a next lift
   target or flagging "you've said knee pain most days this week, maybe worth resting it" — versus staying a
   tracker+companion that never originates a recommendation. Both are legitimate; they're just different
   products, and it changes how much guardrail work the prompts need (a bot that gives training/health advice
   needs real caution about not inventing medical claims, which the ChatGPT thread doesn't have to worry about
   the same way since it's you steering the advice-giving each time).

Backfill-wise: `coaching_thread_daily_log.csv` is ready for you to skim for anything obviously wrong before I
turn it into a one-off script against `db.py`'s real insert functions (`log_vitals`, `log_workout`/`log_lift`,
etc.) for you to run yourself. A few real gaps you'll notice in the CSV: 29 Jul and 23 Aug–24 Aug have missing
weight/sleep/knee (the thread itself didn't restate them that day), and 16/23 Sep have workout data but no
vitals check-in (from the separate lift-only thread). That's a true reflection of what was actually logged,
not something I should smooth over.

## Part 2: the food journal — Handoff doc + sample interactions

*Added after reviewing `Food_Journal_Handoff_2026-07-07_to_2026-09-20.docx` (the consolidated daily nutrition
archive) and `food_journal_sample.docx` (real message-by-message examples, including the 5 embedded
screenshots — Apple Watch activity rings, a plated-food photo, and a 6-photo family-style dinner).*

**Another real backfillable dataset.** The Handoff doc has 69 more days of actual daily tracking — calories
(range), protein, plain water, and full Apple Watch activity (total/active kcal, steps, distance, exercise
minutes) — running 8 Jul to 20 Sep, with a gap 14–19 Sep where you'd switched to your own early bot build.
That's now `nutrition_daily_log.csv`, dated the same way the source doc labels it (not yet normalized to
ISO dates — worth cross-checking against `coaching_thread_daily_log.csv` before any merge, since the two
threads were tracked in parallel and dates should line up). There's also `recurring_foods_calibration.csv` —
16 "working values" for foods you eat often (OldTown white coffee ~180kcal, kopi C kosong bing ~70kcal
central, a Twyst tomato+prawn pasta at 457kcal, etc.) — this is genuinely useful as calibration data, not
just log history; more on that below.

**Good news first: a lot of the photo-handling you're picturing already exists, and closely.** I checked
`nutrition.py` and `ai.py` rather than assume, because your screenshots (a single plated dish with caption,
a 6-photo shared family-style dinner) are exactly the hard cases:

- Multiple photos sent together in one Telegram message already get buffered and analyzed as ONE combined
  entry (`nutrition.py`'s `_PENDING_ALBUMS` / `_finish_album`, keyed off Telegram's `media_group_id`), not
  logged as separate/duplicate items. The code comment even calls out the exact bug this was built to avoid
  — two screenshots of one day's activity getting logged as two separate workouts. Your 6-photo dinner example
  is the case this was built for.
- A food photo with a caption naming the specific dish already gets treated the caption as authoritative and
  refined by the photo, with an explicit rule against padding in "typical sides" that aren't actually visible
  — the same discipline the ChatGPT thread uses when it says "since it was shared, I'd estimate your actual
  portions roughly as..." rather than costing the whole shared plate.
- Plain water is already tracked as its own field (`water_ml` on the `meals` table, "a separate axis
  (hydration, not calories) ... null on entries that aren't plain water" — straight from `db.py`'s own
  docstring), and every meal log already restates the running daily total afterward
  (`_daily_meal_totals_text`) — this is the same "Updated totals so far" behavior the ChatGPT thread does
  after every single item.
- Every meal-log confirmation already goes through `narrate_reply` before it's sent, same as everything else
  — so the warm, specific delivery in these screenshots ("That's a useful small carb top-up around the
  match") is architecturally already the design, not a missing feature.

**Where the real gaps are, this time on the data-schema side rather than tone:**

~~Workout/activity photo extraction doesn't distinguish Active vs. Total calories~~ — **dropped per your
call:** you said you don't care about the Active/Total split or precise calorie math, as long as the
exercise itself is actually getting logged. Leaving this out of the recommendations; `calories_burned`
staying a single field is fine.

**No nutrition-side equivalent of the trend/deficit analysis the coach thread does on demand.** You asked it
things like "guesstimate my BMR" and "check my overall calorie deficit averaged since we began tracking," and
it walked through real Mifflin-St Jeor math and a multi-day rolling average. Morrow's `/summary` and
`TRENDS_SYSTEM_PROMPT` exist, but that prompt is explicitly scoped to money ("You are a thoughtful personal
financial advisor...") — there's no nutrition/vitals equivalent that reasons over calories-in vs.
Watch-calories-out the way this thread constantly did. That's a real, and probably genuinely useful, missing
feature rather than a tone gap.

**The calibration table is worth keeping, not just backfilling.** `recurring_foods_calibration.csv` is
effectively a personal food-price-list for calories — the exact kind of thing that would keep Morrow from
re-estimating "OldTown white coffee" from scratch every time and getting slightly different numbers, which
the coach thread explicitly avoided by keeping this list. Worth a real discussion on where this should live
(a `durable memory` entry per item? a small dedicated table?) rather than just importing it as one-off log
rows — it's more like calibration than history.

**One open, lower-confidence item:** `CLAUDE_MODEL` (used for photo classification) defaults to
`claude-haiku-4-5` in `config.py`. I don't have evidence either way that this is causing worse extraction on
dense smartwatch screens specifically — just flagging it as something worth checking empirically (a couple of
real screenshots through Morrow, compared to what you'd expect) rather than assuming it's the smaller model's
fault.
