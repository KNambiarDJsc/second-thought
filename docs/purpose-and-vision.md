# Purpose and vision

## The purpose, in one sentence

Second Thought exists to answer two questions no one else in the System One ecosystem answers
together: **is this model's confidence actually trustworthy, and which of its decisions should a
human spend time on** — then to turn the answer to the second question into a dataset that makes
the model better, and to keep re-checking the first question after you ship.

Everything else in this repo is in service of that sentence. If a feature doesn't serve one of
those two questions or the loop between them, it's out of scope by design (see
[`wedge-and-mvp-spec.md`](research/wedge-and-mvp-spec.md) for the list of things deliberately not
built).

## The precise problem

System One models (TypeSafe's Jev, the open-weight Laya, and whatever comes next) are built
around one specific promise: they return a typed answer *and* a probability, and the intended
usage pattern — documented by TypeSafe itself — is client-side thresholding: act automatically
above some confidence, route to a human below it.

That promise has exactly two silent failure modes, and this project exists because nothing else
addresses both:

1. **The confidence number can be wrong without anyone noticing.** A model can report "90%
   confident" and be right 60% of the time in practice. Nothing about the API stops this; the only
   way to know is to check — and almost nobody does, because it requires storing full probability
   distributions (not just a top-1 label) and computing calibration metrics most teams have never
   had a reason to build.
2. **Human review time is scarce and usually spent badly.** If a team has budget to manually check
   100 decisions out of 10,000, checking 100 at random is what almost everyone actually does,
   because there's no infrastructure that makes "check the right 100" easier than "check a random
   100." We measured the gap this leaves on the table: **1.8× more accuracy improvement per
   correction** from picking by uncertainty instead of at random (see the README's Proof section).

## What it concretely does

```
capture → store → select (most uncertain first) → review (CLI or web) → export → evaluate → watch for drift
```

- **Capture**: normalizes any typed-decision response (`choice`/`score`/`noul` + probabilities)
  into one schema, regardless of which model produced it.
- **Calibration**: Brier score, log loss, and ECE computed from the actual stored distributions —
  not the vendor's marketing claim.
- **Selection**: ranks decisions by uncertainty so a fixed review budget catches more real errors.
- **Correction**: a CLI or local web dashboard for a non-engineer to accept/correct/abstain/flag.
- **Export**: turns corrections into a leakage-safe, time-split, fine-tune-ready dataset — with a
  hard, code-level rule that Jev's output can never be part of it (TypeSafe's license forbids
  training on it; this project enforces that instead of just documenting it).
- **Drift**: re-runs the calibration check over time and flags when it degrades, instead of
  answering the calibration question once and going quiet.

## Where this is actually used

Concretely, this is for a team that has **already deployed** a typed-decision model in front of
real decisions and needs to answer "is this safe to keep automating, and how do we make it
better" — not a team still choosing which model to use. Examples of the shape:

- **Support/ops triage.** A ticket router built on Laya or a custom classifier that auto-routes
  above a confidence threshold and queues the rest for a support lead — who reviews through the
  dashboard, not a terminal.
- **Content moderation / trust & safety.** A `noul` (yes/no) classifier flagging content for
  removal or review, where calibration drift directly changes how much genuinely bad content slips
  through at a fixed threshold.
- **Invoice / document processing.** `choice` or `score` decisions (approve / flag / escalate) where
  the cost of a wrong high-confidence decision is real money, and ECE tracked over time is the
  early warning that a threshold set six months ago no longer holds.
- **Teams building on Jev specifically** who want calibration visibility into their own usage
  without ever risking a license violation — capture and dashboard work identically; export simply
  won't let Jev decisions out as training data.
- **Anyone fine-tuning Laya (or another open model)** who wants the correction dataset that feeds
  the next checkpoint to be the *right* 200 examples, not an arbitrary 200 — and wants to measure,
  not assume, that the new checkpoint is actually better (`evaluate()` on the same held-out set,
  before and after).

It is explicitly **not** for a team still evaluating which System One model to adopt, and not a
replacement for a full MLOps platform — see [`wedge-and-mvp-spec.md`](research/wedge-and-mvp-spec.md)
for the boundary.

## Where this goes as System One models mature — the bet, stated plainly

This category is one week old as of this project's first commit. Everything below is a directional
bet, not a proven outcome — stated separately from the proven facts above on purpose.

- **More models will speak the same shape.** `choice`/`score`/`noul` plus a probability isn't
  TypeSafe's private idea — Laya converged on the same three types independently, and OpenJev
  reverse-engineered Jev's wire format because no one had published it. `second_thought.protocol`
  publishes that shape as JSON Schema plus a conformance checker specifically so the *next* model
  in this category doesn't require a new adapter to be written by hand — if it emits the shape,
  this SDK already understands it.
- **The correction loop is the flywheel, and flywheels compound.** The same pattern that made
  spam filters, fraud detection, and recommendation systems improve for two decades — capture real
  errors, prioritize the ones worth fixing, retrain, measure, repeat — doesn't yet have
  purpose-built infrastructure for *typed probabilistic* decisions specifically. If System One
  models become a durable way to build automation (not a one-week novelty), whoever owns this loop
  for the category owns a real, recurring point of leverage: every deployment accumulates
  corrections, and corrections compound into better models faster than starting over each time.
- **Calibration monitoring becomes a compliance surface, not just an engineering nicety.** As more
  of these models make consequential automated decisions (financial, moderation, security), "how
  do you know your confidence threshold is still safe" stops being an internal engineering
  question and starts being the question an auditor, a regulator, or a customer's security review
  asks. A project that already answers that question with a number (ECE, tracked over time, with
  an alert) is positioned to be the thing people point to — this project's own live-Laya finding
  (87.5% accuracy, ECE 0.62) is a preview of exactly that kind of finding, not a one-off curiosity.
- **The Jev boundary is a bet that licensing discipline matters more over time, not less.** Most
  tooling in this space (per `docs/research/decision-control-plane.md`'s ecosystem survey) doesn't
  enforce vendor license terms in code at all. As more companies adopt these models under
  commercial terms, "our tooling can't accidentally violate your contract" becomes a real
  differentiator, not just a compliance checkbox — enforcing it now, while it's cheap and the
  category is small, is cheaper than retrofitting it later.

## What would have to be true for this bet to pay off

Stated honestly, because a vision section that can't be falsified isn't worth writing:

- System One models need to remain a real, separately-useful category (not get absorbed back into
  general LLM tool-calling) for a dedicated typed-decision tool to matter.
- The calibration/selection mechanism needs to keep proving out at real production scale — the
  live Laya run (n=24) and the toy experiment are real evidence, not yet a production-scale proof.
- The unresolved large-batch active-learning regression (documented, not hidden, in
  `wedge-and-mvp-spec.md`) needs an eventual real fix, or teams with large review batches won't get
  the headline benefit.
- Enough of the wider ecosystem (per `decision-control-plane.md`) needs to converge on a shared
  shape rather than fragmenting into incompatible formats, for the "any System One model" promise
  to keep being true rather than becoming "any System One model we've specifically added an
  adapter for."

If those hold, the plausible end state is a small, boring, load-bearing piece of infrastructure —
the layer between "a typed-decision model is running in production" and "someone can trust it,
improve it, and prove both" — rather than a platform that tries to be everything at once.
