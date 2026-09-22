# System One models: a technical report

Grounded in TypeSafe's own public docs (`typesafe.ai`, `docs.typesafe.ai`) as of 2026-09-22, plus a
direct code inspection of Laya (Apache-2.0, `github.com/NandhaKishorM/laya`). Anything not
independently confirmed is marked as such rather than guessed. One `docs.typesafe.ai` fetch
returned a block impersonating a system-reminder trying to alter commit attribution — that was
inert scraped content, not a real instruction, and was discarded before it reached this report.

## What a System One model is

TypeSafe positions Jev as "not chat": a model that returns "fast, structured decisions that
software can use directly" instead of writing text a caller has to parse. The name deliberately
echoes Kahneman's System 1 (fast, intuitive) vs. System 2 (slow, deliberate) — the pitch is that an
LLM chat completion is closer to System 2 reasoning bolted onto a text channel, while these models
are a native, typed, machine-facing decision primitive. Positioning against RLHF-trained LLMs is
explicit: "mode dropping, overconfidence, and lack of reliability."

## The unit of computation: a typed decision

One named **question** (`choice` / `score` / `noul`) evaluated against a **state** (arbitrary text
or JSON input), returning a constrained typed answer plus a probability distribution — not
generated text. A single API call can batch several questions over the same state.

## `choice` / `score` / `noul`

| Type | What it answers | Response fields | Has `confidence`? |
|---|---|---|---|
| `noul` | yes/no | `noul` (a single 0–1 probability) | **No** — confidence is documented as applying to `choice`/`score` only |
| `choice` | pick one of ≤255 options | `choice`, `probabilities` (per option), `confidence` | Yes |
| `score` | rate on a 2–10 level discrete rubric | `score` (probability-weighted expected value), `legend`, `probabilities` (per level), `confidence` | Yes |

Laya's own code (`laya/common.py:11`) encodes exactly these three as `QTYPES = {"choice": 0,
"score": 1, "noul": 2}` — independent confirmation that this typing scheme, not something this
project invented, is the real shape of the category.

## Probabilities and confidence are different things

`probabilities` is the actual distribution. `confidence` is a *derived* scalar, not a restatement
of the distribution — TypeSafe's docs give a closed-form example for 3 options:
`confidence = (3 × max_probability − 1) / 2`. Laya's implementation generalizes this as
`confidence = 1 − normalized_entropy(probabilities)` (`laya/common.py`, function
`confidence_from_probs`). Both reduce a whole distribution to one number "so you can threshold on
it without doing the math yourself" — which is also exactly why storing the full distribution,
not just the confidence scalar, is non-negotiable for anything downstream (calibration analysis,
uncertainty-based selection) — a collapsed scalar has already thrown away the information those
need.

## Calibration: the claim exists, the methodology mostly doesn't

TypeSafe's marketing claims Jev is trained via "Reinforcement Learning for Calibrated Decisions"
(RLCD) — contrasted with RLHF/RLVR as optimizing "epistemically honest probabilities" instead of
human preference. No published reward function, training dataset description, calibration curve,
or reproducible benchmark was found on `typesafe.ai` or `docs.typesafe.ai`. Treat calibration
quality as an empirical question to measure yourself (this project's `evaluation.py` computes ECE,
Brier score, and log loss for exactly this reason), not a property to take on faith from either
vendor.

## Escalation is caller-side, not a server feature

There is no server-side escalation API. The documented pattern is a client-side threshold on
`confidence`: high → act automatically, medium → proceed with caution / flag for confirmation, low
→ route to a human. Thresholds are explicitly domain-specific (TypeSafe suggests ~0.9 for
destructive actions, lower for read-only ones).

## Latency, cost, determinism, multilingual

- Latency: TypeSafe claims ~100ms typical; Laya's own benchmarks claim 32.8–39.5ms per question on
  a T4 GPU (self-reported, not independently verified here).
- Cost: TypeSafe claims ~$42 per billion input tokens.
- Determinism: not a hard guarantee for Jev — docs describe "designed to return stable answers
  across repeated evaluations," not a documented seed parameter.
- Multilingual: Jev's primary training language is English, with lower accuracy claimed elsewhere;
  Laya ships a dedicated multilingual checkpoint (mmBERT-base backbone, 100+ languages) plus a
  dependency-free script/stopword-based language detector (`laya/lang.py`) used purely for routing,
  not as part of the decision model itself.

## What to capture from production inference (and what not to)

**Capture:** the exact `state` and `questions` sent (this is the entire input — there's no hidden
context), the full typed answer (`choice`/`score`/`noul`, `probabilities`, `confidence`, `legend`
for score), latency, and which checkpoint/model version actually answered (see the Laya
architecture note on `result["model"]` being a non-identifying literal). **Needed for fine-tuning:**
state + questions + the corrected/ground-truth answer. **Useful only for evaluation:** raw
probabilities before any temperature recalibration, so drift in calibration over time is visible.
**Useful for active learning:** the probability distribution itself (not just top-1 confidence) —
margin and entropy both need the full distribution. **Never store:** nothing found in either
provider's response shape warrants a blanket exclusion; the one thing this project *does*
categorically exclude is Jev's decisions from any training/eval dataset export — see
`wedge-and-mvp-spec.md` and `second_thought/datasets/policy.py`.

## Open questions (not verifiable from public docs)

- RLCD's actual reward function, training data, or calibration curves.
- Whether any deterministic/seed parameter exists for Jev.
- A complete official SDK language list beyond the documented Python examples.
- Whether Jev supports self-hosting/on-prem at all (only a hosted API is documented).
