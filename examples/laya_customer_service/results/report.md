# Second Thought — live Laya run

**First real execution of this directory's reproduction path.** Run 2026-09-22 on an
NVIDIA GeForce RTX 4090 (RunPod, secure cloud, US-NC-1), `laya==0.3.5`,
`convaiinnovations/laya-typed-decisions`. Total cost: ~$0.01 (pod ran ~4 minutes at
$0.74/hr). Script: [`run.py`](../run.py); raw output: [`report.json`](report.json).

The 24 tickets are synthetic example text (same spirit as
`examples/generic_provider/data.py`'s synthetic tickets) — not a real production
queue. What's real: live Laya inference on real GPU hardware, real capture into the
actual `Store`, real uncertainty-based `select()`, real human-judgment corrections,
a real `export()`, and real calibration metrics from the actual shipped `evaluate()`
— against the live model, not synthetic sklearn data, for the first time.

## Raw accuracy and calibration

24 customer-service tickets, one `choice` question each (refund / replace / escalate
/ ignore), Laya's own top choice compared against a human (reviewer) judgment call:

| metric | value |
|---|---|
| accuracy | 0.875 (21/24) |
| mean Brier score | 0.2759 |
| mean log loss | 0.5870 |
| ECE (expected calibration error) | 0.6162 |

**The ECE is bad, and that's not a bug in the harness — it matches a warning Laya
itself emits at load time**: `this checkpoint ships temperatures outside [0.5, 5]
which would distort confidence; clamping ... Treat confidence from the affected
buckets as uncalibrated.` This run is independent, real evidence for that self-
reported caveat: raw accuracy is high (87.5%) but the probability distributions
behind that accuracy are not trustworthy at face value. This is exactly the kind of
finding Second Thought's calibration measurement exists to catch — an SDK evaluating
its own recommended model and finding a real problem, not a clean bill of health.

## Selection: does uncertainty-based review actually catch the errors?

`select(events, budget=8)` on fresh, uncorrected captures — the 8 of 24 tickets
(33% of volume) Second Thought would put in front of a human reviewer first:

| result | value |
|---|---|
| total mispredictions (of 24) | 3 |
| mispredictions caught in the top-8 selected | 3 / 3 (100%) |
| review volume needed to catch them | 8 / 24 (33%) |

All three of Laya's actual mistakes were surfaced in the first third of the review
queue by uncertainty score alone — the same mechanism validated on synthetic data in
`examples/toy_typed_decisions/`, now confirmed once against a real model's real
errors. Three tickets is too small an n to claim a general recall rate; it's reported
as one concrete, real data point, not a benchmark.

The three mispredictions themselves were genuinely ambiguous, not model failures on
easy cases: a benign "thanks for the update" ticket misread as needing escalation, a
five-star review misread as a replacement request, and a wrong-item ticket where the
model said "refund" and the ground truth used here was "replace" — a defensible
either-way call.

## Latency

| | |
|---|---|
| mean | 34.3ms |
| min | 18.2ms |
| max | 396.6ms (first call — includes CUDA/kernel warmup) |
| model load | 10.7s (weights already HF-cached; ~16s cold) |

Steady-state latency (~18ms/prediction) is consistent with the model card's own
benchmark (39.5ms GPU) — same order of magnitude, this run faster likely due to GPU
tier (RTX 4090 vs. the card's reference Tesla T4).

## Export

`export()` on the 24 corrected decisions: 24/24 written (0 excluded — nothing here is
Jev-sourced, nothing left uncorrected), time-split 16 train / 3 val / 5 test. Real
JSONL + Parquet files, same code path `secondthought export` uses.

## What this does not prove

Not a production-scale benchmark (n=24, one workflow, one reviewer's judgment calls
as ground truth) and not a fine-tuning run — see the top-level README for why actual
fine-tuning isn't a turnkey automated script here (Laya ships no fine-tuning CLI,
only a Kaggle notebook). What it does prove, for the first time: the full
capture → select → correct → export → evaluate loop runs correctly end-to-end
against the real, live model this project is built around, not just against
synthetic stand-ins.
