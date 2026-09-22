# Second Thought — toy typed-decision experiment

Synthetic 5-class classification, 2000 pool / 500 held-out test.
Seed set: 30 labels. Then 75 additional corrections in rounds of 5, comparing random selection against Second Thought's default `uncertainty` strategy.

| strategy | accuracy after 75 corrections | gain over seed-only |
|---|---|---|
| random | 0.5600 | +0.0600 |
| uncertainty | 0.6060 | +0.1060 |

This is a synthetic smoke test of the selection mechanism, not a production benchmark — see the module docstring in `run.py`.

## A note on batch size

With `budget_per_round=15` on a harder synthetic config (2 clusters per class, more label noise), uncertainty selection did NOT beat random (0.392 vs random's 0.436 final accuracy, seed=7). That's not hidden here: it matches published findings that naive uncertainty sampling in large batches clusters redundantly near the current decision boundary instead of spreading out. Five follow-up fixes were tried (four embedding choices for `select()`'s `diversify_with` parameter, plus an epsilon-greedy random/uncertainty hybrid) — none held up as a general fix across data seeds. Full numbers in `docs/research/wedge-and-mvp-spec.md`. The practical guidance for now: keep `budget_per_round` small relative to your labeled set (where the headline result above holds), or review one at a time.