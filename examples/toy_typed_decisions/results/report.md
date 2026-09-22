# Second Thought — toy typed-decision experiment

Synthetic 5-class classification, 2000 pool / 500 held-out test.
Seed set: 30 labels. Then 75 additional corrections in rounds of 5, comparing random selection against Second Thought's default `uncertainty` strategy.

| strategy | accuracy after 75 corrections | gain over seed-only |
|---|---|---|
| random | 0.5600 | +0.0600 |
| uncertainty | 0.6060 | +0.1060 |

This is a synthetic smoke test of the selection mechanism, not a production benchmark — see the module docstring in `run.py`.

## A note on batch size

With `budget_per_round=15` (a larger batch relative to the 30-item seed set) on this same dataset, uncertainty selection did NOT beat random (0.436 vs 0.560 after 120 corrections, seed=7). That's not hidden here: it matches published findings that naive uncertainty sampling in large batches clusters redundantly near the current decision boundary instead of spreading out. `select()` ships a `diversify_with` parameter for exactly this case (see `second_thought/selection/selector.py`); a quick test using raw feature vectors as the embedding did not fix it either, suggesting a task-relevant embedding (not raw features) is needed for diversification to help at this batch size — left as an open problem, not papered over. The practical guidance for now: keep `budget_per_round` small relative to your labeled set, or review one at a time.