# The wedge, the MVP spec, and what this project refuses to become

## The wedge

> Infrastructure for learning from typed probabilistic decisions — specifically the intersection
> of calibration measurement, uncertainty-based active learning, and fixed-schema (`choice` /
> `score` / `noul`) decisions, ending in a leakage-safe fine-tune-ready export.

Not "it helps fine-tune models" (too generic — every LLM-ops tool claims that). The competitive
matrix (`competitive-matrix.md`) shows each of the three ingredients exists separately; none of the
surveyed tools combine all three for non-autoregressive, typed decisions specifically. That's the
actual bet, and it's falsifiable — which is why Phase 12's experiment mattered more than any other
part of this project.

## Killing the original brief's assumptions, on purpose

- **"Multiplicative Expected Learning Value"** — rejected. No evidence a product of five heuristic
  signals (uncertainty × disagreement × novelty × impact × representativeness) outperforms a
  transparent weighted sum, and shipping an unvalidated formula dressed as a metric is worse than a
  legible one. `selection/strategies.py`'s `score_uncertainty` is a documented, equal-weight mean of
  two signals (entropy, inverse margin) — nothing more is claimed for it.
- **Novelty / distribution-shift / business-impact / representativeness scoring** — not built.
  Each needs infrastructure this MVP doesn't have (an embedding index with history, a cost model, a
  labeled shift baseline). Faking them with placeholder math would be worse than the honest
  `docs`-documented gap. `select()`'s `diversify_with` parameter is the one extension point built
  for this, and even it didn't cleanly fix the batch-size problem below — reported, not hidden.
- **Jev fine-tuning loop** — not built, on legal grounds, not technical ones. TypeSafe's MCA
  §2.3(b) bans training/distillation on Jev output. `second_thought/datasets/policy.py` enforces
  this in code: `export()` silently excludes Jev-sourced decisions by default, and raises
  `BlockedProviderError` if a caller explicitly asks for them. Jev capture/observability (its own
  calibration review) is unrestricted; its appearance in any training/eval dataset is not.
- **Jev live-API adapter** — not built. This project has never held a TypeSafe API key.
  `adapters/jev.py` is a capture-only, bring-your-own-client normalizer, not a fabricated HTTP
  integration.

## The central experiment, run for real (not fabricated)

`examples/toy_typed_decisions/run.py`: a synthetic 5-class classification problem, comparing random
selection against `second_thought`'s default `uncertainty` strategy, using the actual shipped
`select()`/`evaluate()` code (not a separate hand-rolled comparison). Result, reproducible with the
seed in the script:

| strategy | accuracy after 75 corrections | gain over seed-only |
|---|---|---|
| random | 0.560 | +0.060 |
| uncertainty | 0.606 | +0.106 |

Uncertainty selection extracted ~1.8× the accuracy improvement per correction, consistent across
three different data seeds tested during development (data_seed 1: 0.612 vs 0.630; seed 7: 0.560
vs 0.606; seed 42: 0.556 vs 0.650 — uncertainty ahead every time).

**This result is batch-size-dependent, and that's reported, not hidden.** With a larger
`budget_per_round` (15, vs. the seed set of 30) on the same data, uncertainty selection did **not**
beat random (0.392–0.436 vs. random's 0.436 across variants, seed 7). This matches published
findings that naive uncertainty sampling in large batches queries redundantly near the current
decision boundary instead of spreading out — a real limitation of the shipped strategy, not a bug.

**Follow-up investigation (2026-09-22), reported in full rather than stopping at the first
negative result:** tried five things to fix it, none of which held up as a general fix:

1. `diversify_with` using raw pool features — didn't help (0.386 final, still below random).
2. `diversify_with` using standardized features — small improvement (0.412), still below random.
3. `diversify_with` using a 5-component PCA embedding — similar (0.414), still below random.
4. `diversify_with` using the classifier's own `decision_function` output as a pseudo-embedding
   (closer to "task-relevant structure" than raw features) — best of the diversification attempts
   (0.418), **still below random's 0.436**.
5. An epsilon-greedy hybrid (mixing a fixed fraction of random picks into each uncertainty batch)
   looked promising on the original seed — 25% random + 75% uncertainty reached 0.462, beating both
   pure strategies — but **did not replicate** on two other data seeds (seed 1: hybrid 0.524 vs.
   random 0.568, hybrid loses; seed 42: hybrid 0.440 vs. random 0.458, hybrid loses). Reported as a
   negative result, not shipped as a strategy, specifically because it looked like a fix on the
   first seed tried and stopping there would have been a cherry-picked, non-reproducible claim.

**Second follow-up investigation (2026-09-22, same day, continued rather than treated as closed):**
all five attempts above kept the shipped `_diversify()` algorithm (greedy farthest-first traversal
seeded from the single highest-uncertainty candidate) and only varied what embedding fed into it.
A sixth attempt changed the *algorithm* instead: k-means clustering (k = budget) over the
shortlist's `decision_function` embedding — attempt 4's best-performing embedding — picking the
single highest-uncertainty member of each cluster, instead of farthest-first traversal. Motivation:
farthest-first always seeds from the most extreme uncertain point and walks outward from there,
which can still stay within one region of decision-boundary space; clustering partitions the
shortlist into `budget` regions up front, which is a structurally different way to force spread.

Tested against the same harder config and the same three seeds (1, 7, 42), `budget_per_round=15`:

| seed | random | uncertainty (no diversify) | cluster-diversify | beats random? |
|---|---|---|---|---|
| 1 | 0.4540 | 0.4120 | 0.4240 | No |
| 7 | 0.3380 | 0.3700 | 0.3680 | Yes |
| 42 | 0.3780 | 0.3980 | 0.3820 | Yes |

Beats random on 2 of 3 seeds, loses on seed 1 — **does not close the gap robustly either**, for the
same reason the epsilon-greedy hybrid was rejected: a fix that wins on most seeds tried but loses on
one isn't a general fix, it's noise dressed as one. Not shipped as the default, and not added as a
new `Strategy` — this is a negative result, recorded so the next person doesn't re-run this specific
idea (a fundamentally different diversification algorithm, not just a new embedding) from scratch
either. The exact reproduction script is not checked in (it lived in a scratch investigation, not
`examples/`), but the method is fully specified above: it's a ~15-line change to `_diversify()`
(swap the farthest-first loop for `sklearn.cluster.KMeans(n_clusters=budget)` on the shortlist,
then pick the top-scored candidate per cluster) plus threading a `decision_function`-based embed
function through in place of raw features — reproducible from that description alone.

**Conclusion: large-batch degradation of uncertainty sampling remains an open problem on this
benchmark**, now after six mitigation attempts (five embedding variants of farthest-first, one
structurally different clustering algorithm) rather than five. None closed the gap robustly. This
is documented here instead of quietly dropped so the next person doesn't re-run the same six ideas
from scratch. The practical guidance stands: keep `budget_per_round` small relative to the labeled
set (where uncertainty selection reliably wins, per the headline result above), or review one at a
time.

**Why synthetic data, not real Laya fine-tuning:** reproducing this at Laya's actual scale needs a
GPU, a labeled production-shaped dataset, and enough wall-clock time for several fine-tuning
rounds — none of which this repository can honestly claim to have exercised in the environment it
was built in (no GPU available). `examples/laya_customer_service/README.md` documents exactly how
to reproduce the same loop against real Laya; running it is left to whoever has the compute.

## What this project explicitly does not build

- Not a prompt manager, not an LLM observability dashboard, not a generic annotation/labeling
  platform, not a generic fine-tuning UI, not a RAG framework, not a LangChain/LangGraph
  alternative, not a generic model registry, not GPU orchestration, not distributed infrastructure
  (Kubernetes/Kafka/Redis), not a hosted service.
- Not novelty/distribution-shift/business-impact selection signals (see above — deferred, not
  faked).
- Not a Jev live-API client, and never a Jev training-data pipeline.
- Not a web UI for v0 — the CLI (`secondthought review`, etc.) is the correction interface.

## Correction: this was never architecturally Laya-specific, but read that way

Early docs and the README led with "Laya-first," which overstated how coupled the SDK actually is
to Laya. The core pipeline (`schema.py`, `capture.py`, `selection/`, `correction.py`,
`datasets/`, `evaluation.py`) never imports Laya or assumes its response shape beyond the generic
`{"model", "answers"}` envelope every inspected System One provider shares. What was missing was a
*demonstrated* path for "bring your own model" — `adapters/custom.py`'s `FunctionProvider` and
`examples/generic_provider/` (a plain TF-IDF + logistic regression text classifier, no Laya or Jev
involved) now prove that concretely rather than asserting it. Laya remains the only provider this
project fine-tunes end-to-end for one reason only — licensing (Apache-2.0, unlike Jev) — not
because the SDK privileges it architecturally.

## MVP scope actually shipped

Schema (`schema.py`, versioned, JSON-serializable) → SQLite storage → capture (provider-agnostic
normalizer, since Laya and Jev share a response envelope) → selection (entropy/margin/least-
confidence/uncertainty/random, with an opt-in diversity re-ranking pass) → correction (accept/
correct/abstain/flag, all with provenance) → dataset export (JSONL + Parquet, time-based
train/val/test split, hard policy boundary against Jev) → evaluation (accuracy, Brier, log loss,
ECE, selective accuracy) → a real Laya adapter → a capture-only Jev adapter → a CLI → a runnable,
real, honestly-reported experiment.

## Risks

- **Laya's provenance**: corroborated as likely-real (see `laya-architecture.md`) but not fully
  verified; pin versions, don't trust its self-reported benchmarks.
- **Batch-size sensitivity** of the default selection strategy (above) — a real limitation to fix
  before recommending large `budget_per_round` values in production guidance.
- **Legal boundary enforcement** depends on `provider` strings being set correctly by adapters —
  a custom adapter that mislabels a Jev-sourced decision as something else would bypass the policy
  check. Worth a follow-up: provenance-signing captured decisions, not just trusting the string.

## Open-source posture

Apache-2.0 (matches Laya's own license and this project's dependency on it). Everything shipped
here — schema, storage, selection, correction, export, evaluation, the Laya adapter, the CLI — is
open source with no hosted/free-tier split; there's no commercial product yet to carve boundaries
around. A future hosted offering (managed storage, a team review UI, org-level calibration
dashboards) is plausible but deliberately not designed here — see "what this project does not
build."

## Naming

"Second Thought" — the deliberate, reflective correction layer sitting on top of a System One
(fast, intuitive) model, directly playing on the Kahneman framing TypeSafe's own naming invokes,
without claiming the "System Two" name outright. Checked before committing: free on PyPI (PEP 503
normalization means `second-thought` and `secondthought` are the same package slot), no
GitHub org/repo collision under the account this ships from, no notable trademark or product
collision found in a web search.
