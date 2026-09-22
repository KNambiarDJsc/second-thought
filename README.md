# Second Thought

Learning infrastructure for typed probabilistic decisions from **any System
One model** — the category of fast, non-autoregressive, calibrated decision
models (TypeSafe's Jev, open-weight alternatives like Laya, or your own
fine-tuned classifier) that answer typed questions (`choice` / `score` /
`noul`) with a full probability distribution instead of writing text.

The SDK isn't tied to any one model. `second_thought.adapters.SystemOneProvider`
is a two-method structural interface (`predict()`, `model_info()`), and
`FunctionProvider` wraps any plain function that returns
`{"model", "answers"}` in one line — see `examples/generic_provider/` for a
complete, GPU-free, real (not synthetic) example using a plain
scikit-learn text classifier, no Laya or Jev involved. Laya and Jev get
named adapters (`adapters/laya.py`, `adapters/jev.py`) because they're the
two concrete System One models this project actually inspected, not because
anything is hardcoded to them.

Second Thought captures those decisions, measures whether their
probabilities are actually calibrated, finds the ones most worth a human's
time to review, and turns the corrections into a versioned, leak-free
dataset — without becoming another generic LLM-ops, annotation, or
fine-tuning platform. See `docs/research/` for the research this project is
built on, including why that narrow scope is the point.

```
capture → store (SQLite) → select (uncertainty) → correct (CLI) → export (JSONL/Parquet) → evaluate
```

## Install

```
pip install -e ".[dev]"          # core + tests
pip install -e ".[laya]"          # + the real Laya adapter
pip install -e ".[dashboard]"     # + the local review/drift web UI (`secondthought serve`)
pip install -e ".[experiments]"   # + numpy/scikit-learn, to run examples/toy_typed_decisions
```

## Quickstart

```python
from second_thought import Store, capture, select, correct, export
from second_thought.adapters import FunctionProvider  # or LayaProvider, JevProvider, or your own

provider = FunctionProvider("my-model", my_predict_fn)  # wraps any callable — see below
store = Store(".secondthought/store.db")

result = provider.predict(state, questions)
event = capture(store, result, provider=provider.name, state=state, questions=questions,
                 workflow="customer_service")

candidates = select(list(store.query()), budget=25)
correct(store, candidates[0].event.id, candidates[0].question_id, value="refund")

report = export(list(store.query()), "dataset/")
```

Or via the CLI: `secondthought init`, `select`, `review`, `export`, `evaluate`.

## A local dashboard, continuous calibration drift, and an open wire spec

Three things past the core loop, all built on the same `Store`/`select`/`correct`/
`evaluate` functions above — no parallel implementation, no new business logic:

- **`secondthought serve`** (needs `pip install second-thought[dashboard]`) — a local
  web UI: a review queue for `accept`/`correct`/`abstain` that doesn't need a terminal,
  and a calibration/drift chart per workflow/provider. Every calibration/active-learning
  tool this project found in the wider System One ecosystem (`jevcal`, `jev-align`,
  `Janus`) is CLI- or library-only; this is the one that isn't.
- **`second_thought.drift`** — the same `evaluate()` above, but bucketed over time and
  checked against a control band (a standard Shewhart control chart), so calibration
  degrading in production shows up as a flagged point, not silence. Every calibration
  tool surveyed, including this SDK's own `evaluate()`, was a one-time snapshot before
  this — see `docs/research/decision-control-plane.md` for why that's the real gap.
- **`second_thought.protocol`** (`secondthought protocol validate|schema`) — the typed-
  decision request/response shape this SDK already runs against real providers,
  published as JSON Schema + a standalone conformance checker. TypeSafe's own docs call
  "System One" a category but publish no open spec for it; this is the shape three
  independent real implementations (Jev, Laya, OpenJev) already converge on, usable
  with or without the rest of this SDK.

## Why Laya is the only provider this project fine-tunes end-to-end, and why Jev is capture-only

TypeSafe's Master Customer Agreement (§2.3(b)) prohibits training or
distilling a model on Jev's output, or using it to build a similar/competing
product. `second_thought.datasets.policy` enforces that in code: Jev
decisions can be captured and reviewed for your own observability, but
`export()` will never include them, regardless of which adapter produced
them. Laya is Apache-2.0 and open-weight, so it's the one *named* adapter
this project completed end-to-end for real fine-tuning — see
`docs/research/laya-architecture.md` for how it was actually inspected
before that adapter was written, including a provenance concern about its
GitHub star count that's worth reading before you depend on it. This is a
licensing fact about two specific vendors, not a limitation of the SDK: any
provider you bring yourself (via `FunctionProvider` or your own class) is
just as fine-tunable as Laya, export-wise, as long as it isn't Jev.

## What this deliberately does not build

See `docs/research/wedge-and-mvp-spec.md` for the full list. In short: not a
prompt manager, not a *generic* LLM observability dashboard (the local dashboard
above is scoped to this SDK's own typed-decision loop — a review queue and a
calibration/drift chart, not general tracing/spans/prompt playgrounds), not a
generic annotation tool, not a model registry, not distributed infrastructure.
If a feature request looks like any of those, it's out of scope by design.

## Status

v0.1.0. Three examples, each proving a different claim:

- `examples/toy_typed_decisions/` — a real, runnable smoke test of the core
  selection-vs-random hypothesis on synthetic tabular data.
- `examples/generic_provider/` — a real, runnable, GPU-free proof that the
  SDK works for a model that is neither Laya nor Jev (a plain TF-IDF +
  logistic regression text classifier), including genuine evidence that
  uncertainty-selected tickets are wrong more often than an arbitrary sample
  — i.e. that selection is finding real learning value, not just noise.
- `examples/laya_customer_service/` — documents how to reproduce the same
  loop against real Laya, and now includes a real run of it (2026-09-22, one
  RTX 4090, `laya==0.3.5`): 87.5% accuracy on 24 live-inferred tickets, and
  uncertainty-based `select()` catching 3/3 real mispredictions within the
  first third of the review queue. It also surfaced a real, non-synthetic
  finding: Laya's own load-time warning about out-of-range calibration
  temperatures is corroborated by this run's ECE (0.62) — high accuracy,
  poorly calibrated probabilities. See `results/report.md` for the full
  writeup. This is inference + the full SDK loop, not a fine-tuning run —
  see that directory's README for why an actual fine-tune isn't a turnkey
  automated script here (Laya ships no fine-tuning CLI, only a notebook).

Past the three examples: `second_thought.drift` (continuous calibration monitoring,
not a one-time snapshot), `second_thought/dashboard/` (`secondthought serve` — a
local review + drift UI), and `second_thought.protocol` (`secondthought protocol
validate|schema` — an open, standalone conformance checker for the typed-decision
wire shape) — see `docs/research/decision-control-plane.md` for the ecosystem
research and first-principles case for all three, added 2026-09-22.
