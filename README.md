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
prompt manager, not an LLM observability dashboard, not a generic annotation
tool, not a model registry, not distributed infrastructure. If a feature
request looks like any of those, it's out of scope by design.

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
  loop against real Laya at production scale; that run has not been
  executed as part of building this repository (no GPU available) and its
  README says so.
