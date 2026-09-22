# Second Thought

Learning infrastructure for typed probabilistic decisions from **System One
models** — the new category of fast, non-autoregressive, calibrated
decision models (TypeSafe's Jev, and open-weight alternatives like Laya)
that answer typed questions (`choice` / `score` / `noul`) with a full
probability distribution instead of writing text.

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

store = Store(".secondthought/store.db")

# however you already call your System One model:
result = provider.predict(state, questions)
event = capture(store, result, provider="laya", state=state, questions=questions,
                 workflow="customer_service")

candidates = select(list(store.query()), budget=25)
correct(store, candidates[0].event.id, candidates[0].question_id, value="refund")

report = export(list(store.query()), "dataset/")
```

Or via the CLI: `secondthought init`, `select`, `review`, `export`, `evaluate`.

## Why Laya, and why not Jev for training data

TypeSafe's Master Customer Agreement (§2.3(b)) prohibits training or
distilling a model on Jev's output, or using it to build a similar/competing
product. `second_thought.datasets.policy` enforces that in code: Jev
decisions can be captured and reviewed for your own observability, but
`export()` will never include them. Laya is Apache-2.0 and open-weight, so
it's the adapter this project can responsibly complete end-to-end — see
`docs/research/laya-architecture.md` for how it was actually inspected
before this adapter was written, including a provenance concern about its
GitHub star count that's worth reading before you depend on it.

## What this deliberately does not build

See `docs/research/wedge-and-mvp-spec.md` for the full list. In short: not a
prompt manager, not an LLM observability dashboard, not a generic annotation
tool, not a model registry, not distributed infrastructure. If a feature
request looks like any of those, it's out of scope by design.

## Status

v0.1.0. The toy experiment in `examples/toy_typed_decisions/` is a real,
runnable smoke test of the core selection-vs-random hypothesis on synthetic
data — read its module docstring for exactly what it does and doesn't prove.
`examples/laya_customer_service/` documents how to reproduce the same loop
against real Laya; that run has not been executed as part of building this
repository (no GPU in this environment) and its README says so.
