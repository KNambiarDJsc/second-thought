<div align="center">

# Second Thought

**Learning infrastructure for typed probabilistic decisions from any System One model**

[![CI](https://github.com/KNambiarDJsc/second-thought/actions/workflows/ci.yml/badge.svg)](https://github.com/KNambiarDJsc/second-thought/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

[Quickstart](#quickstart) · [Why](#why-second-thought) · [Dashboard](#a-local-dashboard-continuous-drift-monitoring-and-an-open-wire-spec) · [Proof, not claims](#proof-not-claims) · [Contributing](#contributing)

<br>

<img src="docs/assets/dashboard.jpg" alt="Second Thought's local review dashboard: a ticket the model is unsure about, its confidence, and one-click accept/correct" width="720">

<sub>The local review queue (`secondthought serve`) — real screenshot, not a mockup. This ticket scored 0.890 on uncertainty; the model's own answer was only 48% confident.</sub>

</div>

---

> **A real run against live Laya found 87.5% accuracy paired with an ECE of 0.62 — high accuracy, badly calibrated probabilities, independently matching a load-time warning Laya itself emits.** That's the kind of gap this project exists to catch. [Full writeup →](examples/laya_customer_service/results/report.md)

**System One models** are the new category of fast, non-autoregressive, calibrated decision
models — TypeSafe's [Jev](https://typesafe.ai), open-weight alternatives like
[Laya](https://huggingface.co/convaiinnovations/laya), or your own fine-tuned classifier — that
answer typed questions (`choice` / `score` / `noul`) with a full probability distribution instead
of writing text a caller has to parse.

Second Thought captures those decisions, measures whether their probabilities are *actually*
calibrated, finds the ones most worth a human's time to review, turns the corrections into a
versioned fine-tune-ready dataset, and watches calibration for drift after you ship — without
becoming another generic LLM-ops, annotation, or observability platform.

```
capture → store (SQLite) → select (uncertainty) → correct (CLI/web) → export (JSONL/Parquet) → evaluate → watch for drift
```

The SDK isn't tied to any one model or vendor. `SystemOneProvider` is a two-method structural
interface (`predict()`, `model_info()`); `FunctionProvider` wraps any plain callable that returns
`{"model", "answers"}` in one line. Laya and Jev ship named adapters because they're the two
concrete models this project actually inspected — not because anything is hardcoded to them.

## Why Second Thought

Every calibration tool in the System One ecosystem checks a model's probabilities **once,
offline**, and every correction/review tool needs an **engineer at a terminal**. Second Thought is
built to close both gaps, on real evidence, not a redesigned wrapper around the same idea:

| | Second Thought | LLM-ops platforms<br>(TensorZero, Langfuse, …) | Classic ML monitoring<br>(Evidently, WhyLabs, …) | Weak-supervision tools<br>(Snorkel, Cleanlab, …) |
|---|:---:|:---:|:---:|:---:|
| Stores the full probability distribution | ✅ | ❌ | partial | ✅ |
| ECE (expected calibration error) | ✅ | ❌ | ❌ | ❌ |
| ECE tracked over time, with drift alerting | ✅ | ❌ | ❌ | ❌ |
| Uncertainty-based review queue | ✅ | ❌ | ❌ | ✅ |
| Correction → fine-tune-ready dataset export | ✅ | ✅ | ❌ | partial |
| Local web review UI, no terminal needed | ✅ | varies | varies | varies |
| Typed-decision native (not LLM-text-shaped) | ✅ | ❌ | ✅ | ✅ |

Classic ML monitoring tools are the strongest of the four at general drift monitoring — that's
their core category — but ECE specifically, as either a metric or something tracked over time, was
absent across every tool surveyed (Evidently AI comes closest with log loss). No single tool does
the full row above. See [`docs/research/competitive-matrix.md`](docs/research/competitive-matrix.md)
for the full 19-tool matrix this table is drawn from, sourced from each tool's own docs, not
memory.

## Install

```bash
pip install -e ".[dev]"           # core + tests
pip install -e ".[laya]"          # + the real Laya adapter
pip install -e ".[dashboard]"     # + the local review/drift web UI (`secondthought serve`)
pip install -e ".[experiments]"   # + numpy/scikit-learn, to run examples/toy_typed_decisions
```

*(Not yet on PyPI — `second-thought` is unclaimed there as of this writing. Install from source
until it is.)*

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

Or via the CLI: `secondthought init`, `select`, `review`, `export`, `evaluate`, `serve`,
`protocol validate|schema`.

## A local dashboard, continuous drift monitoring, and an open wire spec

Three things past the core loop, all built on the exact same `Store` / `select` / `correct` /
`evaluate` functions above — no parallel implementation, no new business logic to trust:

- **`secondthought serve`** (`pip install second-thought[dashboard]`) — a local web UI: a review
  queue for `accept` / `correct` / `abstain` that doesn't need a terminal, and a calibration/drift
  chart per workflow and provider. Every calibration/active-learning tool this project found in
  the wider System One ecosystem (`jevcal`, `jev-align`, `Janus`) is CLI- or library-only — this
  is the one that isn't.
- **`second_thought.drift`** — the same `evaluate()` above, bucketed over time and checked against
  a control band (a standard Shewhart control chart, not invented math). Calibration degrading in
  production shows up as a flagged point, not silence. Every calibration tool surveyed — including
  this SDK's own `evaluate()`, before this — was a one-time snapshot.
- **`second_thought.protocol`** (`secondthought protocol validate|schema`) — the typed-decision
  request/response shape this SDK already runs against real providers, published as JSON Schema +
  a standalone conformance checker. TypeSafe calls "System One" a category but ships no open spec
  for it; this is the shape three independent real implementations (Jev, Laya, OpenJev) already
  converge on — usable with or without the rest of this SDK.

See [`docs/research/decision-control-plane.md`](docs/research/decision-control-plane.md) for the
ecosystem research and the first-principles case behind all three.

## Proof, not claims

Every number below is computed by a script in this repo, not hand-edited. Re-run any of them
yourself.

- **A real live model, not a mock.** [`examples/laya_customer_service/`](examples/laya_customer_service/)
  ran the full loop against real Laya on a GPU (2026-09-22, one RTX 4090, `laya==0.3.5`, total
  cost **~$0.01**): **87.5% accuracy** on 24 live-inferred tickets, and uncertainty-based
  `select()` caught **3 of 3 real mispredictions** while reviewing only the first third (8/24) of
  the queue. It also surfaced a genuine finding, not a synthetic one: Laya's own load-time warning
  about out-of-range calibration temperatures is independently corroborated by this run's
  **ECE of 0.62** — high accuracy, poorly calibrated probabilities. Full writeup in
  [`results/report.md`](examples/laya_customer_service/results/report.md).
- **Selection beats random, measured, across seeds.** [`examples/toy_typed_decisions/`](examples/toy_typed_decisions/)
  is a real, runnable comparison on synthetic tabular data: uncertainty selection reached **0.606
  accuracy** after 75 corrections vs. **0.560** for random selection (+0.106 vs. +0.060 gain over
  seed-only) — about **1.8×** the accuracy improvement per correction, consistent across three
  data seeds.
- **Works with zero GPU, zero Laya, zero Jev.** [`examples/generic_provider/`](examples/generic_provider/)
  proves the SDK on a plain scikit-learn TF-IDF classifier — nothing here is hardcoded to one
  vendor's response shape.
- **Negative results are reported, not hidden.** The active-learning regression at large batch
  sizes has six documented, real, failed mitigation attempts in
  [`docs/research/wedge-and-mvp-spec.md`](docs/research/wedge-and-mvp-spec.md) — including one
  that looked like a fix on the first seed tried and was rejected anyway because it didn't
  replicate. It's still an open problem. We'd rather tell you that than quietly drop it.
- **Actually tested.** 72+ tests, `mypy --strict` clean, `ruff` clean, CI green on Linux and
  Windows across Python 3.11–3.13 — [see for yourself](.github/workflows/ci.yml).

## Why Laya is the only provider this project fine-tunes end-to-end, and why Jev is capture-only

TypeSafe's Master Customer Agreement (§2.3(b)) prohibits training or distilling a model on Jev's
output, or using it to build a similar/competing product. `second_thought.datasets.policy`
enforces that **in code**: Jev decisions can be captured and reviewed for your own observability,
but `export()` will never include them, regardless of which adapter produced them. Laya is
Apache-2.0 and open-weight, so it's the one *named* adapter this project completed end-to-end for
real fine-tuning — see [`docs/research/laya-architecture.md`](docs/research/laya-architecture.md)
for how it was actually inspected before that adapter was written, including a provenance concern
about its GitHub star count worth reading before you depend on it. This is a licensing fact about
two specific vendors, not a limitation of the SDK: any provider you bring yourself (via
`FunctionProvider` or your own class) is just as fine-tunable as Laya, export-wise, as long as it
isn't Jev.

## What this deliberately does not build

See [`docs/research/wedge-and-mvp-spec.md`](docs/research/wedge-and-mvp-spec.md) for the full
list and the reasoning behind each one. In short: not a prompt manager, not a *generic* LLM
observability dashboard (the local dashboard above is scoped to this SDK's own typed-decision
loop — a review queue and a calibration/drift chart, not general tracing/spans/prompt
playgrounds), not a generic annotation tool, not a model registry, not distributed
infrastructure. If a feature request looks like any of those, it's out of scope by design — open
an issue and we'll say so plainly rather than let scope creep in quietly.

## Status

**v0.1.0, alpha.** The core loop (capture → store → select → correct → export → evaluate) and the
drift/dashboard/protocol layer on top of it are real, tested, and run against a real model — see
[Proof, not claims](#proof-not-claims) above. Not yet on PyPI. API may still move before v1.

## Contributing

Contributions are genuinely welcome — this is a one-week-old project in a one-week-old model
category, and there's a lot of real ground still uncovered. Start with
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how the codebase fits together, how to run the test
suite, and where to find a good first issue. Bring your own System One model? A `FunctionProvider`
example or a named adapter (`second_thought/adapters/`) is one of the highest-leverage
contributions there is — see [`docs/research/decision-control-plane.md`](docs/research/decision-control-plane.md)
for what's already been surveyed in the wider ecosystem before you start.

[GitHub Discussions](https://github.com/KNambiarDJsc/second-thought/discussions) is open for
questions and ideas; [Issues](https://github.com/KNambiarDJsc/second-thought/issues) for bugs and
concrete feature requests.

## License

[Apache License 2.0](LICENSE).
