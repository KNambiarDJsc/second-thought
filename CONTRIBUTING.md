# Contributing to Second Thought

Thanks for considering it. This is a young project (the repo is days old) in a young category
(TypeSafe's System One models launched 2026-09-15), so there's real, uncovered ground here —
not just typo fixes.

## Before you start

For anything bigger than a small fix, open an [issue](https://github.com/KNambiarDJsc/second-thought/issues)
or a [discussion](https://github.com/KNambiarDJsc/second-thought/discussions) first. It's a lot
cheaper to align on approach before code than after — and this project has already killed several
ideas that looked good on paper (see `docs/research/wedge-and-mvp-spec.md`'s "what this
deliberately does not build"). Saves you a rewrite, saves the maintainer a hard "no."

## Dev setup

```bash
git clone https://github.com/KNambiarDJsc/second-thought.git
cd second-thought
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

`dev` pulls in everything needed to run the full test suite, including the optional pieces
(`numpy`/`scikit-learn` for the experiments, `fastapi`/`uvicorn`/`httpx` for the dashboard). It
does **not** install `laya` (a real model download) — add `pip install -e ".[laya]"` if you're
touching the Laya adapter specifically.

## Before opening a PR

Run the same three checks CI runs, in this order:

```bash
ruff check src/ tests/ examples/
mypy src/second_thought
pytest -q
```

All three must be clean. `mypy --strict` is on for the whole `src/second_thought` package — a
new module without type hints will fail CI, not just get flagged in review.

## What a good PR looks like here

- **Tests that prove the behavior, not just exercise the code.** Look at `tests/test_drift.py`'s
  `test_detect_drift_flags_a_real_calibration_regression` for the house style: build a case that
  would fail without your fix, not just one that happens to pass with it.
- **Real numbers, not hand-edited ones.** If your PR touches an example or a benchmark
  (`examples/*/results/`), regenerate the report by actually running the script — don't hand-edit
  the JSON/Markdown output. `git diff` on a regenerated report should show real computed values.
- **Report negative results too.** If you tried something and it didn't work, say so in the PR —
  see `docs/research/wedge-and-mvp-spec.md`'s "Follow-up investigation" sections for the tone
  this project aims for: six real, documented, failed attempts at one problem, left in the repo
  rather than quietly dropped. A "this didn't work, here's why" PR to the docs is genuinely
  welcome, not just tolerated.
- **No fabricated formulas.** `selection/strategies.py` explicitly rejected an invented
  "Expected Learning Value" scoring formula in favor of documented, standard techniques (entropy,
  margin). `drift.py` uses a named, standard one (Shewhart control charts) for the same reason.
  If a new scoring/detection idea isn't a documented technique, it needs evidence it beats the
  alternatives before it ships as a default — a PR proposing it as an *option* with that evidence
  is welcome regardless.

## Where things live

- `schema.py` — the `DecisionEvent`/`Prediction`/`QuestionSpec` shape everything else is built on.
- `capture.py` → `storage/` → `selection/` → `correction.py` → `datasets/export.py` →
  `evaluation.py` — the core loop, roughly in the order data flows through it.
- `drift.py` — continuous calibration monitoring on top of `evaluation.py`.
- `dashboard/` — the local web UI (`secondthought serve`), a thin HTTP layer with no business
  logic of its own; it calls the same functions above.
- `protocol.py` — the open, standalone wire-shape conformance checker.
- `adapters/` — provider adapters (`laya.py`, `jev.py`, `custom.py`'s `FunctionProvider`).
- `cli/main.py` — the `secondthought` CLI, one Typer command per public operation.
- `docs/research/` — the design rationale. `decision-control-plane.md` and
  `wedge-and-mvp-spec.md` are the two worth reading before a non-trivial change; both document
  real decisions and real research, not aspirational planning.

## Good places to start

- **A new adapter.** `second_thought.adapters.SystemOneProvider` is a two-method structural
  interface (`predict()`, `model_info()`) — no subclassing required. **OpenJev**
  ([razorback16/openjev](https://github.com/razorback16/openjev), Apache-2.0, self-hostable,
  Jev-wire-compatible) is a concrete, real, unclaimed one; a `FunctionProvider`-based example for
  any other real typed-decision model is just as welcome and doesn't need a named adapter class.
- **A new selection strategy.** `selection/strategies.py` — add a scoring function, register it
  in `Strategy`, and cite what you're implementing (a paper, a named technique) in the docstring.
- **A new drift metric or control technique.** `drift.py` currently supports accuracy/Brier/log
  loss/ECE with a Shewhart band; a documented alternative (e.g. a CUSUM or EWMA chart) is welcome
  as an option.
- **The open research problem.** The large-batch active-learning regression in
  `docs/research/wedge-and-mvp-spec.md` has six real, failed mitigation attempts on record. A
  seventh real attempt — tried, measured across the same three seeds, reported honestly whichever
  way it goes — is one of the most valuable contributions this project could get.
- **Running Laya's actual fine-tune step.** `examples/laya_customer_service/README.md` documents
  why this isn't a turnkey script (Laya ships no fine-tuning CLI, only a notebook) — a PR that
  actually does it, with a real before/after evaluation, closes a real gap.

## Code of conduct

Be the kind of contributor whose PRs are a pleasure to review: clear description, tests that
prove the point, and honesty about what you didn't get to. Beyond that, use ordinary judgment —
this project doesn't need more process than that yet.
