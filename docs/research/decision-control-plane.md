# From a CLI tool to a decision control plane: the case for drift.py + the dashboard

Researched and built 2026-09-22, the same week TypeSafe launched Jev and coined "System
One models" (2026-09-15) — this ecosystem is one week old. Grounded in fetches of
`docs.typesafe.ai`, GitHub, and web search, not memory; every claim below cites what it's
based on.

## What's actually out there, one week in

TypeSafe's own docs call "System One" a *category* — "Jev is TypeSafe's flagship model
and the first System One model" — but publish no open protocol, schema, or spec beyond
their own closed SDK and `POST /v1/systemone`. Confirmed by two independent fetches of
`docs.typesafe.ai/concepts/system-one`.

A real ecosystem is already forming around Jev specifically:

- **jevcal** — fits and drift-checks confidence thresholds against labeled data, writes
  a "calibration lock file." A **one-time, offline** fit.
- **jev-align** (Sutro) — active-learning CLI: Jev evaluates rows, surfaces uncertain
  ones for a human, GEPA proposes revised *question definitions* for the user to accept
  or reject.
- **Janus** — measures calibration and does confidence-based routing.
- **OpenJev** — a real, Apache-2.0, self-hostable model (DiffusionGemma 26B-A4B, ~18GB
  in 4-bit, needs 24GB+ VRAM) that speaks Jev's own wire API so Jev's SDKs work against
  it unchanged. 305 stars, active, pre-production. Its own README: "not formally
  standardized beyond wire compatibility" — it reverse-engineered Jev's shape; nobody
  published one to implement against instead.
- MCP servers, a Postgres extension, a Neovim plugin, multiple SDK ports.

Every one of these is **CLI- or library-shaped**, and every calibration tool
(`jevcal`, `evaluate()` in this project, Janus) is a **single point-in-time snapshot**.

## Two gaps, from first principles

The entire value proposition of a System One model rests on one promise: the confidence
score is trustworthy enough to act on automatically above some threshold, and route to a
human below it (TypeSafe's own documented pattern — see
`docs/research/system-one-technical-report.md`, "Escalation is caller-side"). That
promise fails silently in exactly two ways nothing surveyed addresses:

1. **Nobody watches whether it's *still* true.** A threshold fit once (`jevcal`'s own
   pattern: fit + lock file) goes stale the moment the model, checkpoint, or input
   distribution drifts. This project's own real Laya run
   (`examples/laya_customer_service/results/`) found high accuracy (87.5%) with *poor*
   calibration (ECE 0.62) on a single snapshot — the real, open question a production
   deployment needs answered continuously is whether that gap is getting better or
   worse, not just what it was once.
2. **The correction loop is engineer-only.** Every tool above, and this project's own
   `secondthought review` CLI, requires a terminal. The person best placed to correct a
   mis-routed support ticket is usually a support lead, not an engineer.

Neither gap needs a new capability invented from scratch — `evaluate()`, `select()`,
`correct()` already exist and are tested. Both are a matter of exposing what's already
real, continuously and accessibly, instead of once and via CLI.

## What was built

- **`second_thought/drift.py`** — buckets corrected decisions by time window,
  evaluates each with the existing `evaluate()`, and flags a bucket whose metric falls
  outside a control band fit from earlier buckets: a Shewhart control chart, the
  standard decades-old technique for "is a monitored statistic still behaving like it
  used to" — not new math invented for this project (matches this project's own
  standing rule against fabricated formulas; see `selection/strategies.py`'s rejection
  of the "Expected Learning Value" formula for the same reason).
- **`second_thought/dashboard/`** (`secondthought serve`, needs the `dashboard` extra)
  — a local web UI over the *same* `Store`/`select`/`correct`/`evaluate`/`drift`
  functions: a review queue usable without a terminal, and a calibration/drift chart per
  workflow/provider. No new business logic — a thin HTTP + static-file layer.
- **`second_thought/protocol.py`** (`secondthought protocol validate|schema`) — the
  request/response shape `schema.py` already encodes, generated as JSON Schema (from the
  same Pydantic models, not a hand-maintained copy) plus a standalone conformance
  checker. Useful with or without the rest of this SDK: any builder targeting this
  category can check their own API's shape against it, the way OpenJev had to
  reverse-engineer Jev's instead of implementing a published one.

## A decision this project made once, revisited on purpose

`wedge-and-mvp-spec.md` says, for v0: *"Not a web UI for v0 — the CLI (`secondthought
review`, etc.) is the correction interface."* That was the right call for an MVP proving
one falsifiable hypothesis (does uncertainty selection beat random). It stops being the
right call once the ecosystem research above shows the actual adoption ceiling for this
whole category of tooling is CLI-only accessibility, and once there's a second, distinct
capability (drift) that a CLI naturally shows once, not continuously. This isn't
silently walking back that decision — it's named, dated, and the reasoning is on the
page it revises, the same as every other design call this project has documented.

## What this deliberately doesn't do

No new adapters (no OpenJev, no live model calls — kept out of this pass on purpose:
see repo history for why). No hosted/SaaS version — the dashboard is `secondthought
serve`, local-first, reading the same on-disk SQLite `Store` the CLI does, consistent
with this project's standing local-first rule. No new statistical machinery beyond a
named, standard technique (Shewhart bands) — no invented drift-detection formula.
