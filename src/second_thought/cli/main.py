"""The `secondthought` CLI. Every command here has real functionality behind it —
nothing is wired up as a stub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from second_thought.capture import capture as capture_event
from second_thought.correction import abstain as do_abstain
from second_thought.correction import accept as do_accept
from second_thought.correction import correct as do_correct
from second_thought.correction import flag as do_flag
from second_thought.correction import predicted_value
from second_thought.datasets import export as export_dataset
from second_thought.evaluation import evaluate as run_evaluation
from second_thought.schema import DecisionEvent
from second_thought.selection import Strategy
from second_thought.selection import select as select_candidates
from second_thought.storage import Store

app = typer.Typer(help="Learning infrastructure for typed probabilistic decisions.")
console = Console()

DEFAULT_DB = ".secondthought/store.db"


def _store(db: str) -> Store:
    return Store(db)


@app.command()
def init(db: Annotated[str, typer.Option(help="SQLite path to create")] = DEFAULT_DB) -> None:
    """Create a local decision store."""
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with _store(db):
        pass
    console.print(f"[green]Initialized[/green] store at {db}")


@app.command()
def capture(
    responses_jsonl: Annotated[
        str, typer.Argument(help="JSONL file of {provider, state, questions, raw} records")
    ],
    db: Annotated[str, typer.Option()] = DEFAULT_DB,
) -> None:
    """Bulk-load pre-recorded raw provider responses into the store.

    Each line: {"provider": "laya", "state": ..., "questions": {...}, "raw": {...},
    "workflow": "...", "latency_ms": ...}. For live instrumentation, call
    ``second_thought.capture()`` directly from your inference code instead.
    """
    n = 0
    failures: list[tuple[int, str]] = []
    with _store(db) as store:
        for lineno, line in enumerate(
            Path(responses_jsonl).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                # strict=True: a backfill is offline, human-attended work — a
                # malformed line should be reported, not silently dropped the
                # way a live inference-path capture() call would (default
                # strict=False there; see second_thought.capture).
                capture_event(
                    store,
                    rec["raw"],
                    provider=rec["provider"],
                    state=rec["state"],
                    questions=rec["questions"],
                    workflow=rec.get("workflow"),
                    model_version=rec.get("model_version"),
                    latency_ms=rec.get("latency_ms"),
                    strict=True,
                )
                n += 1
            except Exception as exc:  # noqa: BLE001 - reported per line, not swallowed
                failures.append((lineno, str(exc)))
    console.print(f"[green]Captured[/green] {n} decision events into {db}")
    if failures:
        console.print(f"[red]Skipped {len(failures)} malformed line(s)[/red]")
        for lineno, message in failures:
            console.print(f"  line {lineno}: {message}")


@app.command()
def inspect(db: Annotated[str, typer.Option()] = DEFAULT_DB) -> None:
    """Summarize what's in the store: counts by provider and correction status."""
    with _store(db) as store:
        events = list(store.query())
    table = Table(title=f"{db} ({len(events)} events)")
    table.add_column("Provider")
    table.add_column("Workflow")
    table.add_column("Events")
    table.add_column("Corrected questions")
    by_key: dict[tuple[str, str], list[int]] = {}
    for e in events:
        key = (e.provider, e.workflow or "-")
        counts = by_key.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += len(e.corrections)
    for (provider, workflow), (n_events, n_corrected) in sorted(by_key.items()):
        table.add_row(provider, workflow, str(n_events), str(n_corrected))
    console.print(table)


@app.command()
def select(
    budget: Annotated[int, typer.Option(help="How many decisions to select for review")] = 25,
    strategy: Annotated[Strategy, typer.Option()] = Strategy.UNCERTAINTY,
    provider: Annotated[str | None, typer.Option()] = None,
    workflow: Annotated[str | None, typer.Option()] = None,
    db: Annotated[str, typer.Option()] = DEFAULT_DB,
) -> None:
    """Show the top decisions worth a human's time, ranked by ``strategy``."""
    with _store(db) as store:
        events = list(store.query(provider=provider, workflow=workflow))
    candidates = select_candidates(events, strategy=strategy, budget=budget)
    table = Table(title=f"Top {len(candidates)} by {strategy.value}")
    table.add_column("Event ID")
    table.add_column("Question")
    table.add_column("Score", justify="right")
    table.add_column("Prediction")
    for c in candidates:
        pred = c.event.predictions[c.question_id]
        table.add_row(
            c.event.id[:8], c.question_id, f"{c.score:.3f}", str(predicted_value(pred))
        )
    console.print(table)


@app.command()
def review(
    event_id: str,
    question_id: str,
    action: Annotated[str, typer.Argument(help="accept | correct | abstain | flag")],
    value: Annotated[str | None, typer.Option(help="Corrected value, for action=correct")] = None,
    note: Annotated[str | None, typer.Option()] = None,
    reviewer: Annotated[str | None, typer.Option()] = None,
    db: Annotated[str, typer.Option()] = DEFAULT_DB,
) -> None:
    """Apply a review decision to one (event, question) pair."""
    with _store(db) as store:
        event = store.get(event_id) or _resolve_prefix(store, event_id)
        if event is None:
            raise typer.BadParameter(f"no event matching {event_id!r}")
        if action == "accept":
            do_accept(store, event.id, question_id, reviewer=reviewer)
        elif action == "correct":
            if value is None:
                raise typer.BadParameter("--value is required for action=correct")
            do_correct(store, event.id, question_id, value, reviewer=reviewer, note=note)
        elif action == "abstain":
            do_abstain(store, event.id, question_id, reviewer=reviewer, note=note)
        elif action == "flag":
            do_flag(store, event.id, question_id, note or "", reviewer=reviewer)
        else:
            raise typer.BadParameter("action must be one of: accept, correct, abstain, flag")
    console.print(f"[green]Recorded[/green] {action} for {event_id}/{question_id}")


def _resolve_prefix(store: Store, prefix: str) -> DecisionEvent | None:
    matches = [e for e in store.query() if e.id.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


@app.command()
def export(
    out: Annotated[str, typer.Option(help="Output directory")] = "dataset",
    provider: Annotated[list[str] | None, typer.Option("--provider")] = None,
    split: Annotated[str, typer.Option()] = "time",
    db: Annotated[str, typer.Option()] = DEFAULT_DB,
) -> None:
    """Export corrected decisions as a train/val/test correction dataset."""
    with _store(db) as store:
        events = list(store.query())
    report = export_dataset(events, out, providers=provider, split=split)  # type: ignore[arg-type]
    console.print(
        f"[green]Wrote[/green] {report.written_records} records to {out} "
        f"(train={report.split_counts['train']}, val={report.split_counts['val']}, "
        f"test={report.split_counts['test']}); excluded "
        f"{report.excluded_blocked_provider} policy-blocked, "
        f"{report.excluded_uncorrected_questions} uncorrected"
    )


@app.command()
def evaluate(
    provider: Annotated[str | None, typer.Option()] = None,
    question: Annotated[str | None, typer.Option("--question-id")] = None,
    db: Annotated[str, typer.Option()] = DEFAULT_DB,
) -> None:
    """Print accuracy/Brier/log-loss/ECE/selective-accuracy over corrected decisions."""
    with _store(db) as store:
        events = list(store.query(provider=provider))
    report = run_evaluation(events, question_id=question)
    console.print(report)


if __name__ == "__main__":
    app()
