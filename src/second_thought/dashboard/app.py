"""The local review + calibration-drift dashboard.

Every point tool this project's research surveyed in the System One ecosystem
(jevcal, jev-align, Janus) is CLI/library-shaped — reviewing a decision, or
seeing whether calibration has drifted, requires an engineer at a terminal.
That's a real accessibility ceiling: the person best placed to correct a
mis-routed support ticket is usually a support lead, not an engineer running
``secondthought review``. This module doesn't replace the CLI or the Python
API — it's a third, optional way in, for the same local ``Store`` and the
same ``select``/``correct``/``evaluate`` functions the rest of the SDK
already ships and tests. No new business logic lives here; this is a thin
HTTP + static-file layer over ``second_thought``'s existing, already-real
functions (including ``drift.py``, previously unexposed anywhere).

Run with ``secondthought serve`` (see ``second_thought.cli.main``), or
directly:

    uvicorn second_thought.dashboard.app:create_app --factory

Needs the ``dashboard`` extra: ``pip install "second-thought[dashboard]"``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

# Deliberately no `from __future__ import annotations` here: the request-body
# models below are defined locally inside create_app() (so importing
# FastAPI/pydantic can stay lazy, for the optional `dashboard` extra).
# FastAPI resolves a route's parameter types at registration time via
# typing.get_type_hints() against the function's *module* globals; a
# string annotation (what `from __future__ import annotations` produces)
# can't be resolved back to a locally-scoped class that never lives in
# those globals, so FastAPI silently falls back to treating the parameter
# as a query parameter named "body" instead of the JSON request body —
# real annotations (this file's default without that future-import) don't
# have that problem, since the class object itself is already attached
# directly, nothing needs to be resolved from a string.
from second_thought.correction import abstain as do_abstain
from second_thought.correction import accept as do_accept
from second_thought.correction import correct as do_correct
from second_thought.drift import bucketed_evaluation, detect_drift
from second_thought.evaluation import evaluate
from second_thought.schema import DecisionEvent
from second_thought.selection import Strategy, select
from second_thought.storage import Store

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(db_path: str = ".secondthought/store.db") -> "FastAPI":
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "the dashboard needs FastAPI + uvicorn. Install with "
            "`pip install second-thought[dashboard]`."
        ) from exc

    store = Store(db_path)
    app = FastAPI(title="Second Thought")

    class CorrectBody(BaseModel):
        event_id: str
        question_id: str
        value: Any
        reviewer: str | None = None
        note: str | None = None

    class AcceptBody(BaseModel):
        event_id: str
        question_id: str
        reviewer: str | None = None

    class AbstainBody(BaseModel):
        event_id: str
        question_id: str
        reviewer: str | None = None
        note: str | None = None

    def _find(event_id: str) -> DecisionEvent:
        event = store.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail=f"no decision event {event_id!r}")
        return event

    def _row(event: DecisionEvent, question_id: str, score: float | None = None) -> dict[str, Any]:
        prediction = event.predictions[question_id]
        return {
            "event_id": event.id,
            "question_id": question_id,
            "provider": event.provider,
            "workflow": event.workflow,
            "timestamp": event.timestamp,
            "input_state": event.input_state if isinstance(event.input_state, str) else str(event.input_state),
            "instructions": event.questions[question_id].instructions,
            "criteria": event.questions[question_id].criteria,
            "predicted_choice": prediction.choice,
            "predicted_score": prediction.score,
            "predicted_noul": prediction.noul,
            "probabilities": prediction.probabilities,
            "confidence": prediction.confidence,
            "score": score,
            "corrected": question_id in event.corrections,
        }

    @app.get("/api/summary")
    def summary(workflow: str | None = None, provider: str | None = None) -> dict[str, Any]:
        events = list(store.query(workflow=workflow, provider=provider))
        by_workflow: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        for e in events:
            by_workflow[e.workflow or "(none)"] = by_workflow.get(e.workflow or "(none)", 0) + 1
            by_provider[e.provider] = by_provider.get(e.provider, 0) + 1
        report = evaluate(events)
        return {
            "total_events": len(events),
            "corrected_events": sum(1 for e in events if e.has_correction()),
            "by_workflow": by_workflow,
            "by_provider": by_provider,
            "evaluation": {
                "n": report.n, "accuracy": report.accuracy, "mean_brier": report.mean_brier,
                "mean_log_loss": report.mean_log_loss, "ece": report.ece,
            },
        }

    @app.get("/api/queue")
    def queue(
        budget: int = 20, workflow: str | None = None, provider: str | None = None,
        strategy: str = "uncertainty",
    ) -> list[dict[str, Any]]:
        events = list(store.query(workflow=workflow, provider=provider))
        candidates = select(events, strategy=Strategy(strategy), budget=budget)
        return [_row(c.event, c.question_id, score=c.score) for c in candidates]

    @app.post("/api/correct")
    def api_correct(body: CorrectBody) -> dict[str, Any]:
        _find(body.event_id)
        try:
            event = do_correct(
                store, body.event_id, body.question_id, body.value,
                reviewer=body.reviewer, note=body.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _row(event, body.question_id)

    @app.post("/api/accept")
    def api_accept(body: AcceptBody) -> dict[str, Any]:
        _find(body.event_id)
        try:
            event = do_accept(store, body.event_id, body.question_id, reviewer=body.reviewer)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _row(event, body.question_id)

    @app.post("/api/abstain")
    def api_abstain(body: AbstainBody) -> dict[str, Any]:
        _find(body.event_id)
        try:
            event = do_abstain(
                store, body.event_id, body.question_id, reviewer=body.reviewer, note=body.note
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _row(event, body.question_id)

    @app.get("/api/drift")
    def drift(
        metric: str = "ece", bucket_seconds: float = 86400, baseline_windows: int = 3,
        workflow: str | None = None, provider: str | None = None, question_id: str | None = None,
    ) -> dict[str, Any]:
        events = list(store.query(workflow=workflow, provider=provider))
        buckets = bucketed_evaluation(events, bucket_seconds=bucket_seconds, question_id=question_id)
        try:
            report = detect_drift(buckets, metric=metric, baseline_windows=baseline_windows)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "metric": report.metric,
            "control_center": report.control_center,
            "control_limit": report.control_limit,
            "latest_out_of_control": report.latest_out_of_control,
            "points": [
                {
                    "start": p.bucket.start, "end": p.bucket.end, "n": p.bucket.report.n,
                    "value": p.value, "in_control": p.in_control,
                }
                for p in report.points
            ],
        }

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app
