"""Normalize a raw provider response into a ``DecisionEvent`` and persist it.

This is the thin instrumentation layer the API design calls for: it does not
call the model itself (adapters or your own client code do that) — it only
turns whatever a System One provider returned into the project's schema.
"""

from __future__ import annotations

from typing import Any

from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.storage import Store


def normalize(
    raw: dict[str, Any],
    *,
    provider: str,
    state: Any,
    questions: dict[str, dict[str, Any]],
    workflow: str | None = None,
    model_version: str | None = None,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> DecisionEvent:
    """Build a ``DecisionEvent`` from a raw ``{"model", "answers", "usage"}`` response.

    ``questions`` is the same dict you passed to the provider's ``predict()``
    call — it's needed because the response alone doesn't repeat the
    question's instructions/criteria.
    """
    predictions = {}
    for qid, answer in raw.get("answers", {}).items():
        predictions[qid] = Prediction(
            type=QuestionType(answer["type"]),
            choice=answer.get("choice"),
            score=answer.get("score"),
            noul=answer.get("noul"),
            probabilities=answer.get("probabilities"),
            confidence=answer.get("confidence"),
            legend=answer.get("legend"),
        )

    question_specs = {
        qid: QuestionSpec(
            type=QuestionType(qdef["type"]),
            instructions=qdef["instructions"],
            criteria=qdef.get("criteria"),
        )
        for qid, qdef in questions.items()
    }

    event_metadata = dict(metadata or {})
    if "usage" in raw:
        event_metadata["usage"] = raw["usage"]
    if "routing" in raw:
        event_metadata["routing"] = raw["routing"]
    if "shortlist" in raw:
        event_metadata["shortlist"] = raw["shortlist"]

    return DecisionEvent(
        provider=provider,
        model=raw.get("model", provider),
        model_version=model_version,
        workflow=workflow,
        input_state=state,
        questions=question_specs,
        predictions=predictions,
        latency_ms=latency_ms,
        metadata=event_metadata,
    )


def capture(
    store: Store,
    raw: dict[str, Any],
    *,
    provider: str,
    state: Any,
    questions: dict[str, dict[str, Any]],
    workflow: str | None = None,
    model_version: str | None = None,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> DecisionEvent:
    """``normalize()`` and persist in one call — the common instrumentation path."""
    event = normalize(
        raw,
        provider=provider,
        state=state,
        questions=questions,
        workflow=workflow,
        model_version=model_version,
        latency_ms=latency_ms,
        metadata=metadata,
    )
    store.add(event)
    return event
