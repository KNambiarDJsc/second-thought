"""Normalize a raw provider response into a ``DecisionEvent`` and persist it.

This is the thin instrumentation layer the API design calls for: it does not
call the model itself (adapters or your own client code do that) — it only
turns whatever a System One provider returned into the project's schema.
"""

from __future__ import annotations

import logging
from typing import Any

from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.storage import Store

_logger = logging.getLogger("second_thought.capture")


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
    strict: bool = False,
) -> DecisionEvent | None:
    """``normalize()`` and persist in one call — the common instrumentation path.

    Instrumentation must not be able to take down the request path it
    observes. By default (``strict=False``), a malformed ``raw`` response —
    a missing key, an invalid ``QuestionType``, a ``Prediction`` failing its
    probabilities-sum-to-~1 check, or anything else ``normalize()`` can raise
    on — is logged through the ``second_thought.capture`` logger and
    ``capture()`` returns ``None`` instead of letting that error propagate
    into whatever called this, which in the intended usage (inline from an
    inference request) would otherwise mean one bad model response crashes
    the request it's only meant to be observing.

    Pass ``strict=True`` — recommended for tests, backfills, and CLI bulk
    loads where you want to fail loudly on bad data — to let the underlying
    error raise instead.
    """
    try:
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
    except Exception:
        if strict:
            raise
        _logger.exception(
            "second_thought.capture: dropping a malformed %r response instead of "
            "raising into the caller (pass strict=True to raise instead)",
            provider,
        )
        return None
    store.add(event)
    return event
