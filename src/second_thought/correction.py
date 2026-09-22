"""The reviewer's four verbs: accept, correct, abstain, flag."""

from __future__ import annotations

from second_thought.schema import Correction, DecisionEvent, Prediction
from second_thought.storage import Store


def predicted_value(prediction: Prediction) -> str | float:
    if prediction.choice is not None:
        return prediction.choice
    if prediction.score is not None:
        return prediction.score
    if prediction.noul is not None:
        return prediction.noul
    raise ValueError(f"prediction of type {prediction.type!r} has no scalar answer to accept")


def accept(
    store: Store, event_id: str, question_id: str, *, reviewer: str | None = None
) -> DecisionEvent:
    """The reviewer agrees with the model. Still recorded as a correction (source='accept')

    so it counts toward the correction dataset's coverage of reviewed decisions."""
    event = _load(store, event_id)
    value = predicted_value(event.predictions[question_id])
    return _apply(store, event, question_id, value, source="accept", reviewer=reviewer)


def correct(
    store: Store,
    event_id: str,
    question_id: str,
    value: object,
    *,
    reviewer: str | None = None,
    note: str | None = None,
) -> DecisionEvent:
    """The reviewer overrides the model's answer."""
    event = _load(store, event_id)
    return _apply(
        store, event, question_id, value, source="human_correction", reviewer=reviewer, note=note
    )


def abstain(
    store: Store,
    event_id: str,
    question_id: str,
    *,
    reviewer: str | None = None,
    note: str | None = None,
) -> DecisionEvent:
    """The reviewer declines to give an authoritative answer.

    Recorded as metadata, not a ``Correction`` — an abstention has no
    training value and must not appear in a dataset export as if it were a
    label. It's still excluded from future selection so it doesn't keep
    resurfacing.
    """
    event = _load(store, event_id)
    abstained = event.metadata.setdefault("abstained_questions", [])
    if question_id not in abstained:
        abstained.append(question_id)
    if note:
        event.metadata.setdefault("notes", {})[question_id] = note
    store.add(event)
    return event


def flag(
    store: Store, event_id: str, question_id: str, note: str, *, reviewer: str | None = None
) -> DecisionEvent:
    """The reviewer marks the decision ambiguous/worth a second opinion, without resolving it.

    Unlike ``abstain``, a flagged item is NOT excluded from future selection —
    the point is to surface it for someone else to look at, not to drop it.
    """
    event = _load(store, event_id)
    flagged = event.metadata.setdefault("flagged_questions", {})
    flagged[question_id] = {"note": note, "reviewer": reviewer}
    store.add(event)
    return event


def _load(store: Store, event_id: str) -> DecisionEvent:
    event = store.get(event_id)
    if event is None:
        raise KeyError(f"no decision event with id {event_id!r}")
    return event


def _apply(
    store: Store,
    event: DecisionEvent,
    question_id: str,
    value: object,
    *,
    source: str,
    reviewer: str | None,
    note: str | None = None,
) -> DecisionEvent:
    if question_id not in event.predictions:
        raise KeyError(f"event {event.id!r} has no question {question_id!r}")
    event.corrections[question_id] = Correction(
        question_id=question_id, value=value, source=source, reviewer=reviewer, note=note
    )
    store.add(event)
    return event
