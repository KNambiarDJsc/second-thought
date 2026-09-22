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

    def _mutate(event: DecisionEvent) -> None:
        _record_correction(
            event, question_id, predicted_value(_prediction(event, question_id)),
            source="accept", reviewer=reviewer,
        )

    return store.apply(event_id, _mutate)


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

    def _mutate(event: DecisionEvent) -> None:
        _prediction(event, question_id)  # validates question_id exists
        _record_correction(
            event, question_id, value, source="human_correction", reviewer=reviewer, note=note
        )

    return store.apply(event_id, _mutate)


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

    def _mutate(event: DecisionEvent) -> None:
        abstained = event.metadata.setdefault("abstained_questions", [])
        if question_id not in abstained:
            abstained.append(question_id)
        if note:
            event.metadata.setdefault("notes", {})[question_id] = note

    return store.apply(event_id, _mutate)


def flag(
    store: Store, event_id: str, question_id: str, note: str, *, reviewer: str | None = None
) -> DecisionEvent:
    """The reviewer marks the decision ambiguous/worth a second opinion, without resolving it.

    Unlike ``abstain``, a flagged item is NOT excluded from future selection —
    the point is to surface it for someone else to look at, not to drop it.
    """

    def _mutate(event: DecisionEvent) -> None:
        flagged = event.metadata.setdefault("flagged_questions", {})
        flagged[question_id] = {"note": note, "reviewer": reviewer}

    return store.apply(event_id, _mutate)


def _prediction(event: DecisionEvent, question_id: str) -> Prediction:
    if question_id not in event.predictions:
        raise KeyError(f"event {event.id!r} has no question {question_id!r}")
    return event.predictions[question_id]


def _record_correction(
    event: DecisionEvent,
    question_id: str,
    value: object,
    *,
    source: str,
    reviewer: str | None,
    note: str | None = None,
) -> None:
    event.corrections[question_id] = Correction(
        question_id=question_id, value=value, source=source, reviewer=reviewer, note=note
    )
