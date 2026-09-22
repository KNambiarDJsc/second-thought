"""A System One-specific evaluation report, built only from corrected (ground-truthed) decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from second_thought.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss,
    selective_accuracy,
)
from second_thought.correction import predicted_value
from second_thought.schema import Correction, DecisionEvent, Prediction, QuestionType


def _distribution_and_true_label(prediction: Prediction, correction: Correction) -> tuple[dict[str, float], str] | None:
    """Reduce any typed-decision kind to (distribution, true_label key) for Brier/log-loss.

    Returns None when the prediction carries no distribution at all (should
    not happen for well-formed Laya/Jev responses, but a malformed or
    hand-constructed record shouldn't crash evaluation).
    """
    if prediction.type == QuestionType.NOUL:
        if prediction.noul is None:
            return None
        truth = bool(correction.value) if not isinstance(correction.value, float) else correction.value > 0.5
        return {"true": prediction.noul, "false": 1.0 - prediction.noul}, ("true" if truth else "false")

    if prediction.probabilities is None:
        return None

    if prediction.type == QuestionType.CHOICE:
        return prediction.probabilities, str(correction.value)

    # SCORE: probabilities are keyed by stringified level index; match the
    # closest numeric key to the corrected value rather than assuming 0- or
    # 1-indexing, since that convention isn't guaranteed across providers.
    truth_value = float(correction.value)
    closest_key = min(prediction.probabilities, key=lambda k: abs(float(k) - truth_value))
    return prediction.probabilities, closest_key


def is_correct(prediction: Prediction, correction: Correction) -> bool:
    predicted = predicted_value(prediction)
    truth = correction.value
    if prediction.type == QuestionType.SCORE:
        return round(float(predicted)) == round(float(truth))
    if prediction.type == QuestionType.NOUL:
        predicted_bool = float(predicted) > 0.5
        truth_bool = bool(truth) if not isinstance(truth, float) else truth > 0.5
        return predicted_bool == truth_bool
    return bool(predicted == truth)


@dataclass(frozen=True)
class EvaluationReport:
    n: int
    accuracy: float | None
    mean_brier: float | None
    mean_log_loss: float | None
    ece: float | None
    selective: dict[float, tuple[float, float]] = field(default_factory=dict)
    """threshold -> (accuracy_on_covered, coverage)"""


def evaluate(
    events: Sequence[DecisionEvent],
    *,
    question_id: str | None = None,
    thresholds: Sequence[float] = (0.5, 0.7, 0.9),
) -> EvaluationReport:
    """Evaluate every corrected (event, question) pair, optionally filtered to one question id."""
    corrects: list[bool] = []
    confidences: list[float] = []
    briers: list[float] = []
    logs: list[float] = []

    for event in events:
        for qid, correction in event.corrections.items():
            if question_id is not None and qid != question_id:
                continue
            prediction = event.predictions[qid]
            corrects.append(is_correct(prediction, correction))
            if prediction.confidence is not None:
                confidences.append(prediction.confidence)
            reduced = _distribution_and_true_label(prediction, correction)
            if reduced is not None:
                dist, true_label = reduced
                if true_label in dist:
                    briers.append(brier_score(dist, true_label))
                    logs.append(log_loss(dist, true_label))

    n = len(corrects)
    if n == 0:
        return EvaluationReport(n=0, accuracy=None, mean_brier=None, mean_log_loss=None, ece=None)

    selective = {}
    if confidences and len(confidences) == n:
        for t in thresholds:
            selective[t] = selective_accuracy(confidences, corrects, t)

    return EvaluationReport(
        n=n,
        accuracy=sum(corrects) / n,
        mean_brier=(sum(briers) / len(briers)) if briers else None,
        mean_log_loss=(sum(logs) / len(logs)) if logs else None,
        ece=expected_calibration_error(confidences, corrects) if len(confidences) == n else None,
        selective=selective,
    )
