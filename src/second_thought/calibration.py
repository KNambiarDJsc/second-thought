"""Calibration and probability-quality metrics for typed decisions.

These operate on plain (probabilities, ground_truth) pairs rather than on
``DecisionEvent`` objects directly, so they're usable both inside the
evaluation module and against ad-hoc data (e.g. an experiment's held-out set)
without constructing full decision records.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def normalized_entropy(probabilities: dict[str, float]) -> float:
    """Shannon entropy of a distribution, normalized to [0, 1] by log(k).

    0 = fully concentrated on one option (certain), 1 = uniform (maximally
    uncertain). This is the same normalization Laya's own ``confidence``
    field is built from (``confidence = 1 - normalized_entropy``); it's
    reimplemented here, independent of any provider, so it applies uniformly
    to any typed-decision distribution this project captures.
    """
    probs = [p for p in probabilities.values() if p > 0]
    k = len(probabilities)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(k)


def margin(probabilities: dict[str, float]) -> float:
    """Gap between the top two probabilities. Small margin = ambiguous decision."""
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) < 2:
        return 1.0
    return ordered[0] - ordered[1]


def brier_score(probabilities: dict[str, float], true_label: str) -> float:
    """Multiclass Brier score: mean squared error between the distribution and a one-hot truth.

    0 = perfect, 2 = worst possible (matches the standard multiclass
    definition, not the binary 0-1 scaled variant).
    """
    if true_label not in probabilities:
        raise ValueError(f"true_label {true_label!r} is not one of the predicted options")
    return sum(
        (p - (1.0 if option == true_label else 0.0)) ** 2 for option, p in probabilities.items()
    )


def log_loss(probabilities: dict[str, float], true_label: str, eps: float = 1e-12) -> float:
    """Negative log-likelihood of the true label under the predicted distribution."""
    if true_label not in probabilities:
        raise ValueError(f"true_label {true_label!r} is not one of the predicted options")
    p = max(probabilities[true_label], eps)
    return -math.log(p)


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    mean_confidence: float
    empirical_accuracy: float
    count: int


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> list[ReliabilityBin]:
    """Bucket (confidence, correct) pairs into equal-width bins for a reliability diagram."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    bins: list[ReliabilityBin] = []
    width = 1.0 / n_bins
    for i in range(n_bins):
        lo, hi = i * width, (i + 1) * width
        idx = [j for j, c in enumerate(confidences) if (lo <= c < hi) or (hi == 1.0 and c == 1.0)]
        if not idx:
            continue
        mean_conf = sum(confidences[j] for j in idx) / len(idx)
        acc = sum(1 for j in idx if correct[j]) / len(idx)
        bins.append(ReliabilityBin(lo, hi, mean_conf, acc, len(idx)))
    return bins


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> float:
    """ECE: the count-weighted mean absolute gap between confidence and empirical accuracy."""
    bins = reliability_bins(confidences, correct, n_bins)
    total = len(confidences)
    if total == 0:
        return 0.0
    return sum(b.count * abs(b.mean_confidence - b.empirical_accuracy) for b in bins) / total


def selective_accuracy(
    confidences: Sequence[float], correct: Sequence[bool], threshold: float
) -> tuple[float, float]:
    """Accuracy and coverage when only acting on decisions at or above ``threshold`` confidence.

    Returns (accuracy_on_covered, coverage). Coverage is the fraction of all
    decisions that clear the threshold; accuracy is undefined (returned as
    1.0 by convention) when nothing clears it.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    covered = [c for conf, c in zip(confidences, correct, strict=True) if conf >= threshold]
    coverage = len(covered) / len(confidences) if confidences else 0.0
    if not covered:
        return 1.0, 0.0
    accuracy = sum(covered) / len(covered)
    return accuracy, coverage
