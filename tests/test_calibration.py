import math

from second_thought.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss,
    margin,
    normalized_entropy,
    selective_accuracy,
)


def test_normalized_entropy_extremes():
    assert normalized_entropy({"a": 1.0, "b": 0.0}) == 0.0
    assert math.isclose(normalized_entropy({"a": 0.5, "b": 0.5}), 1.0)


def test_margin():
    assert math.isclose(margin({"a": 0.6, "b": 0.4}), 0.2)
    assert margin({"a": 1.0}) == 1.0


def test_brier_score_perfect_and_worst():
    assert brier_score({"a": 1.0, "b": 0.0}, "a") == 0.0
    assert math.isclose(brier_score({"a": 0.0, "b": 1.0}, "a"), 2.0)


def test_log_loss_confident_correct_is_near_zero():
    assert log_loss({"a": 0.999, "b": 0.001}, "a") < 0.01


def test_ece_perfectly_calibrated_is_zero():
    # 10 predictions at confidence 0.7, 7 correct -> matches its own bucket exactly
    confidences = [0.7] * 10
    correct = [True] * 7 + [False] * 3
    assert expected_calibration_error(confidences, correct, n_bins=10) < 1e-9


def test_selective_accuracy_thresholding():
    confidences = [0.9, 0.9, 0.4, 0.4]
    correct = [True, True, False, True]
    acc, coverage = selective_accuracy(confidences, correct, threshold=0.8)
    assert acc == 1.0
    assert coverage == 0.5
