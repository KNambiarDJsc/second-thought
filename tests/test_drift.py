from second_thought.drift import bucketed_evaluation, detect_drift
from second_thought.schema import Correction, DecisionEvent, Prediction, QuestionSpec, QuestionType

DAY = 86400.0


def _event(event_id: str, timestamp: float, *, correct: bool, confidence: float) -> DecisionEvent:
    """A choice decision, well-calibrated when `correct` matches a high `confidence`
    on the predicted class and a low one spread over the rest."""
    predicted = "a"
    true_label = "a" if correct else "b"
    other = 1.0 - confidence
    event = DecisionEvent(
        id=event_id,
        timestamp=timestamp,
        provider="synthetic",
        model="m",
        input_state="x",
        questions={"q1": QuestionSpec(type=QuestionType.CHOICE, instructions="pick", criteria=["a", "b"])},
        predictions={
            "q1": Prediction(
                type=QuestionType.CHOICE, choice=predicted,
                probabilities={"a": confidence, "b": other}, confidence=confidence,
            )
        },
    )
    event.corrections["q1"] = Correction(question_id="q1", value=true_label, source="human_correction")
    return event


def _stable_bucket(day: int, n: int = 20) -> list[DecisionEvent]:
    """A bucket of decisions with consistently good, stable calibration."""
    return [
        _event(f"d{day}-{i}", day * DAY + i, correct=(i % 10 != 0), confidence=0.85)
        for i in range(n)
    ]


def _degraded_bucket(day: int, n: int = 20) -> list[DecisionEvent]:
    """A bucket where the model is still confident but frequently wrong --
    exactly the "high accuracy, bad calibration" shape this project's real
    Laya run actually found."""
    return [
        _event(f"d{day}-{i}", day * DAY + i, correct=(i % 2 == 0), confidence=0.85)
        for i in range(n)
    ]


def test_bucketed_evaluation_splits_by_time_window():
    events = _stable_bucket(0) + _stable_bucket(1) + _stable_bucket(2)
    buckets = bucketed_evaluation(events, bucket_seconds=DAY)
    assert len(buckets) == 3
    assert all(b.report.n == 20 for b in buckets)
    assert buckets[0].start < buckets[1].start < buckets[2].start


def test_bucketed_evaluation_skips_empty_windows():
    # day 0 and day 5 only -- days 1-4 have no activity and must not appear
    # as zero-filled buckets.
    events = _stable_bucket(0) + _stable_bucket(5)
    buckets = bucketed_evaluation(events, bucket_seconds=DAY)
    assert len(buckets) == 2


def test_bucketed_evaluation_ignores_uncorrected_events():
    events = _stable_bucket(0)
    uncorrected = _event("uncorrected", 0.0, correct=True, confidence=0.9)
    uncorrected.corrections.clear()
    buckets = bucketed_evaluation([*events, uncorrected], bucket_seconds=DAY)
    assert sum(b.report.n for b in buckets) == len(events)


def test_detect_drift_flags_no_drift_when_calibration_is_stable():
    events = []
    for day in range(6):
        events += _stable_bucket(day)
    buckets = bucketed_evaluation(events, bucket_seconds=DAY)
    report = detect_drift(buckets, metric="ece", baseline_windows=3)
    assert not report.latest_out_of_control
    assert report.any_out_of_control == []


def test_detect_drift_flags_a_real_calibration_regression():
    events = []
    for day in range(3):
        events += _stable_bucket(day)  # baseline: well-calibrated
    for day in range(3, 6):
        events += _degraded_bucket(day)  # then: high confidence, frequently wrong
    buckets = bucketed_evaluation(events, bucket_seconds=DAY)
    report = detect_drift(buckets, metric="ece", baseline_windows=3)

    assert len(report.points) == 6
    # baseline buckets (the ones the control band was fit from) read in-control
    assert all(p.in_control for p in report.points[:3])
    # the degraded buckets breach the control band
    assert any(not p.in_control for p in report.points[3:])
    assert report.latest_out_of_control


def test_detect_drift_on_accuracy_flags_a_drop():
    events = []
    for day in range(3):
        events += _stable_bucket(day)
    for day in range(3, 5):
        events += _degraded_bucket(day)
    buckets = bucketed_evaluation(events, bucket_seconds=DAY)
    report = detect_drift(buckets, metric="accuracy", baseline_windows=3)
    assert report.latest_out_of_control


def test_detect_drift_returns_empty_report_with_too_few_buckets():
    events = _stable_bucket(0) + _stable_bucket(1)
    buckets = bucketed_evaluation(events, bucket_seconds=DAY)
    report = detect_drift(buckets, metric="ece", baseline_windows=3)
    assert report.points == []


def test_detect_drift_rejects_unknown_metric():
    import pytest

    with pytest.raises(ValueError, match="unknown metric"):
        detect_drift([], metric="not_a_real_metric")
