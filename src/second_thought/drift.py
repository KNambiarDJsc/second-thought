"""Continuous calibration drift detection — the piece this whole ecosystem is missing.

Every calibration tool this project inspected (this project's own ``evaluate()``,
the community's ``jevcal``, ``Janus``) computes accuracy/Brier/ECE as a single
point-in-time snapshot. None of them ask the next question a production
deployment actually needs answered: *is calibration getting worse over time?*
A threshold fit once — ``jevcal``'s own documented "fit + lock file" pattern —
silently goes stale the moment the model, the input distribution, or the
underlying checkpoint drifts, and nothing surveyed watches for that.

This module buckets corrected decisions by time window, evaluates each bucket
independently with the existing ``evaluate()``, and flags a bucket whose
metric falls outside a control band fit from earlier buckets — a Shewhart
control chart, the standard, decades-old technique for exactly this problem
(is a monitored process statistic still behaving like it used to), not a
new formula invented for this project. See ``docs/research/
decision-control-plane.md`` for why this is scoped the way it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean, pstdev

from second_thought.evaluation import EvaluationReport, evaluate
from second_thought.schema import DecisionEvent

DEFAULT_BUCKET_SECONDS = 86400  # one day


@dataclass(frozen=True)
class Bucket:
    start: float
    end: float
    report: EvaluationReport


def bucketed_evaluation(
    events: Sequence[DecisionEvent],
    *,
    bucket_seconds: float = DEFAULT_BUCKET_SECONDS,
    question_id: str | None = None,
) -> list[Bucket]:
    """Evaluate corrected decisions in consecutive time windows instead of all at once.

    Empty windows (no corrected decisions in that span) are omitted rather
    than emitted as zero-filled buckets — a gap in review activity isn't a
    calibration data point.
    """
    corrected = [e for e in events if e.has_correction(question_id)]
    if not corrected:
        return []
    corrected = sorted(corrected, key=lambda e: e.timestamp)

    start = corrected[0].timestamp
    buckets: list[Bucket] = []
    bucket_start = start
    while True:
        bucket_end = bucket_start + bucket_seconds
        window = [e for e in corrected if bucket_start <= e.timestamp < bucket_end]
        if window:
            report = evaluate(window, question_id=question_id)
            buckets.append(Bucket(start=bucket_start, end=bucket_end, report=report))
        if bucket_end > corrected[-1].timestamp:
            break
        bucket_start = bucket_end
    return buckets


DriftMetric = str  # one of "accuracy", "mean_brier", "mean_log_loss", "ece"

# Whether a *rise* in this metric is the bad direction (True) or a *fall* is
# (False) — accuracy dropping is bad; error metrics rising is bad.
_HIGHER_IS_WORSE: dict[str, bool] = {
    "accuracy": False,
    "mean_brier": True,
    "mean_log_loss": True,
    "ece": True,
}


@dataclass(frozen=True)
class DriftPoint:
    bucket: Bucket
    value: float
    in_control: bool
    control_center: float
    control_limit: float


@dataclass(frozen=True)
class DriftReport:
    metric: DriftMetric
    baseline_n_buckets: int
    control_center: float
    control_limit: float
    points: list[DriftPoint] = field(default_factory=list)

    @property
    def latest_out_of_control(self) -> bool:
        return bool(self.points) and not self.points[-1].in_control

    @property
    def any_out_of_control(self) -> list[DriftPoint]:
        return [p for p in self.points if not p.in_control]


def detect_drift(
    buckets: Sequence[Bucket],
    *,
    metric: DriftMetric = "ece",
    baseline_windows: int = 3,
    k: float = 2.0,
) -> DriftReport:
    """Flag buckets whose ``metric`` falls outside a control band fit from the
    first ``baseline_windows`` buckets — mean ± ``k`` standard deviations, the
    textbook Shewhart control-chart band (``k=2`` ~95%, ``k=3`` ~99.7%, under
    a normality assumption this module does not pretend is exact for a
    metric like ECE; it's a legible, standard heuristic, not a rigorous
    hypothesis test — see the module docstring).

    Needs at least ``baseline_windows + 1`` buckets with a non-``None``
    value for ``metric`` to say anything; returns an empty report otherwise.
    """
    if metric not in _HIGHER_IS_WORSE:
        raise ValueError(f"unknown metric {metric!r}; one of {sorted(_HIGHER_IS_WORSE)}")

    values = [(b, getattr(b.report, metric)) for b in buckets]
    values = [(b, v) for b, v in values if v is not None]
    if len(values) <= baseline_windows:
        return DriftReport(metric=metric, baseline_n_buckets=len(values), control_center=0.0, control_limit=0.0)

    baseline = [v for _, v in values[:baseline_windows]]
    center = mean(baseline)
    spread = pstdev(baseline) if len(baseline) > 1 else 0.0
    limit = k * spread
    higher_is_worse = _HIGHER_IS_WORSE[metric]

    points: list[DriftPoint] = []
    for bucket, value in values:
        if higher_is_worse:
            in_control = value <= center + limit
        else:
            in_control = value >= center - limit
        points.append(
            DriftPoint(bucket=bucket, value=value, in_control=in_control, control_center=center, control_limit=limit)
        )

    return DriftReport(
        metric=metric, baseline_n_buckets=baseline_windows, control_center=center, control_limit=limit, points=points
    )
