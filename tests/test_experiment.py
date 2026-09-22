"""Mechanical smoke test for the experiment runner — fast and small.

The scientific claim (uncertainty beats random) is validated by
``examples/toy_typed_decisions/run.py`` at a scale that actually
demonstrates it; this test only checks the plumbing doesn't break, using a
handful of samples so it runs in well under a second.
"""

import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from second_thought.experiments import run_experiment
from second_thought.selection import Strategy


def test_experiment_runs_and_labels_only_grow():
    x, y = make_classification(
        n_samples=120, n_features=5, n_informative=3, n_classes=3, random_state=0
    )
    x_pool, x_test, y_pool, y_test = train_test_split(
        x, y, test_size=30, stratify=y, random_state=0
    )
    results = run_experiment(
        x_pool,
        y_pool,
        x_test,
        y_test,
        seed_size=9,
        budget_per_round=5,
        rounds=2,
        seed=0,
        strategies=(Strategy.RANDOM, Strategy.UNCERTAINTY),
    )
    by_strategy: dict[str, list] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    for rounds in by_strategy.values():
        n_labels = [r.n_labels for r in rounds]
        assert n_labels == sorted(n_labels)  # monotonically non-decreasing
        assert all(0.0 <= r.test_accuracy <= 1.0 for r in rounds)
