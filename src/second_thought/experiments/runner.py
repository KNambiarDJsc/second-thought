"""The central experiment: does Second Thought's selection beat random selection?

This dogfoods the actual shipped code — ``second_thought.selection.select()``
and ``second_thought.evaluation.evaluate()`` — against a synthetic
classification pool, rather than hand-rolling a separate comparison. Every
number this produces is computed, not asserted; see
``examples/toy_typed_decisions/run.py`` for the runnable entry point and
``docs/research/wedge-and-mvp-spec.md`` for why a synthetic pool stands in
for a real Laya fine-tuning loop in this repository (compute/time, not
principle).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from second_thought.evaluation import evaluate
from second_thought.schema import Correction, DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.selection import Strategy, select

if TYPE_CHECKING:
    # Every reference to `np` in this module is a type annotation (parameter,
    # return, or local-variable), never evaluated at runtime because of the
    # `from __future__ import annotations` above — so this module has no
    # actual runtime dependency on numpy being installed; it only operates on
    # whatever ndarray-like objects a caller (which does depend on numpy,
    # e.g. via scikit-learn) passes in. Keeping the import real-but-deferred
    # here, rather than dropping it, keeps mypy/IDE type-checking exact. See
    # the `experiments` extra in pyproject.toml for what running an actual
    # experiment (as opposed to just importing this module) requires.
    import numpy as np

QUESTION_ID = "label"


def _make_event(
    idx: int | str, features: np.ndarray, probabilities: dict[str, float]
) -> DecisionEvent:
    top_choice = max(probabilities, key=lambda k: probabilities[k])
    return DecisionEvent(
        id=str(idx),
        provider="toy-classifier",
        model="toy-logistic-regression",
        input_state={"features": features.tolist()},
        questions={
            QUESTION_ID: QuestionSpec(
                type=QuestionType.CHOICE,
                instructions="classify the input",
                criteria=list(probabilities),
            )
        },
        predictions={
            QUESTION_ID: Prediction(
                type=QuestionType.CHOICE,
                choice=top_choice,
                probabilities=probabilities,
                confidence=max(probabilities.values()),
            )
        },
    )


def _predict_probabilities(clf: object, x_rows: np.ndarray, classes: list[str]) -> list[dict[str, float]]:
    proba = clf.predict_proba(x_rows)  # type: ignore[attr-defined]
    return [dict(zip(classes, row.tolist(), strict=True)) for row in proba]


@dataclass(frozen=True)
class RoundResult:
    strategy: str
    round: int
    n_labels: int
    test_accuracy: float
    test_mean_brier: float | None
    test_ece: float | None


def run_experiment(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed_size: int = 20,
    budget_per_round: int = 10,
    rounds: int = 6,
    seed: int = 0,
    strategies: Sequence[Strategy] = (Strategy.RANDOM, Strategy.UNCERTAINTY),
    diversify: bool = False,
) -> list[RoundResult]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "run_experiment() needs scikit-learn (and numpy, installed with it). "
            "Install with `pip install second-thought[experiments]`."
        ) from exc

    results: list[RoundResult] = []
    features_by_id: dict[str, np.ndarray] = {str(i): x_pool[i] for i in range(len(x_pool))}

    def _embed(event: DecisionEvent) -> list[float]:
        return features_by_id[event.id].tolist()  # type: ignore[no-any-return]

    for strategy in strategies:
        labeled_idx, _ = train_test_split(
            list(range(len(x_pool))),
            train_size=seed_size,
            stratify=y_pool,
            random_state=seed,
        )
        labeled_idx = set(labeled_idx)
        unlabeled_idx = set(range(len(x_pool))) - labeled_idx

        for round_num in range(rounds + 1):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(x_pool[list(labeled_idx)], y_pool[list(labeled_idx)])
            classes = [str(c) for c in clf.classes_]

            test_probs = _predict_probabilities(clf, x_test, classes)
            test_events = []
            for i, probs in enumerate(test_probs):
                event = _make_event(f"test-{i}", x_test[i], probs)
                event.corrections[QUESTION_ID] = Correction(
                    question_id=QUESTION_ID, value=str(y_test[i]), source="ground_truth"
                )
                test_events.append(event)
            report = evaluate(test_events)

            results.append(
                RoundResult(
                    strategy=strategy.value,
                    round=round_num,
                    n_labels=len(labeled_idx),
                    test_accuracy=report.accuracy or 0.0,
                    test_mean_brier=report.mean_brier,
                    test_ece=report.ece,
                )
            )

            if round_num == rounds or not unlabeled_idx:
                break

            pool_list = sorted(unlabeled_idx)
            pool_probs = _predict_probabilities(clf, x_pool[pool_list], classes)
            pool_events = [_make_event(idx, x_pool[idx], probs) for idx, probs in zip(pool_list, pool_probs, strict=True)]

            chosen = select(
                pool_events,
                strategy=strategy,
                budget=min(budget_per_round, len(pool_events)),
                seed=seed + round_num,
                diversify_with=_embed if diversify else None,
            )
            for candidate in chosen:
                idx = int(candidate.event.id)
                labeled_idx.add(idx)
                unlabeled_idx.discard(idx)

    return results


def summarize(results: list[RoundResult]) -> str:
    lines = ["strategy,round,n_labels,test_accuracy,test_mean_brier,test_ece"]
    for r in results:
        lines.append(
            f"{r.strategy},{r.round},{r.n_labels},{r.test_accuracy:.4f},"
            f"{r.test_mean_brier if r.test_mean_brier is not None else ''},"
            f"{r.test_ece if r.test_ece is not None else ''}"
        )
    return "\n".join(lines)
