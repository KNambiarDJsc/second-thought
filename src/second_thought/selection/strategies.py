"""Sample-selection scoring strategies.

Each strategy scores one ``(DecisionEvent, question_id)`` candidate; higher
score = more worth a human's time to review. All strategies here are
uncertainty-based and computed from a single prediction's own probability
distribution, deliberately excluding anything that needs infrastructure this
project doesn't have yet (an embedding index, an outcome-labeled history, a
cost model). Those are real, useful signals — see ``docs/research/wedge-and-mvp-spec.md``
for why they're out of scope for v0 rather than faked with placeholder math.

The brief this project grew from proposed a multiplicative "Expected Learning
Value" combining five signals. That was deliberately not implemented: there is
no evidence multiplication (vs. a weighted sum, or nothing at all) is the
right combinator, and shipping an unvalidated formula dressed up as a metric
is worse than shipping a plain, legible one. ``composite_uncertainty`` below
is an explicit, documented weighted sum of two signals, not a black box.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from enum import Enum

from second_thought.calibration import margin, normalized_entropy
from second_thought.schema import Prediction


class Strategy(str, Enum):
    RANDOM = "random"
    ENTROPY = "entropy"
    MARGIN = "margin"
    LEAST_CONFIDENCE = "least_confidence"
    UNCERTAINTY = "uncertainty"  # default composite: entropy + inverse margin


ScoreFn = Callable[[Prediction], float]


def _distribution_of(prediction: Prediction) -> dict[str, float]:
    """Every typed-decision kind reduced to a probability distribution.

    ``noul`` has no ``probabilities`` map (see the Laya/Jev schema notes) so
    it's synthesized here as a two-outcome distribution over {true, false}.
    """
    if prediction.probabilities is not None:
        return prediction.probabilities
    if prediction.noul is not None:
        return {"true": prediction.noul, "false": 1.0 - prediction.noul}
    raise ValueError(f"prediction of type {prediction.type!r} has no scoreable distribution")


def score_random(prediction: Prediction) -> float:
    """Placeholder scorer; ``select()`` substitutes a seeded RNG closure for this strategy."""
    return random.random()


def score_entropy(prediction: Prediction) -> float:
    return normalized_entropy(_distribution_of(prediction))


def score_margin(prediction: Prediction) -> float:
    """Higher score = smaller margin = more ambiguous."""
    return 1.0 - margin(_distribution_of(prediction))


def score_least_confidence(prediction: Prediction) -> float:
    dist = _distribution_of(prediction)
    top = max(dist.values())
    return 1.0 - top


def score_uncertainty(prediction: Prediction) -> float:
    """Default composite: mean of normalized entropy and inverse margin.

    Both signals are already normalized to [0, 1], so a plain mean is a
    legible combinator with no fitted weights to justify. Entropy captures
    overall spread; margin specifically catches close two-way calls that
    entropy alone can under-weight in high-cardinality choice questions.
    """
    return (score_entropy(prediction) + score_margin(prediction)) / 2.0


_REGISTRY: dict[Strategy, ScoreFn] = {
    Strategy.RANDOM: score_random,
    Strategy.ENTROPY: score_entropy,
    Strategy.MARGIN: score_margin,
    Strategy.LEAST_CONFIDENCE: score_least_confidence,
    Strategy.UNCERTAINTY: score_uncertainty,
}


def get_scorer(strategy: Strategy) -> ScoreFn:
    return _REGISTRY[strategy]
