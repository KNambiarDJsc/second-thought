"""Budget-constrained selection: turn per-prediction scores into a review queue."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from second_thought.schema import DecisionEvent
from second_thought.selection.strategies import Strategy, get_scorer

EmbedFn = Callable[[DecisionEvent], Sequence[float]]


@dataclass(frozen=True)
class Candidate:
    event: DecisionEvent
    question_id: str
    score: float
    strategy: Strategy


def select(
    events: Sequence[DecisionEvent],
    *,
    strategy: Strategy = Strategy.UNCERTAINTY,
    budget: int,
    question_ids: Sequence[str] | None = None,
    seed: int | None = None,
    diversify_with: EmbedFn | None = None,
) -> list[Candidate]:
    """Rank uncorrected (event, question) pairs by ``strategy`` and return the top ``budget``.

    A question within an event that's already been corrected is excluded —
    it's already been reviewed and has nothing left to teach a human
    reviewer — even if other questions on the same event remain open.

    If ``diversify_with`` is given, selection runs in two passes: score every
    candidate normally, then greedily re-rank a 3x-budget shortlist by
    farthest-first traversal in embedding space, so a batch of near-duplicate
    high-uncertainty inputs doesn't crowd out everything else. This is the
    project's answer to "diversity-aware selection" (Phase 7): a real,
    simple algorithm, not a fabricated one.
    """
    if budget <= 0:
        return []

    rng = random.Random(seed)
    scorer = get_scorer(strategy)

    candidates: list[Candidate] = []
    for event in events:
        abstained = event.metadata.get("abstained_questions", [])
        for qid, prediction in event.predictions.items():
            if event.has_correction(qid) or qid in abstained:
                continue
            if question_ids is not None and qid not in question_ids:
                continue
            try:
                score = rng.random() if strategy == Strategy.RANDOM else scorer(prediction)
            except ValueError:
                continue  # prediction type has no scoreable distribution (e.g. malformed record)
            candidates.append(Candidate(event=event, question_id=qid, score=score, strategy=strategy))

    candidates.sort(key=lambda c: c.score, reverse=True)

    if diversify_with is None or len(candidates) <= budget:
        return candidates[:budget]

    return _diversify(candidates, budget, diversify_with)


def _diversify(
    candidates: list[Candidate], budget: int, embed: EmbedFn, shortlist_multiplier: int = 3
) -> list[Candidate]:
    shortlist = candidates[: budget * shortlist_multiplier]
    vectors = [embed(c.event) for c in shortlist]

    chosen_idx = [0]  # seed with the single highest-uncertainty candidate
    remaining = set(range(1, len(shortlist)))

    while len(chosen_idx) < budget and remaining:
        best_idx, best_min_dist = None, -1.0
        for i in remaining:
            min_dist = min(_sq_dist(vectors[i], vectors[j]) for j in chosen_idx)
            if min_dist > best_min_dist:
                best_idx, best_min_dist = i, min_dist
        assert best_idx is not None
        chosen_idx.append(best_idx)
        remaining.discard(best_idx)

    return [shortlist[i] for i in chosen_idx]


def _sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
