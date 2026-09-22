from second_thought.correction import correct
from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.selection import Strategy, select
from second_thought.storage import Store


def _event(event_id: str, probs: dict[str, float]) -> DecisionEvent:
    return DecisionEvent(
        id=event_id,
        provider="laya",
        model="laya-rl-agent",
        input_state=event_id,
        questions={
            "q1": QuestionSpec(type=QuestionType.CHOICE, instructions="pick", criteria=list(probs))
        },
        predictions={
            "q1": Prediction(
                type=QuestionType.CHOICE,
                choice=max(probs, key=lambda k: probs[k]),
                probabilities=probs,
                confidence=max(probs.values()),
            )
        },
    )


def test_uncertainty_ranks_ambiguous_decision_first():
    confident = _event("confident", {"a": 0.98, "b": 0.01, "c": 0.01})
    ambiguous = _event("ambiguous", {"a": 0.34, "b": 0.33, "c": 0.33})
    candidates = select([confident, ambiguous], strategy=Strategy.UNCERTAINTY, budget=2)
    assert candidates[0].event.id == "ambiguous"


def test_corrected_questions_are_excluded(tmp_path):
    store = Store(tmp_path / "s.db")
    event = _event("e1", {"a": 0.5, "b": 0.5})
    store.add(event)
    correct(store, "e1", "q1", "a", reviewer="tester")

    events = list(store.query())
    candidates = select(events, budget=5)
    assert candidates == []
    store.close()


def test_budget_limits_output():
    events = [_event(f"e{i}", {"a": 0.5 + i * 0.01, "b": 0.5 - i * 0.01}) for i in range(10)]
    candidates = select(events, budget=3)
    assert len(candidates) == 3


def test_random_strategy_is_seed_reproducible():
    events = [_event(f"e{i}", {"a": 0.9, "b": 0.1}) for i in range(20)]
    a = select(events, strategy=Strategy.RANDOM, budget=5, seed=42)
    b = select(events, strategy=Strategy.RANDOM, budget=5, seed=42)
    assert [c.event.id for c in a] == [c.event.id for c in b]


def test_diversify_avoids_near_duplicates():
    # Two tight clusters of high-uncertainty points; without diversification
    # the top-budget picks would all come from one cluster.
    events = []
    for i in range(6):
        events.append(_event(f"cluster_a_{i}", {"a": 0.5, "b": 0.5}))
    for i in range(6):
        events.append(_event(f"cluster_b_{i}", {"a": 0.51, "b": 0.49}))

    def embed(event: DecisionEvent) -> list[float]:
        return [0.0, 0.0] if event.id.startswith("cluster_a") else [10.0, 10.0]

    candidates = select(events, strategy=Strategy.UNCERTAINTY, budget=4, diversify_with=embed)
    clusters = {c.event.id.rsplit("_", 1)[0] for c in candidates}
    assert clusters == {"cluster_a", "cluster_b"}
