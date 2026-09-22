import threading

import pytest

from second_thought.correction import correct
from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.storage import Store


def _event(event_id: str, provider: str, workflow: str) -> DecisionEvent:
    return DecisionEvent(
        id=event_id,
        provider=provider,
        model="m",
        workflow=workflow,
        input_state="x",
        questions={"q1": QuestionSpec(type=QuestionType.NOUL, instructions="is it spam?")},
        predictions={"q1": Prediction(type=QuestionType.NOUL, noul=0.8, confidence=0.6)},
    )


def _multi_question_event(event_id: str, n: int) -> DecisionEvent:
    return DecisionEvent(
        id=event_id,
        provider="laya",
        model="m",
        input_state="x",
        questions={
            f"q{i}": QuestionSpec(type=QuestionType.NOUL, instructions="is it spam?")
            for i in range(n)
        },
        predictions={
            f"q{i}": Prediction(type=QuestionType.NOUL, noul=0.5, confidence=0.5)
            for i in range(n)
        },
    )


def test_add_and_get_round_trip(tmp_path):
    store = Store(tmp_path / "s.db")
    event = _event("e1", "laya", "moderation")
    store.add(event)
    assert store.get("e1") == event
    assert store.get("missing") is None
    store.close()


def test_persists_across_reopen(tmp_path):
    db_path = tmp_path / "s.db"
    with Store(db_path) as store:
        store.add(_event("e1", "laya", "moderation"))
    with Store(db_path) as store:
        assert store.get("e1") is not None


def test_query_filters(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event("e1", "laya", "moderation"))
    store.add(_event("e2", "jev", "moderation"))
    store.add(_event("e3", "laya", "triage"))

    assert {e.id for e in store.query(provider="laya")} == {"e1", "e3"}
    assert {e.id for e in store.query(workflow="triage")} == {"e3"}
    assert store.count(provider="jev") == 1
    store.close()


def test_corrected_flag_updates_after_correction(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event("e1", "laya", "moderation"))
    assert store.count(corrected=True) == 0
    correct(store, "e1", "q1", True, reviewer="t")
    assert store.count(corrected=True) == 1
    store.close()


def test_apply_raises_on_missing_event(tmp_path):
    store = Store(tmp_path / "s.db")
    with pytest.raises(KeyError):
        store.apply("missing", lambda event: None)
    store.close()


def test_apply_is_atomic_under_concurrent_threads(tmp_path):
    """Regression test for the read-modify-write race: 20 threads each
    correct a different question on the same event concurrently. Before
    ``Store.apply()``, a get() + mutate + add() shape here would lose most
    of these corrections to last-writer-wins; every one must land."""
    n = 20
    store = Store(tmp_path / "s.db")
    store.add(_multi_question_event("e1", n))

    barrier = threading.Barrier(n)

    def _correct(i: int) -> None:
        barrier.wait()
        correct(store, "e1", f"q{i}", True, reviewer=f"reviewer-{i}")

    threads = [threading.Thread(target=_correct, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    event = store.get("e1")
    assert event is not None
    assert len(event.corrections) == n
    store.close()
