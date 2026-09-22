import pytest

from second_thought.correction import abstain, accept, correct, flag
from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.selection import select
from second_thought.storage import Store


def _event() -> DecisionEvent:
    return DecisionEvent(
        id="e1",
        provider="laya",
        model="m",
        input_state="x",
        questions={
            "q1": QuestionSpec(type=QuestionType.CHOICE, instructions="pick", criteria=["a", "b"])
        },
        predictions={
            "q1": Prediction(
                type=QuestionType.CHOICE, choice="a", probabilities={"a": 0.7, "b": 0.3}
            )
        },
    )


def test_accept_records_the_predicted_value(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event())
    event = accept(store, "e1", "q1", reviewer="tester")
    assert event.corrections["q1"].value == "a"
    assert event.corrections["q1"].source == "accept"
    store.close()


def test_correct_overrides_with_given_value(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event())
    event = correct(store, "e1", "q1", "b", reviewer="tester", note="model missed context")
    assert event.corrections["q1"].value == "b"
    assert event.corrections["q1"].note == "model missed context"
    store.close()


def test_abstain_excludes_from_future_selection_without_a_correction(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event())
    event = abstain(store, "e1", "q1", note="ambiguous ticket")
    assert not event.has_correction("q1")
    assert select([event], budget=5) == []
    store.close()


def test_flag_does_not_exclude_from_selection(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event())
    event = flag(store, "e1", "q1", "needs a second opinion")
    assert select([event], budget=5) != []
    store.close()


def test_correct_unknown_question_raises(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event())
    with pytest.raises(KeyError):
        correct(store, "e1", "does-not-exist", "b")
    store.close()
