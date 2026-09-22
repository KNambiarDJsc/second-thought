import pytest
from pydantic import ValidationError

from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType


def _event(**overrides):
    defaults = {
        "provider": "laya",
        "model": "laya-rl-agent",
        "input_state": "hello",
        "questions": {
            "q1": QuestionSpec(type=QuestionType.CHOICE, instructions="pick", criteria=["a", "b"])
        },
        "predictions": {
            "q1": Prediction(
                type=QuestionType.CHOICE, choice="a", probabilities={"a": 0.6, "b": 0.4}
            )
        },
    }
    defaults.update(overrides)
    return DecisionEvent(**defaults)


def test_round_trips_through_json():
    event = _event()
    restored = DecisionEvent.model_validate_json(event.to_jsonl_record())
    assert restored == event


def test_has_correction_is_per_question():
    event = _event()
    assert not event.has_correction("q1")
    assert not event.fully_corrected()


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValidationError):
        Prediction(type=QuestionType.CHOICE, choice="a", probabilities={"a": 0.1, "b": 0.1})


def test_noul_prediction_has_no_probabilities_field():
    pred = Prediction(type=QuestionType.NOUL, noul=0.9, confidence=0.9)
    assert pred.probabilities is None
