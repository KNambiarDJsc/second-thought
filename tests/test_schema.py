import json

import pytest
from pydantic import ValidationError

import second_thought.schema as schema_module
from second_thought.schema import (
    SCHEMA_VERSION,
    DecisionEvent,
    Prediction,
    QuestionSpec,
    QuestionType,
    register_migration,
)


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


def test_from_stored_json_round_trips_a_current_version_record():
    event = _event()
    restored = DecisionEvent.from_stored_json(event.to_jsonl_record())
    assert restored == event


def test_from_stored_json_migrates_a_record_from_a_registered_old_version():
    """Proves the migration mechanism actually runs, not just that it exists:

    registers a real migration function, stores a record shaped like an
    older schema_version, and confirms from_stored_json() upgrades it
    instead of raising a ValidationError."""

    @register_migration("0.9-test")
    def _upgrade(raw: dict) -> dict:
        raw = dict(raw)
        raw["schema_version"] = SCHEMA_VERSION
        raw["metadata"] = {**raw.get("metadata", {}), "migrated_from": "0.9-test"}
        return raw

    try:
        old_record = json.loads(_event().to_jsonl_record())
        old_record["schema_version"] = "0.9-test"

        migrated = DecisionEvent.from_stored_json(json.dumps(old_record))

        assert migrated.schema_version == SCHEMA_VERSION
        assert migrated.metadata["migrated_from"] == "0.9-test"
        # everything else about the event survived the round trip
        assert migrated.provider == "laya"
        assert migrated.predictions["q1"].choice == "a"
    finally:
        schema_module._RECORD_MIGRATIONS.pop("0.9-test", None)


def test_from_stored_json_raises_clearly_when_no_migration_is_registered():
    old_record = json.loads(_event().to_jsonl_record())
    old_record["schema_version"] = "0.1-nonexistent"

    with pytest.raises(ValueError, match="no migration registered"):
        DecisionEvent.from_stored_json(json.dumps(old_record))
