import json

import pytest

from second_thought.correction import accept, correct
from second_thought.datasets import BlockedProviderError, export
from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.storage import Store


def _event(event_id: str, provider: str, timestamp: float) -> DecisionEvent:
    return DecisionEvent(
        id=event_id,
        provider=provider,
        model="m",
        timestamp=timestamp,
        input_state="x",
        questions={
            "q1": QuestionSpec(type=QuestionType.CHOICE, instructions="pick", criteria=["a", "b"])
        },
        predictions={
            "q1": Prediction(
                type=QuestionType.CHOICE, choice="a", probabilities={"a": 0.6, "b": 0.4}
            )
        },
    )


def test_only_corrected_questions_are_exported(tmp_path):
    store = Store(tmp_path / "s.db")
    corrected = _event("e1", "laya", 1.0)
    uncorrected = _event("e2", "laya", 2.0)
    store.add(corrected)
    store.add(uncorrected)
    correct(store, "e1", "q1", "b", reviewer="t")

    report = export(list(store.query()), tmp_path / "out", formats=("jsonl",))
    assert report.written_records == 1
    assert report.excluded_uncorrected_questions == 1
    store.close()


def test_jev_is_silently_excluded_by_default(tmp_path):
    store = Store(tmp_path / "s.db")
    laya_event = _event("e1", "laya", 1.0)
    jev_event = _event("e2", "jev", 2.0)
    store.add(laya_event)
    store.add(jev_event)
    accept(store, "e1", "q1")
    accept(store, "e2", "q1")

    report = export(list(store.query()), tmp_path / "out", formats=("jsonl",))
    assert report.written_records == 1
    assert report.excluded_blocked_provider == 1
    store.close()


def test_explicit_jev_export_request_raises(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_event("e1", "jev", 1.0))
    accept(store, "e1", "q1")

    with pytest.raises(BlockedProviderError):
        export(list(store.query()), tmp_path / "out", providers=["jev"], formats=("jsonl",))
    store.close()


def test_time_split_puts_earliest_records_in_train(tmp_path):
    store = Store(tmp_path / "s.db")
    for i in range(10):
        store.add(_event(f"e{i}", "laya", float(i)))
        accept(store, f"e{i}", "q1")

    report = export(
        list(store.query()), tmp_path / "out", split="time", train_frac=0.7, val_frac=0.1
    )
    assert report.split_counts == {"train": 7, "val": 1, "test": 2}

    train_records = [
        json.loads(line)
        for line in (tmp_path / "out" / "corrections.train.jsonl").read_text().splitlines()
    ]
    assert {r["event_id"] for r in train_records} == {f"e{i}" for i in range(7)}
    store.close()
