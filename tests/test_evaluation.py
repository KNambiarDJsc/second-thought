from second_thought.correction import correct
from second_thought.evaluation import evaluate
from second_thought.schema import DecisionEvent, Prediction, QuestionSpec, QuestionType
from second_thought.storage import Store


def _choice_event(event_id: str, choice: str, probs: dict[str, float]) -> DecisionEvent:
    return DecisionEvent(
        id=event_id,
        provider="laya",
        model="m",
        input_state="x",
        questions={
            "q1": QuestionSpec(type=QuestionType.CHOICE, instructions="pick", criteria=list(probs))
        },
        predictions={
            "q1": Prediction(type=QuestionType.CHOICE, choice=choice, probabilities=probs, confidence=max(probs.values()))
        },
    )


def test_perfect_predictions_score_perfectly(tmp_path):
    store = Store(tmp_path / "s.db")
    for i in range(5):
        store.add(_choice_event(f"e{i}", "a", {"a": 0.99, "b": 0.01}))
        correct(store, f"e{i}", "q1", "a", reviewer="t")

    report = evaluate(list(store.query()))
    assert report.accuracy == 1.0
    assert report.mean_brier < 0.01
    store.close()


def test_uncorrected_events_are_ignored(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_choice_event("e1", "a", {"a": 0.9, "b": 0.1}))
    report = evaluate(list(store.query()))
    assert report.n == 0
    assert report.accuracy is None
    store.close()


def test_wrong_prediction_lowers_accuracy(tmp_path):
    store = Store(tmp_path / "s.db")
    store.add(_choice_event("e1", "a", {"a": 0.9, "b": 0.1}))
    correct(store, "e1", "q1", "b", reviewer="t")  # model said a, truth is b
    report = evaluate(list(store.query()))
    assert report.accuracy == 0.0
    assert report.mean_brier > 1.0
    store.close()
