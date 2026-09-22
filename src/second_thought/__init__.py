"""Second Thought: learning infrastructure for typed probabilistic decisions.

    result = provider.predict(state, questions)
    event = capture(store, result, provider="laya", state=state, questions=questions)

    candidates = select(list(store.query()), budget=25)
    correct(store, candidates[0].event.id, candidates[0].question_id, value="refund")

    report = export(list(store.query()), "out/")
"""

from second_thought.capture import capture, normalize
from second_thought.correction import abstain, accept, correct, flag
from second_thought.datasets import export
from second_thought.evaluation import EvaluationReport, evaluate
from second_thought.schema import (
    Correction,
    DecisionEvent,
    Outcome,
    Prediction,
    QuestionSpec,
    QuestionType,
)
from second_thought.selection import Strategy, select
from second_thought.storage import Store

__all__ = [
    "Correction",
    "DecisionEvent",
    "EvaluationReport",
    "Outcome",
    "Prediction",
    "QuestionSpec",
    "QuestionType",
    "Store",
    "Strategy",
    "abstain",
    "accept",
    "capture",
    "correct",
    "evaluate",
    "export",
    "flag",
    "normalize",
    "select",
]
