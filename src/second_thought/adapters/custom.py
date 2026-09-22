"""Wrap any typed-decision model as a ``SystemOneProvider`` in one line.

Laya and Jev get named adapters because they're the two concrete System One
models this project actually inspected — not because the SDK is tied to
either. Anything that can answer a typed question with a probability
distribution qualifies: a fine-tuned classifier, a rules engine that outputs
a calibrated confidence, a different vendor's typed-decision API. This class
is the "bring your own model" path, so writing a full class that satisfies
``SystemOneProvider`` structurally is never required for the common case.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FunctionProvider:
    """Adapt a plain function to ``SystemOneProvider`` without writing a class.

        def predict_fn(state, questions):
            ...  # call your model however you like
            return {"model": "my-ticket-classifier-v3", "answers": {...}}

        provider = FunctionProvider("my-ticket-classifier", predict_fn)
        event = capture(store, provider.predict(state, questions), provider=provider.name, ...)

    ``predict_fn`` must return the same envelope Laya and Jev both use:
    ``{"model": ..., "answers": {question_id: {"type": "choice"|"score"|"noul", ...}}}``.
    See ``second_thought.capture.normalize`` for the exact per-type fields expected.
    """

    def __init__(
        self,
        name: str,
        predict_fn: Callable[[Any, dict[str, dict[str, Any]]], dict[str, Any]],
        *,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self._predict_fn = predict_fn
        self._model_info = model_info or {"provider": name}

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return self._predict_fn(state, questions)

    def model_info(self) -> dict[str, Any]:
        return self._model_info
