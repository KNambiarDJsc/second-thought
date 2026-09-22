"""The provider abstraction every System One adapter implements.

Kept intentionally small. Both Laya's native ``Agent.system_one()`` and
TypeSafe's documented Jev API return the same top-level shape —
``{"model": ..., "answers": {question_id: {...}}, "usage": {...}}`` — so a
single normalization function (``second_thought.capture.normalize``) covers
both without either adapter needing to reshape anything itself. That
convergence, not an assumption, is why the interface is this thin.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable


@runtime_checkable
class SystemOneProvider(Protocol):
    """A typed-decision model this project can capture decisions from.

    ``name`` must be stable and lowercase — it's the value stored in
    ``DecisionEvent.provider`` and the exact string ``datasets.policy``
    checks dataset exports against.
    """

    name: ClassVar[str]

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Run inference and return the raw ``{"model", "answers", "usage"}`` response."""
        ...

    def model_info(self) -> dict[str, Any]:
        """Static metadata about the loaded model/checkpoint, for ``DecisionEvent.model_version``."""
        ...
