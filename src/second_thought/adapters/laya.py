"""Adapter for Laya (https://github.com/NandhaKishorM/laya, Apache-2.0).

Wraps the real, inspected API — not an assumed one:
``laya.load(model_id_or_path=...) -> Agent`` and
``Agent.system_one(state, questions) -> dict`` (aliased ``predict``), returning
``{"model": "laya-rl-agent", "answers": {...}, "usage": {...}}``.

Two things worth knowing before you rely on this adapter in production:

1. ``result["model"]`` from Laya itself is a hardcoded literal
   (``"laya-rl-agent"``) — it does not encode which checkpoint/subfolder
   answered. This adapter captures the actual ``model_id_or_path`` you loaded
   as ``model_version`` instead, so ``DecisionEvent.model_version`` is the
   thing that's actually trustworthy for later analysis.
2. Laya's GitHub repo shows an unusually fast star/fork trajectory relative
   to its creation date. Independent corroboration (a multi-year publishing
   history on the ``convaiinnovations`` Hugging Face org, third-party
   coverage, a separate community ONNX port) supports treating it as a real,
   fast-moving launch rather than fabricated — but pin your ``laya`` version
   and don't take its self-reported benchmark numbers as independently
   verified.
"""

from __future__ import annotations

from typing import Any, ClassVar


class LayaProvider:
    name: ClassVar[str] = "laya"

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        *,
        device: str | None = None,
        token: str | None = None,
        subfolder: str | None = None,
    ) -> None:
        try:
            import laya
        except ImportError as exc:
            raise ImportError(
                "Laya is not installed. Install it with `pip install second-thought[laya]` "
                "or `pip install laya` directly."
            ) from exc

        self._agent = laya.load(
            model_id_or_path=model_id_or_path, device=device, token=token, subfolder=subfolder
        )
        self.model_id_or_path = model_id_or_path

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = self._agent.system_one(state, questions)
        return result

    def model_info(self) -> dict[str, Any]:
        return {"provider": self.name, "model_id_or_path": self.model_id_or_path}
