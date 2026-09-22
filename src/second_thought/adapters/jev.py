"""Adapter for TypeSafe's Jev — capture/observability only. No live client is shipped here.

This project has never held a TypeSafe API key and does not implement an
HTTP client against ``api.typesafe.ai``. Faking that integration would
violate the engineering standard this project holds itself to ("no fake Jev
integration"), so bring your own client (the documented Python SDK, or plain
``requests`` against ``POST https://api.typesafe.ai/v1/systemone``) and hand
this adapter the raw JSON response you already got back.

More importantly: ``JevProvider.name == "jev"`` is the exact string
``second_thought.datasets.policy`` blocks from dataset export, per
TypeSafe's MCA §2.3(b) (no training/distillation on Jev output, no building
a similar/competing product from it). Capturing Jev decisions here is fine
for your own calibration/observability review of your own Jev usage; they
will never appear in an exported training/eval dataset from this project.
"""

from __future__ import annotations

from typing import Any, ClassVar


class JevProvider:
    name: ClassVar[str] = "jev"

    def __init__(self, model_version: str = "jev-latest") -> None:
        self.model_version = model_version

    def capture_response(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Pass through a response you already obtained from your own Jev client.

        This exists only so callers go through one documented entry point
        rather than hand-building the dict ``second_thought.capture.normalize``
        expects; it performs no network I/O and no reshaping.
        """
        return raw_response

    def model_info(self) -> dict[str, Any]:
        return {"provider": self.name, "model_version": self.model_version}
