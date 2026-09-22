"""The open wire shape second_thought's providers already share — formalized.

Three independent, real implementations of a "typed decision" API already
converge on this exact request/response shape: TypeSafe's Jev (proprietary,
the model that coined "System One"), Laya (Apache-2.0,
huggingface.co/convaiinnovations/laya-typed-decisions), and OpenJev
(Apache-2.0, github.com/razorback16/openjev, wire-compatible with Jev's own
SDK). None of them publish this as an independent, versioned, open schema a
new model builder can implement against without reverse-engineering an
existing provider's API — OpenJev's own README says as much ("not formally
standardized beyond wire compatibility"). This module is that: the shape
``second_thought.schema`` already encodes and has run against real providers
(see ``examples/laya_customer_service/results/``), published as a
conformance checker any developer can use to check whether their own
model's raw JSON response is typed-decision-shaped — with or without
touching the rest of this SDK.

    from second_thought.protocol import validate_response
    result = validate_response(my_raw_json_response)
    print(result.ok, result.errors, result.warnings)

Or from the CLI: ``secondthought protocol validate response.json``,
``secondthought protocol schema`` (prints the JSON Schema).

``PROTOCOL_VERSION`` here tracks this module's understanding of the shape,
not any one vendor's API version. See ``docs/protocol/SPEC.md`` for the full
write-up, provenance for every field, and how to propose a change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from second_thought.schema import QuestionType

PROTOCOL_VERSION = "0.1"

# Fields each answer type is documented to carry. A ``choice``/``score``
# answer with no ``probabilities`` can't be calibration-scored or
# uncertainty-selected by this SDK (or, presumably, by anything else built
# on this shape) — that's a real conformance requirement, not a style
# preference. ``noul`` documented as never carrying ``confidence`` (it's a
# single 0-1 probability already; a second derived scalar would be
# redundant) is a convention, not a hard requirement — deviating is a
# warning, not an error.
_REQUIRED_FIELDS: dict[QuestionType, tuple[str, ...]] = {
    QuestionType.CHOICE: ("choice", "probabilities"),
    QuestionType.SCORE: ("score", "probabilities"),
    QuestionType.NOUL: ("noul",),
}


class RawAnswer(BaseModel):
    """One provider answer to one question, as it appears on the wire — before
    ``second_thought.capture.normalize()`` turns it into a ``Prediction``.

    Deliberately not just an alias for ``Prediction``: this model enforces
    the per-type field-presence rules a wire response must satisfy to be
    usable, which ``Prediction`` (the already-normalized, already-trusted
    internal shape) doesn't need to re-check.
    """

    model_config = ConfigDict(extra="allow")

    type: QuestionType
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    legend: dict[str, str] | None = None

    @model_validator(mode="after")
    def _required_fields_for_type(self) -> RawAnswer:
        missing = [f for f in _REQUIRED_FIELDS[self.type] if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"a {self.type.value!r} answer is missing required field(s) {missing} "
                f"— required for {self.type.value!r}: {list(_REQUIRED_FIELDS[self.type])}"
            )
        return self


class RawResponseEnvelope(BaseModel):
    """The ``{"model", "answers", ...}`` envelope every inspected provider returns."""

    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, RawAnswer]
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_response(raw: dict[str, Any]) -> ValidationResult:
    """Check whether a raw provider response conforms to the typed-decision shape.

    ``errors`` are structural: the response isn't usable by this SDK (or
    plausibly by anything else built on this shape) as-is. ``warnings`` are
    convention deviations that still parse — e.g. a ``noul`` answer that
    also sets ``confidence``, which every inspected provider omits but
    nothing here strictly forbids.
    """
    try:
        envelope = RawResponseEnvelope.model_validate(raw)
    except ValidationError as exc:
        return ValidationResult(ok=False, errors=[_format_error(e) for e in exc.errors()])

    warnings: list[str] = []
    for qid, answer in envelope.answers.items():
        if answer.type is QuestionType.NOUL and answer.confidence is not None:
            warnings.append(
                f"answer {qid!r}: 'noul' answers conventionally omit 'confidence' "
                "(it's already a single 0-1 probability) — every inspected provider "
                "does; this one sets it anyway, which isn't an error but is unusual"
            )
        probs = answer.probabilities
        if probs is not None and not (0.98 <= sum(probs.values()) <= 1.02):
            warnings.append(
                f"answer {qid!r}: probabilities sum to {sum(probs.values())!r}, not ~1.0"
            )

    return ValidationResult(ok=True, warnings=warnings)


def _format_error(e: Mapping[str, Any]) -> str:
    loc = ".".join(str(p) for p in e["loc"])
    return f"{loc}: {e['msg']}" if loc else str(e["msg"])


def json_schema() -> dict[str, Any]:
    """The response envelope's JSON Schema, generated from the same Pydantic
    models ``validate_response`` actually runs — not a hand-maintained copy
    that can silently drift out of sync with the real validator."""
    return RawResponseEnvelope.model_json_schema()


def print_schema() -> None:
    print(json.dumps(json_schema(), indent=2))


QuestionTypeLiteral = Literal["choice", "score", "noul"]
