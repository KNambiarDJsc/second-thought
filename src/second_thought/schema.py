"""The Decision Event schema: a versioned, JSON-serializable record of one typed decision.

Field names deliberately mirror the vocabulary actually used by System One
providers inspected for this project (Laya's ``Agent.system_one`` output and
TypeSafe's documented Jev API): a decision is typed as ``choice``, ``score``,
or ``noul``, and a probabilistic answer carries a ``probabilities`` map plus a
derived ``confidence`` scalar. This is not a generic "LLM trace" schema.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


class QuestionType(str, Enum):
    """The three typed-decision kinds observed across Laya and Jev."""

    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class QuestionSpec(BaseModel):
    """The question as posed to the provider, before any answer exists."""

    type: QuestionType
    instructions: str
    criteria: Any = None
    """Options dict/list for ``choice``, level labels for ``score``, or None for ``noul``."""


class Prediction(BaseModel):
    """One provider answer to one question, in the provider's own typed shape.

    Exactly the fields that exist for each type are populated; the others
    stay ``None``. This mirrors the providers themselves (e.g. Laya's noul
    answers carry no ``probabilities`` map at all) rather than forcing every
    decision type into one flattened shape.
    """

    type: QuestionType
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    legend: dict[str, str] | None = None

    @field_validator("probabilities")
    @classmethod
    def _probabilities_are_a_distribution(
        cls, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        if v is None:
            return v
        total = sum(v.values())
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"probabilities must sum to ~1.0, got {total!r}")
        return v


class Outcome(BaseModel):
    """What actually happened, when it's observable independently of a human review."""

    value: Any
    source: str
    timestamp: float = Field(default_factory=time.time)


class Correction(BaseModel):
    """A human's corrected answer for one question within a decision, with provenance."""

    question_id: str
    value: Any
    source: str
    reviewer: str | None = None
    timestamp: float = Field(default_factory=time.time)
    note: str | None = None


class DecisionEvent(BaseModel):
    """One captured typed decision: what was asked, what came back, what happened next."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: str = SCHEMA_VERSION
    timestamp: float = Field(default_factory=time.time)

    provider: str
    """Which System One provider produced this: e.g. "laya", "jev", or a custom name.

    This field is the anchor point for dataset-export policy — see
    ``second_thought.datasets.policy``. It is deliberately a plain string set
    by the adapter, not a user-settable trust flag.
    """
    model: str
    model_version: str | None = None
    workflow: str | None = None

    input_state: Any
    questions: dict[str, QuestionSpec]
    predictions: dict[str, Prediction]

    outcome: Outcome | None = None
    corrections: dict[str, Correction] = Field(default_factory=dict)
    """Keyed by question_id — an event with several questions may have some corrected and not others."""

    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_correction(self, question_id: str | None = None) -> bool:
        if question_id is not None:
            return question_id in self.corrections
        return len(self.corrections) > 0

    def fully_corrected(self) -> bool:
        return set(self.corrections) == set(self.predictions)

    def to_jsonl_record(self) -> str:
        return self.model_dump_json()
