"""The Decision Event schema: a versioned, JSON-serializable record of one typed decision.

Field names deliberately mirror the vocabulary actually used by System One
providers inspected for this project (Laya's ``Agent.system_one`` output and
TypeSafe's documented Jev API): a decision is typed as ``choice``, ``score``,
or ``noul``, and a probabilistic answer carries a ``probabilities`` map plus a
derived ``confidence`` scalar. This is not a generic "LLM trace" schema.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"

# Forward migrations for the stored JSON record shape, keyed by the
# `schema_version` they migrate *from*. `SCHEMA_VERSION` has no entry here
# because nothing needs to migrate away from the current version -- this
# registry only exists to have somewhere real to put the *next* one, so a
# future schema_version bump doesn't leave every previously-captured row
# permanently unreadable via a bare ValidationError. See
# `DecisionEvent.from_stored_json` and `register_migration`.
_RECORD_MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_migration(
    from_version: str,
) -> Callable[[Callable[[dict[str, Any]], dict[str, Any]]], Callable[[dict[str, Any]], dict[str, Any]]]:
    """Register a function that upgrades a stored record's raw dict one step forward.

    ``fn`` receives the record as a plain dict shaped like ``from_version``
    and must return a dict shaped like the next version (setting its own
    ``schema_version`` key). Migrations chain automatically: registering
    "1.0" -> "1.1" and "1.1" -> "1.2" lets a "1.0" record migrate all the way
    to "1.2" in one ``from_stored_json`` call.
    """

    def _decorator(
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        _RECORD_MIGRATIONS[from_version] = fn
        return fn

    return _decorator


def _migrate_record(raw: dict[str, Any]) -> dict[str, Any]:
    seen: set[str] = set()
    while raw.get("schema_version") != SCHEMA_VERSION:
        version = raw.get("schema_version")
        if not isinstance(version, str):
            raise TypeError(
                f"stored record has no valid schema_version to migrate from: {version!r}"
            )
        if version in seen:
            raise RuntimeError(f"migration cycle detected at schema_version {version!r}")
        migration = _RECORD_MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(
                f"no migration registered from schema_version {version!r} to "
                f"{SCHEMA_VERSION!r} -- a decision event stored under an older schema "
                "can't be read back until one is added via @register_migration"
            )
        seen.add(version)
        raw = migration(raw)
    return raw


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

    @classmethod
    def from_stored_json(cls, data: str) -> DecisionEvent:
        """Deserialize a record from storage, migrating it forward first if it's stale.

        Storage (``Store``) and dataset export both read records back
        through this instead of ``model_validate_json`` directly, so a
        record captured under an older ``schema_version`` doesn't hard-fail
        with a ``ValidationError`` the moment the schema moves on -- it's
        migrated via ``_RECORD_MIGRATIONS`` first.
        """
        raw = json.loads(data)
        if raw.get("schema_version") != SCHEMA_VERSION:
            raw = _migrate_record(raw)
        return cls.model_validate(raw)
