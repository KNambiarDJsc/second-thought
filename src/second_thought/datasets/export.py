"""Turn corrected decisions into a fine-tune-ready correction corpus.

Every record here has passed through a human review (``accept`` or
``correct``) — see ``second_thought.correction``. Abstentions and
uncorrected decisions are excluded because they carry no ground-truth label.
Every record has also passed the provider policy check in
``second_thought.datasets.policy`` — see that module for why.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from second_thought.datasets.policy import assert_not_explicitly_blocked, is_exportable
from second_thought.schema import DecisionEvent

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ExportReport:
    total_events_seen: int
    excluded_blocked_provider: int
    excluded_uncorrected_questions: int
    written_records: int
    split_counts: dict[SplitName, int]
    paths: dict[str, Path] = field(default_factory=dict)


def _flatten(event: DecisionEvent, question_id: str) -> dict[str, Any]:
    prediction = event.predictions[question_id]
    correction = event.corrections[question_id]
    question = event.questions[question_id]
    return {
        "event_id": event.id,
        "schema_version": event.schema_version,
        "provider": event.provider,
        "model": event.model,
        "model_version": event.model_version,
        "workflow": event.workflow,
        "timestamp": event.timestamp,
        "question_id": question_id,
        "question_type": question.type.value,
        "instructions": question.instructions,
        "criteria": json.dumps(question.criteria),
        "input_state": event.input_state if isinstance(event.input_state, str) else json.dumps(
            event.input_state
        ),
        "predicted_choice": prediction.choice,
        "predicted_score": prediction.score,
        "predicted_noul": prediction.noul,
        "probabilities": json.dumps(prediction.probabilities) if prediction.probabilities else None,
        "confidence": prediction.confidence,
        "correction_value": json.dumps(correction.value)
        if not isinstance(correction.value, str | int | float | bool | type(None))
        else correction.value,
        "correction_source": correction.source,
        "correction_reviewer": correction.reviewer,
        "correction_timestamp": correction.timestamp,
    }


def export(
    events: Sequence[DecisionEvent],
    out_dir: str | Path,
    *,
    providers: Sequence[str] | None = None,
    split: Literal["time", "random"] = "time",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
    formats: Sequence[Literal["jsonl", "parquet"]] = ("jsonl", "parquet"),
    dataset_name: str = "corrections",
) -> ExportReport:
    """Export every corrected, policy-permitted (event, question) pair as a split dataset.

    Splitting is temporal by default (earliest ``train_frac`` of records by
    timestamp go to train, next ``val_frac`` to val, the rest to test) to
    avoid the leakage risk of a random split letting a fine-tuned model
    "see the future" relative to when a correction was actually made.
    """
    if providers is not None:
        assert_not_explicitly_blocked(set(providers))

    excluded_blocked = 0
    excluded_uncorrected = 0
    records: list[dict[str, Any]] = []

    for event in events:
        if providers is not None and event.provider not in providers:
            continue
        if not is_exportable(event.provider):
            excluded_blocked += len(event.predictions)
            continue
        for qid in event.predictions:
            if qid not in event.corrections:
                excluded_uncorrected += 1
                continue
            records.append(_flatten(event, qid))

    if split == "time":
        records.sort(key=lambda r: r["timestamp"])
    else:
        import random

        random.Random(seed).shuffle(records)

    n = len(records)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits: dict[SplitName, list[dict[str, Any]]] = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split_name, split_records in splits.items():
        if "jsonl" in formats:
            p = out_path / f"{dataset_name}.{split_name}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for r in split_records:
                    f.write(json.dumps(r) + "\n")
            paths[f"{split_name}_jsonl"] = p
        if "parquet" in formats:
            p = out_path / f"{dataset_name}.{split_name}.parquet"
            _write_parquet(split_records, p)
            paths[f"{split_name}_parquet"] = p

    return ExportReport(
        total_events_seen=len(events),
        excluded_blocked_provider=excluded_blocked,
        excluded_uncorrected_questions=excluded_uncorrected,
        written_records=n,
        split_counts={k: len(v) for k, v in splits.items()},
        paths=paths,
    )


def _write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        pq.write_table(pa.table({"_empty": []}), path)
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)
