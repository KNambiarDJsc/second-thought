"""Local-first storage. SQLite, one table, the full record kept as JSON.

Per the project's engineering standard (start local-first, SQLite first,
Postgres only if it's ever demonstrably needed), this is deliberately not an
ORM-backed multi-table schema. Structured columns exist only for what's
queried directly (provider/workflow/timestamp/corrected); everything else is
read back through the Pydantic model, which is the actual source of truth
for the record's shape.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Self

from second_thought.schema import DecisionEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_events (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    workflow TEXT,
    timestamp REAL NOT NULL,
    corrected INTEGER NOT NULL DEFAULT 0,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decision_events_provider ON decision_events(provider);
CREATE INDEX IF NOT EXISTS idx_decision_events_workflow ON decision_events(workflow);
CREATE INDEX IF NOT EXISTS idx_decision_events_corrected ON decision_events(corrected);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, event: DecisionEvent) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO decision_events "
            "(id, provider, workflow, timestamp, corrected, record) VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.provider,
                event.workflow,
                event.timestamp,
                int(event.fully_corrected()),
                event.to_jsonl_record(),
            ),
        )
        self._conn.commit()

    def get(self, event_id: str) -> DecisionEvent | None:
        row = self._conn.execute(
            "SELECT record FROM decision_events WHERE id = ?", (event_id,)
        ).fetchone()
        return DecisionEvent.model_validate_json(row[0]) if row else None

    def query(
        self,
        *,
        provider: str | None = None,
        workflow: str | None = None,
        corrected: bool | None = None,
        since: float | None = None,
        limit: int | None = None,
    ) -> Iterator[DecisionEvent]:
        clauses: list[str] = []
        params: list[str | int | float] = []
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if workflow is not None:
            clauses.append("workflow = ?")
            params.append(workflow)
        if corrected is not None:
            clauses.append("corrected = ?")
            params.append(int(corrected))
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT record FROM decision_events {where} ORDER BY timestamp ASC {limit_clause}"
        for (record,) in self._conn.execute(sql, params):
            yield DecisionEvent.model_validate_json(record)

    def count(
        self,
        *,
        provider: str | None = None,
        workflow: str | None = None,
        corrected: bool | None = None,
        since: float | None = None,
    ) -> int:
        return sum(
            1
            for _ in self.query(provider=provider, workflow=workflow, corrected=corrected, since=since)
        )
