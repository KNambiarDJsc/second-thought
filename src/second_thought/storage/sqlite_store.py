"""Local-first storage. SQLite, one table, the full record kept as JSON.

Per the project's engineering standard (start local-first, SQLite first,
Postgres only if it's ever demonstrably needed), this is deliberately not an
ORM-backed multi-table schema. Structured columns exist only for what's
queried directly (provider/workflow/timestamp/corrected); everything else is
read back through the Pydantic model, which is the actual source of truth
for the record's shape.

Thread-safety: a ``Store`` is shared, in the intended usage, between whatever
calls ``capture()`` inline from an inference request path (often a threaded
or async server) and whatever calls ``correct()``/``accept()``/etc. from a
review process. A single ``sqlite3.Connection`` is not safe for concurrent
use from multiple threads, so every method here serializes access through
``self._lock``; WAL mode + a ``busy_timeout`` handle the equivalent problem
across separate *processes* sharing the same database file.

Migrations: this table's own structure (columns/indexes) is versioned via
``PRAGMA user_version`` and ``_DB_MIGRATIONS`` below -- distinct from
``DecisionEvent.schema_version``, which versions the JSON payload stored in
the ``record`` column and is migrated on read via ``DecisionEvent.
from_stored_json`` (see ``second_thought.schema``). Two different things can
each change independently: the SQL table shape, and the record shape stored
inside it.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterator
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
CREATE INDEX IF NOT EXISTS idx_decision_events_timestamp ON decision_events(timestamp);
"""

_UPSERT_SQL = (
    "INSERT OR REPLACE INTO decision_events "
    "(id, provider, workflow, timestamp, corrected, record) VALUES (?, ?, ?, ?, ?, ?)"
)

# How long a connection waits on SQLite's write lock before raising
# "database is locked", when a second *process* (not just thread) holds it.
_BUSY_TIMEOUT_MS = 5000

# Ordered SQL migrations for the table/index structure itself, tracked via
# SQLite's own `PRAGMA user_version` (separate from `DecisionEvent.
# schema_version`, which versions the JSON payload in the `record` column,
# not this table's columns/indexes). Each step must be safe to run against
# whatever a prior version of this file already created -- `_SCHEMA`'s own
# `IF NOT EXISTS` covers step 0 for both a brand-new db and one that
# predates this migration mechanism. Append new steps here (e.g. an `ALTER
# TABLE ... ADD COLUMN`) rather than editing an already-shipped one.
def _migration_0_initial_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


_DB_MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _migration_0_initial_schema,
]


def _migrate_db(conn: sqlite3.Connection) -> None:
    (current,) = conn.execute("PRAGMA user_version").fetchone()
    for step in _DB_MIGRATIONS[current:]:
        step(conn)
    target = len(_DB_MIGRATIONS)
    if current < target:
        conn.execute(f"PRAGMA user_version = {target}")
    conn.commit()


def _row(event: DecisionEvent) -> tuple[str, str, str | None, float, int, str]:
    return (
        event.id,
        event.provider,
        event.workflow,
        event.timestamp,
        int(event.fully_corrected()),
        event.to_jsonl_record(),
    )


def _where_clause(
    *,
    provider: str | None,
    workflow: str | None,
    corrected: bool | None,
    since: float | None,
) -> tuple[str, list[str | int | float]]:
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
    return where, params


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Reentrant so `apply()` can be called safely even if its mutator
        # were ever to call back into another lock-taking Store method.
        self._lock = threading.RLock()
        # check_same_thread=False: safety across threads is provided by
        # self._lock, not by pinning the connection to one thread.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        _migrate_db(self._conn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, event: DecisionEvent) -> None:
        with self._lock:
            self._conn.execute(_UPSERT_SQL, _row(event))
            self._conn.commit()

    def get(self, event_id: str) -> DecisionEvent | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record FROM decision_events WHERE id = ?", (event_id,)
            ).fetchone()
        return DecisionEvent.from_stored_json(row[0]) if row else None

    def apply(
        self, event_id: str, mutator: Callable[[DecisionEvent], None]
    ) -> DecisionEvent:
        """Atomically load, mutate, and persist one event.

        A separate ``get()`` + mutate + ``add()`` (the old shape of every
        ``correction.py`` verb) has a read-modify-write race: two callers
        correcting different questions on the same event concurrently can
        each read the pre-mutation record, and whichever ``add()`` commits
        second silently discards the first correction. ``apply()`` closes
        that by running the read, the caller's mutation, and the write
        under one lock and one SQLite write transaction (``BEGIN
        IMMEDIATE``), so concurrent callers — same process or another
        process sharing this database file — serialize instead of racing.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT record FROM decision_events WHERE id = ?", (event_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"no decision event with id {event_id!r}")
                event = DecisionEvent.from_stored_json(row[0])
                mutator(event)
                self._conn.execute(_UPSERT_SQL, _row(event))
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()
        return event

    def query(
        self,
        *,
        provider: str | None = None,
        workflow: str | None = None,
        corrected: bool | None = None,
        since: float | None = None,
        limit: int | None = None,
    ) -> Iterator[DecisionEvent]:
        where, params = _where_clause(
            provider=provider, workflow=workflow, corrected=corrected, since=since
        )
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT record FROM decision_events {where} ORDER BY timestamp ASC {limit_clause}"
        with self._lock:
            # Materialize under the lock rather than streaming the cursor
            # across it: a generator that yields mid-lock would otherwise
            # hold self._lock paused across whatever the caller does with
            # each event, including calling back into the Store.
            rows = self._conn.execute(sql, params).fetchall()
        for (record,) in rows:
            yield DecisionEvent.from_stored_json(record)

    def count(
        self,
        *,
        provider: str | None = None,
        workflow: str | None = None,
        corrected: bool | None = None,
        since: float | None = None,
    ) -> int:
        where, params = _where_clause(
            provider=provider, workflow=workflow, corrected=corrected, since=since
        )
        sql = f"SELECT COUNT(*) FROM decision_events {where}"
        with self._lock:
            (n,) = self._conn.execute(sql, params).fetchone()
        return int(n)
