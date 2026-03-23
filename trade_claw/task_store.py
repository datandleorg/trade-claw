"""SQLite persistence for user task metadata (API + workers)."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_claw.task_runtime import TASK_DB_PATH

_lock = threading.Lock()

# Statuses where no Celery revoke / stop key is needed for housekeeping.
TASK_TERMINAL_STATUSES = frozenset({"stopped", "completed", "failed"})


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _connect() -> sqlite3.Connection:
    path = Path(TASK_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_events_task_id ON task_events (task_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_events_task_id_id ON task_events (task_id, id)"
        )
        conn.commit()


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    name: str
    status: str
    payload: dict[str, Any] | None
    created_at: str
    updated_at: str


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    raw = row["payload"]
    payload = json.loads(raw) if raw else None
    return TaskRecord(
        task_id=row["task_id"],
        name=row["name"],
        status=row["status"],
        payload=payload,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_task(
    *,
    name: str,
    status: str = "pending",
    payload: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> str:
    init_db()
    tid = task_id or str(uuid.uuid4())
    now = _utc_now_iso()
    payload_json = json.dumps(payload) if payload else None
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks (task_id, name, status, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (tid, name, status, payload_json, now, now),
        )
        conn.commit()
    return tid


def update_status(task_id: str, status: str) -> bool:
    init_db()
    now = _utc_now_iso()
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
            (status, now, task_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_task(task_id: str) -> TaskRecord | None:
    init_db()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT task_id, name, status, payload, created_at, updated_at FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    return _row_to_record(row) if row else None


def list_tasks(limit: int = 100) -> list[TaskRecord]:
    init_db()
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT task_id, name, status, payload, created_at, updated_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def list_all_tasks(max_rows: int = 50_000) -> list[TaskRecord]:
    """All task rows (bounded) for admin stop/purge."""
    init_db()
    cap = min(max(max_rows, 1), 500_000)
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT task_id, name, status, payload, created_at, updated_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (cap,),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def all_task_ids() -> list[str]:
    init_db()
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT task_id FROM tasks").fetchall()
    return [str(r["task_id"]) for r in rows]


def purge_all() -> tuple[int, int]:
    """Delete all events then all tasks. Returns (events_deleted, tasks_deleted)."""
    init_db()
    with _lock, _connect() as conn:
        ev = conn.execute("DELETE FROM task_events").rowcount
        tk = conn.execute("DELETE FROM tasks").rowcount
        conn.commit()
    return int(ev), int(tk)


@dataclass(frozen=True)
class TaskEventRecord:
    id: int
    task_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str


def append_task_event(task_id: str, data: dict[str, Any]) -> int:
    """Persist one event (same payload as Redis/SSE). Returns row id."""
    init_db()
    event_type = str(data.get("type", "unknown"))
    now = _utc_now_iso()
    payload_json = json.dumps(data, default=str)
    with _lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO task_events (task_id, event_type, payload, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (task_id, event_type, payload_json, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_task_events(
    task_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
    order: str = "asc",
    event_type: str | None = None,
) -> list[TaskEventRecord]:
    init_db()
    limit = min(max(limit, 1), 5000)
    offset = max(offset, 0)
    order_sql = "DESC" if order.lower() == "desc" else "ASC"
    with _lock, _connect() as conn:
        if event_type:
            rows = conn.execute(
                f"""
                SELECT id, task_id, event_type, payload, created_at
                FROM task_events
                WHERE task_id = ? AND event_type = ?
                ORDER BY id {order_sql}
                LIMIT ? OFFSET ?
                """,
                (task_id, event_type, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT id, task_id, event_type, payload, created_at
                FROM task_events
                WHERE task_id = ?
                ORDER BY id {order_sql}
                LIMIT ? OFFSET ?
                """,
                (task_id, limit, offset),
            ).fetchall()
    out: list[TaskEventRecord] = []
    for row in rows:
        out.append(
            TaskEventRecord(
                id=row["id"],
                task_id=row["task_id"],
                event_type=row["event_type"],
                payload=json.loads(row["payload"]),
                created_at=row["created_at"],
            )
        )
    return out


def count_task_events(task_id: str, *, event_type: str | None = None) -> int:
    init_db()
    with _lock, _connect() as conn:
        if event_type:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM task_events WHERE task_id = ? AND event_type = ?",
                (task_id, event_type),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
    return int(row["c"]) if row else 0
