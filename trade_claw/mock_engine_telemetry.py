"""Last-scan / last-graph snapshot for Streamlit HUD (same SQLite file as mock_trades)."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_claw.task_runtime import MOCK_TRADES_DB_PATH

_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _connect() -> sqlite3.Connection:
    path = Path(MOCK_TRADES_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_telemetry_table() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mock_engine_telemetry (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.commit()


def read_snapshot() -> dict[str, Any]:
    init_telemetry_table()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT payload FROM mock_engine_telemetry WHERE id = 1"
        ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row["payload"])
    except json.JSONDecodeError:
        return {}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def merge_and_save(
    *,
    last_scan: dict[str, Any],
    graph_state: dict[str, Any] | None = None,
) -> None:
    """
    Upsert row id=1. Always refresh `last_scan`. Replace `last_graph` only when graph_state is not None.
    """
    init_telemetry_table()
    prev = read_snapshot()
    payload: dict[str, Any] = {
        "last_scan": _json_safe(last_scan),
        "last_graph": prev.get("last_graph") or {},
    }
    if graph_state is not None:
        payload["last_graph"] = _json_safe(graph_state)
    raw = json.dumps(payload, default=str)
    now = _utc_now_iso()
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO mock_engine_telemetry (id, updated_at, payload)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, payload = excluded.payload
            """,
            (now, raw),
        )
        conn.commit()
