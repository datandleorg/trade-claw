"""SQLite WAL persistence for autonomous mock option trades (Celery + Streamlit reads)."""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_claw.task_runtime import MOCK_TRADES_DB_PATH

logger = logging.getLogger(__name__)
_lock = threading.Lock()

_SELECT_ROW = """
    trade_id, entry_time, exit_time, instrument, direction, entry_price,
    stop_loss, target, llm_rationale, status, exit_price, realized_pnl,
    lot_size, quantity, index_underlying, entry_bars_json, exit_bars_json,
    entry_underlying_bars_json, exit_underlying_bars_json
"""


def _utc_now_sql() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _connect() -> sqlite3.Connection:
    path = Path(MOCK_TRADES_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _normalize_index_underlying(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    return s or None


def _migrate_mock_trades(conn: sqlite3.Connection) -> None:
    info = conn.execute("PRAGMA table_info(mock_trades)").fetchall()
    cols = {str(r[1]) for r in info}
    if "entry_bars_json" not in cols:
        conn.execute("ALTER TABLE mock_trades ADD COLUMN entry_bars_json TEXT")
    if "exit_bars_json" not in cols:
        conn.execute("ALTER TABLE mock_trades ADD COLUMN exit_bars_json TEXT")
    if "entry_underlying_bars_json" not in cols:
        conn.execute("ALTER TABLE mock_trades ADD COLUMN entry_underlying_bars_json TEXT")
    if "exit_underlying_bars_json" not in cols:
        conn.execute("ALTER TABLE mock_trades ADD COLUMN exit_underlying_bars_json TEXT")
    if "index_underlying" not in cols:
        conn.execute("ALTER TABLE mock_trades ADD COLUMN index_underlying TEXT")


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mock_trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                exit_time DATETIME,
                instrument TEXT,
                direction TEXT,
                entry_price REAL,
                stop_loss REAL,
                target REAL,
                llm_rationale TEXT,
                status TEXT,
                exit_price REAL,
                realized_pnl REAL,
                lot_size INTEGER NOT NULL DEFAULT 1,
                quantity INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        _migrate_mock_trades(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mock_trades_status ON mock_trades (status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mock_trades_entry_time ON mock_trades (entry_time)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mock_trades_exit_time ON mock_trades (exit_time)"
        )
        try:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_mock_trades_one_open_per_index
                ON mock_trades(index_underlying)
                WHERE status = 'OPEN'
                  AND index_underlying IS NOT NULL
                  AND TRIM(index_underlying) != ''
                """
            )
        except sqlite3.OperationalError as e:
            logger.warning(
                "Could not create idx_mock_trades_one_open_per_index (duplicate OPEN rows for same index?): %s",
                e,
            )
        conn.commit()


@dataclass(frozen=True)
class MockTradeRow:
    trade_id: int
    entry_time: str | None
    exit_time: str | None
    instrument: str | None
    direction: str | None
    entry_price: float | None
    stop_loss: float | None
    target: float | None
    llm_rationale: str | None
    status: str | None
    exit_price: float | None
    realized_pnl: float | None
    lot_size: int
    quantity: int
    index_underlying: str | None = None
    entry_bars_json: str | None = None
    exit_bars_json: str | None = None
    entry_underlying_bars_json: str | None = None
    exit_underlying_bars_json: str | None = None


def _row(r: sqlite3.Row) -> MockTradeRow:
    keys = r.keys()
    return MockTradeRow(
        trade_id=int(r["trade_id"]),
        entry_time=r["entry_time"],
        exit_time=r["exit_time"],
        instrument=r["instrument"],
        direction=r["direction"],
        entry_price=r["entry_price"],
        stop_loss=r["stop_loss"],
        target=r["target"],
        llm_rationale=r["llm_rationale"],
        status=r["status"],
        exit_price=r["exit_price"],
        realized_pnl=r["realized_pnl"],
        lot_size=int(r["lot_size"] or 1),
        quantity=int(r["quantity"] or 1),
        index_underlying=r["index_underlying"] if "index_underlying" in keys else None,
        entry_bars_json=r["entry_bars_json"] if "entry_bars_json" in keys else None,
        exit_bars_json=r["exit_bars_json"] if "exit_bars_json" in keys else None,
        entry_underlying_bars_json=(
            r["entry_underlying_bars_json"] if "entry_underlying_bars_json" in keys else None
        ),
        exit_underlying_bars_json=(
            r["exit_underlying_bars_json"] if "exit_underlying_bars_json" in keys else None
        ),
    )


def insert_open_trade(
    *,
    instrument: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    llm_rationale: str,
    lot_size: int,
    quantity: int,
    entry_bars_json: str | None = None,
    index_underlying: str | None = None,
) -> int:
    init_db()
    iu = _normalize_index_underlying(index_underlying)
    if iu and has_open_trade_for_underlying(iu):
        raise ValueError(f"OPEN mock trade already exists for index_underlying={iu!r}")
    now = _utc_now_sql()
    with _lock, _connect() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO mock_trades (
                    entry_time, instrument, direction, entry_price, stop_loss, target,
                    llm_rationale, status, lot_size, quantity, index_underlying, entry_bars_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
                """,
                (
                    now,
                    instrument,
                    direction,
                    entry_price,
                    stop_loss,
                    target,
                    llm_rationale,
                    lot_size,
                    quantity,
                    iu,
                    entry_bars_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise ValueError(
                f"OPEN mock trade already exists for index_underlying={iu!r} (unique index)"
            ) from e


def update_entry_bars_json(trade_id: int, entry_bars_json: str | None) -> bool:
    init_db()
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE mock_trades SET entry_bars_json = ? WHERE trade_id = ?",
            (entry_bars_json, trade_id),
        )
        conn.commit()
        return cur.rowcount > 0


def update_entry_underlying_bars_json(trade_id: int, entry_underlying_bars_json: str | None) -> bool:
    init_db()
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE mock_trades SET entry_underlying_bars_json = ? WHERE trade_id = ?",
            (entry_underlying_bars_json, trade_id),
        )
        conn.commit()
        return cur.rowcount > 0


def close_trade(
    trade_id: int,
    *,
    exit_price: float,
    realized_pnl: float,
    exit_bars_json: str | None = None,
    exit_underlying_bars_json: str | None = None,
) -> bool:
    init_db()
    now = _utc_now_sql()
    with _lock, _connect() as conn:
        set_parts = [
            "exit_time = ?",
            "exit_price = ?",
            "realized_pnl = ?",
            "status = 'CLOSED'",
        ]
        vals: list[Any] = [now, exit_price, realized_pnl]
        if exit_bars_json is not None:
            set_parts.append("exit_bars_json = ?")
            vals.append(exit_bars_json)
        if exit_underlying_bars_json is not None:
            set_parts.append("exit_underlying_bars_json = ?")
            vals.append(exit_underlying_bars_json)
        vals.append(trade_id)
        sql = f"UPDATE mock_trades SET {', '.join(set_parts)} WHERE trade_id = ? AND status = 'OPEN'"
        cur = conn.execute(sql, vals)
        conn.commit()
        return cur.rowcount > 0


def list_open_trades() -> list[MockTradeRow]:
    init_db()
    with _lock, _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {_SELECT_ROW.strip()}
            FROM mock_trades
            WHERE status = 'OPEN'
            ORDER BY trade_id
            """
        ).fetchall()
    return [_row(r) for r in rows]


def has_open_trade() -> bool:
    return len(list_open_trades()) > 0


def sum_realized_pnl_closed() -> float:
    """Sum of ``realized_pnl`` for all CLOSED rows (full ledger, not a page limit)."""
    init_db()
    with _lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0.0)
            FROM mock_trades
            WHERE status = 'CLOSED' AND realized_pnl IS NOT NULL
            """
        ).fetchone()
    return float(row[0] if row and row[0] is not None else 0.0)


def has_open_trade_for_underlying(index_key: str) -> bool:
    """True if any row is OPEN for this index key (normalized uppercase)."""
    iu = _normalize_index_underlying(index_key)
    if not iu:
        return False
    init_db()
    with _lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM mock_trades
            WHERE status = 'OPEN' AND UPPER(TRIM(COALESCE(index_underlying, ''))) = ?
            LIMIT 1
            """,
            (iu,),
        ).fetchone()
    return row is not None


def list_recent_trades(*, limit: int = 200) -> list[MockTradeRow]:
    init_db()
    limit = min(max(limit, 1), 2000)
    with _lock, _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {_SELECT_ROW.strip()}
            FROM mock_trades
            ORDER BY trade_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row(r) for r in rows]


def list_trades_between(
    entry_time_utc_lo: str,
    entry_time_utc_hi_exclusive: str,
    *,
    status: str | None = None,
    limit: int = 5000,
) -> list[MockTradeRow]:
    """
    Trades whose `entry_time` is in [entry_time_utc_lo, entry_time_utc_hi_exclusive).
    Pass bounds from `mock_trade_analytics.ist_date_range_to_utc_sql_bounds`.
    """
    init_db()
    limit = min(max(limit, 1), 50_000)
    with _lock, _connect() as conn:
        if status:
            rows = conn.execute(
                f"""
                SELECT {_SELECT_ROW.strip()}
                FROM mock_trades
                WHERE entry_time >= ? AND entry_time < ? AND status = ?
                ORDER BY trade_id DESC
                LIMIT ?
                """,
                (entry_time_utc_lo, entry_time_utc_hi_exclusive, status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT {_SELECT_ROW.strip()}
                FROM mock_trades
                WHERE entry_time >= ? AND entry_time < ?
                ORDER BY trade_id DESC
                LIMIT ?
                """,
                (entry_time_utc_lo, entry_time_utc_hi_exclusive, limit),
            ).fetchall()
    return [_row(r) for r in rows]


def trades_for_chart() -> list[dict[str, Any]]:
    """Closed trades in chronological order for cumulative PnL."""
    init_db()
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT trade_id, entry_time, exit_time, realized_pnl, status
            FROM mock_trades
            WHERE status = 'CLOSED' AND realized_pnl IS NOT NULL
            ORDER BY trade_id ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]
