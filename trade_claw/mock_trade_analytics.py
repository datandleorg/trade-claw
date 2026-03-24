"""Pandas aggregations for multi-month mock trade analysis (SQLite `mock_trades`).

Timestamps in the DB are stored as naive UTC wall-clock strings (see `mock_trade_store._utc_now_sql`).
Date-range filters in the UI are interpreted as IST calendar days and converted to UTC bounds for queries.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from trade_claw.mock_trade_store import MockTradeRow, list_trades_between

IST = ZoneInfo("Asia/Kolkata")


def ist_date_range_to_utc_sql_bounds(d_start: date, d_end: date) -> tuple[str, str]:
    """Inclusive IST calendar [d_start, d_end] → [start_utc_sql, end_exclusive_utc_sql)."""
    start = datetime.combine(d_start, time(0, 0, 0), tzinfo=IST)
    end_excl = datetime.combine(d_end + timedelta(days=1), time(0, 0, 0), tzinfo=IST)
    return (
        start.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        end_excl.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )


def rows_to_dataframe(rows: list[MockTradeRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = [asdict(r) for r in rows]
    return pd.DataFrame(records)


def load_trades_for_analytics(
    d_start: date,
    d_end: date,
    *,
    status: str | None = None,
    limit: int = 5000,
) -> pd.DataFrame:
    lo, hi_excl = ist_date_range_to_utc_sql_bounds(d_start, d_end)
    rows = list_trades_between(lo, hi_excl, status=status, limit=limit)
    return rows_to_dataframe(rows)


def closed_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["status"].astype(str).str.upper() == "CLOSED"].copy()


def parse_db_ts(series: pd.Series) -> pd.Series:
    """Parse naive UTC strings from DB to timezone-aware UTC."""
    if series.empty:
        return series
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return out


def kpis_closed(df_closed: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "closed_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "win_rate": None,
        "total_pnl": 0.0,
        "avg_pnl": None,
        "best_trade": None,
        "worst_trade": None,
    }
    if df_closed.empty:
        return out
    pnl = pd.to_numeric(df_closed["realized_pnl"], errors="coerce").fillna(0.0)
    out["closed_count"] = int(len(df_closed))
    wins = pnl > 0
    losses = pnl < 0
    out["win_count"] = int(wins.sum())
    out["loss_count"] = int(losses.sum())
    out["win_rate"] = float(out["win_count"] / out["closed_count"]) if out["closed_count"] else None
    out["total_pnl"] = float(pnl.sum())
    out["avg_pnl"] = float(pnl.mean()) if out["closed_count"] else None
    out["best_trade"] = float(pnl.max()) if len(pnl) else None
    out["worst_trade"] = float(pnl.min()) if len(pnl) else None
    return out


def monthly_pnl_summary(df_closed: pd.DataFrame) -> pd.DataFrame:
    if df_closed.empty:
        return pd.DataFrame(columns=["month", "trades", "wins", "total_pnl", "avg_pnl"])
    df = df_closed.copy()
    df["_exit"] = parse_db_ts(df["exit_time"])
    df = df.dropna(subset=["_exit"])
    if df.empty:
        return pd.DataFrame(columns=["month", "trades", "wins", "total_pnl", "avg_pnl"])
    df["_month"] = df["_exit"].dt.tz_convert(IST).dt.to_period("M").astype(str)
    pnl = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    df["_pnl"] = pnl
    g = df.groupby("_month", sort=True)
    agg = g.agg(
        trades=("trade_id", "count"),
        wins=("_pnl", lambda s: int((s > 0).sum())),
        total_pnl=("_pnl", "sum"),
        avg_pnl=("_pnl", "mean"),
    ).reset_index()
    agg = agg.rename(columns={"_month": "month"})
    return agg


def equity_curve_by_exit(df_closed: pd.DataFrame) -> pd.DataFrame:
    if df_closed.empty:
        return pd.DataFrame(columns=["exit_time", "trade_id", "realized_pnl", "cum_pnl"])
    df = df_closed.copy()
    df["_exit"] = parse_db_ts(df["exit_time"])
    df = df.dropna(subset=["_exit"])
    df = df.sort_values("_exit")
    pnl = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    df["_pnl"] = pnl
    df["cum_pnl"] = df["_pnl"].cumsum()
    return df.rename(columns={"_exit": "exit_time_utc"})[
        ["exit_time_utc", "trade_id", "_pnl", "cum_pnl"]
    ].rename(columns={"_pnl": "realized_pnl"})


def direction_breakdown(df_closed: pd.DataFrame) -> pd.DataFrame:
    if df_closed.empty:
        return pd.DataFrame(columns=["direction", "trades", "total_pnl"])
    t = df_closed.copy()
    t["_dir"] = t["direction"].fillna("—")
    pnl = pd.to_numeric(t["realized_pnl"], errors="coerce").fillna(0.0)
    t["_pnl"] = pnl
    g = t.groupby("_dir", as_index=False).agg(
        trades=("trade_id", "count"),
        total_pnl=("_pnl", "sum"),
    )
    return g.rename(columns={"_dir": "direction"})


def index_underlying_breakdown(df_closed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate closed trades by `index_underlying` (multi-index mock engine)."""
    if df_closed.empty or "index_underlying" not in df_closed.columns:
        return pd.DataFrame(columns=["index_underlying", "trades", "total_pnl", "wins"])
    t = df_closed.copy()
    t["_ix"] = t["index_underlying"].fillna("—").astype(str).str.strip()
    t.loc[t["_ix"] == "", "_ix"] = "—"
    pnl = pd.to_numeric(t["realized_pnl"], errors="coerce").fillna(0.0)
    t["_pnl"] = pnl
    g = t.groupby("_ix", as_index=False).agg(
        trades=("trade_id", "count"),
        total_pnl=("_pnl", "sum"),
        wins=("_pnl", lambda s: int((s > 0).sum())),
    )
    return g.rename(columns={"_ix": "index_underlying"})


def trades_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
