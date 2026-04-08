"""Optional minute-bar JSON snapshots for mock trades (Kite, at entry/exit only)."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from trade_claw.fo_support import fetch_underlying_intraday
from trade_claw.market_data import candles_to_dataframe

IST = ZoneInfo("Asia/Kolkata")


def snapshot_bar_count() -> int:
    """Max minute bars to store per leg; 0 disables snapshots."""
    try:
        return max(0, int(os.environ.get("MOCK_ENGINE_SNAPSHOT_BARS", "60")))
    except ValueError:
        return 60


def snapshot_exit_hold_max_bars() -> int:
    """Max bars when storing exit snapshot over the entry→exit hold window (full session ~375)."""
    try:
        return max(1, int(os.environ.get("MOCK_ENGINE_SNAPSHOT_EXIT_MAX_BARS", "500")))
    except ValueError:
        return 500


def _session_bounds(session_d: date) -> tuple[str, str]:
    from_dt = datetime(session_d.year, session_d.month, session_d.day, 9, 15, 0)
    to_dt = datetime(session_d.year, session_d.month, session_d.day, 15, 30, 0)
    return from_dt.strftime("%Y-%m-%d %H:%M:%S"), to_dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_entry_utc(entry_time_sql: str) -> pd.Timestamp | None:
    s = (entry_time_sql or "").strip()
    if not s:
        return None
    if len(s) >= 19:
        s = s[:19]
    t = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t


def _bars_utc_series(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["date"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST, ambiguous="infer", nonexistent="shift_forward")
    return ts.dt.tz_convert(UTC)


def _trim_snapshot_df(
    df: pd.DataFrame,
    *,
    max_bars: int,
    entry_time_sql: str | None,
    exit_hold: bool,
) -> pd.DataFrame:
    """Tail to ``max_bars`` for entry snapshots; exit snapshots = bars from entry time through session end, capped."""
    df = df.sort_values("date").reset_index(drop=True)
    if exit_hold and (entry_time_sql or "").strip():
        entry_u = _parse_entry_utc(entry_time_sql or "")
        cap = snapshot_exit_hold_max_bars()
        if entry_u is not None:
            bar_u = _bars_utc_series(df)
            df = df.loc[bar_u >= entry_u].reset_index(drop=True)
        else:
            cap = max(1, max_bars) if max_bars > 0 else snapshot_exit_hold_max_bars()
    else:
        cap = max(0, max_bars)
    if cap <= 0:
        return df.iloc[0:0]
    if len(df) > cap:
        df = df.iloc[-cap:].reset_index(drop=True)
    return df


def _df_to_ohlc_json_rows(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        d = r["date"]
        rows.append(
            {
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
        )
    return json.dumps(rows)


def fetch_option_minute_bars_json(
    kite,
    nfo_instruments: list,
    tradingsymbol: str,
    session_d: date,
    *,
    max_bars: int,
    entry_time_sql: str | None = None,
) -> str | None:
    exit_hold = bool((entry_time_sql or "").strip())
    if not tradingsymbol or not nfo_instruments:
        return None
    if not exit_hold and max_bars <= 0:
        return None
    token = None
    for i in nfo_instruments:
        if i.get("tradingsymbol") == tradingsymbol:
            token = i.get("instrument_token")
            break
    if token is None:
        return None
    from_str, to_str = _session_bounds(session_d)
    try:
        candles = kite.historical_data(int(token), from_str, to_str, interval="minute")
    except Exception:
        return None
    df = candles_to_dataframe(candles)
    if df is None or df.empty:
        return None
    df = _trim_snapshot_df(
        df,
        max_bars=max_bars,
        entry_time_sql=entry_time_sql,
        exit_hold=exit_hold,
    )
    return _df_to_ohlc_json_rows(df)


def fetch_index_minute_bars_json(
    kite,
    nse_instruments: list,
    session_d: date,
    underlying_key: str,
    *,
    max_bars: int,
    entry_time_sql: str | None = None,
) -> str | None:
    """Index spot 1m session bars (09:15–15:30). Exit: optional ``entry_time_sql`` → hold window, else last ``max_bars``."""
    exit_hold = bool((entry_time_sql or "").strip())
    if not nse_instruments or not (underlying_key or "").strip():
        return None
    if not exit_hold and max_bars <= 0:
        return None
    from_str, to_str = _session_bounds(session_d)
    try:
        df, err = fetch_underlying_intraday(
            kite, underlying_key.upper().strip(), nse_instruments, from_str, to_str, "minute"
        )
    except Exception:
        return None
    if df is None or df.empty or err:
        return None
    df = _trim_snapshot_df(
        df,
        max_bars=max_bars,
        entry_time_sql=entry_time_sql,
        exit_hold=exit_hold,
    )
    return _df_to_ohlc_json_rows(df)


def fetch_nifty_minute_bars_json(
    kite,
    nse_instruments: list,
    session_d: date,
    *,
    max_bars: int,
) -> str | None:
    """Last ``max_bars`` NIFTY 50 1m session bars (09:15–15:30)."""
    return fetch_index_minute_bars_json(
        kite, nse_instruments, session_d, "NIFTY", max_bars=max_bars
    )
