"""Optional minute-bar JSON snapshots for mock trades (Kite, at entry/exit only)."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any

from trade_claw.fo_support import fetch_underlying_intraday
from trade_claw.market_data import candles_to_dataframe


def snapshot_bar_count() -> int:
    """Max minute bars to store per leg; 0 disables snapshots."""
    try:
        return max(0, int(os.environ.get("MOCK_ENGINE_SNAPSHOT_BARS", "60")))
    except ValueError:
        return 60


def _session_bounds(session_d: date) -> tuple[str, str]:
    from_dt = datetime(session_d.year, session_d.month, session_d.day, 9, 15, 0)
    to_dt = datetime(session_d.year, session_d.month, session_d.day, 15, 30, 0)
    return from_dt.strftime("%Y-%m-%d %H:%M:%S"), to_dt.strftime("%Y-%m-%d %H:%M:%S")


def fetch_option_minute_bars_json(
    kite,
    nfo_instruments: list,
    tradingsymbol: str,
    session_d: date,
    *,
    max_bars: int,
) -> str | None:
    if max_bars <= 0 or not tradingsymbol or not nfo_instruments:
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
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) > max_bars:
        df = df.iloc[-max_bars:]
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


def fetch_nifty_minute_bars_json(
    kite,
    nse_instruments: list,
    session_d: date,
    *,
    max_bars: int,
) -> str | None:
    """Last ``max_bars`` NIFTY 1m session bars (09:15–15:30), same window as option snapshots."""
    if max_bars <= 0 or not nse_instruments:
        return None
    from_str, to_str = _session_bounds(session_d)
    try:
        df, err = fetch_underlying_intraday(
            kite, "NIFTY", nse_instruments, from_str, to_str, "minute"
        )
    except Exception:
        return None
    if df is None or df.empty or err:
        return None
    if len(df) > max_bars:
        df = df.iloc[-max_bars:]
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
