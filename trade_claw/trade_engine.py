"""Build mock trade rows from OHLCV + strategy (shared by home and reports)."""
from __future__ import annotations

import pandas as pd

from trade_claw.constants import (
    ALLOCATED_AMOUNT,
    ENVELOPE_EMA_PERIOD,
    ENVELOPE_PCT,
    REPORTS_MIN_BARS_PER_DAY,
)
from trade_claw.strategies import (
    filter_analyses_by_strategy_choice,
    get_applicable_strategies,
    get_strategy_analyses,
    simulate_envelope_trade_close,
    simulate_trade_close,
)


def build_trade_rows_from_analyses(df: pd.DataFrame, analyses: list) -> list[dict]:
    """Simulate trades from pre-filtered analyses list."""
    if df is None or df.empty or len(df) < 2:
        return []
    last_close = float(df["close"].iloc[-1])
    trade_rows: list[dict] = []
    seen_strategy: set[str] = set()
    if not analyses:
        return trade_rows
    for sname, _text, sig in analyses:
        if sname in seen_strategy:
            continue
        entry_bar_idx = sig.get("entry_bar_idx")
        if sig.get("envelope"):
            if not sig.get("signal") or entry_bar_idx is None:
                continue
            if not (0 <= entry_bar_idx < len(df)):
                continue
            entry = float(df.iloc[entry_bar_idx]["close"])
            if entry <= 0:
                continue
            qty = int(ALLOCATED_AMOUNT / entry)
            if qty < 1:
                continue
            closed_at, exit_price, pl, exit_bar_idx = simulate_envelope_trade_close(
                df,
                entry_bar_idx,
                entry,
                sig["signal"],
                qty,
                sig.get("ema_period", ENVELOPE_EMA_PERIOD),
                sig.get("pct", ENVELOPE_PCT),
            )
        elif sig.get("signal") and sig.get("target") is not None and sig.get("stop") is not None:
            if entry_bar_idx is not None and 0 <= entry_bar_idx < len(df):
                entry = float(df.iloc[entry_bar_idx]["close"])
            else:
                entry = last_close
            if entry <= 0:
                continue
            qty = int(ALLOCATED_AMOUNT / entry)
            if qty < 1:
                continue
            target = sig["target"]
            stop = sig["stop"]
            closed_at, exit_price, pl, exit_bar_idx = simulate_trade_close(
                df, entry_bar_idx, entry, target, stop, sig["signal"], qty
            )
        else:
            continue
        value = entry * qty
        trade_rows.append({
            "Strategy": sname,
            "Signal": sig["signal"],
            "Qty": qty,
            "Entry": entry,
            "entry_bar_idx": entry_bar_idx,
            "exit_bar_idx": exit_bar_idx,
            "Closed at": closed_at,
            "Exit": exit_price,
            "P/L": pl,
            "Value": value,
        })
        seen_strategy.add(sname)
    return trade_rows


def build_trade_rows_for_df(df: pd.DataFrame, chosen_interval: str, strategy_for_scrip: str) -> list[dict]:
    """Run filtered strategies on one session DataFrame."""
    if df is None or df.empty or len(df) < 2:
        return []
    strategies = get_applicable_strategies(df, chosen_interval)
    analyses = get_strategy_analyses(df, chosen_interval)
    analyses, _ = filter_analyses_by_strategy_choice(analyses, strategies, strategy_for_scrip)
    return build_trade_rows_from_analyses(df, analyses)


def filter_trade_rows_by_view(trade_rows: list[dict], trade_view: str) -> list[dict]:
    if trade_view == "Long only":
        return [t for t in trade_rows if t.get("Signal") == "BUY"]
    if trade_view == "Short only":
        return [t for t in trade_rows if t.get("Signal") == "SELL"]
    return list(trade_rows)


def split_dataframe_by_trading_day(
    df: pd.DataFrame, min_bars: int = REPORTS_MIN_BARS_PER_DAY
) -> list[tuple]:
    """Return list of (date, DataFrame) for each calendar day in df."""
    if df.empty or "date" not in df.columns:
        return []
    d = df.copy()
    d["_session_date"] = pd.to_datetime(d["date"], errors="coerce").dt.date
    d = d.dropna(subset=["_session_date"])
    out: list[tuple] = []
    for day, g in d.groupby("_session_date"):
        g2 = g.drop(columns=["_session_date"]).reset_index(drop=True)
        if len(g2) >= min_bars:
            out.append((day, g2))
    return sorted(out, key=lambda x: x[0])
