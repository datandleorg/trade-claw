"""
Buy-only \"Institutional floor\" sizing for index ETFs / funds.

Uses 200-day SMA as valuation anchor and 50-day SMA for golden-cross context.
No sell signals — only suggested monthly / lump-sum buy size.
"""
from __future__ import annotations

import pandas as pd

from trade_claw.constants import (
    INSTITUTIONAL_EXTENDED_ABOVE_PCT,
    INSTITUTIONAL_SMA_LONG,
    INSTITUTIONAL_SMA_SHORT,
    INSTITUTIONAL_STANDARD_BUY,
    INSTITUTIONAL_AGGRESSIVE_BUY,
)


def sma_series(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def analyze_institutional_floor(
    df: pd.DataFrame,
    *,
    standard_amt: float = INSTITUTIONAL_STANDARD_BUY,
    aggressive_amt: float = INSTITUTIONAL_AGGRESSIVE_BUY,
    extended_pct: float = INSTITUTIONAL_EXTENDED_ABOVE_PCT,
    sma_short: int = INSTITUTIONAL_SMA_SHORT,
    sma_long: int = INSTITUTIONAL_SMA_LONG,
) -> dict:
    """
    df: daily bars, ascending date, column 'close'.

    Returns dict with ok, metrics, recommendation text, and series for charting.
    """
    out_base = {
        "ok": False,
        "error": None,
        "close": None,
        "sma50": None,
        "sma200": None,
        "pct_vs_200": None,
        "golden": None,
        "recent_golden_cross": None,
        "recommendation": None,
        "amount": None,
        "detail": None,
        "structure_note": None,
        "sma50_series": None,
        "sma200_series": None,
        "dates": None,
    }
    if df is None or df.empty or "close" not in df.columns:
        return {**out_base, "error": "No close prices"}
    c = df["close"].astype(float)
    if len(c) < sma_long:
        return {**out_base, "error": f"Need ≥{sma_long} daily bars; got {len(c)}"}

    s50 = sma_series(c, sma_short)
    s200 = sma_series(c, sma_long)
    last = float(c.iloc[-1])
    v50 = float(s50.iloc[-1])
    v200 = float(s200.iloc[-1])
    if pd.isna(v50) or pd.isna(v200):
        return {**out_base, "error": "SMA not ready"}

    pct_vs_200 = (last / v200 - 1.0) * 100.0
    golden = v50 > v200

    # Golden cross in last 20 sessions (50 crosses above 200)
    cross_up = (s50.shift(1) <= s200.shift(1)) & (s50 > s200)
    recent_golden_cross = bool(cross_up.tail(20).any())

    # Value zone: at or below 200 SMA (small tolerance for "touch")
    if last <= v200 * 1.002:
        rec = "Aggressive buy"
        amount = aggressive_amt
        detail = (
            "Price at or below 200-day SMA vs last ~1 year of data — "
            "historical value zone; consider larger contribution (buy-only, no sell)."
        )
    else:
        rec = "Standard buy"
        amount = standard_amt
        if pct_vs_200 >= extended_pct:
            detail = (
                f"Price is ≥{extended_pct:.0f}% above 200-day SMA — "
                "keep regular plan; avoid chasing with extra size."
            )
        else:
            detail = (
                "Price above 200-day SMA but not extended — "
                "regular accumulation; not in deep value zone."
            )

    structure_note = (
        "Golden cross intact (50 > 200): long-term structure supportive."
        if golden
        else "50-day below 200-day: favour DCA; for lump sums, many wait for 50 to cross above 200."
    )

    dates = df["date"] if "date" in df.columns else df.index

    return {
        "ok": True,
        "error": None,
        "close": last,
        "sma50": v50,
        "sma200": v200,
        "pct_vs_200": pct_vs_200,
        "golden": golden,
        "recent_golden_cross": recent_golden_cross,
        "recommendation": rec,
        "amount": amount,
        "detail": detail,
        "structure_note": structure_note,
        "sma50_series": s50,
        "sma200_series": s200,
        "dates": dates,
        "close_series": c,
    }
