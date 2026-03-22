"""Long-option mock P/L: exit at premium target (high touch) or EOD only — no stop."""
from __future__ import annotations


def simulate_long_option_target_or_eod(df, entry_bar_idx, entry_price: float, target_price: float, qty: float):
    """
    Long option premium: first bar where high >= target_price → Target; else last bar close → EOD.
    P/L = (exit - entry) * qty (always long premium).
    Returns (closed_at, exit_price, pl, exit_bar_idx).
    """
    last_idx = len(df) - 1
    if entry_bar_idx is None or entry_bar_idx >= last_idx or qty <= 0:
        exit_price = float(df["close"].iloc[-1])
        pl = (exit_price - entry_price) * qty
        return "EOD", exit_price, pl, last_idx
    for i in range(entry_bar_idx + 1, len(df)):
        hi = float(df["high"].iloc[i])
        if hi >= target_price:
            pl = (target_price - entry_price) * qty
            return "Target", target_price, pl, i
    exit_price = float(df["close"].iloc[-1])
    pl = (exit_price - entry_price) * qty
    return "EOD", exit_price, pl, last_idx
