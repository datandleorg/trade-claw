"""Long-option mock P/L: exit at premium target (high), stop (low), or EOD."""
from __future__ import annotations


def simulate_long_option_target_stop_eod(
    df,
    entry_bar_idx,
    entry_price: float,
    target_price: float,
    qty: float,
    stop_price: float | None = None,
):
    """
    Long option premium after entry bar:
    - If ``stop_price`` is set (below entry): same-bar ambiguity matches cash **BUY** in ``simulate_trade_close`` —
      if both high >= target and low <= stop → **Target**; elif high >= target → **Target**; elif low <= stop → **Stop**.
    - If no stop: first bar with high >= target_price → **Target** at target_price.
    - Else last bar close → **EOD**.

    P/L = (exit - entry) * qty.
    Returns (closed_at, exit_price, pl, exit_bar_idx).
    """
    last_idx = len(df) - 1
    if entry_bar_idx is None or entry_bar_idx >= last_idx or qty <= 0:
        exit_price = float(df["close"].iloc[-1])
        pl = (exit_price - entry_price) * qty
        return "EOD", exit_price, pl, last_idx

    use_stop = stop_price is not None and stop_price > 0 and stop_price < entry_price

    for i in range(entry_bar_idx + 1, len(df)):
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        if use_stop:
            sl = float(stop_price)
            if hi >= target_price and lo <= sl:
                pl = (target_price - entry_price) * qty
                return "Target", target_price, pl, i
            if hi >= target_price:
                pl = (target_price - entry_price) * qty
                return "Target", target_price, pl, i
            if lo <= sl:
                pl = (sl - entry_price) * qty
                return "Stop", sl, pl, i
        else:
            if hi >= target_price:
                pl = (target_price - entry_price) * qty
                return "Target", target_price, pl, i

    exit_price = float(df["close"].iloc[-1])
    pl = (exit_price - entry_price) * qty
    return "EOD", exit_price, pl, last_idx


def simulate_long_option_target_or_eod(df, entry_bar_idx, entry_price: float, target_price: float, qty: float):
    """Backward-compatible: target or EOD only (no stop)."""
    return simulate_long_option_target_stop_eod(
        df, entry_bar_idx, entry_price, target_price, qty, stop_price=None
    )
