# Mock engine — envelope breakout rules

This document describes **when** the autonomous mock engine may emit a **BULLISH** (long CE) or **BEARISH** (long PE) signal from **index spot** 1-minute candles. Implementation: `trade_claw.mock_market_signal.envelope_breakout_on_last_bar`, invoked from `signal_node` in `trade_claw.mock_trading_graph.py`.

For the full engine pipeline, see [MOCK_ENGINE.md](MOCK_ENGINE.md).

---

## Session data

- Bars are loaded for the **session calendar day** from **09:15 to 15:30** IST (`load_index_session_minute_df` → `fetch_underlying_intraday`, interval `minute`).
- The **signal is evaluated only on the latest completed bar** in that series (same bar the worker sees when the Celery tick runs).

---

## 1. Warmup — at least 22 session bars

Before any breakout logic runs, the dataframe must contain at least:

```text
max(ENVELOPE_EMA_PERIOD + 2, MOCK_ENGINE_MIN_WARMUP_BARS)
```

With the default **EMA period 20** (`ENVELOPE_EMA_PERIOD` in `trade_claw.constants`) and **`MOCK_ENGINE_MIN_WARMUP_BARS = 22`** in `mock_market_signal.py`, this is **22 bars**.

- **EMA + 2**: the envelope uses a 20-period EMA; a **fresh** cross is defined using the **last two** closes vs the bands, so at least **22** rows are required for stable geometry.
- **Explicit 22-bar floor**: if `ENVELOPE_EMA_PERIOD` were increased later, `ema_period + 2` could exceed 22; the `max(...)` keeps both **EMA validity** and a **minimum 22-bar session warmup**.

If there are fewer bars (early session, holiday, or thin data), the function returns **no signal** with a message that mentions warmup.

There is **no** separate wall-clock cutoff (e.g. 09:40); **only** the bar count gate applies for warmup.

---

## 2. Fresh cross on the latest bar

On the **last** row of the dataframe:

- **Bullish (long CE)**: previous close was at or below the **previous** upper band, and **current** close is **above** the **current** upper band.
- **Bearish (long PE)**: previous close was at or above the **previous** lower band, and **current** close is **below** the **current** lower band.

Otherwise there is **no** signal (no “stale” breakout — only the bar that **completes** the cross counts).

---

## 3. Clear break past the band (optional)

After a raw cross passes, an extra check may require the close to sit **clearly** beyond the band, not only barely through it.

- **Bullish**: `(close - upper) / upper ≥ MOCK_ENGINE_BREAKOUT_CLEAR_PCT` (as a fraction of the upper band level; upper is taken in absolute value for the divisor guard).
- **Bearish**: `(lower - close) / lower ≥ MOCK_ENGINE_BREAKOUT_CLEAR_PCT`.

Configuration:

| Env | Meaning |
| :--- | :--- |
| `MOCK_ENGINE_BREAKOUT_CLEAR_PCT` | Decimal fraction (e.g. `0.0002` = **0.02%** of band level). **`0`** disables the check (touch/cross only). Default in code if unset: **0.0002**. Reader: `trade_claw.env_trading_params.mock_engine_breakout_clear_pct` (clamped to a safe max in code). |

If the cross is present but the margin is too small, the function returns **no signal** and explains the required vs actual margin.

---

## Summary

| Rule | Purpose |
| :--- | :--- |
| ≥ `max(ema_period+2, 22)` bars | Warmup + valid EMA envelope on full session series |
| Fresh cross on **last** bar only | No signal on old breaks unless the latest bar confirms a new cross |
| Clear break (unless pct = 0) | Reduce noise from marginal pierces of the band |

---

## Related

- Envelope bandwidth: `MOCK_AGENT_ENVELOPE_PCT` / `fno_envelope_decimal_per_side()` — same geometry as `strategies._envelope_series`.
- Entry **time** window for **new** trades (Celery): still `in_entry_window` (weekdays 09:15–15:19 IST); that is separate from the breakout rules above.
