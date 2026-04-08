# Mock engine — envelope breakout rules

This document describes **when** the autonomous mock engine may emit a **BULLISH** (long CE) or **BEARISH** (long PE) signal from **index spot** 1-minute candles. Implementation: `trade_claw.mock_market_signal.envelope_breakout_on_last_bar`, invoked from `signal_node` in `trade_claw.mock_trading_graph.py`.

For the full engine pipeline, see [MOCK_ENGINE.md](MOCK_ENGINE.md).

---

## Session data

- Bars are loaded for the **session calendar day** from **09:15 to 15:30** IST (`load_index_session_minute_df` → `fetch_underlying_intraday`, interval `minute`).
- The **breakout bar** is either the **latest** row (default) or the **penultimate** row when two-bar confirmation is enabled (`MOCK_ENGINE_BREAKOUT_REQUIRE_CONFIRM_BAR=1`). **Spot** and **signal_bar_time** in the emitted signal use the **latest** bar when confirmation is on, otherwise the same bar as the breakout.

---

## 1. Warmup — bar count

Before any breakout logic runs, the dataframe must contain at least:

```text
max(
  ENVELOPE_EMA_PERIOD + 2 + (1 if confirm_bar else 0),
  MOCK_ENGINE_MIN_WARMUP_BARS + (1 if confirm_bar else 0),
  RANGE_EXPAND_LOOKBACK + 1 + (1 if confirm_bar else 0)   if RANGE_EXPAND_LOOKBACK > 0 else 0,
  VOLUME_LOOKBACK + 1 + (1 if confirm_bar else 0)         if VOLUME_LOOKBACK > 0 else 0,
)
```

(Implementation: `trade_claw.mock_market_signal.envelope_breakout_on_last_bar` — confirm bar adds **+1** because the breakout is evaluated one bar before the last row.)

With default **EMA period 20**, **`MOCK_ENGINE_MIN_WARMUP_BARS = 22`**, and strict filters **off**, this stays **22 bars** (or **23** with `MOCK_ENGINE_BREAKOUT_REQUIRE_CONFIRM_BAR=1`).

If there are fewer bars (early session, holiday, or thin data), the function returns **no signal** with a message that mentions warmup.

There is **no** separate wall-clock cutoff (e.g. 09:40); **only** the bar count gate applies for warmup.

---

## 2. Fresh cross on the breakout bar

On the breakout row index `i_b` (`len-1` by default, or `len-2` when confirm mode is on):

- **Bullish (long CE)**: previous close was at or below the **previous** upper band, and **breakout** close is **above** the **current** upper band.
- **Bearish (long PE)**: previous close was at or above the **previous** lower band, and **breakout** close is **below** the **current** lower band.

Otherwise there is **no** signal (no “stale” breakout — only the bar that **completes** the cross counts).

---

## 3. Breakout penetration (optional)

After the fresh cross is detected on the breakout bar, the engine may require a minimum share of that bar’s **high−low** range to lie **past** the band in the breakout direction (same geometry as the F&O options penetration slider and `strategies._envelope_breakout_penetration_frac`).

| Env | Meaning |
| :--- | :--- |
| `FO_BREAKOUT_PENETRATION_MIN_PCT` | Whole-number **0–100** (% of candle range past the line). **`0`** disables. If **unset**, defaults to **`FO_BREAKOUT_PENETRATION_DEFAULT_PCT`** in code (currently **50**, aligned with the UI). |
| `fo_breakout_penetration_min_pct` | Alias for the same variable (some shells prefer lowercase). |

Reader: `trade_claw.env_trading_params.fo_breakout_penetration_min_frac` (returns a **0–1** fraction). Evaluated **after** the cross, **before** the clear-break check.

---

## 4. Clear break past the band (optional)

After the cross (and optional penetration) pass, an extra check may require the close to sit **clearly** beyond the band, not only barely through it.

- **Bullish**: `(close - upper) / upper ≥ MOCK_ENGINE_BREAKOUT_CLEAR_PCT` (as a fraction of the upper band level; upper is taken in absolute value for the divisor guard).
- **Bearish**: `(lower - close) / lower ≥ MOCK_ENGINE_BREAKOUT_CLEAR_PCT`.

Configuration:

| Env | Meaning |
| :--- | :--- |
| `MOCK_ENGINE_BREAKOUT_CLEAR_PCT` | Decimal fraction (e.g. `0.0002` = **0.02%** of band level). **`0`** disables the check (touch/cross only). Default in code if unset: **0.0002**. Reader: `trade_claw.env_trading_params.mock_engine_breakout_clear_pct` (clamped to a safe max in code). |

If the cross is present but the margin is too small, the function returns **no signal** and explains the required vs actual margin.

---

## 5. Strict “clean breakout” filters (optional)

All of the following are applied **after** the cross, optional penetration, and optional clear-break check. Each knob is **disabled** when unset or set to **`0`** (readers in `trade_claw.env_trading_params`).

| Env | Meaning |
| :--- | :--- |
| `MOCK_ENGINE_BREAKOUT_MIN_BODY_FRAC` | Minimum `\|close−open\| / (high−low)` on the breakout bar (0–1). **`0`** = off. |
| `MOCK_ENGINE_BREAKOUT_MAX_LOWER_WICK_FRAC` | Max lower wick / range: `(min(open,close)−low)/(high−low)`. **`0`** = skip this check. |
| `MOCK_ENGINE_BREAKOUT_MAX_UPPER_WICK_FRAC` | Max upper wick / range. **`0`** = skip. |
| `MOCK_ENGINE_BREAKOUT_RANGE_EXPAND_LOOKBACK` | Integer `N`: require breakout range **>** mean range of the prior `N` bars. **`0`** = off. |
| `MOCK_ENGINE_BREAKOUT_VOLUME_LOOKBACK` | Integer `M`: require breakout volume **>** mean volume of the prior `M` bars. **`0`** = off. If **all** volumes in the lookback window (including the breakout bar) are **zero** (typical index feeds), the volume rule is **skipped**. |
| `MOCK_ENGINE_BREAKOUT_REQUIRE_DIRECTIONAL_BODY` | If `1` / `true`: bull requires `close > open`, bear requires `close < open`. |
| `MOCK_ENGINE_BREAKOUT_REQUIRE_CONFIRM_BAR` | If `1` / `true`: breakout on bar `len−2`, **confirmation** on bar `len−1` (close still outside the band and structure held vs the breakout bar). **Spot** and **signal_bar_time** come from the **confirm** bar. |

---

## Summary

| Rule | Purpose |
| :--- | :--- |
| Warmup bar count | EMA envelope validity + optional confirm / range / volume history |
| Fresh cross on breakout bar | No signal on old breaks unless that bar completes a new cross |
| Penetration (unless 0 / disabled) | Require enough of the breakout bar’s range past the band (aligned with F&O UI) |
| Clear break (unless pct = 0) | Reduce noise from marginal pierces of the band |
| Strict filters (optional) | Body, wicks, range expansion, volume, directional body, 2-bar confirm |

---

## Related

- Envelope bandwidth: per underlying `mock_engine_envelope_decimal_per_side(underlying)` — index keys use optional `MOCK_AGENT_INDEX_ENVELOPE_PCT`, else `MOCK_AGENT_ENVELOPE_PCT` / `fno_envelope_decimal_per_side()` — same geometry as `strategies._envelope_series`.
- Entry **time** window for **new** trades (Celery): still `in_entry_window` (weekdays 09:15–15:19 IST); that is separate from the breakout rules above.
