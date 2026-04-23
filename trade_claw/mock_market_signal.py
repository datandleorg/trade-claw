"""Nifty spot intraday fetch + 20-EMA envelope breakout on the latest bar (mock engine)."""

from __future__ import annotations

import logging
import os
import random
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo

from trade_claw.constants import (
    ENVELOPE_EMA_PERIOD,
    MOCK_ENGINE_SCAN_EQUITY_SYMBOLS,
    MOCK_ENGINE_SCAN_INDEX_KEYS,
)
from trade_claw.env_trading_params import (
    fo_options_default_envelope_bandwidth_pct,
    fo_options_default_option_stop_loss_pct_ui,
    fo_options_default_option_target_pct_ui,
    fno_envelope_decimal_per_side,
    mock_engine_equity_envelope_decimal_per_side,
    mock_engine_index_envelope_decimal_per_side,
    mock_engine_breakout_clear_pct,
    mock_engine_breakout_max_lower_wick_frac,
    mock_engine_breakout_max_upper_wick_frac,
    mock_engine_breakout_min_body_frac,
    mock_engine_breakout_range_expand_lookback,
    mock_engine_breakout_require_confirm_bar,
    mock_engine_breakout_require_directional_body,
    mock_engine_breakout_volume_lookback,
    mock_engine_option_stop_multiplier,
    mock_engine_option_target_multiplier,
    mock_engine_stop_loss_floor_multiplier,
    option_stop_premium_fraction,
    option_target_premium_fraction,
)
from trade_claw.fo_support import fetch_underlying_intraday, underlying_index_tradingsymbol
from trade_claw.strategies import _envelope_series

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

# Session bar count before mock engine may emit an envelope signal (with EMA(20), max(20+2, 22) == 22).
MOCK_ENGINE_MIN_WARMUP_BARS = 22


def mock_agent_envelope_pct() -> float:
    """
    Bandwidth each side of EMA as decimal (e.g. ``0.25`` = 25% above/below EMA; ``0.003`` = 0.3%).
    Env ``MOCK_AGENT_ENVELOPE_PCT`` overrides; else product default for mock engine / agent envelope.
    """
    return fno_envelope_decimal_per_side()


def mock_agent_envelope_pct_for_underlying(u: str) -> float:
    """
    Mock engine scan: bandwidth each side for ``u`` — index underlyings vs equities.
    Index keys are those with an NSE index spot mapping (see :func:`underlying_index_tradingsymbol`);
    others use the equity envelope. Unset per-kind env vars fall back to ``MOCK_AGENT_ENVELOPE_PCT``.
    """
    uu = (u or "").strip().upper()
    if underlying_index_tradingsymbol(uu) is not None:
        return mock_engine_index_envelope_decimal_per_side()
    return mock_engine_equity_envelope_decimal_per_side()


def fo_option_target_pct_runtime() -> float:
    """Target as fraction of premium (see :func:`trade_claw.env_trading_params.option_target_premium_fraction`)."""
    return option_target_premium_fraction()


def fo_option_stop_loss_pct_runtime() -> float:
    """Stop as fraction below entry (see :func:`trade_claw.env_trading_params.option_stop_premium_fraction`)."""
    return option_stop_premium_fraction()


def mock_agent_slippage_points() -> float:
    """Random slippage in index/option points (rupees for Nifty options premium)."""
    lo = float(os.environ.get("MOCK_AGENT_SLIPPAGE_LO", "0.5"))
    hi = float(os.environ.get("MOCK_AGENT_SLIPPAGE_HI", "1.0"))
    if hi < lo:
        lo, hi = hi, lo
    return random.uniform(lo, hi)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_weekday_ist(dt: datetime) -> bool:
    return dt.weekday() < 5


def _ist_wall_time(dt: datetime) -> dtime:
    if dt.tzinfo is None:
        return dt.time()
    return dt.astimezone(IST).time()


def in_entry_window(dt: datetime) -> bool:
    """NSE cash-style: new mock entries Mon–Fri 09:15–15:19 IST."""
    if not is_weekday_ist(dt):
        return False
    t = _ist_wall_time(dt)
    return dtime(9, 15) <= t < dtime(15, 20)


def should_force_square_off(dt: datetime) -> bool:
    if not is_weekday_ist(dt):
        return False
    return _ist_wall_time(dt) >= dtime(15, 20)


def session_date_ist(dt: datetime) -> date:
    return dt.date()


def mock_engine_allowed_underlyings() -> frozenset[str]:
    """Symbols the mock engine may trade: ``MOCK_ENGINE_SCAN_INDEX_KEYS`` + ``MOCK_ENGINE_SCAN_EQUITY_SYMBOLS``."""
    return frozenset(MOCK_ENGINE_SCAN_INDEX_KEYS) | frozenset(MOCK_ENGINE_SCAN_EQUITY_SYMBOLS)


def mock_engine_default_underlyings() -> list[str]:
    """Default scan order: NIFTY, BANKNIFTY, then the fixed equity list (no duplicates)."""
    seen: set[str] = set()
    out: list[str] = []
    for u in list(MOCK_ENGINE_SCAN_INDEX_KEYS) + list(MOCK_ENGINE_SCAN_EQUITY_SYMBOLS):
        ux = u.upper().strip()
        if ux not in seen:
            seen.add(ux)
            out.append(ux)
    return out


def mock_engine_underlyings() -> list[str]:
    """
    Keys scanned each graph run (envelope on spot 1m), in order.
    Env ``MOCK_ENGINE_UNDERLYINGS`` = comma-separated subset of allowed symbols (see
    ``mock_engine_allowed_underlyings``).
    Default = NIFTY + BANKNIFTY + ``MOCK_ENGINE_SCAN_EQUITY_SYMBOLS`` in ``constants.py``.
    """
    allowed = mock_engine_allowed_underlyings()
    raw = (os.environ.get("MOCK_ENGINE_UNDERLYINGS") or "").strip()
    if not raw:
        return mock_engine_default_underlyings()
    out: list[str] = []
    for part in raw.split(","):
        p = part.upper().strip()
        if p in allowed:
            out.append(p)
        elif p:
            logger.warning("MOCK_ENGINE_UNDERLYINGS: unknown key %r ignored", p)
    return out if out else mock_engine_default_underlyings()


def nse_index_ltp_symbol(underlying_key: str) -> str | None:
    """Kite LTP instrument key, e.g. ``NSE:NIFTY 50`` for ``NIFTY``."""
    ts = underlying_index_tradingsymbol(underlying_key.upper().strip())
    return f"NSE:{ts}" if ts else None


def _bar_ohlc(df, i: int) -> tuple[float, float, float, float]:
    row = df.iloc[i]
    return (
        float(row["open"]),
        float(row["high"]),
        float(row["low"]),
        float(row["close"]),
    )


def _candle_range(h: float, l: float) -> float:
    return max(h - l, 1e-12)


def _body_frac(o: float, h: float, l: float, c: float) -> float:
    return abs(c - o) / _candle_range(h, l)


def _lower_wick_frac(o: float, h: float, l: float, c: float) -> float:
    return (min(o, c) - l) / _candle_range(h, l)


def _upper_wick_frac(o: float, h: float, l: float, c: float) -> float:
    return (h - max(o, c)) / _candle_range(h, l)


def _strict_breakout_bar_ok(
    df,
    i_b: int,
    *,
    direction: str,
) -> tuple[bool, str]:
    """Apply optional body / wick / range / volume / directional-body filters at ``i_b``."""
    o, h, l, c = _bar_ohlc(df, i_b)

    min_body = mock_engine_breakout_min_body_frac()
    if min_body > 0:
        if _body_frac(o, h, l, c) < min_body:
            return False, f"body fraction < {min_body:.3f} (need stronger body vs range)"

    cap_lo = mock_engine_breakout_max_lower_wick_frac()
    if cap_lo > 0 and _lower_wick_frac(o, h, l, c) > cap_lo:
        return False, f"lower wick / range > {cap_lo:.3f}"
    cap_hi = mock_engine_breakout_max_upper_wick_frac()
    if cap_hi > 0 and _upper_wick_frac(o, h, l, c) > cap_hi:
        return False, f"upper wick / range > {cap_hi:.3f}"

    if mock_engine_breakout_require_directional_body():
        if direction == "BULLISH" and not (c > o):
            return False, "directional body: need close > open (bull)"
        if direction == "BEARISH" and not (c < o):
            return False, "directional body: need close < open (bear)"

    n_rng = mock_engine_breakout_range_expand_lookback()
    if n_rng > 0:
        if i_b < n_rng:
            return False, "range expansion: insufficient history"
        cur_rng = float(df["high"].iloc[i_b]) - float(df["low"].iloc[i_b])
        prior = df["high"].iloc[i_b - n_rng : i_b] - df["low"].iloc[i_b - n_rng : i_b]
        mean_prior = float(prior.mean())
        if mean_prior <= 1e-12 or cur_rng <= mean_prior:
            return False, f"range not expanded vs prior {n_rng} bars (mean prior {mean_prior:.4f})"

    m_vol = mock_engine_breakout_volume_lookback()
    if m_vol > 0 and "volume" in df.columns:
        try:
            slice_hi = df["volume"].iloc[max(0, i_b - m_vol) : i_b + 1]
            if float(slice_hi.fillna(0).abs().max()) <= 0:
                pass  # index / zero-volume feed: skip volume rule
            else:
                win = df["volume"].iloc[i_b - m_vol : i_b]
                v_b = float(df["volume"].iloc[i_b])
                mean_v = float(win.mean())
                if mean_v > 1e-12 and v_b <= mean_v:
                    return False, f"volume not above prior {m_vol}-bar mean"
        except (TypeError, ValueError, KeyError):
            pass

    return True, ""


def _confirm_bar_ok(
    df,
    i_b: int,
    i_c: int,
    *,
    direction: str,
    upper,
    lower,
) -> tuple[bool, str]:
    """Second bar holds breakout (``i_c`` = latest bar)."""
    o_c, h_c, l_c, c_c = _bar_ohlc(df, i_c)
    u_c = float(upper.iloc[i_c])
    lo_c = float(lower.iloc[i_c])
    _, _, l_b, c_b = _bar_ohlc(df, i_b)
    _, h_b, _, _ = _bar_ohlc(df, i_b)

    if direction == "BULLISH":
        if not (c_c > u_c):
            return False, "confirm: close not above upper band"
        if not (c_c >= c_b or l_c >= l_b):
            return False, "confirm: structure not held (close/low vs breakout bar)"
    else:
        if not (c_c < lo_c):
            return False, "confirm: close not below lower band"
        if not (c_c <= c_b or h_c <= h_b):
            return False, "confirm: structure not held (close/high vs breakout bar)"
    return True, ""


def envelope_breakout_on_last_bar(
    df,
    *,
    ema_period: int = ENVELOPE_EMA_PERIOD,
    pct: float,
) -> tuple[bool, str, dict]:
    """
    True if the session completes an envelope cross on the breakout bar (latest bar, or
    penultimate bar when ``MOCK_ENGINE_BREAKOUT_REQUIRE_CONFIRM_BAR`` is set).

    Requires **warmup**: enough bars for EMA, cross, optional strict filters, and optional confirm.
    Optional **clear break**: ``MOCK_ENGINE_BREAKOUT_CLEAR_PCT``. Strict filters (body, wicks,
    range expansion, volume, directional body) are env-gated; ``0`` / unset disables each.

    BUY cross → BULLISH (long CE); SELL cross → BEARISH (long PE).
    """
    empty: dict = {
        "direction": None,
        "leg": None,
        "spot": None,
        "signal_bar_time": None,
    }
    confirm = mock_engine_breakout_require_confirm_bar()
    extra = 1 if confirm else 0
    n_rng = mock_engine_breakout_range_expand_lookback()
    m_vol = mock_engine_breakout_volume_lookback()
    min_bars = max(ema_period + 2 + extra, MOCK_ENGINE_MIN_WARMUP_BARS + extra)
    if n_rng > 0:
        min_bars = max(min_bars, n_rng + 1 + extra)
    if m_vol > 0:
        min_bars = max(min_bars, m_vol + 1 + extra)

    if df is None or len(df) < min_bars:
        return (
            False,
            f"Warmup: need at least {min_bars} session bars (EMA {ema_period} + envelope cross"
            f"{', confirm bar' if confirm else ''}); got {0 if df is None else len(df)}.",
            empty,
        )

    close = df["close"]
    center, upper, lower = _envelope_series(df, ema_period, pct)
    i_c = len(df) - 1
    i_b = i_c - 1 if confirm else i_c

    c = float(close.iloc[i_b])
    c_prev = float(close.iloc[i_b - 1])
    u = float(upper.iloc[i_b])
    u_prev = float(upper.iloc[i_b - 1])
    lo = float(lower.iloc[i_b])
    lo_prev = float(lower.iloc[i_b - 1])

    if c > u and c_prev <= u_prev:
        direction = "BULLISH"
        leg = "CE"
        side = "above upper"
    elif c < lo and c_prev >= lo_prev:
        direction = "BEARISH"
        leg = "PE"
        side = "below lower"
    else:
        bar_label = "confirm bar" if confirm else "latest bar"
        text = (
            f"EMA({ema_period}) ±{100 * pct:.2f}%: no fresh cross on breakout ({bar_label} index {i_b}). "
            f"Close ₹{c:,.2f}, upper ₹{u:,.2f}, lower ₹{lo:,.2f}."
        )
        return False, text, empty

    clear_eps = mock_engine_breakout_clear_pct()
    if clear_eps > 0:
        if direction == "BULLISH":
            u_den = max(abs(u), 1e-12)
            margin = (c - u) / u_den
            if margin < clear_eps:
                text = (
                    f"EMA({ema_period}) ±{100 * pct:.2f}%: cross above upper but not a clear break "
                    f"(need margin ≥ {100 * clear_eps:.4f}% of upper; got {100 * margin:.4f}%). "
                    f"Close ₹{c:,.2f}, upper ₹{u:,.2f}."
                )
                return False, text, empty
        else:
            lo_den = max(abs(lo), 1e-12)
            margin = (lo - c) / lo_den
            if margin < clear_eps:
                text = (
                    f"EMA({ema_period}) ±{100 * pct:.2f}%: cross below lower but not a clear break "
                    f"(need margin ≥ {100 * clear_eps:.4f}% of lower; got {100 * margin:.4f}%). "
                    f"Close ₹{c:,.2f}, lower ₹{lo:,.2f}."
                )
                return False, text, empty

    ok_strict, strict_reason = _strict_breakout_bar_ok(df, i_b, direction=direction)
    if not ok_strict:
        text = (
            f"EMA({ema_period}) ±{100 * pct:.2f}%: cross {side} rejected (strict filter): {strict_reason}."
        )
        return False, text, empty

    if confirm:
        ok_c, c_reason = _confirm_bar_ok(df, i_b, i_c, direction=direction, upper=upper, lower=lower)
        if not ok_c:
            text = f"EMA({ema_period}) ±{100 * pct:.2f}%: breakout ok but confirm bar failed: {c_reason}."
            return False, text, empty

    i_sig = i_c if confirm else i_b
    spot = float(close.iloc[i_sig])
    ts = df.iloc[i_sig]["date"]
    try:
        signal_bar_time = str(ts)
    except Exception:
        signal_bar_time = None

    ce = float(center.iloc[i_sig])
    u_sig = float(upper.iloc[i_sig])
    lo_sig = float(lower.iloc[i_sig])
    bar_note = "confirm bar" if confirm else "latest bar"
    text = (
        f"EMA({ema_period}) ±{100 * pct:.2f}%: break {side} on breakout bar; spot from {bar_note}. "
        f"Center ₹{ce:,.2f}, spot close ₹{spot:,.2f} → {direction} (long {leg})."
    )
    sig = {
        "direction": direction,
        "leg": leg,
        "spot": spot,
        "signal_bar_time": signal_bar_time,
        "center": ce,
        "upper": u_sig,
        "lower": lo_sig,
    }
    return True, text, sig


def load_index_session_interval_df(
    kite, nse_instruments: list, session_d: date, underlying_key: str, interval: str
):
    """Today's (IST) session 09:15–15:30 bars for the index key (``minute``, ``3minute``, …)."""
    from_dt = datetime(session_d.year, session_d.month, session_d.day, 9, 15, 0)
    to_dt = datetime(session_d.year, session_d.month, session_d.day, 15, 30, 0)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    return fetch_underlying_intraday(
        kite, underlying_key.upper().strip(), nse_instruments, from_str, to_str, interval
    )


def load_index_session_minute_df(kite, nse_instruments: list, session_d: date, underlying_key: str):
    """Today's (IST) session 09:15–15:30 minute bars for the index key (``NIFTY``, ``BANKNIFTY``, …)."""
    return load_index_session_interval_df(
        kite, nse_instruments, session_d, underlying_key, "minute"
    )


def load_nifty_session_minute_df(kite, nse_instruments: list, session_d: date):
    """Today's (IST) session 09:15–15:30 minute bars for NIFTY 50 spot (backward-compatible)."""
    return load_index_session_minute_df(kite, nse_instruments, session_d, "NIFTY")


def top_five_option_instruments(
    nfo_instruments: list,
    *,
    underlying: str,
    session_d: date,
    spot: float,
    leg: str,
) -> tuple[list[dict], str | None]:
    """
    Nearest expiry on/after session_d; five strikes around ATM (steps −2..+2 on strike ladder).
    """
    from trade_claw.fo_support import filter_options_by_underlying, pick_nearest_expiry_on_or_after

    leg = (leg or "").upper().strip()
    if leg not in ("CE", "PE"):
        return [], f"Invalid leg {leg!r}"

    opts = filter_options_by_underlying(nfo_instruments, underlying)
    if not opts:
        return [], f"No NFO options for {underlying}"

    ex = pick_nearest_expiry_on_or_after(opts, session_d)
    if ex is None:
        return [], "No expiry on/after session date"

    same = [i for i in opts if _expiry_eq(i, ex) and i.get("instrument_type") == leg]
    if not same:
        return [], f"No {leg} rows for expiry {ex}"

    try:
        strikes_sorted = sorted({float(i.get("strike", 0)) for i in same})
    except (TypeError, ValueError):
        return [], "Invalid strikes"

    if not strikes_sorted:
        return [], "Empty strike ladder"

    atm_idx = min(range(len(strikes_sorted)), key=lambda j: abs(strikes_sorted[j] - float(spot)))
    take_idx = [atm_idx + k for k in (-2, -1, 0, 1, 2)]
    take_idx = [max(0, min(j, len(strikes_sorted) - 1)) for j in take_idx]
    seen: set[float] = set()
    ordered_strikes: list[float] = []
    for j in take_idx:
        s = strikes_sorted[j]
        if s not in seen:
            seen.add(s)
            ordered_strikes.append(s)

    while len(ordered_strikes) < 5:
        # pad with nearest unused strikes if duplicates collapsed
        for s in strikes_sorted:
            if s not in seen:
                seen.add(s)
                ordered_strikes.append(s)
                break
        else:
            break

    out: list[dict] = []
    for strike in ordered_strikes[:5]:
        rows = [i for i in same if abs(float(i.get("strike", 0)) - strike) < 0.01]
        if not rows:
            continue
        rows.sort(key=lambda x: (x.get("tradingsymbol") or "", x.get("instrument_token") or 0))
        out.append(rows[0])

    if len(out) < 5:
        return out, None
    return out, None


def _expiry_eq(inst: dict, ex: date) -> bool:
    from trade_claw.fo_support import _to_date

    ed = _to_date(inst.get("expiry"))
    return ed == ex
