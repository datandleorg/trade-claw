"""Nifty spot intraday fetch + 20-EMA envelope breakout on the latest bar (mock engine)."""

from __future__ import annotations

import logging
import os
import random
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo

from trade_claw.constants import ENVELOPE_EMA_PERIOD, FO_INDEX_UNDERLYING_KEYS, NIFTY50_SYMBOLS
from trade_claw.env_trading_params import (
    fo_options_default_envelope_bandwidth_pct,
    fo_options_default_option_stop_loss_pct_ui,
    fo_options_default_option_target_pct_ui,
    fno_envelope_decimal_per_side,
    mock_engine_breakout_clear_pct,
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
    """Symbols the mock engine may trade: index keys + Nifty 50 equity symbols."""
    return frozenset(FO_INDEX_UNDERLYING_KEYS) | frozenset(NIFTY50_SYMBOLS)


def mock_engine_default_underlyings() -> list[str]:
    """Default scan order: three indices first, then Nifty 50 (no duplicates)."""
    seen: set[str] = set()
    out: list[str] = []
    for u in list(FO_INDEX_UNDERLYING_KEYS) + list(NIFTY50_SYMBOLS):
        ux = u.upper().strip()
        if ux not in seen:
            seen.add(ux)
            out.append(ux)
    return out


def mock_engine_underlyings() -> list[str]:
    """
    Keys scanned each graph run (envelope on spot 1m), in order.
    Env ``MOCK_ENGINE_UNDERLYINGS`` = comma-separated subset of index keys and/or Nifty 50 symbols.
    Default = indices (``FO_INDEX_UNDERLYING_KEYS``) then all ``NIFTY50_SYMBOLS``.
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


def envelope_breakout_on_last_bar(
    df,
    *,
    ema_period: int = ENVELOPE_EMA_PERIOD,
    pct: float,
) -> tuple[bool, str, dict]:
    """
    True if the **most recent** bar completes an envelope cross (same geometry as `_ma_envelope_analysis`).
    Requires **warmup**: at least ``max(ema_period + 2, MOCK_ENGINE_MIN_WARMUP_BARS)`` session bars.
    Optional **clear break**: close must exceed the band by ``mock_engine_breakout_clear_pct`` of band level
    (env ``MOCK_ENGINE_BREAKOUT_CLEAR_PCT``; ``0`` = touch/cross only).
    BUY cross → BULLISH (long CE); SELL cross → BEARISH (long PE).
    """
    empty: dict = {
        "direction": None,
        "leg": None,
        "spot": None,
        "signal_bar_time": None,
    }
    min_bars = max(ema_period + 2, MOCK_ENGINE_MIN_WARMUP_BARS)
    if df is None or len(df) < min_bars:
        return (
            False,
            f"Warmup: need at least {min_bars} session bars (EMA {ema_period} + envelope cross); "
            f"got {0 if df is None else len(df)}.",
            empty,
        )

    close = df["close"]
    center, upper, lower = _envelope_series(df, ema_period, pct)
    i = len(df) - 1
    c = float(close.iloc[i])
    c_prev = float(close.iloc[i - 1])
    u = float(upper.iloc[i])
    u_prev = float(upper.iloc[i - 1])
    lo = float(lower.iloc[i])
    lo_prev = float(lower.iloc[i - 1])

    if c > u and c_prev <= u_prev:
        direction = "BULLISH"
        leg = "CE"
        side = "above upper"
    elif c < lo and c_prev >= lo_prev:
        direction = "BEARISH"
        leg = "PE"
        side = "below lower"
    else:
        text = (
            f"EMA({ema_period}) ±{100 * pct:.2f}%: no fresh cross on latest bar. "
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

    ts = df.iloc[i]["date"]
    try:
        signal_bar_time = str(ts)
    except Exception:
        signal_bar_time = None

    ce = float(center.iloc[i])
    text = (
        f"EMA({ema_period}) ±{100 * pct:.2f}%: break {side} on latest bar. "
        f"Center ₹{ce:,.2f}, spot close ₹{c:,.2f} → {direction} (long {leg})."
    )
    sig = {
        "direction": direction,
        "leg": leg,
        "spot": c,
        "signal_bar_time": signal_bar_time,
        "center": ce,
        "upper": u,
        "lower": lo,
    }
    return True, text, sig


def load_index_session_minute_df(kite, nse_instruments: list, session_d: date, underlying_key: str):
    """Today's (IST) session 09:15–15:30 minute bars for the index key (``NIFTY``, ``BANKNIFTY``, …)."""
    from_dt = datetime(session_d.year, session_d.month, session_d.day, 9, 15, 0)
    to_dt = datetime(session_d.year, session_d.month, session_d.day, 15, 30, 0)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    return fetch_underlying_intraday(
        kite, underlying_key.upper().strip(), nse_instruments, from_str, to_str, "minute"
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
