"""Nifty spot intraday fetch + 20-EMA envelope breakout on the latest bar (mock engine)."""

from __future__ import annotations

import os
import random
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo

from trade_claw.constants import ENVELOPE_EMA_PERIOD
from trade_claw.fo_support import fetch_underlying_intraday
from trade_claw.strategies import _envelope_series

IST = ZoneInfo("Asia/Kolkata")


def mock_agent_envelope_pct() -> float:
    """Bandwidth each side of EMA as decimal (e.g. 0.005 = 0.5%)."""
    raw = os.environ.get("MOCK_AGENT_ENVELOPE_PCT", "0.002").strip()
    if raw:
        return float(raw)
    return 0.005


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


def envelope_breakout_on_last_bar(
    df,
    *,
    ema_period: int = ENVELOPE_EMA_PERIOD,
    pct: float,
) -> tuple[bool, str, dict]:
    """
    True if the **most recent** bar completes an envelope cross (same geometry as `_ma_envelope_analysis`).
    BUY cross → BULLISH (long CE); SELL cross → BEARISH (long PE).
    """
    empty: dict = {
        "direction": None,
        "leg": None,
        "spot": None,
        "signal_bar_time": None,
    }
    min_bars = ema_period + 2
    if df is None or len(df) < min_bars:
        return False, f"Need at least {min_bars} bars; got {0 if df is None else len(df)}.", empty

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


def load_nifty_session_minute_df(kite, nse_instruments: list, session_d: date):
    """Today's (IST) session 09:15–15:30 minute bars for NIFTY spot index."""
    from_dt = datetime(session_d.year, session_d.month, session_d.day, 9, 15, 0)
    to_dt = datetime(session_d.year, session_d.month, session_d.day, 15, 30, 0)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    return fetch_underlying_intraday(
        kite, "NIFTY", nse_instruments, from_str, to_str, "minute"
    )


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
