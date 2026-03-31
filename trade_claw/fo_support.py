"""NFO option instrument resolution (ATM CE/PE) and underlying spot helpers."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from trade_claw.constants import FO_MAX_EXPIRY_CANDLE_FALLBACKS, INTERVAL_MINUTES
from trade_claw.market_data import candles_to_dataframe, get_instrument_token


def _to_date(d: Any) -> date | None:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def fo_lot_qty_for_allocation(
    entry_premium: float, lot_size: int | None, allocated: float
) -> tuple[int, int, int]:
    """
    Size in **whole exchange lots** only. Returns (n_lots, qty_in_units, lot_size).
    qty_in_units = n_lots * lot_size (option contracts / underlying units per Kite).
    """
    ls = max(1, int(lot_size or 1))
    if entry_premium <= 0 or allocated <= 0:
        return 0, 0, ls
    one_lot_prem = entry_premium * ls
    n = int(allocated // one_lot_prem)
    if n < 1:
        return 0, 0, ls
    return n, n * ls, ls


def is_nfo_option(inst: dict) -> bool:
    return inst.get("exchange") == "NFO" and inst.get("instrument_type") in ("CE", "PE")


# NFO instrument `name` (upper) for each index underlying key (see constants.FO_INDEX_UNDERLYING_KEYS).
_INDEX_NFO_NAMES: dict[str, frozenset[str]] = {
    "NIFTY": frozenset({"NIFTY"}),
    "BANKNIFTY": frozenset({"BANKNIFTY"}),
    "MIDCPNIFTY": frozenset({"MIDCPNIFTY"}),
}


def nfo_index_name_set(underlying: str) -> frozenset[str] | None:
    """If `underlying` is a configured index key, return allowed NFO `name` values (uppercase); else None."""
    return _INDEX_NFO_NAMES.get(underlying.upper().strip())


def filter_options_by_underlying(nfo_instruments: list, underlying: str) -> list[dict]:
    """Filter NFO CE/PE rows for index keys (FO_INDEX_*) or equity underlying symbol."""
    u = underlying.upper().strip()
    index_names = nfo_index_name_set(u)
    out = []
    for i in nfo_instruments:
        if not is_nfo_option(i):
            continue
        name = (i.get("name") or "").upper().strip()
        ts = (i.get("tradingsymbol") or "").upper()
        if index_names is not None:
            if name in index_names:
                out.append(i)
            continue
        if name == u or ts.startswith(u):
            out.append(i)
    return out


def pick_nearest_expiry_on_or_after(options: list[dict], ref: date) -> date | None:
    expiries = set()
    for i in options:
        ed = _to_date(i.get("expiry"))
        if ed and ed >= ref:
            expiries.add(ed)
    return min(expiries) if expiries else None


def _sorted_expiries_on_or_after(options: list[dict], ref: date) -> list[date]:
    expiries = set()
    for i in options:
        ed = _to_date(i.get("expiry"))
        if ed and ed >= ref:
            expiries.add(ed)
    return sorted(expiries)


def _choose_strike_from_ladder(
    strikes_sorted: list[float],
    spot: float,
    leg: str,
    steps_from_atm: int,
    manual_strike: float | None,
) -> float:
    atm_idx = min(range(len(strikes_sorted)), key=lambda i: abs(strikes_sorted[i] - float(spot)))
    if manual_strike is not None and float(manual_strike) > 0:
        tgt = float(manual_strike)
        return min(strikes_sorted, key=lambda s: abs(s - tgt))
    k = int(steps_from_atm)
    if leg == "CE":
        idx = atm_idx + k
    else:
        idx = atm_idx - k
    idx = max(0, min(idx, len(strikes_sorted) - 1))
    return strikes_sorted[idx]


def _instrument_rows_for_strike(book: list[dict], chosen_strike: float, tol: float = 0.01) -> list[dict]:
    """
    All NFO rows in `book` matching the listed strike (within tol).
    If none exact, return single nearest row (same as legacy pick_option_contract).
    Sorted by tradingsymbol for stable ordering when the master lists duplicates.
    """
    exact: list[dict] = []
    best: dict | None = None
    best_d: float | None = None
    for i in book:
        try:
            stv = float(i.get("strike", 0))
        except (TypeError, ValueError):
            continue
        d = abs(stv - chosen_strike)
        if d < tol:
            exact.append(i)
        if best_d is None or d < best_d:
            best_d = d
            best = i
    if exact:
        return sorted(exact, key=lambda x: (x.get("tradingsymbol") or "", x.get("instrument_token") or 0))
    return [best] if best is not None else []


def build_option_trade_candidates(
    nfo_instruments: list,
    underlying: str,
    spot: float,
    session_date: date,
    leg: str,
    steps_from_atm: int = 0,
    manual_strike: float | None = None,
    *,
    max_expiries: int = FO_MAX_EXPIRY_CANDLE_FALLBACKS,
) -> tuple[list[tuple[dict, float, date]], str | None]:
    """
    Ordered list of (instrument, chosen_strike, expiry) to try for historical candles.

    Walks expiries **on or after session_date** (nearest first). For each expiry, recomputes
    ATM / OTM / ITM strike on **that** chain, then emits every master row at that strike
    (duplicate symbols) in stable order — so we can fall back when Kite returns no bars.
    """
    leg = (leg or "").upper().strip()
    if leg not in ("CE", "PE"):
        return [], f"Invalid leg {leg!r}"

    opts = filter_options_by_underlying(nfo_instruments, underlying)
    if not opts:
        return [], f"No NFO options found for {underlying}"

    expiries = _sorted_expiries_on_or_after(opts, session_date)
    if not expiries:
        return [], f"No expiry on/after {session_date} for {underlying}"

    out: list[tuple[dict, float, date]] = []
    for ex in expiries[: max(1, int(max_expiries))]:
        same = [i for i in opts if _to_date(i.get("expiry")) == ex]
        book = [i for i in same if i.get("instrument_type") == leg]
        if not book:
            continue
        try:
            strikes_sorted = sorted({float(i.get("strike", 0)) for i in book})
        except (TypeError, ValueError):
            continue
        if not strikes_sorted:
            continue
        chosen_strike = _choose_strike_from_ladder(
            strikes_sorted, spot, leg, steps_from_atm, manual_strike
        )
        rows = _instrument_rows_for_strike(book, chosen_strike)
        for inst in rows:
            out.append((inst, float(chosen_strike), ex))

    if not out:
        return [], f"No {leg} contracts for {underlying} in first expiries"
    return out, None


def pick_option_contract(
    nfo_instruments: list,
    underlying: str,
    spot: float,
    session_date: date,
    leg: str,
    steps_from_atm: int = 0,
    manual_strike: float | None = None,
) -> tuple[dict | None, str | None, float | None]:
    """
    Pick one CE or PE contract: nearest expiry on/after session_date, then strike from policy or manual.

    - **manual_strike** > 0: choose listed strike nearest to that value (ignores steps_from_atm).
    - Else **steps_from_atm**: ATM=0; for **CE** +1 moves to next higher strike (more OTM call);
      for **PE** +1 moves to next lower strike (more OTM put). Negative steps move ITM.

    Returns (instrument_dict, error_message, chosen_strike_float).
    """
    cands, err = build_option_trade_candidates(
        nfo_instruments,
        underlying,
        spot,
        session_date,
        leg,
        steps_from_atm,
        manual_strike,
        max_expiries=1,
    )
    if err:
        return None, err, None
    inst, strike, _ex = cands[0]
    return inst, None, float(strike)


def pick_atm_ce_pe(
    nfo_instruments: list,
    underlying: str,
    spot: float,
    session_date: date,
) -> tuple[dict | None, dict | None, str | None]:
    """
    Choose nearest expiry on/after session_date, then ATM strike CE and PE.
    Returns (ce_inst, pe_inst, error_message).
    """
    ce, e1, _ = pick_option_contract(nfo_instruments, underlying, spot, session_date, "CE", 0, None)
    if e1:
        return None, None, e1
    pe, e2, _ = pick_option_contract(nfo_instruments, underlying, spot, session_date, "PE", 0, None)
    if e2:
        return None, None, e2
    return ce, pe, None


def _ts_minute(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        try:
            t = t.tz_convert("Asia/Kolkata")
        except Exception:
            pass
        if t.tzinfo is not None:
            t = pd.Timestamp(t.to_pydatetime().replace(tzinfo=None))
    return t.floor("min")


def align_option_entry_bar(
    df_under: pd.DataFrame,
    df_opt: pd.DataFrame,
    entry_bar_idx: int,
    interval_key: str = "minute",
) -> int | None:
    """Map underlying bar index to option row by nearest timestamp (tolerance scales with bar size)."""
    if df_under.empty or df_opt.empty or entry_bar_idx < 0 or entry_bar_idx >= len(df_under):
        return None
    bar_sec = max(60, int(INTERVAL_MINUTES.get(interval_key, 1) * 60))
    # Allow ~1 bar of clock skew between NSE index and NFO series from Kite
    tol_sec = bar_sec + 150
    t0 = _ts_minute(df_under.iloc[entry_bar_idx]["date"])
    best_j, best_sec = None, None
    for j in range(len(df_opt)):
        t1 = _ts_minute(df_opt.iloc[j]["date"])
        sec = abs((t1 - t0).total_seconds())
        if best_sec is None or sec < best_sec:
            best_sec, best_j = sec, j
    if best_j is not None and best_sec is not None and best_sec <= tol_sec:
        return best_j
    if len(df_opt) == len(df_under):
        return int(entry_bar_idx)
    return None


# NSE indices: `tradingsymbol` candidates (first match in instrument master wins).
_INDEX_NSE_SPOT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "NIFTY": ("NIFTY 50",),
    "BANKNIFTY": ("NIFTY BANK",),
    "MIDCPNIFTY": ("NIFTY MIDCAP SELECT", "NIFTY MID SELECT"),
}


def _nse_index_tradingsymbol_fallback(_underlying: str, _nse_instruments: list) -> str | None:
    """If exact tradingsymbol candidates miss, match Kite NSE index rows by segment/name."""
    return None


def underlying_index_tradingsymbol(underlying: str) -> str | None:
    """First NSE cash index symbol for this key, without consulting instrument list (best-effort)."""
    u = underlying.upper().strip()
    cands = _INDEX_NSE_SPOT_CANDIDATES.get(u)
    return cands[0] if cands else None


def _nse_spot_candidates_for_underlying(underlying: str) -> list[str]:
    u = underlying.upper().strip()
    if u in _INDEX_NSE_SPOT_CANDIDATES:
        return list(_INDEX_NSE_SPOT_CANDIDATES[u])
    return [underlying]


def fetch_underlying_intraday(kite, underlying: str, nse_instruments, from_str: str, to_str: str, interval: str):
    """Load intraday OHLC for index (FO_INDEX_* keys) or equity symbol."""
    last_err: str | None = None
    cands = list(_nse_spot_candidates_for_underlying(underlying))
    fb = _nse_index_tradingsymbol_fallback(underlying, nse_instruments)
    if fb and fb not in cands:
        cands.append(fb)
    for sym in cands:
        token = get_instrument_token(sym, nse_instruments)
        if token is None:
            last_err = f"No NSE instrument for {sym}"
            continue
        try:
            candles = kite.historical_data(token, from_str, to_str, interval=interval)
        except Exception as e:
            last_err = str(e)
            continue
        df = candles_to_dataframe(candles)
        if df.empty:
            last_err = f"No underlying candles for {sym}"
            continue
        return df.sort_values("date").reset_index(drop=True), None
    return None, last_err or f"No NSE data for {underlying}"
