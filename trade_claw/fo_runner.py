"""Single-day F&O mock trade runner (underlying signal → 1 option lot). Used by day view and month scan."""
from __future__ import annotations

from datetime import date, datetime

from trade_claw.constants import (
    ENVELOPE_EMA_PERIOD,
    FO_MAX_EXPIRY_CANDLE_FALLBACKS,
    FO_OPTION_TARGET_PCT,
    MA_EMA_FAST,
    MA_EMA_SLOW,
    REPORTS_MIN_BARS_PER_DAY,
)
from trade_claw.fo_support import (
    align_option_entry_bar,
    build_option_trade_candidates,
    fetch_underlying_intraday,
)
from trade_claw.market_data import candles_to_dataframe
from trade_claw.option_trades import simulate_long_option_target_or_eod
from trade_claw.strategies import _ma_ema_crossover_analysis, _ma_envelope_analysis


def run_fo_underlying_one_day(
    *,
    kite,
    nse_instruments,
    nfo_instruments,
    underlying: str,
    name: str,
    session_date: date,
    chosen_interval: str,
    strategy_is_envelope: bool,
    envelope_pct: float,
    strategy_choice: str,
    steps_from_atm: int,
    strike_policy_label: str,
    manual_strike_val: float | None,
    brokerage_per_lot_rt: float,
    taxes_per_lot_rt: float,
    include_chart_data: bool = True,
    min_session_bars: int = REPORTS_MIN_BARS_PER_DAY,
) -> dict:
    """
    Run full F&O mock pipeline for one calendar session day and one underlying.
    Returns a row dict aligned with fo_options table/chart expectations.
    """
    from_dt = datetime(session_date.year, session_date.month, session_date.day, 9, 15, 0)
    to_dt = datetime(session_date.year, session_date.month, session_date.day, 15, 30, 0)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

    def _row(base: dict) -> dict:
        out = {**base, "Session date": session_date}
        if not include_chart_data:
            out["df_u"] = None
            out["df_o"] = None
        return out

    df_u, err_u = fetch_underlying_intraday(kite, underlying, nse_instruments, from_str, to_str, chosen_interval)
    if df_u is None or err_u:
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strategy_choice,
            "Leg": "—",
            "Option": "—",
            "Strike": float("nan"),
            "Strike pick": "—",
            "Note": err_u or "No underlying data",
            "Signal": "—",
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": float("nan"),
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": None,
            "df_o": None,
            "trade": None,
        })

    if len(df_u) < min_session_bars:
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strategy_choice,
            "Leg": "—",
            "Option": "—",
            "Strike": float("nan"),
            "Strike pick": "—",
            "Note": f"Too few bars ({len(df_u)} < {min_session_bars}) — likely holiday or no session",
            "Signal": "—",
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": float("nan"),
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": df_u,
            "df_o": None,
            "trade": None,
        })

    if strategy_is_envelope:
        ok, text, sig = _ma_envelope_analysis(df_u, ema_period=ENVELOPE_EMA_PERIOD, pct=envelope_pct)
        strat_label = "Envelope → option"
    else:
        ok, text, sig = _ma_ema_crossover_analysis(df_u, fast_period=MA_EMA_FAST, slow_period=MA_EMA_SLOW)
        strat_label = "MA cross → option"

    if not ok or not sig.get("signal"):
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strat_label,
            "Leg": "—",
            "Option": "—",
            "Strike": float("nan"),
            "Strike pick": "—",
            "Note": text[:200] + ("…" if len(text) > 200 else ""),
            "Signal": "—",
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": float("nan"),
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": df_u,
            "df_o": None,
            "trade": None,
            "_analysis_text": text,
        })

    spot_signal = sig["signal"]
    entry_bar_idx = sig.get("entry_bar_idx")
    try:
        entry_bar_idx = int(entry_bar_idx) if entry_bar_idx is not None else None
    except (TypeError, ValueError):
        entry_bar_idx = None
    if entry_bar_idx is None or not (0 <= entry_bar_idx < len(df_u)):
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strat_label,
            "Leg": "—",
            "Option": "—",
            "Strike": float("nan"),
            "Strike pick": "—",
            "Note": "Invalid entry bar",
            "Signal": spot_signal,
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": float("nan"),
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": df_u,
            "df_o": None,
            "trade": None,
        })

    _chart_mk = {}
    try:
        ei = int(entry_bar_idx)
    except (TypeError, ValueError):
        ei = None
    if spot_signal in ("BUY", "SELL") and ei is not None and 0 <= ei < len(df_u):
        _chart_mk = {"_chart_entry_bar_idx": ei, "_chart_spot_signal": spot_signal}

    spot = float(df_u.iloc[entry_bar_idx]["close"])
    leg_typ = "CE" if spot_signal == "BUY" else "PE"
    leg_label = "Long CE" if spot_signal == "BUY" else "Long PE"
    strike_pick_label = (
        f"Manual → {manual_strike_val:,.0f}" if manual_strike_val else strike_policy_label.split("—")[0].strip()
    )

    cands, opt_err = build_option_trade_candidates(
        nfo_instruments,
        underlying,
        spot,
        session_date,
        leg_typ,
        steps_from_atm,
        manual_strike_val,
        max_expiries=FO_MAX_EXPIRY_CANDLE_FALLBACKS,
    )
    if opt_err or not cands:
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strat_label,
            "Leg": "—",
            "Option": "—",
            "Strike": float("nan"),
            "Strike pick": strike_pick_label,
            "Note": opt_err or "Option resolve failed",
            "Signal": spot_signal,
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": float("nan"),
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": df_u,
            "df_o": None,
            "trade": None,
            "_analysis_text": text,
            **_chart_mk,
        })

    primary_sym = cands[0][0].get("tradingsymbol", "—")
    candle_errors: list[str] = []
    opt_inst: dict | None = None
    chosen_strike = float("nan")
    df_o = None
    chain_note = ""
    for idx, (inst, stk, ex) in enumerate(cands):
        opt_sym_try = inst.get("tradingsymbol", "—")
        token = inst.get("instrument_token")
        if token is None:
            candle_errors.append(f"{opt_sym_try}:no token")
            continue
        try:
            ocandles = kite.historical_data(int(token), from_str, to_str, interval=chosen_interval)
        except Exception as e:
            candle_errors.append(f"{opt_sym_try}:{str(e)[:48]}")
            continue
        df_try = candles_to_dataframe(ocandles)
        if df_try.empty:
            candle_errors.append(f"{opt_sym_try}:no bars")
            continue
        opt_inst = inst
        chosen_strike = float(stk)
        df_o = df_try.sort_values("date").reset_index(drop=True)
        if idx > 0:
            chain_note = (
                f"Used `{opt_sym_try}` (expiry {ex.isoformat()}) after empty/error on earlier chain row(s)."
            )
        break
    else:
        err_tail = "; ".join(candle_errors[:8])
        if len(candle_errors) > 8:
            err_tail += f" …(+{len(candle_errors) - 8} more)"
        note = "No option candles"
        if err_tail:
            note = f"No option candles — tried {len(candle_errors)} instrument(s): {err_tail}"
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strat_label,
            "Leg": leg_label,
            "Option": primary_sym,
            "Strike": float(cands[0][1]) if cands else float("nan"),
            "Strike pick": strike_pick_label,
            "Note": note,
            "Signal": spot_signal,
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": float("nan"),
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": df_u,
            "df_o": None,
            "trade": None,
            "_analysis_text": text,
            **_chart_mk,
        })

    assert opt_inst is not None and df_o is not None
    opt_sym = opt_inst.get("tradingsymbol", "—")
    opt_entry_idx = align_option_entry_bar(df_u, df_o, entry_bar_idx, chosen_interval)
    if opt_entry_idx is None or opt_entry_idx >= len(df_o):
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strat_label,
            "Leg": leg_label,
            "Option": opt_sym,
            "Strike": float(chosen_strike),
            "Strike pick": strike_pick_label,
            "Note": "Could not align option bar to underlying entry",
            "Signal": spot_signal,
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": float("nan"),
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": df_u,
            "df_o": df_o,
            "trade": None,
            "_analysis_text": text,
            **_chart_mk,
        })

    entry_premium = float(df_o.iloc[opt_entry_idx]["close"])
    if entry_premium <= 0:
        return _row({
            "Underlying": underlying,
            "Name": name,
            "Strategy": strat_label,
            "Leg": leg_label,
            "Option": opt_sym,
            "Strike": float(chosen_strike),
            "Strike pick": strike_pick_label,
            "Note": "Non-positive entry premium",
            "Signal": spot_signal,
            "Lots": 0,
            "Lot size": float("nan"),
            "Qty": 0,
            "Entry": entry_premium,
            "Target prem.": float("nan"),
            "Txn cost": 0.0,
            "P/L gross": 0.0,
            "Closed at": "—",
            "Exit": float("nan"),
            "P/L": 0.0,
            "Value": 0.0,
            "df_u": df_u,
            "df_o": df_o,
            "trade": None,
            "_analysis_text": text,
            **_chart_mk,
        })

    target_prem = entry_premium * (1.0 + FO_OPTION_TARGET_PCT)
    ls = max(1, int(opt_inst.get("lot_size") or 1))
    n_lots = 1
    qty = ls

    closed_at, exit_price, gross_pl, exit_bar_idx = simulate_long_option_target_or_eod(
        df_o, opt_entry_idx, entry_premium, target_prem, float(qty)
    )
    txn_cost = n_lots * (brokerage_per_lot_rt + taxes_per_lot_rt)
    net_pl = gross_pl - txn_cost
    value = entry_premium * qty
    trade = {
        "Strategy": strat_label,
        "Signal": spot_signal,
        "Leg": leg_label,
        "Option": opt_sym,
        "Strike": float(chosen_strike),
        "Strike pick": strike_pick_label,
        "Lots": n_lots,
        "Lot size": ls,
        "Qty": qty,
        "Entry": entry_premium,
        "Target prem.": target_prem,
        "entry_bar_idx": entry_bar_idx,
        "exit_bar_idx": exit_bar_idx,
        "opt_entry_idx": opt_entry_idx,
        "Closed at": closed_at,
        "Exit": exit_price,
        "Txn cost": txn_cost,
        "P/L gross": gross_pl,
        "P/L": net_pl,
        "Value": value,
        "why": text,
        "chain_note": chain_note,
    }
    row_note = f"OK — {chain_note}" if chain_note else "OK"
    out = {
        "Session date": session_date,
        "Underlying": underlying,
        "Name": name,
        "Strategy": strat_label,
        "Leg": leg_label,
        "Option": opt_sym,
        "Strike": float(chosen_strike),
        "Strike pick": strike_pick_label,
        "Note": row_note,
        "Signal": spot_signal,
        "Lots": n_lots,
        "Lot size": ls,
        "Qty": qty,
        "Entry": entry_premium,
        "Target prem.": target_prem,
        "Txn cost": txn_cost,
        "P/L gross": gross_pl,
        "Closed at": closed_at,
        "Exit": exit_price,
        "P/L": net_pl,
        "Value": value,
        "df_u": df_u,
        "df_o": df_o,
        "trade": trade,
        "_analysis_text": text,
        **_chart_mk,
    }
    if not include_chart_data:
        out["df_u"] = None
        out["df_o"] = None
    return out


def iter_calendar_dates_in_month(d: date) -> list[date]:
    """All calendar dates from day 1 through last day of month containing d."""
    import calendar

    _, last = calendar.monthrange(d.year, d.month)
    return [date(d.year, d.month, day) for day in range(1, last + 1)]
