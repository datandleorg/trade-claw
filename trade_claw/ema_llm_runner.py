"""Single-day EMA trend/crossover backtest with mandatory LLM option pick."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from trade_claw.fo_support import (
    align_option_entry_bar,
    build_option_trade_candidates,
    fetch_underlying_intraday,
)
from trade_claw.market_data import candles_to_dataframe
from trade_claw.mock_llm_flow_log import (
    create_flow_run_dir,
    write_flow_candidates_json,
    write_flow_deterministic,
    write_flow_outcome,
)
from trade_claw.mock_market_signal import now_ist
from trade_claw.option_trades import simulate_long_option_target_stop_eod


class EmaLlmPick(BaseModel):
    tradingsymbol: str = Field(description="Exactly one symbol from candidates")
    stop_loss: float = Field(description="Option premium stop in INR, below entry")
    target: float = Field(description="Option premium target in INR, above entry")
    rationale: str = Field(description="Short rationale for strike and risk levels")


@dataclass
class SignalEvent:
    bar_idx: int
    signal: str  # BUY / SELL
    close: float
    trend_ema: float
    fast_ema: float
    slow_ema: float


def _openai_creds() -> tuple[str, str]:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY is missing or placeholder")
    model = (os.environ.get("OPENAI_MODEL") or "gpt-5.4-mini").strip()
    return key, model


def _vwap_series(df_u: pd.DataFrame) -> pd.Series:
    vol = pd.to_numeric(df_u.get("volume"), errors="coerce").fillna(0.0)
    tp = (df_u["high"] + df_u["low"] + df_u["close"]) / 3.0
    cum_vol = vol.cumsum().replace(0.0, pd.NA)
    return ((tp * vol).cumsum() / cum_vol).fillna(method="bfill").fillna(df_u["close"])


def _is_sideways(
    close: pd.Series,
    trend: pd.Series,
    i: int,
    window: int,
    flat_slope_pct: float,
    max_crosses: int,
) -> bool:
    start = max(0, i - window + 1)
    if i - start < 3:
        return False
    trend_now = float(trend.iloc[i])
    trend_then = float(trend.iloc[start])
    if trend_then == 0:
        return False
    slope_pct = abs((trend_now - trend_then) / trend_then) * 100.0
    if slope_pct > flat_slope_pct:
        return False
    c = close.iloc[start : i + 1]
    t = trend.iloc[start : i + 1]
    above = (c > t).astype(int)
    crosses = int((above.diff().fillna(0) != 0).sum())
    return crosses >= max_crosses


def detect_ema_signals(
    df_u: pd.DataFrame,
    *,
    trend_ema_period: int,
    fast_ema_period: int,
    slow_ema_period: int,
    use_vwap_filter: bool,
    use_strong_candle: bool,
    strong_candle_min_body_frac: float,
    sideways_window: int,
    sideways_flat_slope_pct: float,
    sideways_max_crosses: int,
) -> tuple[list[SignalEvent], str]:
    close = pd.to_numeric(df_u["close"], errors="coerce")
    trend = close.ewm(span=trend_ema_period, adjust=False).mean()
    fast = close.ewm(span=fast_ema_period, adjust=False).mean()
    slow = close.ewm(span=slow_ema_period, adjust=False).mean()
    vwap = _vwap_series(df_u) if use_vwap_filter else None

    start = max(trend_ema_period, fast_ema_period, slow_ema_period) + 1
    signals: list[SignalEvent] = []
    skip_sideways = 0
    for i in range(start, len(df_u)):
        c = float(close.iloc[i])
        tr = float(trend.iloc[i])
        fp, fn = float(fast.iloc[i - 1]), float(fast.iloc[i])
        sp, sn = float(slow.iloc[i - 1]), float(slow.iloc[i])
        cross_up = fp <= sp and fn > sn
        cross_dn = fp >= sp and fn < sn
        if not cross_up and not cross_dn:
            continue
        if _is_sideways(close, trend, i, sideways_window, sideways_flat_slope_pct, sideways_max_crosses):
            skip_sideways += 1
            continue
        sig: str | None = None
        if cross_up and c > tr:
            sig = "BUY"
        elif cross_dn and c < tr:
            sig = "SELL"
        if sig is None:
            continue
        if use_vwap_filter and vwap is not None:
            vw = float(vwap.iloc[i])
            if sig == "BUY" and c <= vw:
                continue
            if sig == "SELL" and c >= vw:
                continue
        if use_strong_candle:
            o = float(df_u.iloc[i]["open"])
            h = float(df_u.iloc[i]["high"])
            l = float(df_u.iloc[i]["low"])
            rng = max(1e-9, h - l)
            body = abs(c - o) / rng
            if body < strong_candle_min_body_frac:
                continue
        signals.append(
            SignalEvent(
                bar_idx=i,
                signal=sig,
                close=c,
                trend_ema=tr,
                fast_ema=float(fast.iloc[i]),
                slow_ema=float(slow.iloc[i]),
            )
        )
    return signals, f"Detected {len(signals)} crossover signal(s); skipped sideways={skip_sideways}"


def _option_candidates_for_signal(
    *,
    nfo_instruments: list,
    underlying: str,
    session_date: date,
    signal: str,
    spot: float,
) -> tuple[list[tuple[dict, float, date]], str | None]:
    leg = "CE" if signal == "BUY" else "PE"
    out: list[tuple[dict, float, date]] = []
    seen: set[str] = set()
    for step in (-2, -1, 0, 1, 2):
        cands, err = build_option_trade_candidates(
            nfo_instruments=nfo_instruments,
            underlying=underlying,
            spot=spot,
            session_date=session_date,
            leg=leg,
            steps_from_atm=step,
            manual_strike=None,
            max_expiries=2,
        )
        if err:
            continue
        for inst, strike, ex in cands:
            ts = str(inst.get("tradingsymbol") or "")
            if not ts or ts in seen:
                continue
            seen.add(ts)
            out.append((inst, float(strike), ex))
    if not out:
        return [], f"No option candidates for {underlying} {leg}"
    return out[:12], None


def _llm_pick_for_signal(
    *,
    underlying: str,
    signal: str,
    event: SignalEvent,
    option_rows: list[dict[str, Any]],
    target_pct: float,
    stop_pct: float,
) -> EmaLlmPick:
    key, model = _openai_creds()
    llm = ChatOpenAI(api_key=key, model=model, temperature=0.1, max_completion_tokens=900)
    structured = llm.with_structured_output(EmaLlmPick)
    leg = "CE" if signal == "BUY" else "PE"
    sys = SystemMessage(
        content=(
            "You are selecting one NSE option contract for a long-premium intraday trade. "
            "Return strict JSON: tradingsymbol, stop_loss, target, rationale. "
            "tradingsymbol must be exactly one candidate symbol. "
            "stop_loss must be below candidate entry_premium. target must be above entry_premium. "
            "Use realistic distances and two decimals."
        )
    )
    human = HumanMessage(
        content=(
            f"Underlying={underlying} signal={signal} leg={leg}\n"
            f"Signal bar idx={event.bar_idx}, close={event.close:.2f}, trendEMA={event.trend_ema:.2f}, "
            f"fastEMA={event.fast_ema:.2f}, slowEMA={event.slow_ema:.2f}\n"
            f"User stop_pct={100*stop_pct:.2f}% target_pct={100*target_pct:.2f}%\n"
            f"Candidates JSON:\n{json.dumps(option_rows, indent=2)}"
        )
    )
    return structured.invoke([sys, human])


def _validate_pick(
    *,
    pick: EmaLlmPick,
    entry: float,
    target_pct: float,
    stop_pct: float,
) -> str | None:
    stop = float(pick.stop_loss)
    tgt = float(pick.target)
    if not (stop > 0 and stop < entry and tgt > entry):
        return "LLM stop/target are not around entry premium"
    tgt_min = entry * (1 + max(0.01, target_pct * 0.5))
    tgt_max = entry * (1 + max(0.05, target_pct * 2.5))
    if not (tgt_min <= tgt <= tgt_max):
        return f"LLM target out of bounds [{tgt_min:.2f}, {tgt_max:.2f}]"
    if stop_pct > 0:
        stop_lo = entry * (1 - min(0.95, stop_pct * 2.5))
        stop_hi = entry * (1 - max(0.005, stop_pct * 0.25))
        if not (stop_lo <= stop <= stop_hi):
            return f"LLM stop out of bounds [{stop_lo:.2f}, {stop_hi:.2f}]"
    return None


def _next_underlying_bar_after_ts(df_u: pd.DataFrame, ts_like: Any) -> int:
    """First underlying bar index with timestamp >= ts_like; defaults to end."""
    try:
        t_exit = pd.Timestamp(ts_like)
    except Exception:  # noqa: BLE001
        return max(0, len(df_u) - 1)
    for i, dt in enumerate(df_u["date"]):
        try:
            t_i = pd.Timestamp(dt)
        except Exception:  # noqa: BLE001
            continue
        if t_i >= t_exit:
            return i
    return max(0, len(df_u) - 1)


def run_ema_llm_backtest_one_day(
    *,
    kite,
    nse_instruments: list,
    nfo_instruments: list,
    underlying: str,
    session_date: date,
    chosen_interval: str,
    trend_ema_period: int,
    fast_ema_period: int,
    slow_ema_period: int,
    target_pct: float,
    stop_pct: float,
    use_vwap_filter: bool,
    use_strong_candle: bool,
    strong_candle_min_body_frac: float,
    brokerage_per_trade: float = 40.0,
    sideways_window: int = 14,
    sideways_flat_slope_pct: float = 0.08,
    sideways_max_crosses: int = 3,
) -> dict[str, Any]:
    from_dt = datetime(session_date.year, session_date.month, session_date.day, 9, 15, 0)
    to_dt = datetime(session_date.year, session_date.month, session_date.day, 15, 30, 0)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    df_u, err_u = fetch_underlying_intraday(kite, underlying, nse_instruments, from_str, to_str, chosen_interval)
    if df_u is None or err_u:
        return {"error": err_u or "No underlying data", "df_u": None, "trades": []}
    df_u = df_u.sort_values("date").reset_index(drop=True)
    signals, sig_note = detect_ema_signals(
        df_u,
        trend_ema_period=trend_ema_period,
        fast_ema_period=fast_ema_period,
        slow_ema_period=slow_ema_period,
        use_vwap_filter=use_vwap_filter,
        use_strong_candle=use_strong_candle,
        strong_candle_min_body_frac=strong_candle_min_body_frac,
        sideways_window=sideways_window,
        sideways_flat_slope_pct=sideways_flat_slope_pct,
        sideways_max_crosses=sideways_max_crosses,
    )
    log_root_env = (os.environ.get("MOCK_LLM_PROMPT_LOG_DIR") or "").strip()
    log_root = (
        Path(log_root_env).expanduser().resolve()
        if log_root_env
        else Path("data/mock_llm_flows").resolve()
    )
    log_root.mkdir(parents=True, exist_ok=True)
    out_trades: list[dict[str, Any]] = []
    next_signal_bar_idx = 0
    for ev in signals:
        if int(ev.bar_idx) < int(next_signal_bar_idx):
            out_trades.append(
                {
                    "status": "skipped",
                    "reason": f"Signal skipped: active trade window until bar {next_signal_bar_idx}",
                    "signal": ev,
                    "flow_id": "",
                }
            )
            continue
        flow_id, flow_dir = create_flow_run_dir(log_root, session_d=session_date, ist_now=now_ist())
        write_flow_deterministic(
            flow_dir,
            {
                "strategy_id": "ema_trend_cross_llm",
                "underlying": underlying,
                "signal": ev.signal,
                "signal_bar_idx": ev.bar_idx,
                "signal_bar_time": str(df_u.iloc[ev.bar_idx]["date"]),
                "trend_ema_period": trend_ema_period,
                "fast_ema_period": fast_ema_period,
                "slow_ema_period": slow_ema_period,
                "filters": {
                    "use_vwap_filter": use_vwap_filter,
                    "use_strong_candle": use_strong_candle,
                    "strong_candle_min_body_frac": strong_candle_min_body_frac,
                },
            },
        )
        cands, cand_err = _option_candidates_for_signal(
            nfo_instruments=nfo_instruments,
            underlying=underlying,
            session_date=session_date,
            signal=ev.signal,
            spot=ev.close,
        )
        if cand_err:
            write_flow_outcome(flow_dir, error=cand_err)
            out_trades.append({"status": "skipped", "reason": cand_err, "signal": ev, "flow_id": flow_id})
            continue
        option_rows: list[dict[str, Any]] = []
        option_df_by_symbol: dict[str, pd.DataFrame] = {}
        option_meta_by_symbol: dict[str, dict[str, Any]] = {}
        candle_load_errors: list[str] = []
        for inst, strike, ex in cands:
            ts = str(inst.get("tradingsymbol") or "")
            tok = inst.get("instrument_token")
            if not ts or tok is None:
                continue
            try:
                oc = kite.historical_data(int(tok), from_str, to_str, interval=chosen_interval)
                df_o = candles_to_dataframe(oc).sort_values("date").reset_index(drop=True)
            except Exception as e:  # noqa: BLE001
                candle_load_errors.append(f"{ts}:{e}")
                continue
            if df_o.empty:
                candle_load_errors.append(f"{ts}:no bars")
                continue
            oei = align_option_entry_bar(df_u, df_o, ev.bar_idx, chosen_interval)
            if oei is None or not (0 <= int(oei) < len(df_o)):
                candle_load_errors.append(f"{ts}:align failed")
                continue
            entry = float(df_o.iloc[int(oei)]["close"])
            if entry <= 0:
                candle_load_errors.append(f"{ts}:entry<=0")
                continue
            option_rows.append(
                {
                    "tradingsymbol": ts,
                    "strike": float(strike),
                    "expiry": ex.isoformat(),
                    "entry_premium": round(entry, 2),
                    "lot_size": int(inst.get("lot_size") or 1),
                }
            )
            option_df_by_symbol[ts] = df_o
            option_meta_by_symbol[ts] = {"entry_idx": int(oei), "strike": float(strike), "inst": inst}
        write_flow_candidates_json(flow_dir, option_rows)
        if not option_rows:
            err = "All option candidates failed candle/align checks"
            if candle_load_errors:
                err = f"{err}: {'; '.join(candle_load_errors[:8])}"
            write_flow_outcome(flow_dir, error=err)
            out_trades.append({"status": "skipped", "reason": err, "signal": ev, "flow_id": flow_id})
            continue
        try:
            pick = _llm_pick_for_signal(
                underlying=underlying,
                signal=ev.signal,
                event=ev,
                option_rows=option_rows,
                target_pct=target_pct,
                stop_pct=stop_pct,
            )
        except Exception as e:  # noqa: BLE001
            err = f"LLM failed: {e}"
            write_flow_outcome(flow_dir, error=err)
            out_trades.append({"status": "skipped", "reason": err, "signal": ev, "flow_id": flow_id})
            continue
        pick_obj = {
            "tradingsymbol": pick.tradingsymbol,
            "stop_loss": float(pick.stop_loss),
            "target": float(pick.target),
            "rationale": pick.rationale.strip(),
        }
        sym = pick.tradingsymbol.strip()
        if sym not in option_df_by_symbol:
            err = f"LLM picked unknown symbol: {sym}"
            write_flow_outcome(flow_dir, error=err)
            out_trades.append(
                {
                    "status": "skipped",
                    "reason": err,
                    "signal": ev,
                    "flow_id": flow_id,
                    "llm_output": pick_obj,
                }
            )
            continue
        df_o = option_df_by_symbol[sym]
        meta = option_meta_by_symbol[sym]
        entry_idx = int(meta["entry_idx"])
        entry = float(df_o.iloc[entry_idx]["close"])
        v_err = _validate_pick(pick=pick, entry=entry, target_pct=target_pct, stop_pct=stop_pct)
        if v_err:
            write_flow_outcome(flow_dir, error=v_err)
            out_trades.append(
                {
                    "status": "skipped",
                    "reason": v_err,
                    "signal": ev,
                    "flow_id": flow_id,
                    "llm_output": pick_obj,
                }
            )
            continue
        lot_size = max(1, int(meta["inst"].get("lot_size") or 1))
        qty = lot_size
        closed_at, exit_price, gross_pl, exit_bar_idx = simulate_long_option_target_stop_eod(
            df_o,
            entry_bar_idx=entry_idx,
            entry_price=entry,
            target_price=float(pick.target),
            qty=float(qty),
            stop_price=float(pick.stop_loss),
        )
        net_pl = float(gross_pl) - float(brokerage_per_trade)
        trade = {
            "status": "executed",
            "flow_id": flow_id,
            "flow_dir": str(flow_dir),
            "signal": ev.signal,
            "signal_bar_idx": ev.bar_idx,
            "signal_bar_time": str(df_u.iloc[ev.bar_idx]["date"]),
            "underlying_entry": ev.close,
            "option": sym,
            "strike": float(meta["strike"]),
            "entry": entry,
            "target": float(pick.target),
            "stop": float(pick.stop_loss),
            "closed_at": closed_at,
            "exit": float(exit_price),
            "lot_size": lot_size,
            "qty": qty,
            "txn_cost": float(brokerage_per_trade),
            "pl_gross": float(gross_pl),
            "pl_net": float(net_pl),
            "rationale": pick.rationale.strip(),
            "llm_output": pick_obj,
            "opt_entry_idx": entry_idx,
            "opt_exit_idx": int(exit_bar_idx),
            "df_o": df_o,
        }
        write_flow_outcome(flow_dir, trade_id=1, error=None)
        out_trades.append(trade)
        exit_u_idx = _next_underlying_bar_after_ts(df_u, df_o.iloc[int(exit_bar_idx)]["date"])
        next_signal_bar_idx = int(exit_u_idx) + 1
    return {"error": None, "df_u": df_u, "trades": out_trades, "note": sig_note}
