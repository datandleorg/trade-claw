"""LangGraph: envelope signal → top-5 contracts → LLM pick → mock DB insert."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, TypedDict

from kiteconnect import KiteConnect
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from trade_claw.constants import ENVELOPE_EMA_PERIOD, FO_INDEX_UNDERLYING_LABELS
from trade_claw.env_trading_params import (
    mock_llm_risk_instruction,
    mock_llm_sltp_fallback_to_env,
    mock_llm_sltp_stop_pct_bounds,
    mock_llm_sltp_target_pct_bounds,
    option_stop_premium_fraction,
    option_target_premium_fraction,
)
from trade_claw.fo_support import _to_date
from trade_claw.mock_market_signal import (
    envelope_breakout_on_last_bar,
    load_index_session_minute_df,
    mock_agent_envelope_pct_for_underlying,
    mock_agent_slippage_points,
    mock_engine_option_stop_multiplier,
    mock_engine_option_target_multiplier,
    mock_engine_underlyings,
    top_five_option_instruments,
)
from trade_claw import mock_trade_store
from trade_claw.mock_engine_log import scan_info, scan_warning


def _resolve_entry_sltp(
    entry: float,
    llm_stop: float | None,
    llm_target: float | None,
) -> tuple[float, float, str | None]:
    """
    Validate LLM stop/target vs ``entry`` and env percent bands.
    Returns (stop_loss, target, error). On error, caller skips insert unless handled.
    """
    tgt_lo_p, tgt_hi_p = mock_llm_sltp_target_pct_bounds()
    st_lo_p, st_hi_p = mock_llm_sltp_stop_pct_bounds()
    tgt_min = entry * (1 + tgt_lo_p)
    tgt_max = entry * (1 + tgt_hi_p)
    st_floor = entry * (1 - st_hi_p)
    st_ceil = entry * (1 - st_lo_p)

    def from_env() -> tuple[float, float]:
        return (
            round(entry * mock_engine_option_stop_multiplier(), 2),
            round(entry * mock_engine_option_target_multiplier(), 2),
        )

    if llm_stop is None or llm_target is None:
        if mock_llm_sltp_fallback_to_env():
            s, t = from_env()
            return s, t, None
        return 0.0, 0.0, "LLM must return stop_loss and target (option premium ₹)"

    stop = round(float(llm_stop), 2)
    target = round(float(llm_target), 2)

    ok = (
        stop > 0
        and stop < entry
        and target > entry
        and st_floor <= stop <= st_ceil
        and tgt_min <= target <= tgt_max
    )
    if ok:
        return stop, target, None
    if mock_llm_sltp_fallback_to_env():
        scan_warning(
            "graph_err",
            "EXECUTE LLM SL/TP invalid entry=%.2f stop=%.2f target=%.2f (bands stop [%.2f,%.2f] target [%.2f,%.2f]) → env multipliers",
            entry,
            stop,
            target,
            st_floor,
            st_ceil,
            tgt_min,
            tgt_max,
        )
        s, t = from_env()
        return s, t, None
    return (
        0.0,
        0.0,
        f"Invalid LLM stop/target for entry ₹{entry:.2f}: need ₹{st_floor:.2f}≤stop≤₹{st_ceil:.2f}, "
        f"₹{tgt_min:.2f}≤target≤₹{tgt_max:.2f} (or set MOCK_LLM_SLTP_FALLBACK=1)",
    )
from trade_claw.mock_trade_snapshot import (
    fetch_index_minute_bars_json,
    fetch_option_minute_bars_json,
    snapshot_bar_count,
)


class LLMPick(BaseModel):
    tradingsymbol: str = Field(description="Must be exactly one tradingsymbol from the candidate list")
    rationale: str = Field(description="Short rationale for strike and risk levels")
    stop_loss: float = Field(description="Stop loss for the chosen option, premium in INR (must be below synthetic entry)")
    target: float = Field(description="Take-profit for the chosen option, premium in INR (must be above synthetic entry)")


class TradingState(TypedDict, total=False):
    error: str
    notes: str
    signal_underlying: str
    underlying: str
    direction: str
    leg: str
    spot: float
    signal_bar_time: str
    signal_text: str
    candidates: list[dict[str, Any]]
    llm_tradingsymbol: str
    llm_stop_loss: float
    llm_target: float
    stop_loss: float
    target: float
    llm_rationale: str
    trade_id: int


def _openai_creds() -> tuple[str, str]:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY is missing or placeholder")
    model = (os.environ.get("OPENAI_MODEL") or "gpt-5.4-mini").strip()
    return key, model


def build_mock_trading_graph(
    *,
    kite: KiteConnect,
    session_d: date,
    nse_instruments: list,
    nfo_instruments: list,
    checkpointer: Any | None,
) -> CompiledStateGraph:
    key, model = _openai_creds()
    llm = ChatOpenAI(api_key=key, model=model, temperature=0.2, max_completion_tokens=1200)
    structured = llm.with_structured_output(LLMPick)

    def signal_node(state: TradingState) -> TradingState:
        parts: list[str] = []
        focus = (state.get("signal_underlying") or "").strip().upper()
        allowed = mock_engine_underlyings()
        if focus:
            if focus not in allowed:
                scan_warning(
                    "graph_err",
                    "SIGNAL bad signal_underlying=%r not in MOCK_ENGINE_UNDERLYINGS",
                    focus,
                )
                return {
                    "notes": f"signal_underlying {focus!r} not in MOCK_ENGINE_UNDERLYINGS",
                    "signal_text": f"signal_underlying {focus!r} not in MOCK_ENGINE_UNDERLYINGS",
                }
            indices_to_scan = [focus]
        else:
            indices_to_scan = list(allowed)
        for u in indices_to_scan:
            df, err = load_index_session_minute_df(kite, nse_instruments, session_d, u)
            if df is None or err:
                parts.append(f"{u}: {err or 'no data'}")
                continue
            envelope_pct = mock_agent_envelope_pct_for_underlying(u)
            ok, text, sig = envelope_breakout_on_last_bar(
                df, ema_period=ENVELOPE_EMA_PERIOD, pct=envelope_pct
            )
            if ok and sig.get("direction"):
                scan_info(
                    "signal",
                    "SIGNAL underlying=%s direction=%s leg=%s spot=%.2f bar=%s",
                    u,
                    sig["direction"],
                    sig["leg"],
                    float(sig["spot"] or 0),
                    sig.get("signal_bar_time") or "",
                )
                return {
                    "underlying": u,
                    "direction": sig["direction"],
                    "leg": sig["leg"],
                    "spot": float(sig["spot"] or 0),
                    "signal_bar_time": sig.get("signal_bar_time") or "",
                    "signal_text": text,
                }
            parts.append(f"{u}: {text}")
        combined = " | ".join(parts)
        if len(combined) > 6000:
            combined = combined[:5997] + "..."
        return {"notes": combined, "signal_text": combined}

    def candidates_node(state: TradingState) -> TradingState:
        if state.get("error") or not state.get("direction"):
            return {}
        idx = (state.get("underlying") or "NIFTY").strip().upper()
        cands, err = top_five_option_instruments(
            nfo_instruments,
            underlying=idx,
            session_d=session_d,
            spot=float(state.get("spot") or 0),
            leg=str(state.get("leg") or ""),
        )
        if err or not cands:
            scan_warning(
                "graph_err",
                "CANDIDATES fail underlying=%s err=%s",
                idx,
                err or "No option candidates",
            )
            return {"error": err or "No option candidates"}
        keys = [f"NFO:{i.get('tradingsymbol', '')}" for i in cands if i.get("tradingsymbol")]
        ltps: dict[str, Any] = {}
        try:
            ltps = kite.ltp(keys) if keys else {}
        except Exception as e:  # noqa: BLE001
            scan_warning("ltp_warn", "kite.ltp candidates batch failed: %s", e)
        enriched: list[dict[str, Any]] = []
        for i in cands:
            ts = i.get("tradingsymbol") or ""
            ltp = None
            if ts:
                row = ltps.get(f"NFO:{ts}")
                if isinstance(row, dict):
                    ltp = row.get("last_price")
            ed = _to_date(i.get("expiry"))
            dte = (ed - session_d).days if ed else None
            enriched.append(
                {
                    "tradingsymbol": ts,
                    "strike": float(i.get("strike", 0)),
                    "instrument_type": i.get("instrument_type"),
                    "expiry": str(ed) if ed else "",
                    "dte_days": dte,
                    "lot_size": int(i.get("lot_size") or 1),
                    "ltp": float(ltp) if ltp is not None else None,
                }
            )
        return {"candidates": enriched}

    def llm_node(state: TradingState) -> TradingState:
        if state.get("error") or not state.get("candidates"):
            return {}
        cands = state["candidates"]
        allowed = {c["tradingsymbol"] for c in cands if c.get("tradingsymbol")}
        idx = (state.get("underlying") or "NIFTY").strip().upper()
        idx_label = FO_INDEX_UNDERLYING_LABELS.get(idx, idx)
        tgt_lo_p, tgt_hi_p = mock_llm_sltp_target_pct_bounds()
        st_lo_p, st_hi_p = mock_llm_sltp_stop_pct_bounds()
        sug_tgt = option_target_premium_fraction()
        sug_stp = option_stop_premium_fraction()
        risk_extra = mock_llm_risk_instruction()
        risk_block = f"\n\nAdditional risk guidance from operator:\n{risk_extra}" if risk_extra else ""
        sys = SystemMessage(
            content=(
                "You choose one **NSE F&O option** contract (index or single-stock underlying) for a "
                "**long premium only** mock trade "
                "(calls for bullish, puts for bearish — wrong type already removed in code). "
                "Pick the best strike/expiry among the candidates using DTE, moneyness vs spot, and liquidity (LTP). "
                "You must also output **stop_loss** and **target** as option **premium prices in Indian rupees (₹)** "
                "for the **same contract** you pick (the chosen row's LTP is your reference). "
                "Synthetic long entry will be approximately **LTP minus ~0.5–1.0 ₹ slippage** — "
                "your stop_loss must be **strictly below** that entry and target **strictly above** it: "
                "stop_loss < entry < target. "
                f"Suggested distance from entry: target about +{100 * sug_tgt:.1f}% above entry, "
                f"stop about −{100 * sug_stp:.1f}% below entry (guidance only). "
                f"Hard bounds (validated in code): target between +{100 * tgt_lo_p:.1f}% and +{100 * tgt_hi_p:.1f}% "
                f"above entry; stop between −{100 * st_hi_p:.1f}% and −{100 * st_lo_p:.1f}% below entry "
                f"(i.e. stop premium in that band below entry). "
                "Use two decimal places mentally; output numeric rupee levels."
                f"{risk_block}"
            )
        )
        human = HumanMessage(
            content=(
                f"Underlying: {idx} ({idx_label}). Spot ~ {state.get('spot', 0):.2f}. "
                f"Direction: {state.get('direction')}. Leg: {state.get('leg')}.\n"
                f"Candidates (JSON):\n{json.dumps(cands, indent=2)}"
            )
        )
        try:
            pick: LLMPick = structured.invoke([sys, human])
        except Exception as e:  # noqa: BLE001
            scan_warning("graph_err", "LLM invoke failed underlying=%s: %s", idx, e)
            return {"error": f"LLM failed: {e}"}
        if pick.tradingsymbol not in allowed:
            scan_warning(
                "graph_err",
                "LLM bad symbol underlying=%s picked=%r",
                idx,
                pick.tradingsymbol,
            )
            return {"error": f"LLM picked unknown symbol {pick.tradingsymbol!r}"}
        scan_info(
            "llm_pick",
            "LLM_PICK underlying=%s symbol=%s stop=%.2f target=%.2f",
            idx,
            pick.tradingsymbol,
            float(pick.stop_loss),
            float(pick.target),
        )
        return {
            "llm_tradingsymbol": pick.tradingsymbol,
            "llm_rationale": pick.rationale.strip(),
            "llm_stop_loss": float(pick.stop_loss),
            "llm_target": float(pick.target),
        }

    def execute_node(state: TradingState) -> TradingState:
        if state.get("error") or not state.get("llm_tradingsymbol"):
            return {}
        tsym = state["llm_tradingsymbol"]
        cands = state.get("candidates") or []
        meta = next((c for c in cands if c.get("tradingsymbol") == tsym), None)
        if not meta:
            scan_warning("graph_err", "EXECUTE missing metadata for symbol=%s", tsym)
            return {"error": "Execute: symbol metadata missing"}
        lot_size = max(1, int(meta.get("lot_size") or 1))
        try:
            ltp_row = kite.ltp([f"NFO:{tsym}"])
            raw = ltp_row.get(f"NFO:{tsym}", {})
            ltp = float(raw.get("last_price") or 0)
        except Exception as e:  # noqa: BLE001
            scan_warning("graph_err", "EXECUTE LTP fetch failed %s: %s", tsym, e)
            return {"error": f"LTP fetch failed: {e}"}
        if ltp <= 0:
            ltp = float(meta.get("ltp") or 0)
        if ltp <= 0:
            scan_warning("graph_err", "EXECUTE no LTP for %s", tsym)
            return {"error": "No LTP for execution"}
        slip = mock_agent_slippage_points()
        entry = max(0.01, ltp - slip)
        stop_loss, target, sltp_err = _resolve_entry_sltp(
            entry,
            state.get("llm_stop_loss"),
            state.get("llm_target"),
        )
        if sltp_err:
            scan_warning("graph_err", "EXECUTE SL/TP %s", sltp_err)
            return {"error": sltp_err}
        qty_units = lot_size  # mock: always 1 exchange lot (PnL uses units = lot_size)
        idx = (state.get("underlying") or "NIFTY").strip().upper()
        try:
            tid = mock_trade_store.insert_open_trade(
                instrument=tsym,
                direction=str(state.get("direction") or ""),
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                llm_rationale=str(state.get("llm_rationale") or ""),
                lot_size=lot_size,
                quantity=qty_units,
                index_underlying=idx,
            )
        except ValueError as e:
            scan_warning("graph_err", "EXECUTE insert rejected: %s", e)
            return {"error": str(e)}
        nb = snapshot_bar_count()
        if nb > 0:
            snap = fetch_option_minute_bars_json(
                kite, nfo_instruments, tsym, session_d, max_bars=nb
            )
            if snap:
                mock_trade_store.update_entry_bars_json(tid, snap)
            try:
                u_snap = fetch_index_minute_bars_json(
                    kite, nse_instruments, session_d, idx, max_bars=nb
                )
                if u_snap:
                    mock_trade_store.update_entry_underlying_bars_json(tid, u_snap)
            except Exception as e:  # noqa: BLE001
                scan_warning("ltp_warn", "entry underlying snapshot failed: %s", e)
        scan_info(
            "open",
            "OPEN trade_id=%s index=%s instrument=%s entry=%.2f stop=%.2f target=%.2f qty=%s",
            tid,
            idx,
            tsym,
            entry,
            stop_loss,
            target,
            qty_units,
        )
        return {"trade_id": tid, "stop_loss": stop_loss, "target": target}

    def route_after_signal(state: TradingState) -> str:
        if state.get("error"):
            return "end"
        if state.get("direction") and state.get("leg") and state.get("underlying"):
            return "candidates"
        return "end"

    def route_after_candidates(state: TradingState) -> str:
        if state.get("error") or not state.get("candidates"):
            return "end"
        return "llm"

    def route_after_llm(state: TradingState) -> str:
        if state.get("error") or not state.get("llm_tradingsymbol"):
            return "end"
        return "execute"

    g = StateGraph(TradingState)
    g.add_node("signal", signal_node)
    g.add_node("candidates", candidates_node)
    g.add_node("llm", llm_node)
    g.add_node("execute", execute_node)
    g.add_edge(START, "signal")
    g.add_conditional_edges("signal", route_after_signal, {"candidates": "candidates", "end": END})
    g.add_conditional_edges("candidates", route_after_candidates, {"llm": "llm", "end": END})
    g.add_conditional_edges("llm", route_after_llm, {"execute": "execute", "end": END})
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer)


def invoke_mock_graph(
    *,
    kite: KiteConnect,
    checkpointer: Any,
    session_d: date,
    nse_instruments: list,
    nfo_instruments: list,
    initial_state: dict[str, Any] | None = None,
) -> TradingState:
    import uuid

    graph = build_mock_trading_graph(
        kite=kite,
        session_d=session_d,
        nse_instruments=nse_instruments,
        nfo_instruments=nfo_instruments,
        checkpointer=checkpointer,
    )
    thread_id = f"mock_{session_d.isoformat()}_{uuid.uuid4().hex[:12]}"
    seed: dict[str, Any] = dict(initial_state) if initial_state else {}
    return graph.invoke(seed, config={"configurable": {"thread_id": thread_id}})
