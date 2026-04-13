"""LangGraph: envelope signal → top-5 contracts → LLM pick → mock DB insert."""

from __future__ import annotations

import base64
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from kiteconnect import KiteConnect
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from trade_claw.constants import ENVELOPE_EMA_PERIOD, FO_INDEX_UNDERLYING_LABELS
from trade_claw.env_trading_params import (
    mock_llm_attach_underlying_chart,
    mock_llm_prompt_log_dir,
    mock_llm_risk_instruction,
    mock_llm_sltp_fallback_to_env,
    mock_llm_sltp_stop_pct_bounds,
    mock_llm_sltp_target_pct_bounds,
    mock_llm_vision_model,
    mock_engine_option_target_price,
    mock_engine_option_target_rupees_above_entry,
    option_stop_premium_fraction,
    option_target_premium_fraction,
)
from trade_claw.mock_llm_flow_log import (
    create_flow_run_dir,
    update_flow_deterministic,
    write_flow_candidates_json,
    write_flow_deterministic,
    write_flow_outcome,
)
from trade_claw.mock_llm_prompt_log import (
    write_breakout_assessment_log,
    write_llm_invoke_error_log,
    write_llm_structured_output_log,
    write_llm_turn_snapshot,
    write_signal_llm_turn_log,
)
from trade_claw.mock_llm_signal_chart import underlying_session_chart_png_bytes
from trade_claw.fo_support import _to_date
from trade_claw.mock_market_signal import (
    envelope_breakout_on_last_bar,
    load_index_session_minute_df,
    now_ist,
    mock_agent_envelope_pct_for_underlying,
    mock_agent_slippage_points,
    mock_engine_option_stop_multiplier,
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
    When ``MOCK_ENGINE_OPTION_TARGET_RUPEES`` is set, the stored target is always entry + that amount;
    only the stop is checked against the stop band (LLM target is not persisted).
    Returns (stop_loss, target, error). On error, caller skips insert unless handled.
    """
    fixed_tgt_rs = mock_engine_option_target_rupees_above_entry()
    tgt_lo_p, tgt_hi_p = mock_llm_sltp_target_pct_bounds()
    st_lo_p, st_hi_p = mock_llm_sltp_stop_pct_bounds()
    tgt_min = entry * (1 + tgt_lo_p)
    tgt_max = entry * (1 + tgt_hi_p)
    st_floor = entry * (1 - st_hi_p)
    st_ceil = entry * (1 - st_lo_p)
    env_target = mock_engine_option_target_price(entry)

    def from_env() -> tuple[float, float]:
        return (
            round(entry * mock_engine_option_stop_multiplier(), 2),
            env_target,
        )

    if llm_stop is None or llm_target is None:
        if mock_llm_sltp_fallback_to_env():
            s, t = from_env()
            return s, t, None
        return 0.0, 0.0, "LLM must return stop_loss and target (option premium ₹)"

    stop = round(float(llm_stop), 2)
    target = round(float(llm_target), 2)

    if fixed_tgt_rs is not None:
        stop_valid = stop > 0 and stop < entry and st_floor <= stop <= st_ceil
        if stop_valid:
            return stop, env_target, None
        if mock_llm_sltp_fallback_to_env():
            scan_warning(
                "graph_err",
                "EXECUTE LLM SL/TP invalid entry=%.2f stop=%.2f (fixed target +₹%.2f; bands stop [%.2f,%.2f]) → env stop + fixed target",
                entry,
                stop,
                fixed_tgt_rs,
                st_floor,
                st_ceil,
            )
            s, t = from_env()
            return s, t, None
        return (
            0.0,
            0.0,
            f"Invalid LLM stop for entry ₹{entry:.2f}: need ₹{st_floor:.2f}≤stop≤₹{st_ceil:.2f} "
            f"(or set MOCK_LLM_SLTP_FALLBACK=1). Target is fixed at ₹{env_target:.2f} (+₹{fixed_tgt_rs:,.0f} premium).",
        )

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


class LLMBreakoutAssessment(BaseModel):
    breakout: bool = Field(
        description="True only if the latest completed bar shows a fresh EMA-envelope breakout (close outside band)"
    )
    direction: str = Field(
        description="BULLISH or BEARISH when breakout is true; empty string when false",
    )
    leg: str = Field(description="CE for BULLISH, PE for BEARISH, empty when no breakout")
    rationale: str = Field(description="Brief reasoning from the chart (1–4 sentences)")
    signal_bar_time: str = Field(
        default="",
        description="Breakout bar timestamp if visible (ISO-like), else empty",
    )


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
    signal_llm_rationale: str
    signal_envelope_pct: float
    trade_id: int
    llm_flow_id: str
    llm_flow_dir: str
    vision_validation_status: str


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

    _LLM_BREAKOUT_REJECT = "__llm_breakout_reject__"
    _VISION_INVOKE_FALLBACK = "__vision_invoke_fallback__"

    def _llm_validate_envelope_breakout(
        u: str,
        df: pd.DataFrame,
        pct: float,
        parts: list[str],
        provisional: dict[str, Any],
        vision_log_dir: Path | None,
        *,
        chart_png: bytes | None = None,
    ) -> dict[str, Any] | str:
        """
        Vision **validation** after a deterministic envelope signal. Returns:
        - ``dict`` → confirmed signal state;
        - ``_LLM_BREAKOUT_REJECT`` → model vetoes (no trade this underlying);
        - ``_VISION_INVOKE_FALLBACK`` → API error (caller uses deterministic fallback).
        """
        idx_label = FO_INDEX_UNDERLYING_LABELS.get(u, u)
        if chart_png is None:
            chart_png = underlying_session_chart_png_bytes(
                df,
                envelope_pct=pct,
                signal_bar_time=None,
                underlying_label=idx_label,
                direction="",
                assessment_mode=True,
            )
        if chart_png is None:
            return _VISION_INVOKE_FALLBACK
        pd_dir = str(provisional.get("direction") or "")
        pd_leg = str(provisional.get("leg") or "")
        sys_b = SystemMessage(
            content=(
                f"You review an NSE underlying **1-minute** spot chart for **{idx_label}**. "
                f"Candles are OHLC; curves are **EMA {ENVELOPE_EMA_PERIOD}** and **±{100 * pct:.3f}%** envelope "
                "(same geometry as the mock engine). A **yellow vertical line** marks the **latest bar**.\n"
                "The **rule-based mock engine** has already flagged a **fresh envelope breakout**: "
                f"**{pd_dir}** (long **{pd_leg}**). Your job is to **confirm or reject** that assessment from the image.\n"
                "Set **breakout=true** only if the **last completed candle** clearly supports that breakout: "
                "**BULLISH** = close clearly **above** the upper band; **BEARISH** = close clearly **below** the lower band.\n"
                "If you disagree, the chart is ambiguous, or the move is not a decisive close outside the band, "
                "set **breakout=false**. When **breakout=true**, **direction** / **leg** must match what you see "
                "(BULLISH/CE or BEARISH/PE). **rationale** must cite the chart."
            )
        )
        last_close = float(df["close"].iloc[-1])
        hum_txt = (
            f"Underlying {u} ({idx_label}). Last close ≈ {last_close:.2f}.\n\n"
            f"**Deterministic rule summary:**\n{provisional.get('text', '')}\n\n"
            "Confirm (**breakout=true**) or reject (**breakout=false**) this breakout using the chart image."
        )
        b64 = base64.standard_b64encode(chart_png).decode("ascii")
        human = HumanMessage(
            content=[
                {"type": "text", "text": hum_txt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
        invoke_model = mock_llm_vision_model() or model
        llm_br = ChatOpenAI(
            api_key=key, model=invoke_model, temperature=0.1, max_completion_tokens=700
        ).with_structured_output(LLMBreakoutAssessment)
        vdir = vision_log_dir
        if vdir is not None:
            sys_t = sys_b.content if isinstance(sys_b.content, str) else json.dumps(sys_b.content)
            write_llm_turn_snapshot(
                vdir,
                system_text=sys_t,
                human_text=hum_txt,
                chart_png=chart_png,
                meta={
                    "llm_flow": "breakout_validation",
                    "underlying": u,
                    "invoke_model": invoke_model,
                    "envelope_pct": pct,
                    "envelope_ema_period": ENVELOPE_EMA_PERIOD,
                    "provisional_direction": pd_dir,
                    "provisional_leg": pd_leg,
                },
            )
        try:
            out: LLMBreakoutAssessment = llm_br.invoke([sys_b, human])
        except Exception as e:  # noqa: BLE001
            scan_warning("graph_err", "LLM vision validate failed underlying=%s: %s", u, e)
            if vdir is not None:
                write_llm_invoke_error_log(vdir, str(e), exc_type=type(e).__name__)
            return _VISION_INVOKE_FALLBACK
        rat = (out.rationale or "").strip()
        if vdir is not None:
            write_breakout_assessment_log(vdir, out, invoke_path="multimodal")
        if not out.breakout:
            parts.append(f"{u}: LLM veto breakout — {rat[:320]}" if rat else f"{u}: LLM veto breakout")
            return _LLM_BREAKOUT_REJECT
        d = (out.direction or "").strip().upper()
        leg_m = (out.leg or "").strip().upper()
        direction: str | None = None
        leg: str | None = None
        if "BULL" in d or leg_m == "CE":
            direction, leg = "BULLISH", "CE"
        elif "BEAR" in d or leg_m == "PE":
            direction, leg = "BEARISH", "PE"
        if direction is None or leg is None:
            scan_warning(
                "graph_err",
                "LLM validate inconsistent underlying=%s direction=%r leg=%r",
                u,
                out.direction,
                out.leg,
            )
            parts.append(f"{u}: LLM validate parse failed — {rat[:200]}" if rat else f"{u}: LLM validate parse failed")
            return _LLM_BREAKOUT_REJECT
        spot = float(provisional.get("spot") or df["close"].iloc[-1])
        sbt = (out.signal_bar_time or "").strip()
        if not sbt:
            sbt = str(provisional.get("signal_bar_time") or "").strip()
        if not sbt:
            last_dt = df["date"].iloc[-1]
            sbt = last_dt.isoformat() if hasattr(last_dt, "isoformat") else str(last_dt)
        scan_info(
            "signal",
            "SIGNAL_VISION_OK underlying=%s direction=%s leg=%s spot=%.2f bar=%s",
            u,
            direction,
            leg,
            spot,
            sbt,
        )
        return {
            "underlying": u,
            "direction": direction,
            "leg": leg,
            "spot": spot,
            "signal_bar_time": sbt,
            "signal_text": f"{u}: vision confirmed — {rat}" if rat else f"{u}: vision confirmed",
            "signal_llm_rationale": rat,
            "signal_envelope_pct": pct,
        }

    def signal_node(state: TradingState) -> TradingState:
        parts: list[str] = []
        last_scanned_pct: float | None = None
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
        log_root_str = mock_llm_prompt_log_dir()
        log_root = Path(log_root_str).expanduser().resolve() if log_root_str else None

        for u in indices_to_scan:
            df, err = load_index_session_minute_df(kite, nse_instruments, session_d, u)
            if df is None or err:
                parts.append(f"{u}: {err or 'no data'}")
                continue
            pct = mock_agent_envelope_pct_for_underlying(u)
            last_scanned_pct = pct
            ok, text, sig = envelope_breakout_on_last_bar(
                df, ema_period=ENVELOPE_EMA_PERIOD, pct=pct
            )
            if not ok or not sig.get("direction"):
                parts.append(f"{u}: {text}")
                continue

            provisional = {
                "direction": sig["direction"],
                "leg": sig["leg"],
                "spot": float(sig["spot"] or 0),
                "signal_bar_time": sig.get("signal_bar_time") or "",
                "text": text,
            }
            flow_dir_p: Path | None = None
            flow_id_str: str | None = None
            if log_root is not None:
                flow_id_str, flow_dir_p = create_flow_run_dir(
                    log_root, session_d=session_d, ist_now=now_ist()
                )
                write_flow_deterministic(
                    flow_dir_p,
                    {
                        "underlying": u,
                        "envelope_pct": pct,
                        "text": text,
                        "sig": {
                            "direction": sig.get("direction"),
                            "leg": sig.get("leg"),
                            "spot": float(sig["spot"] or 0),
                            "signal_bar_time": sig.get("signal_bar_time"),
                        },
                        "vision_validation_status": "pending_vision",
                    },
                )

            vision_sub = flow_dir_p / "vision" if flow_dir_p is not None else None
            chart_png: bytes | None = None
            if mock_llm_attach_underlying_chart():
                chart_png = underlying_session_chart_png_bytes(
                    df,
                    envelope_pct=pct,
                    signal_bar_time=None,
                    underlying_label=FO_INDEX_UNDERLYING_LABELS.get(u, u),
                    direction="",
                    assessment_mode=True,
                )

            if chart_png is not None:
                vres = _llm_validate_envelope_breakout(
                    u, df, pct, parts, provisional, vision_sub, chart_png=chart_png
                )
                if vres == _LLM_BREAKOUT_REJECT:
                    if flow_dir_p is not None:
                        update_flow_deterministic(
                            flow_dir_p, {"vision_validation_status": "vetoed_vision"}
                        )
                    continue
                if vres == _VISION_INVOKE_FALLBACK:
                    if flow_dir_p is not None:
                        update_flow_deterministic(
                            flow_dir_p, {"vision_validation_status": "fallback_error"}
                        )
                    scan_info(
                        "signal",
                        "SIGNAL underlying=%s direction=%s leg=%s spot=%.2f bar=%s (vision fallback)",
                        u,
                        sig["direction"],
                        sig["leg"],
                        float(sig["spot"] or 0),
                        sig.get("signal_bar_time") or "",
                    )
                    fb: dict[str, Any] = {
                        "underlying": u,
                        "direction": sig["direction"],
                        "leg": sig["leg"],
                        "spot": float(sig["spot"] or 0),
                        "signal_bar_time": sig.get("signal_bar_time") or "",
                        "signal_text": text,
                        "signal_envelope_pct": pct,
                        "signal_llm_rationale": (
                            "Rule-based envelope breakout only: vision LLM call failed; "
                            "using deterministic signal."
                        ),
                        "vision_validation_status": "fallback_error",
                    }
                    if flow_id_str and flow_dir_p is not None:
                        fb["llm_flow_id"] = flow_id_str
                        fb["llm_flow_dir"] = str(flow_dir_p)
                    return fb
                if isinstance(vres, dict):
                    if flow_dir_p is not None:
                        update_flow_deterministic(
                            flow_dir_p, {"vision_validation_status": "confirmed_vision"}
                        )
                    vres["vision_validation_status"] = "confirmed_vision"
                    if flow_id_str and flow_dir_p is not None:
                        vres["llm_flow_id"] = flow_id_str
                        vres["llm_flow_dir"] = str(flow_dir_p)
                    return vres

            if flow_dir_p is not None:
                update_flow_deterministic(
                    flow_dir_p, {"vision_validation_status": "skipped_no_chart"}
                )
            scan_info(
                "signal",
                "SIGNAL underlying=%s direction=%s leg=%s spot=%.2f bar=%s (vision skipped)",
                u,
                sig["direction"],
                sig["leg"],
                float(sig["spot"] or 0),
                sig.get("signal_bar_time") or "",
            )
            skip_out: dict[str, Any] = {
                "underlying": u,
                "direction": sig["direction"],
                "leg": sig["leg"],
                "spot": float(sig["spot"] or 0),
                "signal_bar_time": sig.get("signal_bar_time") or "",
                "signal_text": text,
                "signal_envelope_pct": pct,
                "signal_llm_rationale": (
                    "Rule-based envelope breakout only: vision validation skipped "
                    "(chart attach off or PNG unavailable)."
                ),
                "vision_validation_status": "skipped_no_chart",
            }
            if flow_id_str and flow_dir_p is not None:
                skip_out["llm_flow_id"] = flow_id_str
                skip_out["llm_flow_dir"] = str(flow_dir_p)
            return skip_out

        combined = " | ".join(parts)
        if len(combined) > 6000:
            combined = combined[:5997] + "..."
        out_end: dict[str, Any] = {"notes": combined, "signal_text": combined}
        if last_scanned_pct is not None:
            out_end["signal_envelope_pct"] = last_scanned_pct
        return out_end

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
        out_c: dict[str, Any] = {"candidates": enriched}
        lf = (state.get("llm_flow_dir") or "").strip()
        if lf:
            write_flow_candidates_json(Path(lf), enriched)
        return out_c

    def llm_node(state: TradingState) -> TradingState:
        if state.get("error") or not state.get("candidates"):
            return {}
        cands = state["candidates"]
        allowed = {c["tradingsymbol"] for c in cands if c.get("tradingsymbol")}
        idx = (state.get("underlying") or "NIFTY").strip().upper()
        idx_label = FO_INDEX_UNDERLYING_LABELS.get(idx, idx)
        envelope_pct_u = mock_agent_envelope_pct_for_underlying(idx)
        fixed_tgt_rs = mock_engine_option_target_rupees_above_entry()
        tgt_lo_p, tgt_hi_p = mock_llm_sltp_target_pct_bounds()
        st_lo_p, st_hi_p = mock_llm_sltp_stop_pct_bounds()
        sug_tgt = option_target_premium_fraction()
        sug_stp = option_stop_premium_fraction()
        risk_extra = mock_llm_risk_instruction()
        risk_block = f"\n\nAdditional risk guidance from operator:\n{risk_extra}" if risk_extra else ""
        if fixed_tgt_rs is not None:
            tgt_hint = (
                f"Take-profit on option premium is **fixed by configuration** at **+₹{fixed_tgt_rs:,.0f}** above "
                "synthetic entry; code stores that level regardless of your target field. "
                "Still output **target** ≈ entry + that amount (must be > entry) for the schema. "
            )
            bounds_txt = (
                f"Hard bounds (validated in code): **stop** between −{100 * st_hi_p:.1f}% and −{100 * st_lo_p:.1f}% "
                "below entry (stop premium in that band). "
            )
        else:
            tgt_hint = (
                f"Suggested distance from entry: target about +{100 * sug_tgt:.1f}% above entry, "
                f"stop about −{100 * sug_stp:.1f}% below entry (guidance only). "
            )
            bounds_txt = (
                f"Hard bounds (validated in code): target between +{100 * tgt_lo_p:.1f}% and +{100 * tgt_hi_p:.1f}% "
                f"above entry; stop between −{100 * st_hi_p:.1f}% and −{100 * st_lo_p:.1f}% below entry "
                f"(i.e. stop premium in that band below entry). "
            )
        base_sys = (
            "You choose one **NSE F&O option** contract (index or single-stock underlying) for a "
            "**long premium only** mock trade "
            "(calls for bullish, puts for bearish — wrong type already removed in code). "
            "Pick the best strike/expiry among the candidates using DTE, moneyness vs spot, and liquidity (LTP). "
            "You must also output **stop_loss** and **target** as option **premium prices in Indian rupees (₹)** "
            "for the **same contract** you pick (the chosen row's LTP is your reference). "
            "Synthetic long entry will be approximately **LTP minus ~0.5–1.0 ₹ slippage** — "
            "your stop_loss must be **strictly below** that entry and target **strictly above** it: "
            "stop_loss < entry < target. "
            f"{tgt_hint}"
            f"{bounds_txt}"
            "Use two decimal places mentally; output numeric rupee levels."
            f"{risk_block}"
        )
        chart_png: bytes | None = None
        if mock_llm_attach_underlying_chart():
            df_chart, err_chart = load_index_session_minute_df(kite, nse_instruments, session_d, idx)
            if df_chart is not None and not err_chart:
                chart_png = underlying_session_chart_png_bytes(
                    df_chart,
                    envelope_pct=envelope_pct_u,
                    signal_bar_time=(state.get("signal_bar_time") or "") or None,
                    underlying_label=idx_label,
                    direction=str(state.get("direction") or ""),
                )
                if chart_png is None:
                    scan_warning(
                        "graph_err",
                        "LLM chart: PNG export returned nothing underlying=%s "
                        "(Kaleido needs Chromium — install chromium in Docker image; see Dockerfile).",
                        idx,
                    )
            elif err_chart:
                scan_warning(
                    "graph_err",
                    "LLM chart: reload minute df failed underlying=%s: %s",
                    idx,
                    err_chart,
                )
        chart_note = (
            "\n\nAn attached **chart image** shows today's underlying **1-minute** spot session with the **same "
            "EMA envelope** (period and ±% bandwidth) as the breakout signal rule. Use it together with **spot**, "
            "**direction**, and the **candidate table** for strike and liquidity, and to reason about **premium** "
            "stop and target levels."
        )
        sys_plain = SystemMessage(content=base_sys)
        sys = (
            SystemMessage(content=base_sys + chart_note)
            if chart_png
            else sys_plain
        )
        sig_chart_rat = (state.get("signal_llm_rationale") or "").strip()
        sig_block = (
            f"\nChart-based breakout assessment (already decided):\n{sig_chart_rat}\n"
            if sig_chart_rat
            else ""
        )
        text_block = (
            f"Underlying: {idx} ({idx_label}). Spot ~ {state.get('spot', 0):.2f}. "
            f"Direction: {state.get('direction')}. Leg: {state.get('leg')}."
            f"{sig_block}"
            f"\nCandidates (JSON):\n{json.dumps(cands, indent=2)}"
        )
        if chart_png:
            b64 = base64.standard_b64encode(chart_png).decode("ascii")
            human = HumanMessage(
                content=[
                    {"type": "text", "text": text_block},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            )
        else:
            human = HumanMessage(content=text_block)
        invoke_model = (mock_llm_vision_model() or model) if chart_png else model
        llm_invoke = ChatOpenAI(
            api_key=key, model=invoke_model, temperature=0.2, max_completion_tokens=1200
        ).with_structured_output(LLMPick)
        prompt_log_run_dir: Path | None = None
        log_root = mock_llm_prompt_log_dir()
        flow_dir_s = (state.get("llm_flow_dir") or "").strip()
        meta_strike: dict[str, Any] = {
            "llm_flow": "strike_pick",
            "openai_model_default": model,
            "invoke_model": invoke_model,
            "direction": state.get("direction"),
            "leg": state.get("leg"),
            "spot": state.get("spot"),
            "signal_bar_time": state.get("signal_bar_time"),
            "signal_text": (state.get("signal_text") or "")[:4000],
            "envelope_pct": envelope_pct_u,
            "envelope_ema_period": ENVELOPE_EMA_PERIOD,
            "signal_llm_rationale": (state.get("signal_llm_rationale") or "")[:4000],
            "vision_validation_status": (state.get("vision_validation_status") or "")[:120],
        }
        sys_content = sys.content if isinstance(sys.content, str) else json.dumps(sys.content)
        if flow_dir_s:
            strike_dir = Path(flow_dir_s) / "strike"
            if write_llm_turn_snapshot(
                strike_dir,
                system_text=sys_content,
                human_text=text_block,
                chart_png=chart_png,
                meta=meta_strike,
            ):
                prompt_log_run_dir = strike_dir
                scan_info("llm_prompt_log", "LLM strike trace underlying=%s dir=%s", idx, strike_dir)
            else:
                scan_warning(
                    "graph_err",
                    "LLM strike log write failed underlying=%s dir=%s",
                    idx,
                    strike_dir,
                )
        elif log_root:
            prompt_log_run_dir = write_signal_llm_turn_log(
                Path(log_root),
                ist_now=now_ist(),
                session_d=session_d,
                underlying=idx,
                system_text=sys_content,
                human_text=text_block,
                chart_png=chart_png,
                meta=meta_strike,
            )
            if prompt_log_run_dir:
                scan_info("llm_prompt_log", "LLM prompt trace underlying=%s dir=%s", idx, prompt_log_run_dir)
            else:
                scan_warning(
                    "graph_err",
                    "LLM prompt log write failed underlying=%s root=%s",
                    idx,
                    log_root,
                )
        llm_invoke_path = "multimodal" if chart_png else "text_only"
        try:
            pick: LLMPick = llm_invoke.invoke([sys, human])
        except Exception as e:  # noqa: BLE001
            if chart_png:
                scan_warning(
                    "graph_err",
                    "LLM multimodal structured invoke failed underlying=%s model=%s: %s; retry text-only",
                    idx,
                    invoke_model,
                    e,
                )
                if prompt_log_run_dir:
                    try:
                        plain_sys = (
                            sys_plain.content
                            if isinstance(sys_plain.content, str)
                            else json.dumps(sys_plain.content)
                        )
                        (prompt_log_run_dir / "retry_system.txt").write_text(plain_sys, encoding="utf-8")
                        (prompt_log_run_dir / "retry_human.txt").write_text(text_block, encoding="utf-8")
                        (prompt_log_run_dir / "retry_note.txt").write_text(
                            f"Multimodal invoke failed; text-only retry.\n{type(e).__name__}: {e}\n",
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
                human_plain = HumanMessage(content=text_block)
                try:
                    llm_invoke_path = "text_retry"
                    pick = structured.invoke([sys_plain, human_plain])
                except Exception as e2:  # noqa: BLE001
                    scan_warning("graph_err", "LLM text-only retry failed underlying=%s: %s", idx, e2)
                    if prompt_log_run_dir:
                        write_llm_invoke_error_log(
                            prompt_log_run_dir,
                            f"LLM failed after text retry: {e2}",
                            exc_type=type(e2).__name__,
                        )
                    return {"error": f"LLM failed: {e2}"}
            else:
                scan_warning("graph_err", "LLM invoke failed underlying=%s: %s", idx, e)
                if prompt_log_run_dir:
                    write_llm_invoke_error_log(
                        prompt_log_run_dir,
                        f"LLM invoke failed: {e}",
                        exc_type=type(e).__name__,
                    )
                return {"error": f"LLM failed: {e}"}
        if prompt_log_run_dir:
            write_llm_structured_output_log(
                prompt_log_run_dir,
                pick,
                invoke_path=llm_invoke_path,
                symbol_in_candidate_list=pick.tradingsymbol in allowed,
            )
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
        flow_out_p: Path | None = None
        lf_e = (state.get("llm_flow_dir") or "").strip()
        if lf_e and mock_llm_prompt_log_dir():
            flow_out_p = Path(lf_e)

        def _flow_done(tid: int | None = None, err: str | None = None) -> None:
            if flow_out_p is not None:
                write_flow_outcome(flow_out_p, trade_id=tid, error=err)

        tsym = state["llm_tradingsymbol"]
        cands = state.get("candidates") or []
        meta = next((c for c in cands if c.get("tradingsymbol") == tsym), None)
        if not meta:
            scan_warning("graph_err", "EXECUTE missing metadata for symbol=%s", tsym)
            _flow_done(err="Execute: symbol metadata missing")
            return {"error": "Execute: symbol metadata missing"}
        lot_size = max(1, int(meta.get("lot_size") or 1))
        try:
            ltp_row = kite.ltp([f"NFO:{tsym}"])
            raw = ltp_row.get(f"NFO:{tsym}", {})
            ltp = float(raw.get("last_price") or 0)
        except Exception as e:  # noqa: BLE001
            scan_warning("graph_err", "EXECUTE LTP fetch failed %s: %s", tsym, e)
            _flow_done(err=f"LTP fetch failed: {e}")
            return {"error": f"LTP fetch failed: {e}"}
        if ltp <= 0:
            ltp = float(meta.get("ltp") or 0)
        if ltp <= 0:
            scan_warning("graph_err", "EXECUTE no LTP for %s", tsym)
            _flow_done(err="No LTP for execution")
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
            _flow_done(err=sltp_err)
            return {"error": sltp_err}
        qty_units = lot_size  # mock: always 1 exchange lot (PnL uses units = lot_size)
        idx = (state.get("underlying") or "NIFTY").strip().upper()
        sig_r = (state.get("signal_llm_rationale") or "").strip()
        pick_r = (state.get("llm_rationale") or "").strip()
        if sig_r and pick_r:
            combined_rat = f"[Breakout / chart] {sig_r}\n[Strike / risk] {pick_r}"
        elif pick_r:
            combined_rat = pick_r
        else:
            combined_rat = sig_r
        try:
            tid = mock_trade_store.insert_open_trade(
                instrument=tsym,
                direction=str(state.get("direction") or ""),
                entry_price=entry,
                stop_loss=stop_loss,
                target=target,
                llm_rationale=combined_rat,
                lot_size=lot_size,
                quantity=qty_units,
                index_underlying=idx,
            )
        except ValueError as e:
            scan_warning("graph_err", "EXECUTE insert rejected: %s", e)
            _flow_done(err=str(e))
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
        _flow_done(tid=tid)
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
