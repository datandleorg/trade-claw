"""LLM-only BANKNIFTY mock engine graph (supervisor + optional vision tool)."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from kiteconnect import KiteConnect
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from trade_claw.constants import FO_INDEX_UNDERLYING_LABELS
from trade_claw.llm_mock_agent_memory import format_transcript_for_worker
from trade_claw.env_trading_params import (
    mock_engine_option_stop_multiplier,
    mock_engine_option_target_price,
    mock_llm_prompt_log_dir,
)
from trade_claw.fo_support import _to_date
from trade_claw.mock_engine_log import scan_info, scan_warning
from trade_claw.mock_llm_flow_log import create_flow_run_dir, write_flow_outcome
from trade_claw.mock_llm_signal_chart import underlying_session_chart_png_bytes
from trade_claw.mock_market_signal import (
    load_index_session_minute_df,
    mock_agent_slippage_points,
    now_ist,
    top_five_option_instruments,
)
from trade_claw import mock_trade_store

_UNDERLYING = "BANKNIFTY"


class SupervisorIntent(BaseModel):
    mode: Literal["SEARCH", "MANAGE"] = Field(description="SEARCH when no open position, MANAGE when open exists")
    use_vision: bool = Field(description="Deprecated for BANKNIFTY graph; vision analysis now runs every tick")
    focus: str = Field(description="One-line reason for current focus")
    risk_plan: str = Field(description="Brief risk plan for this run")


class VisionAnalysis(BaseModel):
    market_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(description="Bias from chart")
    confidence: float = Field(description="0..1 confidence score", ge=0.0, le=1.0)
    analysis: str = Field(description="Concise technical chart analysis")
    suggested_leg: Literal["CE", "PE", "NONE"] = Field(description="Leg bias for premium buying")


class SupervisorDecision(BaseModel):
    action: Literal["HOLD", "ENTER", "EXIT"] = Field(description="Decision to take this minute")
    rationale: str = Field(description="Decision rationale")
    leg: Literal["CE", "PE", "NONE"] = Field(default="NONE")
    tradingsymbol: str = Field(default="")
    target: float | None = Field(default=None, description="Target premium for ENTER")
    stop_loss: float | None = Field(default=None, description="Stop premium for ENTER")
    exit_reason: str = Field(default="")


@dataclass
class BankNiftyContext:
    kite: KiteConnect
    session_d: date
    nse_instruments: list
    nfo_instruments: list
    flow_id: str | None
    flow_dir: Path | None


def _openai_key() -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY is missing or placeholder")
    return key


def _supervisor_model() -> str:
    return (os.environ.get("LLM_MOCK_SUPERVISOR_MODEL") or "gpt-5-mini").strip()


def _vision_model() -> str:
    # Anthropic current aliases/IDs are in the Claude 4.x family.
    return (os.environ.get("LLM_MOCK_VISION_MODEL") or "claude-haiku-4-5").strip()


def _engine_enabled() -> bool:
    return (os.environ.get("LLM_MOCK_ENGINE_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")


def _llm_mock_profit_exit_hint_inr() -> float:
    """If estimated close P/L reaches this (₹), prompts tell the supervisor to favour EXIT."""
    raw = (os.environ.get("LLM_MOCK_PROFIT_EXIT_HINT_INR") or "3000").strip()
    try:
        return float(max(0.0, min(50_000_000.0, float(raw))))
    except ValueError:
        return 3000.0


def _banknifty_open_position_context(
    kite: KiteConnect,
    open_trade: mock_trade_store.MockTradeRow | None,
) -> dict[str, Any] | None:
    if open_trade is None or not (open_trade.instrument or "").strip():
        return None
    sym = str(open_trade.instrument).strip()
    try:
        row = kite.ltp([f"NFO:{sym}"]).get(f"NFO:{sym}") or {}
        v = row.get("last_price")
        opt_ltp = float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return {"instrument": sym, "option_ltp": None, "estimated_close_pnl_inr": None, "error": "ltp_failed"}
    if opt_ltp is None or opt_ltp <= 0 or open_trade.entry_price is None:
        return {"instrument": sym, "option_ltp": opt_ltp, "estimated_close_pnl_inr": None}
    qty = max(1, int(open_trade.quantity or 1))
    slip = mock_agent_slippage_points()
    exit_est = max(0.01, opt_ltp - slip)
    pnl = (exit_est - float(open_trade.entry_price)) * qty
    return {
        "instrument": sym,
        "option_ltp": round(opt_ltp, 2),
        "quantity": qty,
        "entry_price": float(open_trade.entry_price),
        "estimated_close_pnl_inr": round(pnl, 2),
        "stored_stop_premium": open_trade.stop_loss,
        "stored_target_premium": open_trade.target,
    }


def _build_flow_dir(session_d: date) -> tuple[str | None, Path | None]:
    root = mock_llm_prompt_log_dir()
    if not root:
        return None, None
    try:
        return create_flow_run_dir(Path(root), session_d=session_d, ist_now=now_ist())
    except OSError:
        return None, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except OSError:
        pass


def _invoke_supervisor_intent(
    ctx: BankNiftyContext,
    *,
    open_trade: mock_trade_store.MockTradeRow | None,
    position_ctx: dict[str, Any] | None = None,
) -> SupervisorIntent:
    llm = ChatOpenAI(api_key=_openai_key(), model=_supervisor_model(), temperature=0.2).with_structured_output(SupervisorIntent)
    mode = "MANAGE" if open_trade else "SEARCH"
    otxt = (
        f"Open trade: instrument={open_trade.instrument} entry={open_trade.entry_price} stop={open_trade.stop_loss} target={open_trade.target}"
        if open_trade
        else "No open trade for BANKNIFTY."
    )
    hint = _llm_mock_profit_exit_hint_inr()
    policy = (
        f"Operator policy: **conservative** risk — modest targets, avoid greedy holds. "
        f"If an open position’s estimated close P/L is around **₹{hint:,.0f}** or more, MANAGE mode should bias toward **locking gains** (exit readiness) rather than stretching for home runs. "
        "If tape may be turning **against** the open leg’s path to its stored target (even subtly), bias toward **exit readiness**, not hero holds."
    )
    mem_tail = (format_transcript_for_worker() or "").strip()
    mem_block = (
        "\n\n--- Persistent operator/agent chat (tail; influence HOLD/ENTER/EXIT when relevant) ---\n"
        f"{mem_tail}\n--- End chat memory ---\n"
        if mem_tail
        else ""
    )
    ctx_line = ""
    if position_ctx and position_ctx.get("estimated_close_pnl_inr") is not None:
        ctx_line = f"\nLive position context (JSON): {json.dumps(position_ctx, default=str)}"
    msg = HumanMessage(
        content=(
            f"Underlying={_UNDERLYING}. Mode={mode}. Session date={ctx.session_d}.\n"
            f"{otxt}\n"
            f"{policy}{mem_block}{ctx_line}\n"
            "Decide intent for this minute. You may request vision tool when chart context is useful."
        )
    )
    out: SupervisorIntent = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a trading supervisor for a mock intraday BANKNIFTY options book. "
                    "Be **conservative**: protect open P/L; if conditions may be shifting **against** the trade’s path to its "
                    "take-profit, favour managing toward exit rather than holding for the full target. Return concise intent."
                )
            ),
            msg,
        ]
    )
    if ctx.flow_dir is not None:
        _write_json(
            ctx.flow_dir / "supervisor" / "intent.json",
            {"mode": out.mode, "use_vision": out.use_vision, "focus": out.focus, "risk_plan": out.risk_plan},
        )
    return out


def _vision_tool(ctx: BankNiftyContext, df_u: pd.DataFrame) -> tuple[VisionAnalysis | None, bytes | None]:
    png = underlying_session_chart_png_bytes(
        df_u,
        envelope_pct=0.003,
        signal_bar_time=None,
        underlying_label=FO_INDEX_UNDERLYING_LABELS.get(_UNDERLYING, _UNDERLYING),
        direction="",
        assessment_mode=True,
    )
    if png is None:
        return None, None
    b64 = base64.standard_b64encode(png).decode("ascii")
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    model_raw = _vision_model()
    model_candidates: list[str] = [
        m
        for m in [
            model_raw,
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
        ]
        if m
    ]
    # Preserve order, dedupe.
    seen: set[str] = set()
    models = [m for m in model_candidates if not (m in seen or seen.add(m))]
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze BANKNIFTY 1m chart and return bias, confidence, and suggested leg."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    )
    last_err: Exception | None = None
    out: VisionAnalysis | None = None
    chosen_model: str | None = None
    for m in models:
        try:
            llm = ChatAnthropic(
                api_key=api_key,
                model=m,
                temperature=0.1,
                max_tokens=1200,
            ).with_structured_output(VisionAnalysis)
            out = llm.invoke([SystemMessage(content="You are a technical chart analysis tool."), msg])
            chosen_model = m
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            scan_warning("graph_err", "Vision model failed model=%s err=%s", m, e)
            continue
    if out is None:
        raise RuntimeError(f"All vision model attempts failed. last_error={last_err}")
    if ctx.flow_dir is not None:
        vp = ctx.flow_dir / "vision"
        try:
            vp.mkdir(parents=True, exist_ok=True)
            (vp / "chart.png").write_bytes(png)
        except OSError:
            pass
        _write_json(vp / "analysis.json", {"model": chosen_model, **out.model_dump()})
    return out, png


def _candidate_symbols(ctx: BankNiftyContext, *, leg: str, spot: float) -> list[dict[str, Any]]:
    cands, err = top_five_option_instruments(
        ctx.nfo_instruments,
        underlying=_UNDERLYING,
        session_d=ctx.session_d,
        spot=spot,
        leg=leg,
    )
    if err or not cands:
        return []
    keys = [f"NFO:{c.get('tradingsymbol')}" for c in cands if c.get("tradingsymbol")]
    try:
        ltps = ctx.kite.ltp(keys) if keys else {}
    except Exception:  # noqa: BLE001
        ltps = {}
    out: list[dict[str, Any]] = []
    for c in cands:
        ts = str(c.get("tradingsymbol") or "")
        row = ltps.get(f"NFO:{ts}") if ts else {}
        ltp = row.get("last_price") if isinstance(row, dict) else None
        ed = _to_date(c.get("expiry"))
        out.append(
            {
                "tradingsymbol": ts,
                "strike": float(c.get("strike") or 0),
                "instrument_type": c.get("instrument_type"),
                "expiry": str(ed) if ed else "",
                "lot_size": int(c.get("lot_size") or 1),
                "ltp": float(ltp) if ltp is not None else None,
            }
        )
    if ctx.flow_dir is not None:
        _write_json(ctx.flow_dir / "supervisor" / "candidates.json", {"candidates": out})
    return out


def _invoke_supervisor_decision(
    ctx: BankNiftyContext,
    *,
    open_trade: mock_trade_store.MockTradeRow | None,
    vision: VisionAnalysis | None,
    spot: float,
    candidates: list[dict[str, Any]],
    position_ctx: dict[str, Any] | None = None,
) -> SupervisorDecision:
    llm = ChatOpenAI(api_key=_openai_key(), model=_supervisor_model(), temperature=0.2).with_structured_output(SupervisorDecision)
    otxt = (
        {
            "trade_id": open_trade.trade_id,
            "instrument": open_trade.instrument,
            "entry_price": open_trade.entry_price,
            "stop_loss": open_trade.stop_loss,
            "target": open_trade.target,
        }
        if open_trade
        else None
    )
    vtxt = vision.model_dump() if vision else None
    hint = _llm_mock_profit_exit_hint_inr()
    prompt = {
        "underlying": _UNDERLYING,
        "spot": spot,
        "open_trade": otxt,
        "position_context": position_ctx,
        "vision_analysis": vtxt,
        "candidates": candidates[:5],
        "profit_lock_hint_inr": hint,
        "rules": [
            "If no open trade: HOLD or ENTER.",
            "If open trade: HOLD or EXIT only.",
            "For ENTER choose one tradingsymbol from candidates.",
            "Risk style: **conservative**. For ENTER, set stop_loss and target as option **premium** ₹ levels — "
            "prefer **modest** target vs entry (realistic intraday take-profit), not stretched moonshots.",
            "When `position_context.estimated_close_pnl_inr` is **≥ profit_lock_hint_inr** (or very close), **strongly prefer EXIT** "
            "this minute if the book can close near current prices — crystallize gains unless vision shows **very** strong "
            "continuation with clear room and you accept giveback risk. If unsure, EXIT.",
            "Counter-trend / giveback rule (MANAGE): infer the open leg from `open_trade.instrument` (CE vs PE). "
            "Long **CE** needs underlying strength toward the stored **target** premium; long **PE** needs weakness. "
            "If `vision_analysis` or spot context shows **even a slight** shift **opposite** to that (e.g. CE: softening "
            "bullish bias, creeping BEARISH/NEUTRAL, loss of momentum toward target; PE: the mirror), **prefer EXIT** to "
            "take profits — do **not** hold for the full stored target when the tape argues against you. A small adverse "
            "hint is enough when you already have green P/L; if P/L is flat or small green and the turn is credible, EXIT.",
            "If `agent_memory_transcript` is non-empty, it is the **persisted operator ↔ companion chat** — treat it as "
            "**live human steer** (intent, risk, exit bias). Integrate with vision and rules; if it conflicts with holding "
            "for the mechanical target, **follow the operator thread**.",
        ],
    }
    mem_full = (format_transcript_for_worker() or "").strip()
    prompt["agent_memory_transcript"] = mem_full if mem_full else None
    sys = (
        "You are the intraday supervisor for a **mock** BANKNIFTY **long premium** book (long CE or long PE only).\n"
        "**Conservative mandate:** bank realistic gains; avoid squeezing every rupee out of a move.\n"
        "At **ENTER**, keep target/stop modest vs entry.\n"
        "With an **open** position: (1) If estimated close P/L in `position_context` reaches or nears `profit_lock_hint_inr`, "
        "strongly favour **EXIT** to lock gains unless continuation is overwhelmingly clear. "
        "(2) **Trend vs target:** the stored `target` premium is only a guide — if vision/spot shows **even a slight** move "
        "or bias **against** the direction that would carry the option toward that target (downward pressure vs a long CE’s "
        "path; upward pressure vs a long PE’s path; fading momentum; conflicting bias vs the open leg), **take profits with "
        "EXIT** rather than holding for the trigger. Prefer leaving money on the table over riding a reversal.\n"
        "When the JSON payload includes non-null `agent_memory_transcript`, treat it as **authoritative persisted human "
        "and companion context** — adjust focus, risk, and HOLD/ENTER/EXIT accordingly; it overrides generic stretch-for-target behaviour.\n"
        "Return exactly one action: HOLD, ENTER, or EXIT, obeying the JSON rules."
    )
    out: SupervisorDecision = llm.invoke(
        [SystemMessage(content=sys), HumanMessage(content=json.dumps(prompt, default=str))]
    )
    if ctx.flow_dir is not None:
        _write_json(ctx.flow_dir / "supervisor" / "decision.json", out.model_dump())
    return out


def _entry_sltp(entry: float, stop_loss: float | None, target: float | None) -> tuple[float, float]:
    st = float(stop_loss) if stop_loss is not None else 0.0
    tg = float(target) if target is not None else 0.0
    if st > 0 and st < entry and tg > entry:
        return round(st, 2), round(tg, 2)
    return round(entry * mock_engine_option_stop_multiplier(), 2), mock_engine_option_target_price(entry)


def invoke_llm_banknifty_graph(
    *,
    kite: KiteConnect,
    session_d: date,
    nse_instruments: list,
    nfo_instruments: list,
) -> dict[str, Any]:
    if not _engine_enabled():
        return {"underlying": _UNDERLYING, "skipped": "llm_engine_disabled"}

    flow_id, flow_dir = _build_flow_dir(session_d)
    ctx = BankNiftyContext(
        kite=kite,
        session_d=session_d,
        nse_instruments=nse_instruments,
        nfo_instruments=nfo_instruments,
        flow_id=flow_id,
        flow_dir=flow_dir,
    )
    out: dict[str, Any] = {
        "underlying": _UNDERLYING,
        "llm_flow_id": flow_id,
        "llm_flow_dir": str(flow_dir) if flow_dir else None,
        "engine_name": "llm_banknifty",
    }
    open_trade = next((t for t in mock_trade_store.list_open_trades() if (t.index_underlying or "").upper() == _UNDERLYING), None)
    position_ctx = _banknifty_open_position_context(kite, open_trade)
    intent = _invoke_supervisor_intent(ctx, open_trade=open_trade, position_ctx=position_ctx)
    out["run_mode"] = intent.mode.lower()
    out["supervisor_focus"] = intent.focus
    out["tools_called"] = []
    out["vision_called"] = False

    df_u, err_u = load_index_session_minute_df(kite, nse_instruments, session_d, _UNDERLYING)
    if df_u is None or err_u:
        out["error"] = err_u or "No BANKNIFTY minute data"
        return out
    spot = float(df_u["close"].iloc[-1])
    out["spot"] = spot
    out["position_context"] = position_ctx

    # Always run the vision model for technical analysis on each tick.
    vision: VisionAnalysis | None = None
    out["vision_called"] = True
    out["tools_called"].append("vision_tool")
    try:
        vision, _png = _vision_tool(ctx, df_u)
    except Exception as e:  # noqa: BLE001
        scan_warning("graph_err", "BANKNIFTY vision tool failed: %s", e)
        out["vision_error"] = str(e)
        vision = None

    leg = "CE"
    if vision is not None and vision.suggested_leg in ("CE", "PE"):
        leg = vision.suggested_leg
    candidates = _candidate_symbols(ctx, leg=leg, spot=spot)
    out["candidates"] = candidates
    out["tools_called"].append("kite_candidates_tool")

    decision = _invoke_supervisor_decision(
        ctx,
        open_trade=open_trade,
        vision=vision,
        spot=spot,
        candidates=candidates,
        position_ctx=position_ctx,
    )
    out["decision"] = decision.action
    out["llm_rationale"] = decision.rationale

    if decision.action == "HOLD":
        write_flow_outcome(flow_dir, trade_id=None, error=None) if flow_dir else None
        return out

    if decision.action == "EXIT":
        if open_trade is None:
            out["error"] = "EXIT requested but no open BANKNIFTY trade"
            write_flow_outcome(flow_dir, trade_id=None, error=out["error"]) if flow_dir else None
            return out
        ltp = None
        if open_trade.instrument:
            try:
                row = kite.ltp([f"NFO:{open_trade.instrument}"]).get(f"NFO:{open_trade.instrument}") or {}
                v = row.get("last_price")
                ltp = float(v) if v is not None else None
            except Exception as e:  # noqa: BLE001
                scan_warning("ltp_warn", "BANKNIFTY EXIT ltp failed: %s", e)
        if ltp is None or ltp <= 0:
            out["error"] = "No LTP for EXIT execution"
            write_flow_outcome(flow_dir, trade_id=open_trade.trade_id, error=out["error"]) if flow_dir else None
            return out
        exit_px = max(0.01, ltp - mock_agent_slippage_points())
        pnl = (exit_px - float(open_trade.entry_price or 0)) * int(open_trade.quantity or 1)
        ok = mock_trade_store.close_trade(open_trade.trade_id, exit_price=exit_px, realized_pnl=pnl)
        if not ok:
            out["error"] = "Failed to close open trade"
            write_flow_outcome(flow_dir, trade_id=open_trade.trade_id, error=out["error"]) if flow_dir else None
            return out
        out["closed_trade_id"] = open_trade.trade_id
        out["exit_reason"] = decision.exit_reason or "llm_supervisor_exit"
        scan_info("exit", "BANKNIFTY LLM exit trade_id=%s reason=%s", open_trade.trade_id, out["exit_reason"])
        write_flow_outcome(flow_dir, trade_id=open_trade.trade_id, error=None) if flow_dir else None
        return out

    # ENTER
    if open_trade is not None:
        out["error"] = "ENTER requested but open trade already exists"
        write_flow_outcome(flow_dir, trade_id=open_trade.trade_id, error=out["error"]) if flow_dir else None
        return out
    picked = (decision.tradingsymbol or "").strip()
    meta = next((c for c in candidates if c.get("tradingsymbol") == picked), None)
    if not meta:
        out["error"] = f"Invalid tradingsymbol for ENTER: {picked!r}"
        write_flow_outcome(flow_dir, trade_id=None, error=out["error"]) if flow_dir else None
        return out
    try:
        row = kite.ltp([f"NFO:{picked}"]).get(f"NFO:{picked}") or {}
        ltp = float(row.get("last_price") or 0.0)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"ENTER LTP fetch failed: {e}"
        write_flow_outcome(flow_dir, trade_id=None, error=out["error"]) if flow_dir else None
        return out
    if ltp <= 0:
        out["error"] = "ENTER no LTP"
        write_flow_outcome(flow_dir, trade_id=None, error=out["error"]) if flow_dir else None
        return out
    entry = max(0.01, ltp - mock_agent_slippage_points())
    stop_loss, target = _entry_sltp(entry, decision.stop_loss, decision.target)
    lot_size = max(1, int(meta.get("lot_size") or 1))
    trade_id = mock_trade_store.insert_open_trade(
        instrument=picked,
        direction="BUY" if "CE" in picked.upper() else "SELL",
        entry_price=entry,
        stop_loss=stop_loss,
        target=target,
        llm_rationale=decision.rationale,
        lot_size=lot_size,
        quantity=lot_size,
        index_underlying=_UNDERLYING,
    )
    out["trade_id"] = trade_id
    out["llm_tradingsymbol"] = picked
    out["stop_loss"] = stop_loss
    out["target"] = target
    scan_info("open", "BANKNIFTY LLM open trade_id=%s symbol=%s", trade_id, picked)
    write_flow_outcome(flow_dir, trade_id=trade_id, error=None) if flow_dir else None
    return out
