"""Claude Haiku ReAct agent for pure-LLM F&O mock decisions: vision chart + TA tools + structured submit."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from trade_claw.constants import FO_MAX_EXPIRY_CANDLE_FALLBACKS
from trade_claw.fo_claude_chart import merge_overlay_spec, png_to_data_url, render_session_chart_png
from trade_claw.fo_support import align_option_entry_bar, build_option_trade_candidates, fetch_underlying_intraday
from trade_claw.market_data import candles_to_dataframe
from trade_claw.option_trades import simulate_long_option_target_stop_eod

logger = logging.getLogger("trade_claw.fo_claude_react_agent")

MAX_AGENT_TURNS = 28


def _session_window(session_date: date) -> tuple[str, str]:
    from_dt = datetime(session_date.year, session_date.month, session_date.day, 9, 15, 0)
    to_dt = datetime(session_date.year, session_date.month, session_date.day, 15, 30, 0)
    return from_dt.strftime("%Y-%m-%d %H:%M:%S"), to_dt.strftime("%Y-%m-%d %H:%M:%S")


def _vwap_series(df: pd.DataFrame) -> pd.Series:
    vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum = vol.cumsum().replace(0.0, pd.NA)
    return ((tp * vol).cumsum() / cum).bfill().fillna(df["close"])


def _rsi_last(close: pd.Series, period: int = 14) -> float | None:
    if len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_loss.where(avg_loss != 0, 1e-10).rdiv(avg_gain)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


@dataclass
class FoClaudeContext:
    df_u: pd.DataFrame
    kite: Any
    nfo_instruments: list
    underlying: str
    underlying_label: str
    session_date: date
    from_str: str
    to_str: str
    chosen_interval: str
    steps_from_atm: int
    manual_strike_val: float | None
    option_target_pct: float
    option_stop_loss_pct: float
    brokerage_per_lot_rt: float
    taxes_per_lot_rt: float
    allowed_tradingsymbols: set[str]
    overlay_spec: dict[str, Any] = field(default_factory=dict)
    current_png: bytes | None = None
    final_decision: dict[str, Any] | None = None
    observability: list[dict[str, Any]] = field(default_factory=list)
    run_dir: Path | None = None
    _step: int = 0

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        self._step += 1
        ev = {"step": self._step, "kind": kind, **payload}
        self.observability.append(ev)
        if self.run_dir is not None:
            try:
                p = self.run_dir / f"step_{self._step:03d}_{kind}.json"
                p.write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
            except OSError:
                pass


def _build_allowlist(
    *,
    nfo_instruments: list,
    underlying: str,
    session_date: date,
    spot: float,
    steps_from_atm: int,
    manual_strike_val: float | None,
) -> set[str]:
    out: set[str] = set()
    for leg in ("CE", "PE"):
        cands, err = build_option_trade_candidates(
            nfo_instruments,
            underlying,
            spot,
            session_date,
            leg,
            steps_from_atm,
            manual_strike_val,
            max_expiries=FO_MAX_EXPIRY_CANDLE_FALLBACKS,
        )
        if err:
            continue
        for inst, _stk, _ex in cands[:40]:
            ts = inst.get("tradingsymbol")
            if ts:
                out.add(str(ts))
    return out


class SubmitFoDecisionArgs(BaseModel):
    action: Literal["TRADE", "NO_TRADE"]
    intraday_strategy: str = Field(default="", description="Name of intraday playbook, e.g. VWAP reclaim, EMA cross")
    strategy_explanation: str = Field(default="", description="2–6 sentences tying chart and indicators")
    trigger_description: str = Field(default="", description="Exact trigger: bar/time/level")
    rationale_full: str = Field(default="", description="Strong structured rationale")
    alternatives_considered: str = Field(default="", description="Why not other leg or flat")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    leg: Literal["CE", "PE"] | None = None
    tradingsymbol: str | None = None
    entry_bar_index: int | None = Field(
        default=None,
        description="Underlying bar index at trigger (0-based); default last bar",
    )

    @model_validator(mode="after")
    def _trade_requires(self) -> SubmitFoDecisionArgs:
        if self.action == "TRADE":
            if self.leg not in ("CE", "PE"):
                raise ValueError("TRADE requires leg CE or PE")
            if not (self.tradingsymbol or "").strip():
                raise ValueError("TRADE requires tradingsymbol")
            for fld, val in (
                ("intraday_strategy", self.intraday_strategy),
                ("strategy_explanation", self.strategy_explanation),
                ("trigger_description", self.trigger_description),
                ("rationale_full", self.rationale_full),
            ):
                if not (val or "").strip() or len(str(val).strip()) < 8:
                    raise ValueError(f"TRADE requires substantive {fld}")
        else:
            if len((self.rationale_full or "").strip()) < 10:
                raise ValueError("NO_TRADE requires rationale_full (why flat)")
        return self


def run_claude_fo_react_pipeline(
    *,
    kite: Any,
    nse_instruments: list,
    nfo_instruments: list,
    underlying: str,
    name: str,
    session_date: date,
    chosen_interval: str,
    steps_from_atm: int,
    strike_policy_label: str,
    manual_strike_val: float | None,
    brokerage_per_lot_rt: float,
    taxes_per_lot_rt: float,
    option_target_pct: float,
    option_stop_loss_pct: float,
    anthropic_api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Returns dict: success, error_message, row, trace, observability, chart_pngs."""
    from_str, to_str = _session_window(session_date)
    df_u, err_u = fetch_underlying_intraday(
        kite, underlying, nse_instruments, from_str, to_str, chosen_interval
    )
    if df_u is None or err_u:
        return {
            "success": False,
            "error_message": err_u or "No underlying data",
            "row": None,
            "trace": [],
            "observability": [],
            "chart_pngs": [],
        }
    df_u = df_u.sort_values("date").reset_index(drop=True)
    spot = float(df_u.iloc[-1]["close"])
    allowed = _build_allowlist(
        nfo_instruments=nfo_instruments,
        underlying=underlying,
        session_date=session_date,
        spot=spot,
        steps_from_atm=steps_from_atm,
        manual_strike_val=manual_strike_val,
    )
    if not allowed:
        return {
            "success": False,
            "error_message": "Could not build option allowlist for this underlying/date",
            "row": None,
            "trace": [{"step": "allowlist", "error": "empty"}],
            "observability": [],
            "chart_pngs": [],
        }

    obs_root = (os.environ.get("CLAUDE_FO_OBSERVABILITY_DIR") or "").strip()
    run_dir = (
        Path(obs_root).expanduser().resolve() / session_date.isoformat() / uuid.uuid4().hex[:12]
        if obs_root
        else Path("data/claude_fo_runs").resolve() / session_date.isoformat() / uuid.uuid4().hex[:12]
    )
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        run_dir = None

    ctx = FoClaudeContext(
        df_u=df_u,
        kite=kite,
        nfo_instruments=nfo_instruments,
        underlying=underlying,
        underlying_label=underlying,
        session_date=session_date,
        from_str=from_str,
        to_str=to_str,
        chosen_interval=chosen_interval,
        steps_from_atm=steps_from_atm,
        manual_strike_val=manual_strike_val,
        option_target_pct=option_target_pct,
        option_stop_loss_pct=option_stop_loss_pct,
        brokerage_per_lot_rt=brokerage_per_lot_rt,
        taxes_per_lot_rt=taxes_per_lot_rt,
        allowed_tradingsymbols=allowed,
        run_dir=run_dir,
    )

    ctx.overlay_spec = {"ema_periods": [9, 21, 50], "show_vwap": True}
    png0 = render_session_chart_png(
        df_u,
        underlying_label=f"{underlying} ({session_date})",
        overlay_spec=ctx.overlay_spec,
    )
    ctx.current_png = png0
    chart_pngs: list[bytes] = []
    if png0:
        chart_pngs.append(png0)
        if run_dir:
            try:
                (run_dir / "chart_00_initial.png").write_bytes(png0)
            except OSError:
                pass
        ctx.log("chart", {"label": "initial", "chart_index": 0, "bytes": len(png0)})

    tools = _make_tools(ctx, name, strike_policy_label)
    api_key = (anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return {
            "success": False,
            "error_message": "ANTHROPIC_API_KEY is missing",
            "row": None,
            "trace": [],
            "observability": ctx.observability,
            "chart_pngs": chart_pngs,
            "run_dir": str(run_dir) if run_dir else None,
        }
    model_name = (model or os.environ.get("CLAUDE_MODEL") or "claude-haiku-4-5").strip()
    llm = ChatAnthropic(
        api_key=api_key,
        model=model_name,
        temperature=0.2,
        max_tokens=int(os.environ.get("CLAUDE_MAX_TOKENS", "8192")),
    )
    llm_bind = llm.bind_tools(tools)

    system_text = _system_prompt()
    user_text = _user_payload_text(
        underlying=underlying,
        name=name,
        session_date=session_date,
        chosen_interval=chosen_interval,
        strike_policy_label=strike_policy_label,
        steps_from_atm=steps_from_atm,
        manual_strike_val=manual_strike_val,
        option_target_pct=option_target_pct,
        option_stop_loss_pct=option_stop_loss_pct,
        brokerage_per_lot_rt=brokerage_per_lot_rt,
        taxes_per_lot_rt=taxes_per_lot_rt,
        n_bars=len(df_u),
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if png0:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": png_to_data_url(png0)},
            }
        )
    messages: list[Any] = [
        SystemMessage(content=system_text),
        HumanMessage(content=content),
    ]

    trace: list[dict[str, Any]] = []
    turn = 0
    while turn < MAX_AGENT_TURNS and ctx.final_decision is None:
        turn += 1
        trace.append({"step": "turn", "n": turn})
        try:
            ai: AIMessage = llm_bind.invoke(messages)
        except Exception as e:
            logger.exception("Claude invoke failed: %s", e)
            trace.append({"step": "error", "error": str(e)})
            return {
                "success": False,
                "error_message": f"Claude API error: {e}",
                "row": None,
                "trace": trace,
                "observability": ctx.observability,
                "chart_pngs": chart_pngs,
                "run_dir": str(run_dir) if run_dir else None,
            }

        text_out = (ai.content or "") if isinstance(ai.content, str) else str(ai.content)
        if text_out:
            ctx.log("assistant_text", {"text": text_out[:12000], "turn": turn})
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            trace.append({"step": "no_tool_calls", "turn": turn})
            messages.append(
                HumanMessage(
                    content="You must use tools: call technical tools as needed, "
                    "refresh chart with apply_chart_overlays if you rely on visuals, "
                    "then call submit_fo_decision exactly once."
                )
            )
            continue

        for tc in tool_calls:
            name, args, tid = _parse_tool_call(tc)
            trace.append({"step": "tool_call", "name": name, "arguments": args})
            ctx.log("tool_call", {"name": name, "arguments": args})
            result = _dispatch_tool(name, args, ctx, tools_by_name={t.name: t for t in tools})
            trace.append({"step": "tool_result", "name": name, "result_preview": str(result)[:2000]})
            ctx.log("tool_result", {"name": name, "result": str(result)[:8000]})
            msg_content = str(result)[:120_000]
            messages.append(ToolMessage(content=msg_content, tool_call_id=tid))
            if name == "apply_chart_overlays" and ctx.current_png:
                chart_pngs.append(ctx.current_png)
                if ctx.run_dir:
                    try:
                        (ctx.run_dir / f"chart_{len(chart_pngs):02d}_overlay.png").write_bytes(ctx.current_png)
                    except OSError:
                        pass
                ctx.log(
                    "chart",
                    {
                        "label": "after_apply_chart_overlays",
                        "chart_index": len(chart_pngs) - 1,
                        "bytes": len(ctx.current_png),
                    },
                )
                messages.append(
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Updated chart image after apply_chart_overlays (use this for reasoning).",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": png_to_data_url(ctx.current_png)},
                            },
                        ]
                    )
                )

        if ctx.final_decision is not None:
            break

    if ctx.final_decision is None:
        return {
            "success": False,
            "error_message": "Agent did not call submit_fo_decision",
            "row": None,
            "trace": trace,
            "observability": ctx.observability,
            "chart_pngs": chart_pngs,
            "run_dir": str(run_dir) if run_dir else None,
        }

    dec = ctx.final_decision
    if dec.get("action") == "NO_TRADE":
        row = _no_trade_row(
            underlying=underlying,
            name=name,
            session_date=session_date,
            df_u=df_u,
            decision=dec,
            strat_label="Claude ReAct (no trade)",
        )
        return {
            "success": True,
            "error_message": None,
            "row": row,
            "trace": trace,
            "observability": ctx.observability,
            "chart_pngs": chart_pngs,
            "final_decision": dec,
            "run_dir": str(run_dir) if run_dir else None,
        }

    sym = (dec.get("tradingsymbol") or "").strip()
    leg = (dec.get("leg") or "").strip().upper()
    if sym not in allowed:
        return {
            "success": False,
            "error_message": f"Invalid or disallowed tradingsymbol: {sym!r}",
            "row": None,
            "trace": trace,
            "observability": ctx.observability,
            "chart_pngs": chart_pngs,
            "final_decision": dec,
            "run_dir": str(run_dir) if run_dir else None,
        }
    if leg not in ("CE", "PE"):
        return {
            "success": False,
            "error_message": "TRADE requires leg CE or PE",
            "row": None,
            "trace": trace,
            "observability": ctx.observability,
            "chart_pngs": chart_pngs,
            "final_decision": dec,
            "run_dir": str(run_dir) if run_dir else None,
        }

    opt_inst = _find_instrument(nfo_instruments, sym)
    if not opt_inst:
        return {
            "success": False,
            "error_message": f"Instrument not found in master: {sym}",
            "row": None,
            "trace": trace,
            "observability": ctx.observability,
            "chart_pngs": chart_pngs,
            "final_decision": dec,
            "run_dir": str(run_dir) if run_dir else None,
        }

    entry_bar_idx = dec.get("entry_bar_index")
    try:
        entry_bar_idx = int(entry_bar_idx) if entry_bar_idx is not None else len(df_u) - 1
    except (TypeError, ValueError):
        entry_bar_idx = len(df_u) - 1
    entry_bar_idx = max(0, min(entry_bar_idx, len(df_u) - 1))

    spot_signal = "BUY" if leg == "CE" else "SELL"
    rationale = dec.get("rationale_full") or ""
    strat_label = f"Claude: {dec.get('intraday_strategy', 'custom')[:80]}"

    exec_out = _execute_mock_option(
        kite=kite,
        opt_inst=opt_inst,
        df_u=df_u,
        entry_bar_idx=entry_bar_idx,
        spot_signal=spot_signal,
        strat_label=strat_label,
        analysis_text=dec.get("strategy_explanation") or "",
        from_str=from_str,
        to_str=to_str,
        chosen_interval=chosen_interval,
        brokerage_per_lot_rt=brokerage_per_lot_rt,
        taxes_per_lot_rt=taxes_per_lot_rt,
        underlying=underlying,
        name=name,
        session_date=session_date,
        final_rationale=rationale,
        option_target_pct=option_target_pct,
        option_stop_loss_pct=option_stop_loss_pct,
        chart_mk={
            "_chart_entry_bar_idx": entry_bar_idx,
            "_chart_spot_signal": spot_signal,
        },
    )
    if exec_out.get("error"):
        return {
            "success": False,
            "error_message": exec_out["error"],
            "row": None,
            "trace": trace,
            "observability": ctx.observability,
            "chart_pngs": chart_pngs,
            "final_decision": dec,
            "run_dir": str(run_dir) if run_dir else None,
        }
    row = exec_out["row"]
    row["trade"] = dict(row.get("trade") or {})
    row["trade"]["agent_decision"] = dec
    return {
        "success": True,
        "error_message": None,
        "row": row,
        "trace": trace,
        "observability": ctx.observability,
        "chart_pngs": chart_pngs,
        "final_decision": dec,
        "run_dir": str(run_dir) if run_dir else None,
    }


def _parse_tool_call(tc: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(tc, dict):
        name = tc.get("name") or ""
        args = tc.get("args") or tc.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tid = tc.get("id") or "call"
        return name, args, tid
    name = getattr(tc, "name", "") or ""
    args = getattr(tc, "args", None) or {}
    tid = getattr(tc, "id", None) or "call"
    return name, args, tid


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    ctx: FoClaudeContext,
    tools_by_name: dict[str, StructuredTool],
) -> Any:
    t = tools_by_name.get(name)
    if t is None:
        return {"error": f"unknown tool {name}"}
    try:
        return t.invoke(args)
    except Exception as e:
        logger.exception("tool %s failed: %s", name, e)
        return {"error": str(e)[:500]}


def _find_instrument(nfo: list, tradingsymbol: str) -> dict | None:
    for i in nfo:
        if i.get("tradingsymbol") == tradingsymbol:
            return i
    return None


def _system_prompt() -> str:
    return """You are an expert Indian equity index/stock **intraday F&O mock backtest** assistant.

Hard rules:
- **Mock only**: never instruct live orders. This is simulation.
- **Long premium only**: TRADE means long **CE** (bullish view) or long **PE** (bearish view).
- **Think deeply** before deciding: reason step-by-step in your visible text about trend, key levels, VWAP, EMA structure, and risk. Use tools to verify numbers and to align the chart with your thesis.
- Call **session_ohlc_tail**, **indicator_snapshot**, **swing_levels**, **option_candidates** as needed.
- When you want the chart to show specific overlays (EMA periods, VWAP on/off, horizontal lines, trigger bar vertical line, annotations), call **apply_chart_overlays** with a JSON `spec_delta`. The tool returns a fresh PNG; you will receive the updated image in the next user message.
- When finished, call **submit_fo_decision** exactly once with strong rationale and the **intraday_strategy** name (e.g. "VWAP reclaim + EMA9>21", "Opening range breakout", "Trend + pullback to 21 EMA").
- If you choose NO_TRADE, explain clearly why (chop, late session, conflicting signals)."""


def _user_payload_text(
    *,
    underlying: str,
    name: str,
    session_date: date,
    chosen_interval: str,
    strike_policy_label: str,
    steps_from_atm: int,
    manual_strike_val: float | None,
    option_target_pct: float,
    option_stop_loss_pct: float,
    brokerage_per_lot_rt: float,
    taxes_per_lot_rt: float,
    n_bars: int,
) -> str:
    return json.dumps(
        {
            "underlying": underlying,
            "name": name,
            "session_date": session_date.isoformat(),
            "interval": chosen_interval,
            "bars_loaded": n_bars,
            "strike_policy": strike_policy_label,
            "steps_from_atm": steps_from_atm,
            "manual_strike": manual_strike_val,
            "option_target_pct_ui": round(100 * option_target_pct, 2),
            "option_stop_loss_pct_ui": round(100 * option_stop_loss_pct, 2),
            "brokerage_per_lot_rt": brokerage_per_lot_rt,
            "taxes_per_lot_rt": taxes_per_lot_rt,
            "instructions": (
                "Pick any valid intraday logic; do not assume a fixed rule engine. "
                "Use tools + chart. submit_fo_decision.tradingsymbol MUST be one of the symbols "
                "returned by option_candidates for the chosen leg."
            ),
        },
        indent=2,
    )


def _make_tools(ctx: FoClaudeContext, name: str, strike_policy_label: str) -> list[StructuredTool]:
    def session_ohlc_tail(n_bars: int = 40) -> str:
        n = max(5, min(int(n_bars), 120))
        tail = ctx.df_u.tail(n)
        rows = []
        for _, r in tail.iterrows():
            rows.append(
                {
                    "date": str(r["date"]),
                    "o": round(float(r["open"]), 4),
                    "h": round(float(r["high"]), 4),
                    "l": round(float(r["low"]), 4),
                    "c": round(float(r["close"]), 4),
                }
            )
        return json.dumps({"bars": rows, "count": len(rows)})

    def indicator_snapshot(
        ema_periods: str = "9,21,50",
        rsi_period: int = 14,
    ) -> str:
        df = ctx.df_u
        close = pd.to_numeric(df["close"], errors="coerce")
        parts = [int(x.strip()) for x in ema_periods.split(",") if x.strip().isdigit()][:8]
        ema_out = {}
        for p in parts:
            if len(df) >= p:
                ema_out[str(p)] = round(float(close.ewm(span=p, adjust=False).mean().iloc[-1]), 4)
        vw = _vwap_series(df)
        last_c = float(close.iloc[-1])
        last_vw = float(vw.iloc[-1])
        rsi_v = _rsi_last(close, rsi_period)
        payload = {
            "last_close": last_c,
            "vwap_last": last_vw,
            "distance_from_vwap_pct": round(100.0 * (last_c - last_vw) / last_vw, 4) if last_vw else None,
            "ema_last": ema_out,
            "rsi_last": round(rsi_v, 2) if rsi_v is not None else None,
        }
        return json.dumps(payload)

    def swing_levels(lookback: int = 30) -> str:
        lb = max(5, min(int(lookback), len(ctx.df_u)))
        seg = ctx.df_u.tail(lb)
        hi = float(seg["high"].max())
        lo = float(seg["low"].min())
        return json.dumps({"session_tail_high": hi, "session_tail_low": lo, "lookback_bars": lb})

    def option_candidates(leg: str) -> str:
        leg_u = leg.upper().strip()
        if leg_u not in ("CE", "PE"):
            return json.dumps({"error": "leg must be CE or PE"})
        spot = float(ctx.df_u.iloc[-1]["close"])
        cands, err = build_option_trade_candidates(
            ctx.nfo_instruments,
            ctx.underlying,
            spot,
            ctx.session_date,
            leg_u,
            ctx.steps_from_atm,
            ctx.manual_strike_val,
            max_expiries=3,
        )
        if err:
            return json.dumps({"error": err})
        out = []
        for inst, stk, ex in cands[:15]:
            out.append(
                {
                    "tradingsymbol": inst.get("tradingsymbol"),
                    "strike": float(stk),
                    "expiry": ex.isoformat(),
                    "lot_size": int(inst.get("lot_size") or 1),
                }
            )
        return json.dumps({"leg": leg_u, "candidates": out, "strike_policy": strike_policy_label})

    def apply_chart_overlays(spec_delta_json: str) -> str:
        try:
            delta = json.loads(spec_delta_json) if spec_delta_json.strip() else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid JSON: {e}"})
        ctx.overlay_spec = merge_overlay_spec(ctx.overlay_spec, delta)
        png = render_session_chart_png(
            ctx.df_u,
            underlying_label=f"{ctx.underlying} ({ctx.session_date})",
            overlay_spec=ctx.overlay_spec,
        )
        ctx.current_png = png
        if png:
            return json.dumps(
                {
                    "ok": True,
                    "png_size_bytes": len(png),
                    "overlay": ctx.overlay_spec,
                    "note": "A follow-up message will attach the updated chart image for your next reasoning step.",
                },
                default=str,
            )
        return json.dumps({"ok": False, "error": "PNG export failed (kaleido?)", "overlay": ctx.overlay_spec})

    def submit_fo_decision(**kwargs: Any) -> str:
        try:
            parsed = SubmitFoDecisionArgs(**kwargs)
        except Exception as e:
            return json.dumps({"error": f"validation: {e}"})
        d = parsed.model_dump()
        ctx.final_decision = d
        return json.dumps({"ok": True, "recorded": True, "summary": d.get("action")})

    return [
        StructuredTool.from_function(session_ohlc_tail, name="session_ohlc_tail", description="Last N underlying OHLC bars as JSON."),
        StructuredTool.from_function(indicator_snapshot, name="indicator_snapshot", description="EMA last, VWAP, RSI for the session dataframe."),
        StructuredTool.from_function(swing_levels, name="swing_levels", description="Recent high/low in tail window."),
        StructuredTool.from_function(option_candidates, name="option_candidates", description="List tradingsymbol candidates for CE or PE."),
        StructuredTool.from_function(
            apply_chart_overlays,
            name="apply_chart_overlays",
            description="Merge JSON spec_delta into chart overlays and re-render PNG. Pass JSON string keys: ema_periods, show_vwap, hlines, vline_bar_idx, annotations.",
        ),
        StructuredTool.from_function(
            submit_fo_decision,
            name="submit_fo_decision",
            description="Final decision. Required fields per schema. Call once.",
            args_schema=SubmitFoDecisionArgs,
        ),
    ]


def _no_trade_row(
    *,
    underlying: str,
    name: str,
    session_date: date,
    df_u: pd.DataFrame,
    decision: dict[str, Any],
    strat_label: str,
) -> dict[str, Any]:
    return {
        "Session date": session_date,
        "Underlying": underlying,
        "Name": name,
        "Strategy": strat_label,
        "Leg": "—",
        "Option": "—",
        "Strike": float("nan"),
        "Strike pick": "—",
        "Note": (decision.get("rationale_full") or "NO_TRADE")[:500],
        "Signal": "—",
        "Lots": 0,
        "Lot size": float("nan"),
        "Qty": 0,
        "Entry": float("nan"),
        "Target prem.": float("nan"),
        "Stop prem.": float("nan"),
        "Txn cost": 0.0,
        "P/L gross": 0.0,
        "Closed at": "—",
        "Exit": float("nan"),
        "P/L": 0.0,
        "Value": 0.0,
        "df_u": df_u,
        "df_o": None,
        "trade": None,
        "agent_decision": decision,
    }


def _execute_mock_option(
    *,
    kite: Any,
    opt_inst: dict,
    df_u: pd.DataFrame,
    entry_bar_idx: int,
    spot_signal: str,
    strat_label: str,
    analysis_text: str,
    from_str: str,
    to_str: str,
    chosen_interval: str,
    brokerage_per_lot_rt: float,
    taxes_per_lot_rt: float,
    underlying: str,
    name: str,
    session_date: date,
    final_rationale: str,
    option_target_pct: float,
    option_stop_loss_pct: float,
    chart_mk: dict,
) -> dict[str, Any]:
    token = opt_inst.get("instrument_token")
    opt_sym = opt_inst.get("tradingsymbol", "—")
    leg_label = "Long CE" if spot_signal == "BUY" else "Long PE"
    try:
        strike = float(opt_inst.get("strike", 0))
    except (TypeError, ValueError):
        strike = float("nan")
    if token is None:
        return {"error": "Missing option token"}
    try:
        ocandles = kite.historical_data(int(token), from_str, to_str, interval=chosen_interval)
    except Exception as e:
        return {"error": str(e)[:200]}
    df_o = candles_to_dataframe(ocandles)
    if df_o.empty:
        return {"error": "No option candles"}
    df_o = df_o.sort_values("date").reset_index(drop=True)
    opt_entry_idx = align_option_entry_bar(df_u, df_o, entry_bar_idx, chosen_interval)
    if opt_entry_idx is None or opt_entry_idx >= len(df_o):
        return {"error": "Could not align option bar to underlying entry"}
    entry_premium = float(df_o.iloc[opt_entry_idx]["close"])
    if entry_premium <= 0:
        return {"error": "Non-positive entry premium"}
    _otp = float(option_target_pct)
    _osl = float(option_stop_loss_pct)
    target_prem = entry_premium * (1.0 + _otp)
    stop_prem: float | None = None
    if _osl > 0:
        sp = entry_premium * (1.0 - _osl)
        stop_prem = sp if sp > 0 else None
    ls = max(1, int(opt_inst.get("lot_size") or 1))
    n_lots = 1
    qty = ls
    closed_at, exit_price, gross_pl, exit_bar_idx = simulate_long_option_target_stop_eod(
        df_o, opt_entry_idx, entry_premium, target_prem, float(qty), stop_price=stop_prem
    )
    txn_cost = n_lots * (brokerage_per_lot_rt + taxes_per_lot_rt)
    net_pl = gross_pl - txn_cost
    value = entry_premium * qty
    strike_pick = str(opt_sym).strip() or "—"
    trade = {
        "Strategy": strat_label,
        "Signal": spot_signal,
        "Leg": leg_label,
        "Option": opt_sym,
        "Strike": strike,
        "Strike pick": strike_pick,
        "Lots": n_lots,
        "Lot size": ls,
        "Qty": qty,
        "Entry": entry_premium,
        "Target prem.": target_prem,
        "Stop prem.": stop_prem if stop_prem is not None else float("nan"),
        "entry_bar_idx": entry_bar_idx,
        "exit_bar_idx": exit_bar_idx,
        "opt_entry_idx": opt_entry_idx,
        "Closed at": closed_at,
        "Exit": exit_price,
        "Txn cost": txn_cost,
        "P/L gross": gross_pl,
        "P/L": net_pl,
        "Value": value,
        "why": analysis_text,
        "agent_rationale": final_rationale,
    }
    row = {
        "Session date": session_date,
        "Underlying": underlying,
        "Name": name,
        "Strategy": strat_label,
        "Leg": leg_label,
        "Option": opt_sym,
        "Strike": strike,
        "Strike pick": strike_pick,
        "Note": f"OK — {strike_pick}",
        "Signal": spot_signal,
        "Lots": n_lots,
        "Lot size": ls,
        "Qty": qty,
        "Entry": entry_premium,
        "Target prem.": target_prem,
        "Stop prem.": stop_prem if stop_prem is not None else float("nan"),
        "Txn cost": txn_cost,
        "P/L gross": gross_pl,
        "Closed at": closed_at,
        "Exit": exit_price,
        "P/L": net_pl,
        "Value": value,
        "df_u": df_u,
        "df_o": df_o,
        "trade": trade,
        "_analysis_text": analysis_text,
        **chart_mk,
    }
    return {"row": row}
