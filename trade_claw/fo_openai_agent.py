"""OpenAI ReAct agent for F&O mock option selection. Allowlisted Kite reads only — never order APIs."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from openai import OpenAI

from trade_claw.constants import (
    ENVELOPE_EMA_PERIOD,
    FO_OPTION_TARGET_PCT,
    MA_EMA_FAST,
    MA_EMA_SLOW,
)
from trade_claw.fo_support import (
    align_option_entry_bar,
    fetch_underlying_intraday,
    filter_options_by_underlying,
    pick_option_contract,
    _to_date,
)
from trade_claw.market_data import candles_to_dataframe
from trade_claw.option_trades import simulate_long_option_target_or_eod
from trade_claw.strategies import _ma_ema_crossover_analysis, _ma_envelope_analysis

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
MAX_AGENT_TURNS = 12
MAX_OHLC_ROWS_TO_MODEL = 60
MAX_SEARCH_RESULTS = 35

logger = logging.getLogger("trade_claw.fo_openai_agent")


def configure_fo_agent_logging() -> None:
    """
    Attach a stream handler for this module if none exist (Streamlit often leaves root at WARNING).
    Set FO_AGENT_LOG_LEVEL=DEBUG|INFO|WARNING (default INFO).
    """
    if logger.handlers:
        lvl_name = os.environ.get("FO_AGENT_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, lvl_name, logging.INFO))
        return
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [fo_agent] %(message)s")
    )
    logger.addHandler(h)
    logger.propagate = False
    lvl_name = os.environ.get("FO_AGENT_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, lvl_name, logging.INFO))


@dataclass
class FoAgentResult:
    success: bool
    error_message: str | None = None
    intent: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    row: dict[str, Any] | None = None
    final_rationale: str | None = None


def _session_window(session_date: date) -> tuple[str, str]:
    from_dt = datetime(session_date.year, session_date.month, session_date.day, 9, 15, 0)
    to_dt = datetime(session_date.year, session_date.month, session_date.day, 15, 30, 0)
    return from_dt.strftime("%Y-%m-%d %H:%M:%S"), to_dt.strftime("%Y-%m-%d %H:%M:%S")


def compute_deterministic_intent(
    *,
    kite,
    nse_instruments: list,
    underlying: str,
    session_date: date,
    chosen_interval: str,
    strategy_is_envelope: bool,
    envelope_pct: float,
    strategy_choice: str,
) -> dict[str, Any]:
    from_str, to_str = _session_window(session_date)
    logger.info(
        "Deterministic intent: fetch underlying %s session=%s interval=%s window=%s..%s",
        underlying,
        session_date.isoformat(),
        chosen_interval,
        from_str,
        to_str,
    )
    df_u, err_u = fetch_underlying_intraday(
        kite, underlying, nse_instruments, from_str, to_str, chosen_interval
    )
    base: dict[str, Any] = {
        "underlying": underlying,
        "from_str": from_str,
        "to_str": to_str,
        "session_date": session_date.isoformat(),
        "chosen_interval": chosen_interval,
        "strategy_choice": strategy_choice,
        "df_u": df_u,
        "err_u": err_u,
        "has_signal": False,
        "spot_signal": None,
        "entry_bar_idx": None,
        "spot": None,
        "leg_type": None,
        "strat_label": None,
        "analysis_text": None,
        "_chart_mk": {},
    }
    if df_u is None or err_u:
        base["analysis_text"] = err_u or "No underlying data"
        logger.warning("Underlying fetch failed or empty: %s err=%s", underlying, err_u)
        return base

    logger.info(
        "Underlying Kite historical_data OK: %s bars=%s",
        underlying,
        len(df_u),
    )

    if strategy_is_envelope:
        ok, text, sig = _ma_envelope_analysis(df_u, ema_period=ENVELOPE_EMA_PERIOD, pct=envelope_pct)
        strat_label = "Envelope → option"
    else:
        ok, text, sig = _ma_ema_crossover_analysis(
            df_u, fast_period=MA_EMA_FAST, slow_period=MA_EMA_SLOW
        )
        strat_label = "MA cross → option"
    base["strat_label"] = strat_label
    base["analysis_text"] = text

    if not ok or not sig.get("signal"):
        logger.info("Strategy: no signal (ok=%s)", ok)
        return base

    spot_signal = sig["signal"]
    entry_bar_idx = sig.get("entry_bar_idx")
    try:
        entry_bar_idx = int(entry_bar_idx) if entry_bar_idx is not None else None
    except (TypeError, ValueError):
        entry_bar_idx = None
    if entry_bar_idx is None or not (0 <= entry_bar_idx < len(df_u)):
        base["analysis_text"] = (text or "") + " (invalid entry bar)"
        logger.warning(
            "Strategy: invalid entry_bar_idx=%s len_bars=%s",
            entry_bar_idx,
            len(df_u),
        )
        return base

    leg_type = "CE" if spot_signal == "BUY" else "PE"
    spot = float(df_u.iloc[entry_bar_idx]["close"])
    ei = int(entry_bar_idx)
    _chart_mk: dict[str, Any] = {}
    if spot_signal in ("BUY", "SELL") and 0 <= ei < len(df_u):
        _chart_mk = {"_chart_entry_bar_idx": ei, "_chart_spot_signal": spot_signal}

    base.update(
        {
            "has_signal": True,
            "spot_signal": spot_signal,
            "entry_bar_idx": entry_bar_idx,
            "spot": spot,
            "leg_type": leg_type,
            "_chart_mk": _chart_mk,
        }
    )
    logger.info(
        "Strategy signal: %s leg=%s entry_bar=%s spot=%.4f",
        spot_signal,
        leg_type,
        entry_bar_idx,
        spot,
    )
    return base


def _validate_nfo_choice(
    token: int,
    nfo: list,
    underlying: str,
    leg_type: str,
) -> tuple[dict | None, str | None]:
    by_token: dict[int, dict] = {}
    for i in nfo:
        t = i.get("instrument_token")
        if t is not None:
            by_token[int(t)] = i
    inst = by_token.get(int(token))
    if inst is None:
        return None, f"Unknown instrument_token {token}"
    if inst.get("exchange") != "NFO":
        return None, "Instrument is not NFO"
    if inst.get("instrument_type") != leg_type:
        return None, f"Expected {leg_type}, got {inst.get('instrument_type')}"
    allowed = {
        int(x["instrument_token"])
        for x in filter_options_by_underlying(nfo, underlying)
        if x.get("instrument_token") is not None
    }
    if int(token) not in allowed:
        return None, "Token is not in the option chain for this underlying"
    return inst, None


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_nfo_options",
            "description": (
                "Search listed NFO options. Only expiries on or after session_date. "
                "Use to list CE or PE contracts for the underlying."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "underlying": {"type": "string"},
                    "instrument_type": {"type": "string", "enum": ["CE", "PE"]},
                    "session_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "strike_near": {
                        "type": "number",
                        "description": "Optional; sort by distance to this strike first",
                    },
                    "max_results": {"type": "integer"},
                },
                "required": ["underlying", "instrument_type", "session_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ohlc_bars",
            "description": (
                "Read-only intraday OHLC for an instrument_token (session window). "
                "Does not place orders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument_token": {"type": "integer"},
                    "max_bars": {"type": "integer", "description": "Last N bars (capped)"},
                },
                "required": ["instrument_token"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_mock_trade_choice",
            "description": (
                "Record the final contract for MOCK backtest only. No real order. Call once when decided."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument_token": {"type": "integer"},
                    "tradingsymbol": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["instrument_token", "tradingsymbol", "rationale"],
            },
        },
    },
]


class _ToolContext:
    def __init__(
        self,
        *,
        kite,
        nfo_instruments: list,
        from_str: str,
        to_str: str,
        chosen_interval: str,
        underlying: str,
        leg_type: str,
    ):
        self.kite = kite
        self.nfo_instruments = nfo_instruments
        self.from_str = from_str
        self.to_str = to_str
        self.chosen_interval = chosen_interval
        self.underlying = underlying.upper().strip()
        self.leg_type = leg_type.upper().strip()
        self.final_choice: dict[str, Any] | None = None

    def search_nfo_options(self, args: dict) -> dict:
        u = str(args.get("underlying", "")).upper().strip()
        if u != self.underlying:
            logger.warning("[kite-tool/MCP-style] search_nfo_options rejected wrong underlying=%s", u)
            return {"error": f"underlying must be {self.underlying} for this session"}
        it = str(args.get("instrument_type", "")).upper()
        if it not in ("CE", "PE"):
            logger.warning("[kite-tool] search_nfo_options bad instrument_type=%s", it)
            return {"error": "instrument_type must be CE or PE"}
        try:
            sd = date.fromisoformat(str(args["session_date"])[:10])
        except ValueError:
            logger.warning("[kite-tool] search_nfo_options bad session_date=%s", args.get("session_date"))
            return {"error": "session_date must be YYYY-MM-DD"}
        strike_near = args.get("strike_near")
        try:
            max_r = int(args.get("max_results", 30))
        except (TypeError, ValueError):
            max_r = 30
        max_r = max(1, min(max_r, MAX_SEARCH_RESULTS))

        opts = filter_options_by_underlying(self.nfo_instruments, u)
        rows = [
            i
            for i in opts
            if i.get("instrument_type") == it
            and _to_date(i.get("expiry")) is not None
            and _to_date(i.get("expiry")) >= sd
        ]
        if not rows:
            logger.info(
                "[kite-tool/MCP-style] search_nfo_options no rows underlying=%s type=%s from=%s",
                u,
                it,
                sd.isoformat(),
            )
            return {"options": [], "note": "No contracts for filters"}

        def sort_key(inst: dict) -> tuple:
            ed = _to_date(inst.get("expiry")) or date.max
            try:
                strike = float(inst.get("strike", 0))
            except (TypeError, ValueError):
                strike = 0.0
            if strike_near is not None:
                try:
                    sn = float(strike_near)
                    dist = abs(strike - sn)
                except (TypeError, ValueError):
                    dist = 0.0
            else:
                dist = 0.0
            return (ed, dist, strike, inst.get("tradingsymbol") or "")

        rows.sort(key=sort_key)
        out = []
        for i in rows[:max_r]:
            ed = _to_date(i.get("expiry"))
            out.append(
                {
                    "instrument_token": int(i["instrument_token"]),
                    "tradingsymbol": i.get("tradingsymbol"),
                    "strike": float(i.get("strike", 0)),
                    "expiry": ed.isoformat() if ed else None,
                    "lot_size": int(i.get("lot_size") or 1),
                }
            )
        logger.info(
            "[kite-tool/MCP-style] search_nfo_options underlying=%s type=%s session_date=%s "
            "max_results=%s -> %s row(s)",
            u,
            it,
            sd.isoformat(),
            max_r,
            len(out),
        )
        logger.debug("[kite-tool] search_nfo_options sample: %s", out[:3] if out else [])

        return {"options": out, "count": len(out)}

    def get_ohlc_bars(self, args: dict) -> dict:
        try:
            token = int(args["instrument_token"])
        except (KeyError, TypeError, ValueError):
            logger.warning("[kite-tool/MCP-style] get_ohlc_bars invalid args: %s", args)
            return {"error": "invalid instrument_token"}
        try:
            max_bars = min(int(args.get("max_bars", 50)), MAX_OHLC_ROWS_TO_MODEL)
        except (TypeError, ValueError):
            max_bars = 50
        logger.info(
            "[kite-tool/MCP-style] get_ohlc_bars Kite.historical_data token=%s interval=%s max_bars=%s",
            token,
            self.chosen_interval,
            max_bars,
        )
        try:
            candles = self.kite.historical_data(
                token, self.from_str, self.to_str, interval=self.chosen_interval
            )
        except Exception as e:
            logger.exception("[kite-tool] historical_data failed token=%s: %s", token, e)
            return {"error": str(e)[:200], "bars": []}
        df = candles_to_dataframe(candles)
        if df.empty:
            logger.warning("[kite-tool] get_ohlc_bars empty dataframe token=%s", token)
            return {"bars": [], "note": "empty"}
        df = df.sort_values("date").reset_index(drop=True)
        tail = df.tail(max_bars)
        bars = []
        for _, r in tail.iterrows():
            bars.append(
                {
                    "date": str(r["date"]),
                    "open": round(float(r["open"]), 4),
                    "high": round(float(r["high"]), 4),
                    "low": round(float(r["low"]), 4),
                    "close": round(float(r["close"]), 4),
                }
            )
        logger.info(
            "[kite-tool/MCP-style] get_ohlc_bars OK token=%s returned_bars=%s (from %s total rows)",
            token,
            len(bars),
            len(df),
        )
        return {"bars": bars, "count": len(bars)}

    def submit_mock_trade_choice(self, args: dict) -> dict:
        try:
            token = int(args["instrument_token"])
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "invalid instrument_token"}
        ts = str(args.get("tradingsymbol", "")).strip()
        rationale = str(args.get("rationale", ""))[:8000]
        inst, err = _validate_nfo_choice(token, self.nfo_instruments, self.underlying, self.leg_type)
        if err:
            logger.warning(
                "[kite-tool/MCP-style] submit_mock_trade_choice rejected token=%s: %s",
                token,
                err,
            )
            return {"ok": False, "error": err}
        if ts and inst.get("tradingsymbol") != ts:
            # token wins; warn in result
            pass
        self.final_choice = {
            "instrument_token": token,
            "tradingsymbol": inst.get("tradingsymbol"),
            "rationale": rationale,
            "inst": inst,
        }
        logger.info(
            "[kite-tool/MCP-style] submit_mock_trade_choice ACCEPTED token=%s symbol=%s leg=%s "
            "rationale_len=%s",
            token,
            inst.get("tradingsymbol"),
            self.leg_type,
            len(rationale),
        )
        return {
            "ok": True,
            "message": "Mock choice recorded. Stop and wait — do not call submit_mock_trade_choice again.",
        }


def _build_system_prompt() -> str:
    return """You assist with INDIAN F&O **mock backtesting** only.

Hard rules:
- Never suggest live orders, real brokerage placement, or any Kite order API.
- Only use the provided tools: search contracts, read OHLC, submit_mock_trade_choice.
- submit_mock_trade_choice does NOT send orders; it only records the contract for simulation.

Task: Given the underlying strategy signal (BUY → long CE, SELL → long PE), pick one NFO option
that is appropriate for the session date (expiry on or after session). Prefer sensible near-ATM
strikes unless the analysis implies otherwise. You may fetch OHLC on candidates.

When satisfied, call submit_mock_trade_choice exactly once with instrument_token from search results."""


def _execute_mock_trade(
    *,
    kite,
    opt_inst: dict,
    df_u,
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

    logger.info(
        "[mock-exec] Kite.historical_data option token=%s symbol=%s interval=%s",
        token,
        opt_sym,
        chosen_interval,
    )
    try:
        ocandles = kite.historical_data(int(token), from_str, to_str, interval=chosen_interval)
    except Exception as e:
        logger.exception("[mock-exec] option historical_data failed: %s", e)
        return {"error": str(e)[:200]}

    df_o = candles_to_dataframe(ocandles)
    if df_o.empty:
        logger.warning("[mock-exec] no option candles token=%s", token)
        return {"error": "No option candles"}
    df_o = df_o.sort_values("date").reset_index(drop=True)
    opt_entry_idx = align_option_entry_bar(df_u, df_o, entry_bar_idx, chosen_interval)
    if opt_entry_idx is None or opt_entry_idx >= len(df_o):
        return {"error": "Could not align option bar to underlying entry"}

    entry_premium = float(df_o.iloc[opt_entry_idx]["close"])
    if entry_premium <= 0:
        return {"error": "Non-positive entry premium"}

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
    strike_pick = "LLM agent"

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
        "Note": "OK — LLM mock",
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
        "_analysis_text": analysis_text,
        **chart_mk,
    }
    logger.info(
        "[mock-exec] done symbol=%s closed_at=%s net_pl=%.2f gross=%.2f txn=%.2f",
        opt_sym,
        closed_at,
        net_pl,
        gross_pl,
        txn_cost,
    )
    return {"row": row}


def run_fo_agent_pipeline(
    *,
    kite,
    nse_instruments: list,
    nfo_instruments: list,
    underlying: str,
    name: str,
    session_date: date,
    chosen_interval: str,
    strategy_is_envelope: bool,
    envelope_pct: float,
    strategy_choice: str,
    brokerage_per_lot_rt: float,
    taxes_per_lot_rt: float,
    openai_api_key: str,
    model: str | None = None,
) -> FoAgentResult:
    configure_fo_agent_logging()
    trace: list[dict[str, Any]] = []
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    logger.info(
        "=== F&O agent pipeline start underlying=%s session=%s model=%s strategy=%s ===",
        underlying,
        session_date.isoformat(),
        model,
        strategy_choice,
    )

    intent = compute_deterministic_intent(
        kite=kite,
        nse_instruments=nse_instruments,
        underlying=underlying,
        session_date=session_date,
        chosen_interval=chosen_interval,
        strategy_is_envelope=strategy_is_envelope,
        envelope_pct=envelope_pct,
        strategy_choice=strategy_choice,
    )

    intent_for_trace = {k: v for k, v in intent.items() if k != "df_u"}
    trace.append({"step": "deterministic_intent", "intent": intent_for_trace})

    if intent.get("df_u") is None or intent.get("err_u"):
        logger.warning("Pipeline stop: no underlying data (%s)", intent.get("err_u"))
        return FoAgentResult(
            success=False,
            error_message=intent.get("analysis_text") or intent.get("err_u") or "No underlying data",
            intent=intent,
            trace=trace,
        )

    if not intent.get("has_signal"):
        logger.info("Pipeline stop: no strategy signal, skipping OpenAI")
        return FoAgentResult(
            success=False,
            error_message="No strategy signal for this session — OpenAI agent was not run.",
            intent=intent,
            trace=trace,
        )

    from_str = intent["from_str"]
    to_str = intent["to_str"]
    leg_type = intent["leg_type"]
    assert leg_type is not None

    hint_line = ""
    try:
        spo = float(intent["spot"])
        inst0, e0, st0 = pick_option_contract(
            nfo_instruments, underlying, spo, session_date, leg_type, 0, None
        )
        if inst0 and not e0:
            hint_line = (
                f"Deterministic ATM hint: {inst0.get('tradingsymbol')} "
                f"token={inst0.get('instrument_token')} strike={st0}. "
                "You may choose this or another valid contract.\n"
            )
    except Exception:
        pass

    user_payload = {
        "underlying": underlying,
        "session_date": intent["session_date"],
        "strategy": intent["strat_label"],
        "spot_signal": intent["spot_signal"],
        "required_option_type": leg_type,
        "spot_at_signal": intent["spot"],
        "entry_bar_index": intent["entry_bar_idx"],
        "analysis_excerpt": (intent.get("analysis_text") or "")[:2000],
        "interval": chosen_interval,
        "instructions": hint_line + "Use tools to pick one contract, then submit_mock_trade_choice once.",
    }
    user_content = json.dumps(user_payload, indent=2)

    ctx = _ToolContext(
        kite=kite,
        nfo_instruments=nfo_instruments,
        from_str=from_str,
        to_str=to_str,
        chosen_interval=chosen_interval,
        underlying=underlying,
        leg_type=leg_type,
    )

    client = OpenAI(api_key=openai_api_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    for turn in range(MAX_AGENT_TURNS):
        trace.append({"step": "openai_turn", "turn": turn + 1})
        logger.info(
            "[LLM] chat.completions.create turn=%s/%s model=%s messages_in=%s (api_key=***)",
            turn + 1,
            MAX_AGENT_TURNS,
            model,
            len(messages),
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            logger.exception("[LLM] OpenAI API request failed: %s", e)
            trace.append({"step": "openai_error", "error": str(e)})
            return FoAgentResult(
                success=False,
                error_message=f"OpenAI API error: {e}",
                intent=intent,
                trace=trace,
            )

        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        usage_d: dict[str, Any] = {}
        if usage is not None:
            usage_d = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        rid = getattr(resp, "id", None)
        logger.info(
            "[LLM] response id=%s finish_reason=%s usage=%s",
            rid,
            finish,
            usage_d,
        )
        msg = choice.message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                logger.info(
                    "[LLM] assistant tool_call name=%s id=%s args_len=%s",
                    tc.function.name,
                    tc.id,
                    len(tc.function.arguments or ""),
                )
        elif msg.content:
            logger.debug("[LLM] assistant text (truncated): %s", (msg.content or "")[:400])
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            trace.append({"step": "assistant_no_tools", "content": (msg.content or "")[:600]})
            logger.info("[LLM] no tool_calls this turn (finish=%s)", finish)
            if ctx.final_choice:
                break
            continue

        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fargs = {}
            trace.append({"step": "tool_call", "name": fname, "arguments": fargs})

            if fname == "search_nfo_options":
                result = ctx.search_nfo_options(fargs)
            elif fname == "get_ohlc_bars":
                result = ctx.get_ohlc_bars(fargs)
            elif fname == "submit_mock_trade_choice":
                result = ctx.submit_mock_trade_choice(fargs)
            else:
                result = {"error": f"unknown tool {fname}"}

            preview = json.dumps(result, default=str)
            if len(preview) > 1200:
                preview = preview[:1200] + "…"
            trace.append({"step": "tool_result", "name": fname, "result_preview": preview})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:15000],
                }
            )

        if ctx.final_choice:
            logger.info("[LLM] loop exit after submit_mock_trade_choice (turn %s)", turn + 1)
            break
    else:
        if not ctx.final_choice:
            logger.warning("[LLM] max turns reached without submit_mock_trade_choice")
            return FoAgentResult(
                success=False,
                error_message="Agent did not call submit_mock_trade_choice in time.",
                intent=intent,
                trace=trace,
            )

    if not ctx.final_choice or not ctx.final_choice.get("inst"):
        return FoAgentResult(
            success=False,
            error_message="No valid mock trade choice recorded.",
            intent=intent,
            trace=trace,
        )

    rationale = ctx.final_choice.get("rationale") or ""
    exec_out = _execute_mock_trade(
        kite=kite,
        opt_inst=ctx.final_choice["inst"],
        df_u=intent["df_u"],
        entry_bar_idx=int(intent["entry_bar_idx"]),
        spot_signal=str(intent["spot_signal"]),
        strat_label=str(intent["strat_label"]),
        analysis_text=str(intent.get("analysis_text") or ""),
        from_str=from_str,
        to_str=to_str,
        chosen_interval=chosen_interval,
        brokerage_per_lot_rt=brokerage_per_lot_rt,
        taxes_per_lot_rt=taxes_per_lot_rt,
        underlying=underlying,
        name=name,
        session_date=session_date,
        final_rationale=rationale,
        chart_mk=intent.get("_chart_mk") or {},
    )
    if exec_out.get("error"):
        logger.error("Pipeline stop: mock execution failed: %s", exec_out["error"])
        return FoAgentResult(
            success=False,
            error_message=exec_out["error"],
            intent=intent,
            trace=trace,
            final_rationale=rationale,
        )

    logger.info("=== F&O agent pipeline SUCCESS underlying=%s ===", underlying)
    return FoAgentResult(
        success=True,
        intent=intent,
        trace=trace,
        row=exec_out["row"],
        final_rationale=rationale,
    )


def get_openai_api_key() -> str | None:
    """Resolve API key from environment (dotenv loaded by app)."""
    return os.environ.get("OPENAI_API_KEY") or None
