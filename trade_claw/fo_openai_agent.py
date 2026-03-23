"""OpenAI ReAct agent for F&O mock option selection. Allowlisted Kite reads only — never order APIs."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from mcp import ClientSession
from openai import OpenAI

from trade_claw.constants import (
    ENVELOPE_EMA_PERIOD,
    FO_OPTION_STOP_LOSS_PCT,
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
from trade_claw.option_trades import simulate_long_option_target_stop_eod
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


# OpenAI tool list = Zerodha Kite MCP tool names/schemas for reads only (no place_order / GTT / etc.).
_KITE_INTERVAL_ENUM = [
    "minute",
    "day",
    "3minute",
    "5minute",
    "10minute",
    "15minute",
    "30minute",
    "60minute",
]

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_instruments",
            "description": (
                "Zerodha Kite MCP: search instruments. For this session use filter_on 'underlying' "
                "and query 'NFO:<UNDERLYING>' or '<UNDERLYING>' (NFO implied). "
                "Results are restricted server-side to the session underlying and required CE/PE leg only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. NFO:NIFTY)",
                    },
                    "filter_on": {
                        "type": "string",
                        "description": "Filter field: id, name, isin, tradingsymbol, underlying",
                        "enum": ["id", "name", "isin", "tradingsymbol", "underlying"],
                    },
                    "from": {
                        "type": "number",
                        "description": "Pagination start index (0-based). Default 0",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max instruments returned from search (MCP pagination)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_data",
            "description": (
                "Zerodha Kite MCP: historical OHLC for one instrument_token. "
                "Use the session from_date, to_date, and interval from the user message exactly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument_token": {
                        "type": "number",
                        "description": "Token from search_instruments",
                    },
                    "from_date": {
                        "type": "string",
                        "description": "YYYY-MM-DD HH:MM:SS",
                    },
                    "to_date": {
                        "type": "string",
                        "description": "YYYY-MM-DD HH:MM:SS",
                    },
                    "interval": {
                        "type": "string",
                        "description": "Candle interval",
                        "enum": _KITE_INTERVAL_ENUM,
                    },
                    "continuous": {
                        "type": "boolean",
                        "description": (
                            "For NFO **options** (CE/PE) use **false** — Kite returns "
                            "'invalid interval for continuous data' if true with minute bars. "
                            "Continuous is for rolled futures, not single option contracts."
                        ),
                    },
                    "oi": {
                        "type": "boolean",
                        "description": "Include open interest",
                    },
                },
                "required": ["instrument_token", "from_date", "to_date", "interval"],
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


def _norm_session_dt(s: str) -> str:
    t = str(s).strip().replace("T", " ")
    if len(t) >= 19:
        return t[:19]
    return t


def _underlying_from_instruments_query(query: str) -> str | None:
    q = query.strip().upper()
    if not q:
        return None
    if ":" in q:
        parts = q.split(":", 1)
        if len(parts) != 2:
            return None
        return parts[1].strip()
    return q


def _parse_mcp_expiry_date(s: Any) -> date | None:
    if s is None:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _mcp_row_matches_underlying(row: dict[str, Any], u: str) -> bool:
    """Match MCP instrument row to session underlying (same rules as filter_options_by_underlying)."""
    name = (row.get("name") or "").upper().strip()
    ts = (row.get("tradingsymbol") or "").upper()
    uu = u.upper().strip()
    if uu == "NIFTY":
        return name == "NIFTY"
    if uu == "BANKNIFTY":
        return name == "BANKNIFTY"
    return name == uu or ts.startswith(uu)


def _token_allowed_for_fo_session(
    token: int,
    *,
    nfo_instruments: list,
    underlying: str,
    leg_type: str,
    session_date: date,
) -> bool:
    for i in filter_options_by_underlying(nfo_instruments, underlying):
        try:
            if int(i.get("instrument_token", 0)) != token:
                continue
        except (TypeError, ValueError):
            continue
        if i.get("instrument_type") != leg_type:
            continue
        ed = _to_date(i.get("expiry"))
        if ed is not None and ed >= session_date:
            return True
    return False


class _LocalToolContext:
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
        session_date: date,
    ):
        self.kite = kite
        self.nfo_instruments = nfo_instruments
        self.from_str = from_str
        self.to_str = to_str
        self.chosen_interval = chosen_interval
        self.underlying = underlying.upper().strip()
        self.leg_type = leg_type.upper().strip()
        self.session_date = session_date
        self.final_choice: dict[str, Any] | None = None

    def _validate_get_historical_data_args(self, args: dict) -> str | None:
        try:
            token = int(args["instrument_token"])
        except (KeyError, TypeError, ValueError):
            return "invalid instrument_token"
        if not _token_allowed_for_fo_session(
            token,
            nfo_instruments=self.nfo_instruments,
            underlying=self.underlying,
            leg_type=self.leg_type,
            session_date=self.session_date,
        ):
            return "instrument_token not allowed for this session (wrong chain / leg / expiry)"
        try:
            fds = _norm_session_dt(str(args["from_date"]))
            tds = _norm_session_dt(str(args["to_date"]))
        except KeyError:
            return "from_date and to_date are required"
        if fds != _norm_session_dt(self.from_str) or tds != _norm_session_dt(self.to_str):
            return (
                f"from_date and to_date must match session window exactly: "
                f"{self.from_str} .. {self.to_str}"
            )
        interval = str(args.get("interval", "")).strip()
        if interval != self.chosen_interval:
            return f"interval must be {self.chosen_interval!r} for this session"
        return None

    def _post_filter_search_rows(self, rows_raw: list[dict[str, Any]], u: str) -> list[dict[str, Any]]:
        """NFO options only: session underlying, leg CE/PE, expiry >= session_date."""
        it = self.leg_type
        sd = self.session_date
        rows: list[dict[str, Any]] = []
        for row in rows_raw:
            if (row.get("exchange") or "").upper() != "NFO":
                continue
            if (row.get("instrument_type") or "").upper() != it:
                continue
            if row.get("active") is False:
                continue
            if not _mcp_row_matches_underlying(row, u):
                continue
            ed = _parse_mcp_expiry_date(row.get("expiry_date"))
            if ed is None:
                ed = _to_date(row.get("expiry"))
            if ed is None or ed < sd:
                continue
            rows.append(row)

        def sort_key(inst: dict[str, Any]) -> tuple:
            ed = _parse_mcp_expiry_date(inst.get("expiry_date")) or _to_date(inst.get("expiry")) or date.max
            try:
                strike = float(inst.get("strike", 0))
            except (TypeError, ValueError):
                strike = 0.0
            return (ed, strike, inst.get("tradingsymbol") or "")

        rows.sort(key=sort_key)
        return rows

    def _rows_to_option_summaries(self, rows: list[dict[str, Any]], max_r: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in rows[:max_r]:
            ed = _parse_mcp_expiry_date(i.get("expiry_date")) or _to_date(i.get("expiry"))
            try:
                tok = int(i.get("instrument_token", 0))
            except (TypeError, ValueError):
                continue
            if tok <= 0:
                continue
            out.append(
                {
                    "instrument_token": tok,
                    "tradingsymbol": i.get("tradingsymbol"),
                    "strike": float(i.get("strike", 0)),
                    "expiry": ed.isoformat() if ed else None,
                    "lot_size": int(i.get("lot_size") or 1),
                }
            )
        return out

    def search_instruments(self, args: dict) -> dict:
        q = str(args.get("query", "")).strip()
        if not q:
            return {"error": "query is required", "options": []}
        fo = str(args.get("filter_on") or "underlying").strip().lower()
        if fo != "underlying":
            return {
                "error": "This F&O session only allows filter_on='underlying' with query NFO:<UNDERLYING>.",
                "options": [],
            }
        u = _underlying_from_instruments_query(q)
        if not u or u != self.underlying:
            logger.warning("[kite-tool] search_instruments rejected query=%s session=%s", q, self.underlying)
            return {
                "error": f"query must resolve to underlying {self.underlying} (e.g. NFO:{self.underlying})",
                "options": [],
            }
        try:
            mcp_from = int(args["from"]) if args.get("from") is not None else 0
        except (TypeError, ValueError):
            mcp_from = 0
        mcp_from = max(0, mcp_from)
        try:
            search_limit = (
                int(args["limit"])
                if args.get("limit") is not None
                else int(os.environ.get("KITE_MCP_SEARCH_LIMIT", "500"))
            )
        except (TypeError, ValueError):
            search_limit = 500
        search_limit = max(1, min(search_limit, 5000))

        opts = filter_options_by_underlying(self.nfo_instruments, u)
        rows_src: list[dict[str, Any]] = []
        for i in opts:
            rows_src.append(
                {
                    "exchange": i.get("exchange"),
                    "instrument_type": i.get("instrument_type"),
                    "instrument_token": i.get("instrument_token"),
                    "tradingsymbol": i.get("tradingsymbol"),
                    "name": i.get("name"),
                    "strike": i.get("strike"),
                    "lot_size": i.get("lot_size"),
                    "expiry_date": None,
                    "expiry": i.get("expiry"),
                    "active": i.get("active", True),
                }
            )
        filtered = self._post_filter_search_rows(rows_src, u)
        page = filtered[mcp_from : mcp_from + search_limit]
        max_r = min(MAX_SEARCH_RESULTS, len(page))
        out = self._rows_to_option_summaries(page, max_r)

        if not out:
            logger.info(
                "[kite-tool] search_instruments no rows underlying=%s leg=%s session_date=%s",
                u,
                self.leg_type,
                self.session_date.isoformat(),
            )
            return {"options": [], "note": "No contracts for filters"}

        logger.info(
            "[kite-tool] search_instruments underlying=%s leg=%s session_date=%s -> %s row(s)",
            u,
            self.leg_type,
            self.session_date.isoformat(),
            len(out),
        )
        return {"options": out, "count": len(out)}

    def get_historical_data(self, args: dict) -> dict:
        verr = self._validate_get_historical_data_args(args)
        if verr:
            logger.warning("[kite-tool] get_historical_data rejected: %s", verr)
            return {"error": verr, "data": []}
        token = int(args["instrument_token"])
        interval = str(args.get("interval", "")).strip()
        raw_cont = args.get("continuous", False)
        if isinstance(raw_cont, str):
            raw_cont = raw_cont.lower() in ("1", "true", "yes")
        # Kite: continuous=True is not valid for option intraday (InputException: invalid interval for continuous data).
        continuous = False
        if raw_cont:
            logger.info(
                "[kite-tool] get_historical_data forcing continuous=False for option token=%s (LLM sent true)",
                token,
            )
        oi = args.get("oi", False)
        if isinstance(oi, str):
            oi = oi.lower() in ("1", "true", "yes")

        logger.info(
            "[kite-tool] get_historical_data Kite.historical_data token=%s interval=%s continuous=False",
            token,
            interval,
        )
        try:
            candles = self.kite.historical_data(
                token,
                self.from_str,
                self.to_str,
                interval=interval,
                continuous=continuous,
                oi=bool(oi),
            )
        except Exception as e:
            logger.exception("[kite-tool] historical_data failed token=%s: %s", token, e)
            return {"error": str(e)[:200], "data": []}
        df = candles_to_dataframe(candles)
        if df.empty:
            logger.warning("[kite-tool] get_historical_data empty token=%s", token)
            return {"data": [], "note": "empty"}
        df = df.sort_values("date").reset_index(drop=True)
        tail = df.tail(MAX_OHLC_ROWS_TO_MODEL)
        out: list[dict[str, Any]] = []
        for _, r in tail.iterrows():
            out.append(
                {
                    "date": str(r["date"]),
                    "open": round(float(r["open"]), 4),
                    "high": round(float(r["high"]), 4),
                    "low": round(float(r["low"]), 4),
                    "close": round(float(r["close"]), 4),
                }
            )
        logger.info(
            "[kite-tool] get_historical_data OK token=%s returned=%s (from %s total)",
            token,
            len(out),
            len(df),
        )
        return {"data": out, "count": len(out)}

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


class _AgentToolContext:
    """
    Dispatches Zerodha Kite MCP tool names (search_instruments, get_historical_data) to MCP or KiteConnect.
    Only these read tools + submit_mock_trade_choice are exposed to the LLM (no order APIs).
    """

    def __init__(
        self,
        *,
        local: _LocalToolContext,
        mcp_session: ClientSession | None,
    ) -> None:
        self._local = local
        self._mcp = mcp_session

    @property
    def final_choice(self) -> dict[str, Any] | None:
        return self._local.final_choice

    async def dispatch_tool(self, fname: str, fargs: dict[str, Any]) -> dict[str, Any]:
        if fname == "search_instruments":
            if self._mcp is not None:
                return await self._search_instruments_via_mcp(fargs)
            return self._local.search_instruments(fargs)
        if fname == "get_historical_data":
            if self._mcp is not None:
                return await self._get_historical_data_via_mcp(fargs)
            return self._local.get_historical_data(fargs)
        if fname == "submit_mock_trade_choice":
            return self._local.submit_mock_trade_choice(fargs)
        return {"error": f"unknown tool {fname}"}

    async def _search_instruments_via_mcp(self, args: dict[str, Any]) -> dict[str, Any]:
        from trade_claw.kite_mcp_client import (
            TOOL_SEARCH_INSTRUMENTS,
            extract_instruments_list,
            mcp_call_tool,
        )

        q = str(args.get("query", "")).strip()
        if not q:
            return {"error": "query is required", "options": []}
        fo = str(args.get("filter_on") or "underlying").strip().lower()
        if fo != "underlying":
            return {
                "error": "This F&O session only allows filter_on='underlying' with query NFO:<UNDERLYING>.",
                "options": [],
            }
        u = _underlying_from_instruments_query(q)
        if not u or u != self._local.underlying:
            logger.warning("[Kite MCP] search_instruments rejected query=%s session=%s", q, self._local.underlying)
            return {
                "error": f"query must resolve to underlying {self._local.underlying} (e.g. NFO:{self._local.underlying})",
                "options": [],
            }
        try:
            ll_from = int(args["from"]) if args.get("from") is not None else 0
        except (TypeError, ValueError):
            ll_from = 0
        ll_from = max(0, ll_from)
        try:
            ll_limit = (
                int(args["limit"])
                if args.get("limit") is not None
                else int(os.environ.get("KITE_MCP_SEARCH_LIMIT", "500"))
            )
        except (TypeError, ValueError):
            ll_limit = 500
        ll_limit = max(1, min(ll_limit, 5000))
        fetch_limit = min(5000, max(ll_limit, int(os.environ.get("KITE_MCP_SEARCH_LIMIT", "500"))))

        mcp_args: dict[str, Any] = {
            "query": q,
            "filter_on": "underlying",
            "from": 0,
            "limit": fetch_limit,
        }
        assert self._mcp is not None
        parsed = await mcp_call_tool(self._mcp, TOOL_SEARCH_INSTRUMENTS, mcp_args)
        if isinstance(parsed, dict) and parsed.get("error"):
            return {"error": str(parsed.get("error")), "options": []}

        rows_raw = extract_instruments_list(parsed)
        filtered = self._local._post_filter_search_rows(rows_raw, u)
        page = filtered[ll_from : ll_from + ll_limit]
        out = self._local._rows_to_option_summaries(page, MAX_SEARCH_RESULTS)

        if not out:
            logger.info(
                "[Kite MCP] search_instruments no rows underlying=%s leg=%s session_date=%s",
                u,
                self._local.leg_type,
                self._local.session_date.isoformat(),
            )
            return {"options": [], "note": "No contracts for filters"}

        logger.info(
            "[Kite MCP] search_instruments underlying=%s leg=%s session_date=%s -> %s row(s)",
            u,
            self._local.leg_type,
            self._local.session_date.isoformat(),
            len(out),
        )
        return {"options": out, "count": len(out)}

    async def _get_historical_data_via_mcp(self, args: dict[str, Any]) -> dict[str, Any]:
        from trade_claw.kite_mcp_client import (
            TOOL_GET_HISTORICAL,
            extract_historical_candles,
            mcp_call_tool,
        )

        verr = self._local._validate_get_historical_data_args(args)
        if verr:
            return {"error": verr, "data": []}

        token = int(args["instrument_token"])
        raw_cont = args.get("continuous", False)
        if isinstance(raw_cont, str):
            raw_cont = raw_cont.lower() in ("1", "true", "yes")
        continuous = False
        if raw_cont:
            logger.info(
                "[Kite MCP] get_historical_data forcing continuous=False for option token=%s (LLM sent true)",
                token,
            )
        oi = args.get("oi", False)
        if isinstance(oi, str):
            oi = oi.lower() in ("1", "true", "yes")

        mcp_args = {
            "instrument_token": token,
            "from_date": str(args["from_date"]).strip()[:19],
            "to_date": str(args["to_date"]).strip()[:19],
            "interval": str(args.get("interval", "")).strip(),
            "continuous": continuous,
            "oi": bool(oi),
        }
        assert self._mcp is not None
        logger.info(
            "[Kite MCP] get_historical_data token=%s interval=%s",
            token,
            mcp_args["interval"],
        )
        parsed = await mcp_call_tool(self._mcp, TOOL_GET_HISTORICAL, mcp_args)
        if isinstance(parsed, dict) and parsed.get("error"):
            return {"error": str(parsed.get("error")), "data": []}
        candles = extract_historical_candles(parsed)
        if not candles:
            return {"data": [], "note": "empty"}

        def _candle_sort_key(c: dict[str, Any]) -> str:
            d = c.get("date")
            return str(d) if d is not None else ""

        try:
            candles_sorted = sorted(candles, key=_candle_sort_key)
        except Exception:
            candles_sorted = candles
        tail = candles_sorted[-MAX_OHLC_ROWS_TO_MODEL:]
        out: list[dict[str, Any]] = []
        for c in tail:
            try:
                out.append(
                    {
                        "date": str(c.get("date")),
                        "open": round(float(c.get("open", 0)), 4),
                        "high": round(float(c.get("high", 0)), 4),
                        "low": round(float(c.get("low", 0)), 4),
                        "close": round(float(c.get("close", 0)), 4),
                    }
                )
            except (TypeError, ValueError):
                continue
        logger.info(
            "[Kite MCP] get_historical_data OK token=%s returned=%s (from %s total)",
            token,
            len(out),
            len(candles),
        )
        return {"data": out, "count": len(out)}


def _build_system_prompt(*, use_kite_mcp: bool = False) -> str:
    transport = (
        "Data tools call the **Zerodha Kite MCP server** over the wire when enabled. "
        if use_kite_mcp
        else "Data tools use the same parameters as Zerodha Kite MCP; execution is in-process KiteConnect. "
    )
    return f"""You assist with INDIAN F&O **mock backtesting** only.

Hard rules:
- Never suggest live orders, real brokerage placement, or any Kite order API.
- You only have these tools: **search_instruments**, **get_historical_data**, **submit_mock_trade_choice**.
  (place_order, GTT, and other write tools are NOT exposed.)
- submit_mock_trade_choice does NOT send orders; it only records the contract for simulation.

{transport}

Task: Given the underlying strategy signal (BUY → long CE, SELL → long PE), pick one NFO option
that is appropriate for the session date (expiry on or after session). Prefer sensible near-ATM
strikes unless the analysis implies otherwise. Use **search_instruments** (filter_on=underlying,
query NFO:UNDERLYING) then **get_historical_data** with the exact session window from the user payload.
For **option** tokens use **continuous=false** (required by Kite for intraday option candles).

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
    stop_prem: float | None = None
    if FO_OPTION_STOP_LOSS_PCT > 0:
        sp = entry_premium * (1.0 - FO_OPTION_STOP_LOSS_PCT)
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
    # Use Kite tradingsymbol (e.g. NIFTY2632424800PE), not a placeholder label.
    strike_pick = str(opt_sym).strip() or "—"
    note_ok = f"OK — {strike_pick}" if strike_pick != "—" else "OK"

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
        "Note": note_ok,
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
    logger.info(
        "[mock-exec] done symbol=%s closed_at=%s net_pl=%.2f gross=%.2f txn=%.2f",
        opt_sym,
        closed_at,
        net_pl,
        gross_pl,
        txn_cost,
    )
    return {"row": row}


async def _async_openai_agent_loop(
    *,
    ctx: _AgentToolContext,
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """
    Returns (ok, error_message). On success, ctx.final_choice is set.
    """
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
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            logger.exception("[LLM] OpenAI API request failed: %s", e)
            trace.append({"step": "openai_error", "error": str(e)})
            return False, f"OpenAI API error: {e}"

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

            result = await ctx.dispatch_tool(fname, fargs)

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
            return False, "Agent did not call submit_mock_trade_choice in time."

    if not ctx.final_choice or not ctx.final_choice.get("inst"):
        return False, "No valid mock trade choice recorded."
    return True, None


async def _async_fo_agent_phase(
    *,
    trace: list[dict[str, Any]],
    intent: dict[str, Any],
    local_ctx: _LocalToolContext,
    use_mcp: bool,
    kite_api_key: str | None,
    kite_access_token: str | None,
    openai_api_key: str,
    model: str,
    user_content: str,
    kite,
    underlying: str,
    name: str,
    session_date: date,
    chosen_interval: str,
    from_str: str,
    to_str: str,
    brokerage_per_lot_rt: float,
    taxes_per_lot_rt: float,
) -> FoAgentResult:
    from trade_claw.kite_mcp_client import kite_mcp_client_session

    client = OpenAI(api_key=openai_api_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(use_kite_mcp=use_mcp)},
        {"role": "user", "content": user_content},
    ]

    if use_mcp:
        ak = (kite_api_key or os.environ.get("KITE_API_KEY") or "").strip() or None
        at = (kite_access_token or os.environ.get("KITE_ACCESS_TOKEN") or "").strip() or None
        if not ak:
            logger.warning("[Kite MCP] KITE_API_KEY not passed; relying on child process environment")
        if not at:
            logger.warning("[Kite MCP] access token not passed; relying on child process environment")
        try:
            async with kite_mcp_client_session(
                kite_api_key=ak,
                kite_access_token=at,
            ) as session:
                ctx = _AgentToolContext(local=local_ctx, mcp_session=session)
                ok, err = await _async_openai_agent_loop(
                    ctx=ctx,
                    client=client,
                    model=model,
                    messages=messages,
                    trace=trace,
                )
        except Exception as e:
            logger.exception("[Kite MCP] session failed: %s", e)
            trace.append({"step": "kite_mcp_error", "error": str(e)})
            return FoAgentResult(
                success=False,
                error_message=f"Kite MCP connection failed: {e}",
                intent=intent,
                trace=trace,
            )
    else:
        ctx = _AgentToolContext(local=local_ctx, mcp_session=None)
        ok, err = await _async_openai_agent_loop(
            ctx=ctx,
            client=client,
            model=model,
            messages=messages,
            trace=trace,
        )

    if not ok:
        return FoAgentResult(
            success=False,
            error_message=err or "OpenAI agent failed",
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
    kite_api_key: str | None = None,
    kite_access_token: str | None = None,
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
        "get_historical_data": {
            "from_date": from_str,
            "to_date": to_str,
            "interval": chosen_interval,
            "continuous": False,
            "oi": False,
        },
        "search_instruments": {
            "query": f"NFO:{underlying}",
            "filter_on": "underlying",
        },
        "instructions": (
            hint_line
            + "Tools (Kite MCP names): search_instruments with query/filter_on as above; "
            + "get_historical_data with exact from_date, to_date, interval from get_historical_data; "
            + "then submit_mock_trade_choice once. No order tools are available."
        ),
    }
    user_content = json.dumps(user_payload, indent=2)

    from trade_claw.kite_mcp_client import kite_mcp_enabled

    use_mcp = kite_mcp_enabled()
    trace.append({"step": "kite_mcp", "enabled": use_mcp})

    local_ctx = _LocalToolContext(
        kite=kite,
        nfo_instruments=nfo_instruments,
        from_str=from_str,
        to_str=to_str,
        chosen_interval=chosen_interval,
        underlying=underlying,
        leg_type=leg_type,
        session_date=session_date,
    )

    timeout_s = float(os.environ.get("FO_AGENT_ASYNC_TIMEOUT", "600"))

    def _thread_main() -> FoAgentResult:
        return asyncio.run(
            _async_fo_agent_phase(
                trace=trace,
                intent=intent,
                local_ctx=local_ctx,
                use_mcp=use_mcp,
                kite_api_key=kite_api_key,
                kite_access_token=kite_access_token,
                openai_api_key=openai_api_key,
                model=model,
                user_content=user_content,
                kite=kite,
                underlying=underlying,
                name=name,
                session_date=session_date,
                chosen_interval=chosen_interval,
                from_str=from_str,
                to_str=to_str,
                brokerage_per_lot_rt=brokerage_per_lot_rt,
                taxes_per_lot_rt=taxes_per_lot_rt,
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_thread_main).result(timeout=timeout_s)


def get_openai_api_key() -> str | None:
    """Resolve API key from environment (dotenv loaded by app)."""
    return os.environ.get("OPENAI_API_KEY") or None
