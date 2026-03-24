"""Orchestration for one Celery tick: exits, 15:20 square-off, optional new LangGraph entry."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from trade_claw.kite_headless import get_kite_headless
from trade_claw.mock_market_signal import (
    in_entry_window,
    mock_agent_envelope_pct,
    mock_agent_slippage_points,
    mock_engine_underlyings,
    nse_index_ltp_symbol,
    now_ist,
    session_date_ist,
    should_force_square_off,
)
from trade_claw.mock_trading_graph import invoke_mock_graph
from trade_claw.task_runtime import MOCK_TRADES_DB_PATH
from trade_claw import mock_engine_telemetry
from trade_claw import mock_trade_store
from trade_claw.mock_trade_snapshot import (
    fetch_index_minute_bars_json,
    fetch_option_minute_bars_json,
    snapshot_bar_count,
)
from trade_claw.constants import ENVELOPE_EMA_PERIOD
from trade_claw.mock_engine_log import scan_exception, scan_info, scan_warning

logger = logging.getLogger(__name__)


def _graph_result_telemetry_payload(underlying_key: str, result: dict[str, Any]) -> dict[str, Any]:
    """Subset of LangGraph state for `last_scan` / `last_graph` (JSON-safe via telemetry helper)."""
    return {
        "underlying": underlying_key,
        "error": result.get("error"),
        "trade_id": result.get("trade_id"),
        "signal_text": result.get("signal_text") or result.get("notes"),
        "direction": result.get("direction"),
        "leg": result.get("leg"),
        "spot": result.get("spot"),
        "signal_bar_time": result.get("signal_bar_time"),
        "llm_tradingsymbol": result.get("llm_tradingsymbol"),
        "stop_loss": result.get("stop_loss"),
        "target": result.get("target"),
        "llm_rationale": result.get("llm_rationale"),
        "candidates": result.get("candidates"),
    }


def _session_d_from_entry_sql(entry_time: str | None) -> date:
    if not entry_time:
        return session_date_ist(now_ist())
    try:
        dt = datetime.strptime(str(entry_time)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        return dt.date()
    except ValueError:
        return session_date_ist(now_ist())


def _ltp(kite, nfo_symbol: str) -> float | None:
    key = f"NFO:{nfo_symbol}"
    try:
        row = kite.ltp([key]).get(key) or {}
        v = row.get("last_price")
        return float(v) if v is not None else None
    except Exception as e:  # noqa: BLE001
        scan_warning("ltp_warn", "LTP failed for %s: %s", nfo_symbol, e)
        return None


def _close_at_ltp(
    kite,
    row: mock_trade_store.MockTradeRow,
    *,
    reason: str,
) -> bool:
    slip = mock_agent_slippage_points()
    ltp = _ltp(kite, row.instrument or "")
    if ltp is None or ltp <= 0:
        return False
    exit_px = max(0.01, ltp - slip)
    entry = float(row.entry_price or 0)
    qty = int(row.quantity or 1)
    pnl = (exit_px - entry) * qty
    exit_snap: str | None = None
    exit_under_snap: str | None = None
    nb = snapshot_bar_count()
    if nb > 0:
        sd = _session_d_from_entry_sql(row.entry_time)
        if row.instrument:
            try:
                nfo = kite.instruments("NFO")
                exit_snap = fetch_option_minute_bars_json(
                    kite, nfo, row.instrument, sd, max_bars=nb
                )
            except Exception:  # noqa: BLE001
                exit_snap = None
        try:
            nse = kite.instruments("NSE")
            ukey = (row.index_underlying or "NIFTY").strip().upper()
            exit_under_snap = fetch_index_minute_bars_json(
                kite, nse, sd, ukey, max_bars=nb
            )
        except Exception:  # noqa: BLE001
            exit_under_snap = None
    ok = mock_trade_store.close_trade(
        row.trade_id,
        exit_price=exit_px,
        realized_pnl=pnl,
        exit_bars_json=exit_snap,
        exit_underlying_bars_json=exit_under_snap,
    )
    if ok:
        ev = (
            "exit_target"
            if reason == "target"
            else "exit_stop"
            if reason == "stop"
            else "exit_square"
            if "square" in reason
            else "trade_closed"
        )
        scan_info(
            ev,
            "CLOSE trade_id=%s index=%s instrument=%s reason=%s pnl=%.2f",
            row.trade_id,
            row.index_underlying or "—",
            row.instrument or "—",
            reason,
            pnl,
        )
    return ok


def process_stop_target_exits(kite) -> int:
    """Close OPEN trades when option LTP hits LLM target (above) or stop (below)."""
    n = 0
    for row in mock_trade_store.list_open_trades():
        ltp = _ltp(kite, row.instrument or "")
        if ltp is None:
            continue
        entry = float(row.entry_price or 0)
        tgt = float(row.target or 0)
        stp = float(row.stop_loss or 0)
        if tgt > entry and ltp >= tgt:
            if _close_at_ltp(kite, row, reason="target"):
                n += 1
        elif stp > 0 and stp < entry and ltp <= stp:
            if _close_at_ltp(kite, row, reason="stop"):
                n += 1
    return n


def force_square_off_all(kite) -> int:
    """15:20 IST: close every OPEN mock position at synthetic LTP."""
    n = 0
    for row in mock_trade_store.list_open_trades():
        if _close_at_ltp(kite, row, reason="square_off_1520"):
            n += 1
    return n


def _open_trades_payload() -> list[dict[str, Any]]:
    return [
        {
            "trade_id": t.trade_id,
            "instrument": t.instrument,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "stop_loss": t.stop_loss,
            "target": t.target,
            "status": t.status,
            "quantity": t.quantity,
            "lot_size": t.lot_size,
            "llm_rationale": t.llm_rationale,
            "index_underlying": t.index_underlying,
        }
        for t in mock_trade_store.list_open_trades()
    ]


def run_scan() -> dict[str, Any]:
    """
    Called from Celery Beat every minute (IST weekday morning window).
    Order: SL/target checks → 15:20 square-off → optional LangGraph entry per index with no OPEN row.
    """
    mock_trade_store.init_db()
    mock_engine_telemetry.init_telemetry_table()
    dt = now_ist()
    session_d: date = session_date_ist(dt)
    kite: Any = None
    out: dict[str, Any] = {
        "ist": dt.isoformat(),
        "session_date": session_d.isoformat(),
        "exits_sl_target": 0,
        "exits_square_off": 0,
        "graph": None,
        "skipped": None,
    }

    scan_info(
        "tick_start",
        "TICK start ist=%s session_date=%s weekday=%s",
        out["ist"],
        session_d.isoformat(),
        dt.weekday(),
    )

    def finalize(graph_state: dict[str, Any] | None = None) -> None:
        out["agent_envelope_pct"] = mock_agent_envelope_pct()
        out["agent_ema_period"] = ENVELOPE_EMA_PERIOD
        out["open_trades_detail"] = _open_trades_payload()
        if kite is not None:
            ot = mock_trade_store.list_open_trades()
            try:
                ukey = (
                    (ot[0].index_underlying or "NIFTY").strip().upper()
                    if ot
                    else mock_engine_underlyings()[0]
                )
                sym = nse_index_ltp_symbol(ukey)
                if sym:
                    nl = kite.ltp([sym]).get(sym) or {}
                    lp = nl.get("last_price")
                    out["nifty_ltp"] = float(lp) if lp is not None else None
                else:
                    out["nifty_ltp"] = None
            except Exception:  # noqa: BLE001
                out["nifty_ltp"] = None
            if ot and ot[0].instrument:
                out["open_option_ltp"] = _ltp(kite, ot[0].instrument)
            else:
                out["open_option_ltp"] = None
            out["open_options_ltps"] = [
                {
                    "trade_id": t.trade_id,
                    "index_underlying": t.index_underlying,
                    "instrument": t.instrument,
                    "ltp": _ltp(kite, t.instrument or "") if t.instrument else None,
                }
                for t in ot
            ]
        else:
            out["nifty_ltp"] = None
            out["open_option_ltp"] = None
            out["open_options_ltps"] = []
        mock_engine_telemetry.merge_and_save(last_scan=dict(out), graph_state=graph_state)

    try:
        kite = get_kite_headless()
    except ValueError as e:
        out["skipped"] = str(e)
        scan_warning("skip_kite", "SKIP no_kite_session reason=%s", e)
        finalize(None)
        return out

    if dt.weekday() >= 5:
        out["skipped"] = "weekend"
        scan_info("skip_session", "SKIP weekend (no entries Sat/Sun)")
        finalize(None)
        return out

    out["exits_sl_target"] = process_stop_target_exits(kite)
    if out["exits_sl_target"]:
        scan_info(
            "exits_sl",
            "EXITS_SL_TARGET count=%s (target/stop hits this tick)",
            out["exits_sl_target"],
        )

    if should_force_square_off(dt):
        out["exits_square_off"] = force_square_off_all(kite)
        out["skipped"] = "after_square_off_window"
        scan_info(
            "exit_square",
            "SQUARE_OFF_1520 closes=%s (IST >= 15:20), skip new entries this tick",
            out["exits_square_off"],
        )
        finalize(None)
        return out

    if not in_entry_window(dt):
        out["skipped"] = "outside_entry_window"
        scan_info(
            "skip_session",
            "SKIP outside_entry_window (new entries only 09:15–15:19 IST Mon–Fri)",
        )
        finalize(None)
        return out

    try:
        nse = kite.instruments("NSE")
        nfo = kite.instruments("NFO")
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"instruments: {e}"
        scan_exception("graph_err", "instruments_load_failed: %s", e)
        finalize(None)
        return out

    db_file = str(Path(MOCK_TRADES_DB_PATH).resolve())
    graph_runs: list[dict[str, Any]] = []
    try:
        with SqliteSaver.from_conn_string(db_file) as checkpointer:
            for u in mock_engine_underlyings():
                if mock_trade_store.has_open_trade_for_underlying(u):
                    graph_runs.append({"underlying": u, "skipped": "open_position"})
                    scan_info("skip_graph_idx", "GRAPH skip underlying=%s (already OPEN)", u)
                    continue
                try:
                    result = invoke_mock_graph(
                        kite=kite,
                        checkpointer=checkpointer,
                        session_d=session_d,
                        nse_instruments=nse,
                        nfo_instruments=nfo,
                        initial_state={"signal_underlying": u},
                    )
                except Exception as e:  # noqa: BLE001
                    scan_exception("graph_err", "graph_invoke_failed underlying=%s", u)
                    graph_runs.append({"underlying": u, "error": str(e)})
                    continue
                payload = _graph_result_telemetry_payload(u, dict(result))
                graph_runs.append(payload)
                scan_info(
                    "graph_done",
                    "GRAPH done underlying=%s trade_id=%s error=%s",
                    u,
                    result.get("trade_id"),
                    result.get("error"),
                )
    except Exception as e:  # noqa: BLE001
        scan_exception("graph_fatal", "graph_checkpointer_failed: %s", e)
        out["skipped"] = str(e)
        finalize(None)
        return out

    per_underlying = {str(r["underlying"]): r for r in graph_runs if r.get("underlying")}
    out["graph"] = {"runs": graph_runs, "per_underlying": per_underlying}
    graph_telemetry: dict[str, Any] = {
        "tick_ist": out["ist"],
        "runs": graph_runs,
        "per_underlying": per_underlying,
    }
    finalize(graph_telemetry)
    return out


def run_scan_safe() -> dict[str, Any]:
    try:
        return run_scan()
    except Exception as e:  # noqa: BLE001
        scan_exception("graph_fatal", "run_scan_safe fatal: %s", e)
        mock_trade_store.init_db()
        mock_engine_telemetry.init_telemetry_table()
        mock_engine_telemetry.merge_and_save(
            last_scan={
                "error": str(e),
                "ist": now_ist().isoformat(),
                "skipped": "fatal",
            },
            graph_state=None,
        )
        return {"error": str(e), "ist": now_ist().isoformat()}
