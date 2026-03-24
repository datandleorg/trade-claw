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
    now_ist,
    session_date_ist,
    should_force_square_off,
)
from trade_claw.mock_trading_graph import invoke_mock_graph
from trade_claw.task_runtime import MOCK_TRADES_DB_PATH
from trade_claw import mock_engine_telemetry
from trade_claw import mock_trade_store
from trade_claw.mock_trade_snapshot import (
    fetch_nifty_minute_bars_json,
    fetch_option_minute_bars_json,
    snapshot_bar_count,
)
from trade_claw.constants import ENVELOPE_EMA_PERIOD

logger = logging.getLogger(__name__)


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
        logger.warning("LTP failed for %s: %s", nfo_symbol, e)
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
            exit_under_snap = fetch_nifty_minute_bars_json(
                kite, nse, sd, max_bars=nb
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
        logger.info("Closed trade %s (%s) pnl=%.2f", row.trade_id, reason, pnl)
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
        }
        for t in mock_trade_store.list_open_trades()
    ]


def run_scan() -> dict[str, Any]:
    """
    Called from Celery Beat every minute (IST weekday morning window).
    Order: SL/target checks → 15:20 square-off → optional new graph entry if flat and in window.
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

    logger.info(
        "mock_market_scan start ist=%s session_date=%s weekday=%s",
        out["ist"],
        session_d.isoformat(),
        dt.weekday(),
    )

    def finalize(graph_state: dict[str, Any] | None = None) -> None:
        out["agent_envelope_pct"] = mock_agent_envelope_pct()
        out["agent_ema_period"] = ENVELOPE_EMA_PERIOD
        out["open_trades_detail"] = _open_trades_payload()
        if kite is not None:
            try:
                nl = kite.ltp(["NSE:NIFTY 50"]).get("NSE:NIFTY 50") or {}
                lp = nl.get("last_price")
                out["nifty_ltp"] = float(lp) if lp is not None else None
            except Exception:  # noqa: BLE001
                out["nifty_ltp"] = None
            ot = mock_trade_store.list_open_trades()
            if ot and ot[0].instrument:
                out["open_option_ltp"] = _ltp(kite, ot[0].instrument)
            else:
                out["open_option_ltp"] = None
        else:
            out["nifty_ltp"] = None
            out["open_option_ltp"] = None
        mock_engine_telemetry.merge_and_save(last_scan=dict(out), graph_state=graph_state)

    try:
        kite = get_kite_headless()
    except ValueError as e:
        out["skipped"] = str(e)
        logger.warning("mock_market_scan skipped=no_kite_session reason=%s", e)
        finalize(None)
        return out

    if dt.weekday() >= 5:
        out["skipped"] = "weekend"
        logger.info("mock_market_scan skipped=weekend (no entries Sat/Sun)")
        finalize(None)
        return out

    out["exits_sl_target"] = process_stop_target_exits(kite)
    if out["exits_sl_target"]:
        logger.info(
            "mock_market_scan closed %s open position(s) on target/stop",
            out["exits_sl_target"],
        )

    if should_force_square_off(dt):
        out["exits_square_off"] = force_square_off_all(kite)
        out["skipped"] = "after_square_off_window"
        logger.info(
            "mock_market_scan skipped=after_square_off_window square_off_closes=%s (IST >= 15:20)",
            out["exits_square_off"],
        )
        finalize(None)
        return out

    if not in_entry_window(dt):
        out["skipped"] = "outside_entry_window"
        logger.info(
            "mock_market_scan skipped=outside_entry_window (new entries only 09:15–15:19 IST Mon–Fri)"
        )
        finalize(None)
        return out

    if mock_trade_store.has_open_trade():
        out["skipped"] = "open_position"
        logger.info("mock_market_scan skipped=open_position (flat required for new graph entry)")
        finalize(None)
        return out

    try:
        nse = kite.instruments("NSE")
        nfo = kite.instruments("NFO")
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"instruments: {e}"
        logger.exception("mock_market_scan skipped=instruments_load_failed")
        finalize(None)
        return out

    db_file = str(Path(MOCK_TRADES_DB_PATH).resolve())
    try:
        with SqliteSaver.from_conn_string(db_file) as checkpointer:
            result = invoke_mock_graph(
                kite=kite,
                checkpointer=checkpointer,
                session_d=session_d,
                nse_instruments=nse,
                nfo_instruments=nfo,
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("mock_market_scan skipped=graph_invoke_failed")
        out["skipped"] = str(e)
        finalize(None)
        return out

    out["graph"] = {
        "error": result.get("error"),
        "trade_id": result.get("trade_id"),
        "signal_text": result.get("signal_text") or result.get("notes"),
    }
    logger.info(
        "mock_market_scan graph_invoke_done trade_id=%s error=%s",
        result.get("trade_id"),
        result.get("error"),
    )
    finalize(dict(result))
    return out


def run_scan_safe() -> dict[str, Any]:
    try:
        return run_scan()
    except Exception as e:  # noqa: BLE001
        logger.exception("mock_market_scan fatal (run_scan_safe)")
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
