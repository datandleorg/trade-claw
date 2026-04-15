"""Orchestration for LLM-only BANKNIFTY mock engine tick."""

from __future__ import annotations

from datetime import date
from typing import Any

from trade_claw import mock_engine_telemetry
from trade_claw import mock_trade_store
from trade_claw.kite_headless import get_kite_headless
from trade_claw.mock_engine_log import scan_exception, scan_info, scan_warning
from trade_claw.mock_engine_run import (
    _graph_result_telemetry_payload,
    _open_trades_payload,
    _sound_alerts_from_graph_runs,
    force_square_off_all,
    in_entry_window,
    mock_agent_envelope_pct,
    mock_agent_envelope_pct_for_underlying,
    now_ist,
    process_stop_target_exits,
    session_date_ist,
    should_force_square_off,
)
from trade_claw.llm_mock_agent_memory import agent_memory_stats
from trade_claw.mock_llm_banknifty_graph import invoke_llm_banknifty_graph

_UNDERLYING = "BANKNIFTY"


def run_llm_banknifty_scan() -> dict[str, Any]:
    mock_trade_store.init_db()
    mock_engine_telemetry.init_telemetry_table()
    dt = now_ist()
    session_d: date = session_date_ist(dt)
    kite: Any = None
    out: dict[str, Any] = {
        "engine_name": "llm_banknifty",
        "ist": dt.isoformat(),
        "session_date": session_d.isoformat(),
        "exits_sl_target": 0,
        "exits_square_off": 0,
        "graph": None,
        "skipped": None,
    }
    scan_info("tick_start", "LLM_BANKNIFTY tick start ist=%s", out["ist"])

    def finalize(graph_state: dict[str, Any] | None = None) -> None:
        out["agent_envelope_pct"] = mock_agent_envelope_pct()
        out["agent_envelope_pct_index"] = mock_agent_envelope_pct_for_underlying(_UNDERLYING)
        out["open_trades_detail"] = _open_trades_payload()
        mock_engine_telemetry.merge_and_save(last_scan=dict(out), graph_state=graph_state)

    try:
        kite = get_kite_headless()
    except ValueError as e:
        out["skipped"] = str(e)
        scan_warning("skip_kite", "LLM_BANKNIFTY skip no_kite_session reason=%s", e)
        finalize(None)
        return out

    if dt.weekday() >= 5:
        out["skipped"] = "weekend"
        finalize(None)
        return out

    out["exits_sl_target"] = process_stop_target_exits(kite)
    if should_force_square_off(dt):
        out["exits_square_off"] = force_square_off_all(kite)
        out["skipped"] = "after_square_off_window"
        finalize(None)
        return out

    if not in_entry_window(dt):
        out["skipped"] = "outside_entry_window"
        finalize(None)
        return out

    try:
        nse = kite.instruments("NSE")
        nfo = kite.instruments("NFO")
    except Exception as e:  # noqa: BLE001
        out["skipped"] = f"instruments: {e}"
        scan_exception("graph_err", "LLM_BANKNIFTY instruments_load_failed: %s", e)
        finalize(None)
        return out

    try:
        result = invoke_llm_banknifty_graph(
            kite=kite,
            session_d=session_d,
            nse_instruments=nse,
            nfo_instruments=nfo,
        )
    except Exception as e:  # noqa: BLE001
        scan_exception("graph_err", "LLM_BANKNIFTY graph invoke failed: %s", e)
        result = {"underlying": _UNDERLYING, "error": str(e)}

    payload = _graph_result_telemetry_payload(_UNDERLYING, dict(result))
    payload.update(agent_memory_stats())
    payload["engine_name"] = "llm_banknifty"
    payload["run_mode"] = result.get("run_mode")
    payload["decision"] = result.get("decision")
    payload["tools_called"] = result.get("tools_called") or []
    payload["vision_called"] = bool(result.get("vision_called"))
    payload["supervisor_focus"] = result.get("supervisor_focus")
    payload["closed_trade_id"] = result.get("closed_trade_id")
    payload["exit_reason"] = result.get("exit_reason")
    graph_runs = [payload]
    out["graph"] = {"runs": graph_runs, "per_underlying": {_UNDERLYING: payload}}
    out["sound_alerts"] = _sound_alerts_from_graph_runs(graph_runs, out["ist"])
    graph_state = {
        "engine_name": "llm_banknifty",
        "tick_ist": out["ist"],
        "runs": graph_runs,
        "per_underlying": {_UNDERLYING: payload},
    }
    finalize(graph_state)
    return out


def run_llm_banknifty_scan_safe() -> dict[str, Any]:
    try:
        return run_llm_banknifty_scan()
    except Exception as e:  # noqa: BLE001
        scan_exception("graph_fatal", "run_llm_banknifty_scan_safe fatal: %s", e)
        mock_trade_store.init_db()
        mock_engine_telemetry.init_telemetry_table()
        mock_engine_telemetry.merge_and_save(
            last_scan={"engine_name": "llm_banknifty", "error": str(e), "ist": now_ist().isoformat(), "skipped": "fatal"},
            graph_state=None,
        )
        return {"engine_name": "llm_banknifty", "error": str(e), "ist": now_ist().isoformat()}
