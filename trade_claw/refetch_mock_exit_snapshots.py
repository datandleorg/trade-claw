"""Refetch hold-window exit OHLC from Kite and rewrite ``mock_trades`` snapshot columns."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from trade_claw import mock_trade_store
from trade_claw.kite_headless import get_kite_headless
from trade_claw.mock_trade_snapshot import refetch_exit_snapshot_json_for_trade, snapshot_bar_count


def refetch_and_persist_exit_snapshots(trade_id: int, *, kite: Any) -> dict[str, Any]:
    """
    Re-download option + underlying minute bars for ``trade_id`` and UPDATE ``exit_bars_json`` /
    ``exit_underlying_bars_json``. If a leg returns no data (e.g. expired option), the previous JSON
    for that leg is kept.
    """
    mock_trade_store.init_db()
    row = mock_trade_store.get_trade_by_id(trade_id)
    if row is None:
        return {"ok": False, "trade_id": trade_id, "error": "trade_not_found"}
    if snapshot_bar_count() <= 0:
        return {
            "ok": False,
            "trade_id": trade_id,
            "error": "MOCK_ENGINE_SNAPSHOT_BARS is 0; enable snapshots first",
        }
    nse = kite.instruments("NSE")
    nfo = kite.instruments("NFO")
    new_o, new_u = refetch_exit_snapshot_json_for_trade(
        kite, row, nse_instruments=nse, nfo_instruments=nfo
    )
    out_o = new_o if new_o is not None else row.exit_bars_json
    out_u = new_u if new_u is not None else row.exit_underlying_bars_json
    ok = mock_trade_store.update_exit_snapshots_json(
        trade_id,
        exit_bars_json=out_o,
        exit_underlying_bars_json=out_u,
    )
    return {
        "ok": ok,
        "trade_id": trade_id,
        "refetched_option": new_o is not None,
        "refetched_underlying": new_u is not None,
    }


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Refetch exit snapshot JSON from Kite for mock trades (same hold-window logic as the worker). "
            "Requires Kite session (e.g. Streamlit login or KITE_ACCESS_TOKEN)."
        )
    )
    p.add_argument(
        "trade_ids",
        nargs="*",
        type=int,
        help="One or more trade_id values",
    )
    p.add_argument(
        "--recent-closed",
        type=int,
        metavar="N",
        help="Also refetch the N most recent CLOSED trades (by trade_id desc)",
    )
    args = p.parse_args(argv)
    ids: list[int] = []
    ids.extend(args.trade_ids)
    if args.recent_closed is not None:
        n = max(1, min(int(args.recent_closed), 2000))
        rows = mock_trade_store.list_recent_trades(limit=n + 500)
        closed = [r for r in rows if str(r.status or "").upper() == "CLOSED"][:n]
        ids.extend(r.trade_id for r in closed)
    ids = sorted(set(ids))
    if not ids:
        p.error("pass at least one trade_id or use --recent-closed N")
    try:
        kite = get_kite_headless()
    except ValueError as e:
        print(f"kite: {e}", file=sys.stderr)
        return 1
    exit_code = 0
    for tid in ids:
        r = refetch_and_persist_exit_snapshots(tid, kite=kite)
        print(r)
        if not r.get("ok"):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
