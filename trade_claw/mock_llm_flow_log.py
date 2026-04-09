"""Unified per-signal flow logs under ``MOCK_LLM_PROMPT_LOG_DIR`` (``flow_<id>/`` trees)."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


def create_flow_run_dir(
    base_dir: Path,
    *,
    session_d: date,
    ist_now: datetime,
) -> tuple[str, Path]:
    """Returns ``(flow_id, flow_dir)`` where ``flow_dir`` is ``…/<date>/<HHMM>/flow_<id>/``."""
    flow_id = uuid.uuid4().hex[:12]
    day = session_d.isoformat()
    minute_key = ist_now.strftime("%H%M")
    flow_dir = base_dir.expanduser().resolve() / day / minute_key / f"flow_{flow_id}"
    flow_dir.mkdir(parents=True, exist_ok=True)
    return flow_id, flow_dir


def write_flow_deterministic(flow_dir: Path, payload: dict[str, Any]) -> None:
    try:
        (flow_dir / "deterministic.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except OSError:
        pass


def update_flow_deterministic(flow_dir: Path, updates: dict[str, Any]) -> None:
    p = flow_dir / "deterministic.json"
    base: dict[str, Any] = {}
    try:
        if p.exists():
            base = json.loads(p.read_text(encoding="utf-8"))
        base.update(updates)
        p.write_text(json.dumps(base, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass


def write_flow_candidates_json(flow_dir: Path, candidates: list[dict[str, Any]]) -> None:
    try:
        slim = []
        for c in candidates[:20]:
            slim.append(
                {
                    "tradingsymbol": c.get("tradingsymbol"),
                    "strike": c.get("strike"),
                    "instrument_type": c.get("instrument_type"),
                    "expiry": c.get("expiry"),
                    "dte_days": c.get("dte_days"),
                    "lot_size": c.get("lot_size"),
                    "ltp": c.get("ltp"),
                }
            )
        (flow_dir / "candidates.json").write_text(
            json.dumps(slim, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except OSError:
        pass


def write_flow_outcome(flow_dir: Path, *, trade_id: int | None = None, error: str | None = None) -> None:
    try:
        payload: dict[str, Any] = {"trade_id": trade_id, "error": error}
        (flow_dir / "outcome.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except OSError:
        pass


def list_flow_dirs_in_date_range(
    log_root: Path,
    start_d: date,
    end_d: date,
) -> list[Path]:
    """All ``flow_*`` directories under ``log_root/<YYYY-MM-DD>/<HHMM>/`` for dates in range (inclusive)."""
    out: list[Path] = []
    cur = start_d
    root = log_root.expanduser().resolve()
    while cur <= end_d:
        day_dir = root / cur.isoformat()
        if day_dir.is_dir():
            try:
                for minute_dir in sorted(day_dir.iterdir(), key=lambda p: p.name, reverse=True):
                    if not minute_dir.is_dir():
                        continue
                    for child in minute_dir.iterdir():
                        if child.is_dir() and child.name.startswith("flow_"):
                            out.append(child)
            except OSError:
                pass
        cur += timedelta(days=1)
    try:
        out.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    except OSError:
        pass
    return out


def read_json_if_exists(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
