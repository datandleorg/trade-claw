"""Persist F&O Options page runs to JSON for later analysis (params + full mock trade + OHLC series)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Override with FO_OPTIONS_RUNS_DIR=/path
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "fo_options_runs"


def get_fo_options_runs_dir() -> Path:
    raw = (os.environ.get("FO_OPTIONS_RUNS_DIR") or "").strip()
    p = Path(raw).expanduser() if raw else _DEFAULT_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_fo_options_snapshot_paths() -> list[Path]:
    """All `*.json` snapshots under the runs directory, newest file mtime first."""
    d = get_fo_options_runs_dir()
    if not d.is_dir():
        return []
    paths = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def load_fo_options_snapshot(path: Path | str) -> dict[str, Any]:
    """Load one snapshot JSON (raises on missing file or invalid JSON)."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def json_sanitize(o: Any) -> Any:
    """Make nested structures JSON-safe (dates, NaN, numpy scalars, etc.)."""
    if o is None:
        return None
    if isinstance(o, (str, int, bool)):
        return o
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, date) and not isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): json_sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_sanitize(x) for x in o]
    if isinstance(o, pd.DataFrame):
        return dataframe_to_jsonable(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        try:
            return json_sanitize(o.item())
        except Exception:
            return str(o)
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass
    return str(o)


def dataframe_to_jsonable(df: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    if df is None:
        return None
    if df.empty:
        return []
    try:
        s = df.to_json(orient="records", date_format="iso", default_handler=str)
        return json.loads(s)
    except Exception:
        dfc = df.copy()
        for col in dfc.columns:
            if pd.api.types.is_datetime64_any_dtype(dfc[col]):
                dfc[col] = dfc[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        recs = dfc.to_dict(orient="records")
        return json_sanitize(recs)


def pack_fo_runner_row(row: dict[str, Any]) -> dict[str, Any]:
    """One `run_fo_underlying_one_day` row: DataFrames as record lists, rest sanitized."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, pd.DataFrame):
            out[k] = dataframe_to_jsonable(v)
        else:
            out[k] = json_sanitize(v)
    return out


def fingerprint_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_fo_options_snapshot(
    *,
    params: dict[str, Any],
    rows_out: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> Path:
    """Write one JSON file per call. Caller should only invoke when params changed vs last persist."""
    fp = fingerprint_params(params)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    u = str(params.get("underlying", "UNK")).replace("/", "-")[:32]
    sd = str(params.get("session_date", "nodate")).replace("/", "-")[:16]
    fname = f"fo_{u}_{sd}_{ts}_{fp}.json"
    path = get_fo_options_runs_dir() / fname
    packed = [pack_fo_runner_row(r) for r in rows_out]
    payload = {
        "schema_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "params_fingerprint": fp,
        "params": json_sanitize(params),
        "metrics": json_sanitize(metrics),
        "runs": packed,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
