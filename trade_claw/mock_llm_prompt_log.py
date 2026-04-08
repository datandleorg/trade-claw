"""On-disk trace of mock-engine LLM prompts after a signal (per IST minute folder layout)."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any


def write_signal_llm_turn_log(
    base_dir: Path,
    *,
    ist_now: datetime,
    session_d: date,
    underlying: str,
    system_text: str,
    human_text: str,
    chart_png: bytes | None,
    meta: dict[str, Any],
) -> Path | None:
    """
    Create a directory under ``base_dir / session_date / HHMM /`` and write system text, human text,
    optional ``chart.png``, and ``meta.json``. Returns the run directory, or ``None`` on failure.
    """
    try:
        day = session_d.isoformat()
        minute_key = ist_now.strftime("%H%M")
        sub = f"{underlying}_{ist_now.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = base_dir.expanduser().resolve() / day / minute_key / sub
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "system.txt").write_text(system_text, encoding="utf-8")
        (run_dir / "human.txt").write_text(human_text, encoding="utf-8")
        if chart_png:
            (run_dir / "chart.png").write_bytes(chart_png)
        summary = {
            **meta,
            "human_text_path": "human.txt",
            "system_text_path": "system.txt",
            "chart_png_path": "chart.png" if chart_png else None,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        human_parts: list[dict[str, Any]] = [{"type": "text", "text": human_text}]
        if chart_png:
            human_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,<omitted — see chart.png in this folder>",
                    },
                }
            )
        (run_dir / "human_message_structure.json").write_text(
            json.dumps(human_parts, indent=2),
            encoding="utf-8",
        )
        return run_dir
    except OSError:
        return None


def write_llm_structured_output_log(
    run_dir: Path,
    pick: Any,
    *,
    invoke_path: str,
    symbol_in_candidate_list: bool,
) -> None:
    """Append ``llm_output.json`` (structured pick + trace fields) to an existing run directory."""
    try:
        dump = pick.model_dump() if hasattr(pick, "model_dump") else dict(pick)
        payload = {
            **dump,
            "_llm_invoke_path": invoke_path,
            "_symbol_in_candidate_list": symbol_in_candidate_list,
        }
        (run_dir / "llm_output.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except OSError:
        pass


def write_llm_invoke_error_log(run_dir: Path, message: str, *, exc_type: str | None = None) -> None:
    """Write ``llm_error.json`` when the LLM call fails before a structured pick is returned."""
    try:
        payload: dict[str, Any] = {"error": message}
        if exc_type:
            payload["exception_type"] = exc_type
        (run_dir / "llm_error.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass
