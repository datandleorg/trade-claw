"""Persisted operator/agent chat for BANKNIFTY LLM mock — Streamlit writes; Celery supervisor reads each tick."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_claw import mock_trade_store
from trade_claw.mock_engine_telemetry import read_snapshot

_UNDERLYING = "BANKNIFTY"
_MAX_MESSAGES = 80
_WORKER_TRANSCRIPT_TAIL = 28
_WORKER_TRANSCRIPT_MAX_CHARS = 8000


def agent_memory_path() -> Path:
    raw = (os.environ.get("LLM_MOCK_AGENT_MEMORY_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / "llm_mock_agent_memory.json"


def agent_chat_model_name() -> str:
    return (
        (os.environ.get("LLM_MOCK_AGENT_CHAT_MODEL") or "").strip()
        or (os.environ.get("LLM_MOCK_SUPERVISOR_MODEL") or "").strip()
        or "gpt-5-mini"
    )


def load_messages() -> list[dict[str, Any]]:
    p = agent_memory_path()
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    msgs = raw.get("messages")
    if not isinstance(msgs, list):
        return []
    out: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content, "ts": m.get("ts")})
    return out[-_MAX_MESSAGES:]


def save_messages(messages: list[dict[str, Any]]) -> None:
    p = agent_memory_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    trimmed = messages[-_MAX_MESSAGES:]
    for m in trimmed:
        m.setdefault("ts", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    payload = {"updated_at": datetime.now(UTC).isoformat(), "messages": trimmed}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def clear_messages() -> None:
    save_messages([])


def message_count() -> int:
    return len(load_messages())


def agent_memory_stats() -> dict[str, int]:
    m = load_messages()
    return {"agent_memory_message_count": len(m), "agent_memory_approx_chars": sum(len(str(x.get("content", ""))) for x in m)}


def format_transcript_for_worker(
    *,
    max_chars: int = _WORKER_TRANSCRIPT_MAX_CHARS,
    tail_messages: int = _WORKER_TRANSCRIPT_TAIL,
) -> str:
    """Plain-text tail for supervisor prompts (token-safe)."""
    msgs = load_messages()[-tail_messages:]
    if not msgs:
        return ""
    lines = [f"{str(m.get('role', '')).upper()}: {m.get('content', '')}" for m in msgs]
    s = "\n".join(lines)
    if len(s) <= max_chars:
        return s
    return "...[truncated head]\n" + s[-max_chars:]


def _json_snip(obj: Any, limit: int) -> str:
    s = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    if len(s) > limit:
        return s[: limit - 20] + "\n…[truncated]"
    return s


def build_book_context_snapshot(
    *,
    kite: Any,
    nse_instruments: list | None,
    nfo_instruments: list | None,
    spot_fn: Any,
    opt_ltp_fn: Any,
) -> str:
    """Fresh read-only snapshot for companion LLM (Streamlit)."""
    mock_trade_store.init_db()
    parts: list[str] = []

    snap = read_snapshot()
    ls = snap.get("last_scan") or {}
    lg = snap.get("last_graph") or {}
    if ls:
        parts.append("## Telemetry `last_scan`\n" + _json_snip(ls, 14_000))
    if lg:
        parts.append("## Telemetry `last_graph` (truncated)\n" + _json_snip(lg, 10_000))

    open_bn = [
        t
        for t in mock_trade_store.list_open_trades()
        if (t.index_underlying or "").upper() == _UNDERLYING
    ]
    rows: list[dict[str, Any]] = []
    for t in open_bn:
        ltp = opt_ltp_fn(kite, t.instrument) if opt_ltp_fn else None
        ep = float(t.entry_price or 0.0)
        qty = int(t.quantity or 0)
        unreal = ((ltp - ep) * qty) if (ltp is not None and ep > 0 and qty > 0) else None
        rows.append(
            {
                "trade_id": t.trade_id,
                "instrument": t.instrument,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "target": t.target,
                "quantity": t.quantity,
                "option_ltp": ltp,
                "unrealised_pnl_inr_est": round(unreal, 2) if unreal is not None else None,
                "llm_rationale_excerpt": (t.llm_rationale or "")[:800],
            }
        )
    parts.append("## Open BANKNIFTY mock positions\n" + _json_snip(rows, 6_000))

    spot = spot_fn(kite) if spot_fn else None
    parts.append(f"## BANKNIFTY spot\nApprox **₹{spot:,.2f}**" if spot is not None else "## BANKNIFTY spot\nUnavailable")

    parts.append(
        "## Note\nRead-only context. "
        f"NSE={bool(nse_instruments)} NFO={bool(nfo_instruments)}."
    )
    return "\n\n".join(parts)


def invoke_agent_companion_reply(
    *,
    user_message: str,
    prior_chat_messages: list[dict[str, Any]],
    context_block: str,
) -> str:
    """One companion turn; same transcript file is read by the minute supervisor."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "your_openai_api_key_here":
        return "OPENAI_API_KEY is missing or placeholder — cannot run chat."

    sys = (
        "You are a **companion** for the BANKNIFTY LLM mock book operator.\n"
        "This chat is **persisted** and the **same transcript** (including your replies) is shown to the **autonomous "
        "supervisor** on each Celery tick — keep answers actionable and consistent with conservative mock risk.\n"
        "You do **not** place orders or change the database directly.\n"
        "Ground answers in the **CURRENT BOOK CONTEXT** below; if data is missing, say so.\n"
        "Be concise unless the user asks for depth.\n\n"
        "--- CURRENT BOOK CONTEXT ---\n"
        f"{context_block}\n"
        "--- END CONTEXT ---"
    )

    llm = ChatOpenAI(api_key=key, model=agent_chat_model_name(), temperature=0.25)
    msgs: list = [SystemMessage(content=sys)]
    for m in prior_chat_messages[-40:]:
        role = m.get("role")
        content = str(m.get("content") or "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=user_message.strip()))
    try:
        out = llm.invoke(msgs)
        text = (getattr(out, "content", None) or "").strip()
        return text or "(Empty model response.)"
    except Exception as e:  # noqa: BLE001
        return f"Chat request failed: {e!s}"[:2_000]
