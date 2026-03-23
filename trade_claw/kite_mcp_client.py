"""
Connect to Zerodha Kite MCP (official tools: search_instruments, get_historical_data, …).

Transport (pick one):
- **stdio** (default when KITE_MCP_ENABLED=1): spawn e.g. `npx mcp-remote https://mcp.kite.trade/mcp`
  or a self-built `kite-mcp-server` binary. Passes KITE_API_KEY / KITE_ACCESS_TOKEN in the child env.
- **HTTP**: set KITE_MCP_STREAMABLE_URL to a streamable MCP endpoint (optional headers JSON).
- **Tool log**: set **KITE_MCP_TOOL_OUTPUT_FILE** to append each `tools/call` result as one JSON line (JSONL).

Does not call place_order. Tool names match zerodha/kite-mcp-server (see mcp/market_tools.go).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.types import TextContent

logger = logging.getLogger("trade_claw.kite_mcp_client")

_mcp_tool_output_lock = threading.Lock()

DEFAULT_MCP_REMOTE_ARGS = ["mcp-remote", "https://mcp.kite.trade/mcp"]
TOOL_SEARCH_INSTRUMENTS = os.environ.get("KITE_MCP_TOOL_SEARCH", "search_instruments")
TOOL_GET_HISTORICAL = os.environ.get("KITE_MCP_TOOL_HISTORICAL", "get_historical_data")


def kite_mcp_enabled() -> bool:
    v = (os.environ.get("KITE_MCP_ENABLED") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("KITE_MCP_STREAMABLE_URL", "").strip():
        return True
    if os.environ.get("KITE_MCP_COMMAND", "").strip():
        return True
    return False


def _merge_stdio_env(
    *,
    kite_api_key: str | None,
    kite_access_token: str | None,
) -> dict[str, str]:
    out: dict[str, str] = dict(get_default_environment())
    for k, v in os.environ.items():
        if v is None:
            continue
        if not isinstance(v, str):
            continue
        out[k] = v
    if kite_api_key:
        out["KITE_API_KEY"] = str(kite_api_key).strip()
    if kite_access_token:
        out["KITE_ACCESS_TOKEN"] = str(kite_access_token).strip()
    return out


def _stdio_parameters(
    *,
    kite_api_key: str | None,
    kite_access_token: str | None,
) -> StdioServerParameters:
    cmd = (os.environ.get("KITE_MCP_COMMAND") or "npx").strip()
    raw_args = os.environ.get("KITE_MCP_ARGS")
    if raw_args:
        args = json.loads(raw_args)
        if not isinstance(args, list):
            raise ValueError("KITE_MCP_ARGS must be a JSON array of strings")
        args = [str(a) for a in args]
    else:
        args = list(DEFAULT_MCP_REMOTE_ARGS)
    cwd = os.environ.get("KITE_MCP_CWD") or None
    return StdioServerParameters(
        command=cmd,
        args=args,
        env=_merge_stdio_env(
            kite_api_key=kite_api_key,
            kite_access_token=kite_access_token,
        ),
        cwd=cwd,
    )


def parse_call_tool_result(result) -> Any:
    """Turn CallToolResult into JSON-like Python or error dict."""
    if result.isError:
        parts: list[str] = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text", "")))
        return {"error": " ".join(parts).strip() or "MCP tool error", "isError": True}
    if result.structuredContent is not None:
        return result.structuredContent
    texts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            texts.append(block.text)
        elif hasattr(block, "text"):
            texts.append(str(getattr(block, "text", "")))
    raw = "".join(texts).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:8000]}


def _mcp_tool_output_path() -> str | None:
    p = (os.environ.get("KITE_MCP_TOOL_OUTPUT_FILE") or "").strip()
    return p or None


def _append_mcp_tool_output_record(
    *,
    name: str,
    arguments: dict[str, Any],
    parsed: Any,
) -> None:
    """
    Append one JSON object per line to **KITE_MCP_TOOL_OUTPUT_FILE** (JSONL).
    Optional **KITE_MCP_TOOL_OUTPUT_MAX_CHARS**: max serialized length of the `output`
    field only; if exceeded, `output` is replaced with a truncation wrapper.
    """
    path = _mcp_tool_output_path()
    if not path:
        return
    out_payload: Any = parsed
    max_ch = (os.environ.get("KITE_MCP_TOOL_OUTPUT_MAX_CHARS") or "").strip()
    if max_ch:
        try:
            lim = int(max_ch)
        except ValueError:
            lim = 0
        if lim > 0:
            out_s = json.dumps(parsed, default=str, ensure_ascii=False)
            if len(out_s) > lim:
                out_payload = {
                    "_truncated": True,
                    "_output_serialized_chars": len(out_s),
                    "_limit": lim,
                    "preview": out_s[:lim] + ("…" if len(out_s) > lim else ""),
                }
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": name,
        "arguments": arguments,
        "output": out_payload,
    }
    raw = json.dumps(record, default=str, ensure_ascii=False)
    line = raw + "\n"
    try:
        with _mcp_tool_output_lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError as e:
        logger.warning("[Kite MCP] could not write tool output file %s: %s", path, e)


async def mcp_call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    logger.info("[Kite MCP] tools/call name=%s args_keys=%s", name, list(arguments.keys()))
    result = await session.call_tool(name, arguments)
    parsed = parse_call_tool_result(result)
    if isinstance(parsed, dict) and parsed.get("isError"):
        logger.warning("[Kite MCP] tool error name=%s detail=%s", name, parsed.get("error", "")[:300])
    else:
        plen = len(json.dumps(parsed, default=str)) if parsed is not None else 0
        logger.info("[Kite MCP] tool OK name=%s payload_chars≈%s", name, plen)
    _append_mcp_tool_output_record(name=name, arguments=dict(arguments), parsed=parsed)
    return parsed


def extract_instruments_list(parsed: Any) -> list[dict[str, Any]]:
    """Normalize search_instruments JSON into a list of instrument dicts."""
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if not isinstance(parsed, dict):
        return []
    if isinstance(parsed.get("data"), list):
        return [x for x in parsed["data"] if isinstance(x, dict)]
    if isinstance(parsed.get("instruments"), list):
        return [x for x in parsed["instruments"] if isinstance(x, dict)]
    if isinstance(parsed.get("items"), list):
        return [x for x in parsed["items"] if isinstance(x, dict)]
    for k in ("results", "records"):
        if isinstance(parsed.get(k), list):
            return [x for x in parsed[k] if isinstance(x, dict)]
    return []


def extract_historical_candles(parsed: Any) -> list[dict[str, Any]]:
    """Normalize get_historical_data response to list of candle dicts."""
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("data"), list):
            return [x for x in parsed["data"] if isinstance(x, dict)]
        if isinstance(parsed.get("candles"), list):
            return [x for x in parsed["candles"] if isinstance(x, dict)]
    return []


async def _init_session_and_log_tools(session: ClientSession) -> None:
    await session.initialize()
    listed = await session.list_tools()
    names = [t.name for t in listed.tools]
    logger.info("[Kite MCP] connected; server tools (%s): %s", len(names), names)


@asynccontextmanager
async def kite_mcp_client_session(
    *,
    kite_api_key: str | None,
    kite_access_token: str | None,
) -> AsyncIterator[ClientSession]:
    """
    Yields an initialized ClientSession.

    - If **KITE_MCP_STREAMABLE_URL** is set: streamable HTTP (e.g. hosted endpoint + headers).
    - Else: **stdio** subprocess (default `npx mcp-remote https://mcp.kite.trade/mcp` or self-hosted binary).
    """
    stream_url = os.environ.get("KITE_MCP_STREAMABLE_URL", "").strip()
    if stream_url:
        headers: dict[str, str] = {}
        hj = os.environ.get("KITE_MCP_HTTP_HEADERS_JSON", "").strip()
        if hj:
            headers.update(json.loads(hj))
        timeout_s = float(os.environ.get("KITE_MCP_HTTP_TIMEOUT", "120"))
        logger.info("[Kite MCP] streamable HTTP connect url=%s", stream_url)
        async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(timeout_s)) as http_client:
            async with streamable_http_client(stream_url, http_client=http_client) as streams:
                read, write, _ = streams
                async with ClientSession(read, write) as session:
                    await _init_session_and_log_tools(session)
                    yield session
        return

    params = _stdio_parameters(
        kite_api_key=kite_api_key,
        kite_access_token=kite_access_token,
    )
    logger.info(
        "[Kite MCP] stdio connect command=%s args=%s",
        params.command,
        params.args,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _init_session_and_log_tools(session)
            yield session
