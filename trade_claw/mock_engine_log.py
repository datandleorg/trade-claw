"""ANSI-colored logs for mock market scan / trade lifecycle (stderr TTY).

Disable with ``NO_COLOR=1`` or ``MOCK_ENGINE_LOG_COLOR=0``. Safe for log files (no escape codes).
"""

from __future__ import annotations

import logging
import os
import sys

_RESET = "\033[0m"
_SCAN_LOGGER = logging.getLogger("trade_claw.mock_market_scan")


def use_ansi_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("MOCK_ENGINE_LOG_COLOR", "1").strip().lower() in ("0", "false", "no"):
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def _paint(text: str, sgr: str) -> str:
    if not use_ansi_color():
        return text
    return f"\033[{sgr}m{text}{_RESET}"


# event_key -> SGR (foreground / style)
_EVENT_SGR = {
    "tick_start": "1;94",
    "exit_target": "92",
    "exit_stop": "93",
    "exit_square": "95",
    "trade_closed": "92",
    "exits_sl": "96",
    "skip_session": "90",
    "skip_kite": "33",
    "skip_graph_idx": "90",
    "graph_done": "36",
    "graph_err": "91",
    "graph_fatal": "1;91",
    "signal": "1;96",
    "llm_pick": "35",
    "open": "1;92",
    "celery": "1;94",
    "ltp_warn": "33",
}


def scan_line(event: str, message: str) -> str:
    tag = _paint("[mock_market_scan]", "1;96")
    esc = _EVENT_SGR.get(event, "37")
    badge = _paint(f"[{event.upper()}]", esc)
    return f"{tag} {badge} {message}"


def _emit(level: int, event: str, fmt: str, *args: object) -> None:
    try:
        body = fmt % args if args else fmt
    except TypeError:
        body = fmt
    _SCAN_LOGGER.log(level, "%s", scan_line(event, body))


def scan_debug(event: str, fmt: str, *args: object) -> None:
    _emit(logging.DEBUG, event, fmt, *args)


def scan_info(event: str, fmt: str, *args: object) -> None:
    _emit(logging.INFO, event, fmt, *args)


def scan_warning(event: str, fmt: str, *args: object) -> None:
    _emit(logging.WARNING, event, fmt, *args)


def scan_error(event: str, fmt: str, *args: object) -> None:
    _emit(logging.ERROR, event, fmt, *args)


def scan_exception(event: str, fmt: str, *args: object) -> None:
    try:
        body = fmt % args if args else fmt
    except TypeError:
        body = fmt
    _SCAN_LOGGER.exception("%s", scan_line(event, body))
