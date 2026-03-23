"""Central logging setup for API, Celery workers, and task helpers (env-driven)."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent: reads TASK_LOG_LEVEL or LOG_LEVEL (default INFO)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("TASK_LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.environ.get(
        "TASK_LOG_FORMAT",
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    datefmt = os.environ.get("TASK_LOG_DATEFMT", "%Y-%m-%dT%H:%M:%S")

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, level))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Namespaced logger under `trade_claw.*`."""
    configure_logging()
    return logging.getLogger(name if name.startswith("trade_claw.") else f"trade_claw.{name}")
