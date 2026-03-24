"""Shared runtime settings for Celery workers, FastAPI, and SQLite (from env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)

_data_dir = Path(__file__).resolve().parent.parent / "data"
_default_db = _data_dir / "tasks.db"
TASK_DB_PATH = os.environ.get("TASK_DB_PATH", str(_default_db))

_default_mock_db = _data_dir / "mock_engine.db"
MOCK_TRADES_DB_PATH = os.environ.get("MOCK_TRADES_DB_PATH", str(_default_mock_db))

EVENT_CHANNEL_PREFIX = "trade_claw"
PAUSE_KEY_PREFIX = f"{EVENT_CHANNEL_PREFIX}:task"


def pause_key(task_id: str) -> str:
    return f"{PAUSE_KEY_PREFIX}:{task_id}:pause"


def stop_key(task_id: str) -> str:
    return f"{PAUSE_KEY_PREFIX}:{task_id}:stop"


def event_channel(task_id: str) -> str:
    return f"{EVENT_CHANNEL_PREFIX}:task:{task_id}:events"
