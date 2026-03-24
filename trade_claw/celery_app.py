"""Celery application: Redis broker/backend."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from trade_claw.task_runtime import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

app = Celery(
    "trade_claw",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=False,
    task_track_started=True,
    beat_schedule={
        "mock-market-scan": {
            "task": "trade_claw.scan_mock_market",
            "schedule": crontab(minute="*", hour="9-15", day_of_week="1-5"),
        },
    },
)

# Side effect: registers @app.task definitions on this Celery instance
import trade_claw.worker_tasks  # noqa: E402, F401
