"""Redis pub/sub: workers publish JSON events; FastAPI streams them over SSE."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import redis as redis_sync
import redis.asyncio as redis_async

from trade_claw.task_runtime import REDIS_URL, event_channel

_log = logging.getLogger("trade_claw.events")

_sync_redis: redis_sync.Redis | None = None


def redis_sync_client() -> redis_sync.Redis:
    global _sync_redis
    if _sync_redis is None:
        _sync_redis = redis_sync.from_url(REDIS_URL, decode_responses=True)
    return _sync_redis


def publish_task_event(task_id: str, data: dict[str, Any]) -> None:
    payload = json.dumps(data, default=str)
    redis_sync_client().publish(event_channel(task_id), payload)
    try:
        from trade_claw import task_store

        task_store.append_task_event(task_id, data)
    except Exception:
        _log.exception("append_task_event failed task_id=%s", task_id)


async def subscribe_task_events(task_id: str) -> AsyncIterator[str]:
    """Yield raw JSON strings suitable for SSE `data:` lines."""
    r = redis_async.from_url(REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    ch = event_channel(task_id)
    await pubsub.subscribe(ch)
    try:
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
            if msg is None:
                yield ": keepalive\n\n"
                continue
            if msg.get("type") == "message" and msg.get("data"):
                yield f"data: {msg['data']}\n\n"
            await asyncio.sleep(0)
    finally:
        await pubsub.unsubscribe(ch)
        await pubsub.close()
        await r.aclose()
