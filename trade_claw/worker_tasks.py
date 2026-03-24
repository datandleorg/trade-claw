"""Celery tasks: stream_demo loop with pause/stop via Redis."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Literal

from trade_claw import task_store
from trade_claw.celery_app import app
from trade_claw.event_pubsub import publish_task_event, redis_sync_client
from langgraph.graph.state import CompiledStateGraph

from trade_claw.joke_langgraph_agent import DEFAULT_JOKE_MODEL, build_joke_agent_graph
from trade_claw.mock_engine_run import run_scan_safe
from trade_claw.task_runtime import pause_key, stop_key

logger = logging.getLogger(__name__)

WaitOutcome = Literal["stop", "pause"] | None


def _wait_between_ticks(r, sk: str, pk: str, seconds: float = 1.0, step: float = 0.2) -> WaitOutcome:
    """Sleep in short steps so Redis stop/pause is seen during the tick delay."""
    deadline = time.monotonic() + seconds
    while True:
        remain = deadline - time.monotonic()
        if remain <= 0:
            break
        if r.get(sk):
            return "stop"
        if r.get(pk):
            return "pause"
        time.sleep(min(step, remain))
    return None


def _controlled_work_loop(
    task_id: str,
    max_ticks: int,
    delay_seconds: float,
    each_tick: Callable[[int], None],
    *,
    started_extra: dict | None = None,
) -> dict:
    """
    One Celery task owns the full run: iterations, delay, pause/stop via Redis.
    """
    r = redis_sync_client()
    pk = pause_key(task_id)
    sk = stop_key(task_id)

    def emit(event_type: str, **extra: object) -> None:
        publish_task_event(task_id, {"type": event_type, "task_id": task_id, **extra})

    task_store.update_status(task_id, "running")
    emit(
        "started",
        max_ticks=max_ticks,
        delay_seconds=delay_seconds,
        **(started_extra or {}),
    )

    try:
        for i in range(max_ticks):
            if r.get(sk):
                r.delete(sk)
                task_store.update_status(task_id, "stopped")
                emit("stopped", tick=i)
                return {"status": "stopped", "tick": i}

            if r.get(pk):
                task_store.update_status(task_id, "paused")
                emit("paused", tick=i)
                while r.get(pk):
                    if r.get(sk):
                        break
                    time.sleep(0.4)
                if r.get(sk):
                    r.delete(sk)
                    task_store.update_status(task_id, "stopped")
                    emit("stopped", tick=i)
                    return {"status": "stopped", "tick": i}
                task_store.update_status(task_id, "running")
                emit("resumed", tick=i)

            each_tick(i)

            between = _wait_between_ticks(r, sk, pk, delay_seconds)
            if between == "stop":
                r.delete(sk)
                task_store.update_status(task_id, "stopped")
                emit("stopped", tick=i)
                return {"status": "stopped", "tick": i}
            if between == "pause":
                continue

        task_store.update_status(task_id, "completed")
        emit("completed", ticks=max_ticks)
        return {"status": "completed", "ticks": max_ticks}
    except Exception as e:  # noqa: BLE001
        task_store.update_status(task_id, "failed")
        emit("failed", error=str(e))
        raise


@app.task(bind=True, name="trade_claw.stream_demo")
def stream_demo(self, max_ticks: int = 3600, label: str = "demo") -> dict:
    task_id = self.request.id

    def each_tick(i: int) -> None:
        publish_task_event(
            task_id,
            {"type": "tick", "task_id": task_id, "tick": i, "label": label},
        )

    return _controlled_work_loop(
        task_id,
        max_ticks,
        1.0,
        each_tick,
        started_extra={"label": label},
    )


def _joke_openai_credentials() -> tuple[str, str]:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY is missing or placeholder; set it in .env")
    model = (os.environ.get("OPENAI_MODEL") or DEFAULT_JOKE_MODEL).strip()
    return key, model


def _generate_one_joke(graph: CompiledStateGraph) -> str:
    result = graph.invoke({})
    text = (result.get("joke") or "").strip()
    if not text:
        raise RuntimeError("Empty joke response from model")
    logger.info("Joke: %s", text)
    return text


@app.task(bind=True, name="trade_claw.llm_joke_agent")
def llm_joke_agent(
    self,
    max_ticks: int = 360,
    label: str = "jokes",
    interval_seconds: float = 10.0,
) -> dict:
    """LangGraph joke agent: one joke per tick via `interval_seconds` until max_ticks or stop/pause."""
    task_id = self.request.id
    api_key, model = _joke_openai_credentials()
    joke_graph = build_joke_agent_graph(api_key=api_key, model=model)

    def each_tick(i: int) -> None:
        joke = _generate_one_joke(joke_graph)
        publish_task_event(
            task_id,
            {
                "type": "joke",
                "task_id": task_id,
                "tick": i,
                "label": label,
                "model": model,
                "joke": joke,
            },
        )

    return _controlled_work_loop(
        task_id,
        max_ticks,
        float(interval_seconds),
        each_tick,
        started_extra={"label": label, "model": model, "interval_seconds": interval_seconds},
    )


@app.task(name="trade_claw.scan_mock_market")
def scan_mock_market() -> dict:
    """Periodic mock Nifty options scan (Celery Beat). See mock_engine_run.run_scan."""
    logger.info("Celery Beat: scan_mock_market task started")
    result = run_scan_safe()
    if result.get("error"):
        logger.error(
            "Celery Beat: scan_mock_market finished with error ist=%s err=%s",
            result.get("ist"),
            result.get("error"),
        )
    elif result.get("skipped"):
        logger.info(
            "Celery Beat: scan_mock_market finished ist=%s skipped=%s",
            result.get("ist"),
            result.get("skipped"),
        )
    else:
        g = result.get("graph") or {}
        logger.info(
            "Celery Beat: scan_mock_market finished ist=%s trade_id=%s graph_error=%s",
            result.get("ist"),
            g.get("trade_id"),
            g.get("error"),
        )
    return result

