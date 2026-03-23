"""FastAPI server: task control APIs and SSE over Redis pub/sub."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from trade_claw.celery_app import app as celery_app
from trade_claw.event_pubsub import publish_task_event, redis_sync_client, subscribe_task_events
from trade_claw.task_runtime import pause_key, stop_key
from trade_claw import task_store
from trade_claw.worker_tasks import llm_joke_agent, stream_demo


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(task_store.init_db)
    yield


app = FastAPI(title="Trade Claw Tasks", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_TASK_NAMES = frozenset({"stream_demo", "llm_joke_agent"})


class StartTaskBody(BaseModel):
    name: str = Field(default="stream_demo", description="Logical task name stored in SQLite")
    label: str = Field(default="demo", description="Passed to the worker for event payloads")
    max_ticks: int = Field(default=3600, ge=1, le=1_000_000)
    interval_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=3600.0,
        description="Delay between jokes for llm_joke_agent (ignored for stream_demo)",
    )


class TaskEventOut(BaseModel):
    id: int
    task_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str

    @classmethod
    def from_record(cls, r: task_store.TaskEventRecord) -> TaskEventOut:
        return cls(
            id=r.id,
            task_id=r.task_id,
            event_type=r.event_type,
            payload=r.payload,
            created_at=r.created_at,
        )


class TaskEventsHistoryOut(BaseModel):
    task_id: str
    total: int
    limit: int
    offset: int
    order: str
    events: list[TaskEventOut]


class TaskOut(BaseModel):
    task_id: str
    name: str
    status: str
    payload: dict[str, Any] | None
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, r: task_store.TaskRecord) -> TaskOut:
        return cls(
            task_id=r.task_id,
            name=r.name,
            status=r.status,
            payload=r.payload,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )


def _worker_stop_and_revoke(task_id: str) -> None:
    r = redis_sync_client()
    r.set(stop_key(task_id), "1", ex=86400)
    celery_app.control.revoke(task_id, terminate=False)


def _clear_task_redis_keys(task_id: str) -> None:
    redis_sync_client().delete(pause_key(task_id), stop_key(task_id))


class StopAllOut(BaseModel):
    stopped_task_ids: list[str]
    skipped_terminal: int


class PurgeAllOut(BaseModel):
    stopped_task_ids: list[str]
    events_deleted: int
    tasks_deleted: int


@app.post("/api/tasks/start", response_model=TaskOut)
async def start_task(body: StartTaskBody) -> TaskOut:
    if body.name not in ALLOWED_TASK_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task name {body.name!r}. Allowed: {sorted(ALLOWED_TASK_NAMES)}",
        )

    if body.name == "stream_demo":
        payload: dict[str, Any] = {
            "label": body.label,
            "max_ticks": body.max_ticks,
        }
    else:
        payload = {
            "label": body.label,
            "max_ticks": body.max_ticks,
            "interval_seconds": body.interval_seconds,
        }

    task_id = await asyncio.to_thread(
        task_store.create_task,
        name=body.name,
        status="pending",
        payload=payload,
    )

    def _enqueue() -> None:
        if body.name == "stream_demo":
            stream_demo.apply_async(
                task_id=task_id,
                kwargs={"max_ticks": body.max_ticks, "label": body.label},
            )
        else:
            llm_joke_agent.apply_async(
                task_id=task_id,
                kwargs={
                    "max_ticks": body.max_ticks,
                    "label": body.label,
                    "interval_seconds": body.interval_seconds,
                },
            )

    await asyncio.to_thread(_enqueue)
    await asyncio.to_thread(task_store.update_status, task_id, "running")
    publish_task_event(task_id, {"type": "queued", "task_id": task_id, **payload})
    rec = await asyncio.to_thread(task_store.get_task, task_id)
    if rec is None:
        raise HTTPException(status_code=500, detail="Task row missing after start")
    return TaskOut.from_record(rec)


@app.post("/api/tasks/{task_id}/stop", response_model=TaskOut)
async def stop_task(task_id: str) -> TaskOut:
    rec = await asyncio.to_thread(task_store.get_task, task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    if rec.status in ("stopped", "completed", "failed"):
        return TaskOut.from_record(rec)

    def _stop() -> None:
        _worker_stop_and_revoke(task_id)

    await asyncio.to_thread(_stop)
    publish_task_event(task_id, {"type": "stop_requested", "task_id": task_id})
    await asyncio.to_thread(task_store.update_status, task_id, "stopping")
    rec2 = await asyncio.to_thread(task_store.get_task, task_id)
    assert rec2 is not None
    return TaskOut.from_record(rec2)


@app.post("/api/tasks/{task_id}/pause", response_model=TaskOut)
async def pause_task(task_id: str) -> TaskOut:
    rec = await asyncio.to_thread(task_store.get_task, task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    if rec.status in ("completed", "failed", "stopped"):
        raise HTTPException(status_code=409, detail=f"Cannot pause task in status {rec.status}")

    def _pause() -> None:
        redis_sync_client().set(pause_key(task_id), "1", ex=86400)

    await asyncio.to_thread(_pause)
    publish_task_event(task_id, {"type": "pause_requested", "task_id": task_id})
    await asyncio.to_thread(task_store.update_status, task_id, "pausing")
    rec2 = await asyncio.to_thread(task_store.get_task, task_id)
    assert rec2 is not None
    return TaskOut.from_record(rec2)


@app.post("/api/tasks/{task_id}/resume", response_model=TaskOut)
async def resume_task(task_id: str) -> TaskOut:
    rec = await asyncio.to_thread(task_store.get_task, task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    if rec.status in ("completed", "failed", "stopped"):
        raise HTTPException(status_code=409, detail=f"Cannot resume task in status {rec.status}")

    def _resume() -> None:
        redis_sync_client().delete(pause_key(task_id))

    await asyncio.to_thread(_resume)
    publish_task_event(task_id, {"type": "resume_requested", "task_id": task_id})
    await asyncio.to_thread(task_store.update_status, task_id, "running")
    rec2 = await asyncio.to_thread(task_store.get_task, task_id)
    assert rec2 is not None
    return TaskOut.from_record(rec2)


@app.get("/api/tasks/{task_id}", response_model=TaskOut)
async def get_task(task_id: str) -> TaskOut:
    rec = await asyncio.to_thread(task_store.get_task, task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return TaskOut.from_record(rec)


@app.get("/api/tasks", response_model=list[TaskOut])
async def list_tasks(limit: int = 100) -> list[TaskOut]:
    rows = await asyncio.to_thread(task_store.list_tasks, limit)
    return [TaskOut.from_record(r) for r in rows]


@app.post("/api/tasks/stop-all", response_model=StopAllOut)
async def stop_all_tasks() -> StopAllOut:
    """Signal stop + revoke for every task not already terminal (SQLite)."""

    def _run() -> StopAllOut:
        rows = task_store.list_all_tasks()
        stopped: list[str] = []
        skipped = 0
        for rec in rows:
            if rec.status in task_store.TASK_TERMINAL_STATUSES:
                skipped += 1
                continue
            _worker_stop_and_revoke(rec.task_id)
            publish_task_event(
                rec.task_id,
                {"type": "stop_requested", "task_id": rec.task_id, "bulk": True},
            )
            task_store.update_status(rec.task_id, "stopping")
            stopped.append(rec.task_id)
        return StopAllOut(stopped_task_ids=stopped, skipped_terminal=skipped)

    return await asyncio.to_thread(_run)


@app.post("/api/tasks/purge-all", response_model=PurgeAllOut)
async def purge_all_tasks() -> PurgeAllOut:
    """Stop + revoke every known task id, clear Redis pause/stop keys, then wipe task_events and tasks."""

    def _run() -> PurgeAllOut:
        ids = task_store.all_task_ids()
        for tid in ids:
            _worker_stop_and_revoke(tid)
        for tid in ids:
            _clear_task_redis_keys(tid)
        ev, tk = task_store.purge_all()
        return PurgeAllOut(
            stopped_task_ids=list(ids),
            events_deleted=ev,
            tasks_deleted=tk,
        )

    return await asyncio.to_thread(_run)


@app.get("/api/tasks/{task_id}/events/history", response_model=TaskEventsHistoryOut)
async def task_events_history(
    task_id: str,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    order: Literal["asc", "desc"] = Query(
        default="asc",
        description="asc = oldest first (timeline); desc = newest first",
    ),
    event_type: str | None = Query(
        default=None,
        description="Optional filter on payload `type` (e.g. tick, stopped)",
    ),
) -> TaskEventsHistoryOut:
    rec = await asyncio.to_thread(task_store.get_task, task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    def _load() -> tuple[list[task_store.TaskEventRecord], int]:
        rows = task_store.list_task_events(
            task_id,
            limit=limit,
            offset=offset,
            order=order,
            event_type=event_type,
        )
        total = task_store.count_task_events(task_id, event_type=event_type)
        return rows, total

    rows, total = await asyncio.to_thread(_load)
    return TaskEventsHistoryOut(
        task_id=task_id,
        total=total,
        limit=limit,
        offset=offset,
        order=order,
        events=[TaskEventOut.from_record(r) for r in rows],
    )


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str) -> StreamingResponse:
    rec = await asyncio.to_thread(task_store.get_task, task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    async def gen():
        async for chunk in subscribe_task_events(task_id):
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class CeleryPurgeQueuesOut(BaseModel):
    """Result of discarding messages from the broker default task queue."""

    tasks_discarded: int
    detail: str = Field(
        default="Messages removed from the broker queue only (not SQLite). "
        "Producers: Celery Beat (periodic), API/worker `apply_async`, or any other "
        "process using the same Redis broker URL."
    )


@app.post("/api/celery/purge-queues", response_model=CeleryPurgeQueuesOut)
async def celery_purge_queues() -> CeleryPurgeQueuesOut:
    """
    Drop **all waiting tasks** on the Celery broker (default queue). Same as
    `uv run celery -A trade_claw.celery_app purge`. Stop workers first if you
    want a clean slate, then restart after this call.
    """

    def _purge() -> int:
        return int(celery_app.control.purge())

    n = await asyncio.to_thread(_purge)
    return CeleryPurgeQueuesOut(tasks_discarded=n)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
