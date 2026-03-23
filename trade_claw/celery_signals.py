"""Celery lifecycle hooks — log worker and task signals."""

from __future__ import annotations

import logging
import signal as std_signal
from typing import Any

from celery import signals

log = logging.getLogger("trade_claw.celery.signal")


def _task_name(sender: Any) -> str:
    if sender is None:
        return "?"
    return getattr(sender, "name", type(sender).__name__)


@signals.before_task_publish.connect
def _before_task_publish(
    sender: str | None = None,
    headers: dict | None = None,
    body: tuple | None = None,
    **kwargs: Any,
) -> None:
    log.debug(
        "before_task_publish sender=%s headers=%s body_summary=%s extra_keys=%s",
        sender,
        headers,
        (body[0] if body else None, "…") if body else None,
        list(kwargs),
    )


@signals.after_task_publish.connect
def _after_task_publish(sender: str | None = None, headers: dict | None = None, **kwargs: Any) -> None:
    log.debug("after_task_publish sender=%s headers=%s extra_keys=%s", sender, headers, list(kwargs))


@signals.task_prerun.connect
def _task_prerun(
    sender: Any = None,
    task_id: str | None = None,
    task: Any = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    **extra: Any,
) -> None:
    log.info(
        "task_prerun name=%s task_id=%s args=%s kwargs=%s",
        _task_name(sender or task),
        task_id,
        args,
        kwargs,
    )
    log.debug("task_prerun extra=%s", extra)


@signals.task_postrun.connect
def _task_postrun(
    sender: Any = None,
    task_id: str | None = None,
    task: Any = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    retval: Any = None,
    state: str | None = None,
    **extra: Any,
) -> None:
    log.info(
        "task_postrun name=%s task_id=%s state=%s retval=%s",
        _task_name(sender or task),
        task_id,
        state,
        retval,
    )
    log.debug("task_postrun args=%s kwargs=%s extra=%s", args, kwargs, extra)


@signals.task_success.connect
def _task_success(sender: Any = None, result: Any = None, **kwargs: Any) -> None:
    log.info("task_success name=%s result=%s kwargs_keys=%s", _task_name(sender), result, list(kwargs))


@signals.task_failure.connect
def _task_failure(
    sender: Any = None,
    task_id: str | None = None,
    exception: BaseException | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    traceback: Any = None,
    einfo: Any = None,
    **extra: Any,
) -> None:
    log.error(
        "task_failure name=%s task_id=%s exception=%s args=%s kwargs=%s",
        _task_name(sender),
        task_id,
        exception,
        args,
        kwargs,
        exc_info=exception is not None,
    )
    log.debug("task_failure einfo=%s extra=%s", einfo, extra)


@signals.task_retry.connect
def _task_retry(
    sender: Any = None,
    task_id: str | None = None,
    reason: str | None = None,
    einfo: Any = None,
    **kwargs: Any,
) -> None:
    log.warning(
        "task_retry name=%s task_id=%s reason=%s",
        _task_name(sender),
        task_id,
        reason,
    )


@signals.task_revoked.connect
def _task_revoked(
    request: Any = None,
    terminated: bool | None = None,
    signum: int | None = None,
    expired: bool | None = None,
    **kwargs: Any,
) -> None:
    tid = getattr(request, "id", None) if request is not None else None
    log.warning(
        "task_revoked task_id=%s terminated=%s signum=%s (%s) expired=%s extra_keys=%s",
        tid,
        terminated,
        signum,
        std_signal.Signals(signum).name if signum is not None else None,
        expired,
        list(kwargs),
    )


@signals.worker_init.connect
def _worker_init(sender: Any = None, **kwargs: Any) -> None:
    log.info("worker_init sender=%s kwargs_keys=%s", sender, list(kwargs))


@signals.worker_process_init.connect
def _worker_process_init(**kwargs: Any) -> None:
    log.info("worker_process_init kwargs_keys=%s", list(kwargs))


@signals.worker_ready.connect
def _worker_ready(sender: Any = None, **kwargs: Any) -> None:
    log.info("worker_ready sender=%s kwargs_keys=%s", sender, list(kwargs))


@signals.worker_shutdown.connect
def _worker_shutdown(sender: Any = None, **kwargs: Any) -> None:
    log.info("worker_shutdown sender=%s kwargs_keys=%s", sender, list(kwargs))
