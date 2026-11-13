import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from pydantic import BaseModel

TASK_TIMEOUT_SECONDS = 300


class Outcome[T](BaseModel):
    """What one job produced: its result, or why it failed. One failure never sinks the batch."""

    index: int
    result: T | None = None
    error: str | None = None


def default_workers(jobs: int) -> int:
    return max(1, min(jobs, os.process_cpu_count() or 1))


def run_parallel[T](
    function: Callable[..., T],
    jobs: Sequence[tuple[Any, ...]],
    workers: int | None = None,
    timeout: float = TASK_TIMEOUT_SECONDS,
) -> list[Outcome[T]]:
    """Run `function(*job)` for every job in separate processes, results in job order.

    `function` and every argument and result must be picklable, so it has to be defined at the top
    level of a module. Processes are started with `spawn`, which is safe with threads and with
    Nautilus' native code. A single job, or one worker, runs in this process.
    """
    count = workers or default_workers(len(jobs))
    if count <= 1 or len(jobs) <= 1:
        return [attempt(function, index, job) for index, job in enumerate(jobs)]
    outcomes: dict[int, Outcome[T]] = {}
    pool = ProcessPoolExecutor(count, mp_context=multiprocessing.get_context("spawn"))
    futures = {pool.submit(function, *job): index for index, job in enumerate(jobs)}
    try:
        for future in as_completed(futures, timeout=timeout):
            index = futures[future]
            try:
                outcomes[index] = Outcome(index=index, result=future.result())
            except BrokenProcessPool as exc:
                outcomes[index] = Outcome(index=index, error=f"A worker died: {exc}")
            except Exception as exc:
                outcomes[index] = Outcome(index=index, error=f"{type(exc).__name__}: {exc}")
    except TimeoutError:
        stop_workers(pool)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return [
        outcomes.get(index, Outcome(index=index, error=f"Did not finish in {timeout:.0f}s"))
        for index in range(len(jobs))
    ]


def stop_workers(pool: ProcessPoolExecutor) -> None:
    """End workers still busy at the deadline; Python 3.14 has a public call, 3.13 does not."""
    if hasattr(pool, "terminate_workers"):
        pool.terminate_workers()
        return
    for process in list(getattr(pool, "_processes", {}).values()):
        process.terminate()


def attempt[T](function: Callable[..., T], index: int, job: tuple[Any, ...]) -> Outcome[T]:
    try:
        return Outcome(index=index, result=function(*job))
    except Exception as exc:
        return Outcome(index=index, error=f"{type(exc).__name__}: {exc}")
