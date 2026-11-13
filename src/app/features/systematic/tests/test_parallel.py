import os
import time

from app.features.systematic import parallel


def square(number: int) -> int:
    return number * number


def explode(number: int) -> int:
    if number == 2:
        raise ValueError("two is not allowed")
    return number


def process_id(_: int) -> int:
    return os.getpid()


def sleepy(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


def die(_: int) -> int:
    os._exit(1)


def test_results_come_back_in_job_order_from_separate_processes() -> None:
    outcomes = parallel.run_parallel(square, [(n,) for n in range(6)], workers=3)
    assert [o.result for o in outcomes] == [0, 1, 4, 9, 16, 25]
    assert all(o.error is None for o in outcomes)


def test_the_work_really_runs_in_other_processes() -> None:
    outcomes = parallel.run_parallel(process_id, [(n,) for n in range(4)], workers=2)
    assert os.getpid() not in {o.result for o in outcomes}


def test_one_failing_job_is_reported_and_the_others_still_finish() -> None:
    outcomes = parallel.run_parallel(explode, [(n,) for n in range(4)], workers=2)
    assert [o.result for o in outcomes] == [0, 1, None, 3]
    assert outcomes[2].error == "ValueError: two is not allowed"


def test_a_single_job_or_worker_runs_here_without_a_pool() -> None:
    assert parallel.run_parallel(process_id, [(1,)])[0].result == os.getpid()
    serial = parallel.run_parallel(process_id, [(1,), (2,)], workers=1)
    assert {o.result for o in serial} == {os.getpid()}
    assert parallel.run_parallel(explode, [(2,)])[0].error == "ValueError: two is not allowed"


def test_jobs_that_overrun_the_timeout_are_reported_not_waited_for() -> None:
    outcomes = parallel.run_parallel(sleepy, [(0.0,), (30.0,)], workers=2, timeout=2)
    assert outcomes[0].result == 0.0
    assert outcomes[1].error == "Did not finish in 2s"


def test_a_worker_that_dies_is_reported_as_an_error() -> None:
    outcomes = parallel.run_parallel(die, [(1,), (2,)], workers=2)
    assert all(o.result is None and o.error for o in outcomes)


def test_no_jobs_is_not_an_error() -> None:
    assert parallel.run_parallel(square, []) == []
    assert parallel.default_workers(0) == 1
