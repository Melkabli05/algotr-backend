import asyncio
import json
import time
from collections.abc import Callable

from alpaca.common.exceptions import APIError
from requests import HTTPError, Response


def api_error(message: str, status: int | None = None) -> APIError:
    http_error = None
    if status is not None:
        response = Response()
        response.status_code = status
        http_error = HTTPError(response=response)
    return APIError(json.dumps({"message": message}), http_error)  # type: ignore[no-untyped-call]


async def until(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    """Wait for background work to reach a state, however slowly a busy machine gets there."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("The condition was not met in time")
        await asyncio.sleep(0.02)
