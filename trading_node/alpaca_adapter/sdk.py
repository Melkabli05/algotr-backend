import asyncio
from asyncio import AbstractEventLoop
from collections.abc import Callable
from typing import Any

from alpaca.common.exceptions import APIError

DEFINITIVE_REJECTION_STATUSES = range(400, 500)
RETRYABLE_STATUS = 429


async def call[T](function: Callable[..., object], expected: type[T], *args: Any) -> T:
    """Run a blocking alpaca-py call off the event loop and check the model type it returned."""
    result = await asyncio.to_thread(function, *args)
    if not isinstance(result, expected):
        raise TypeError(f"Expected {expected.__name__} from Alpaca, got {type(result).__name__}")
    return result


async def call_list[T](function: Callable[..., object], expected: type[T], *args: Any) -> list[T]:
    """Run a blocking alpaca-py call returning a list and check every item's model type."""
    result = await asyncio.to_thread(function, *args)
    if not isinstance(result, list) or not all(isinstance(item, expected) for item in result):
        raise TypeError(f"Expected a list of {expected.__name__} from Alpaca")
    return result


def call_soon(loop: AbstractEventLoop | None, callback: Callable[..., None], *args: Any) -> None:
    """Schedule a callback on the node's loop from a stream thread."""
    if loop is not None:
        loop.call_soon_threadsafe(callback, *args)


def error_reason(exc: APIError) -> str:
    try:
        return str(exc.message)
    except (ValueError, KeyError, TypeError):
        return f"Alpaca returned {exc.status_code}"


def is_definitive_rejection(exc: APIError) -> bool:
    """A 4xx answer means Alpaca refused the command; anything else leaves the outcome unknown."""
    return exc.status_code in DEFINITIVE_REJECTION_STATUSES and exc.status_code != RETRYABLE_STATUS
