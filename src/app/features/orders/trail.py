import logging
import time
from collections.abc import Callable
from typing import Any

from alpaca.common.exceptions import APIError
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from requests import RequestException

from app.core import audit
from app.core.errors import REJECTED_STATUSES, broker_message
from app.features.orders.risk import RiskRefusedError
from app.features.orders.service import AlgoRefusedError, ChangeRefusedError

logger = logging.getLogger(__name__)

ACTIONS = {
    ("POST", "/orders"): "submit",
    ("POST", "/orders/iceberg"): "iceberg",
    ("POST", "/orders/algo"): "algo",
}


def log_of(request: Request) -> audit.AuditLog:
    shared: audit.AuditLog | None = getattr(request.app.state, "audit", None)
    return shared or audit.AuditLog(audit.AUDIT_PATH)


def write(
    log: audit.AuditLog,
    action: str,
    request: dict[str, Any],
    outcome: str,
    reason: str | None,
    result: dict[str, Any] | None = None,
) -> None:
    """One entry per attempt; a fault in the log never undoes or blocks what was sent."""
    try:
        log.record(
            "order",
            {
                "action": action,
                "request": request,
                "outcome": outcome,
                "reason": reason,
                "result": result,
            },
        )
    except Exception:
        logger.exception("Could not write the audit entry")


def audited[T](
    log: audit.AuditLog,
    action: str,
    request: dict[str, Any],
    call: Callable[[], T],
    summary: Callable[[T], dict[str, Any]],
) -> T:
    """Run `call` and record how it ended, with the round trip to Alpaca in milliseconds."""
    started = time.perf_counter()
    try:
        result = call()
    except (RiskRefusedError, ChangeRefusedError, AlgoRefusedError) as exc:
        write(log, action, request, "refused", str(exc))
        raise
    except APIError as exc:
        outcome = "refused" if exc.status_code in REJECTED_STATUSES else "failed"
        write(log, action, request, outcome, broker_message(exc))
        raise
    except RequestException as exc:
        write(log, action, request, "failed", str(exc)[:300])
        raise
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    write(log, action, request, "accepted", None, {**summary(result), "ms": elapsed})
    return result


def action_of(request: Request) -> str | None:
    if request.method == "PATCH" and request.url.path.startswith("/orders/"):
        return "change"
    return ACTIONS.get((request.method, request.url.path))


async def refused_before_routing(request: Request, exc: RequestValidationError) -> Response:
    """Record an order request that failed validation, then answer as FastAPI would."""
    action = action_of(request)
    if action is not None:
        reasons = "; ".join(
            f"{'.'.join(map(str, item['loc'][1:]))}: {item['msg']}" for item in exc.errors()
        )
        body = jsonable_encoder(exc.body) if isinstance(exc.body, dict) else {"body": str(exc.body)}
        write(log_of(request), action, body, "refused", reasons)
    response: JSONResponse = await request_validation_exception_handler(request, exc)
    return response
