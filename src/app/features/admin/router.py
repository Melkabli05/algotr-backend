from typing import Any

from fastapi import APIRouter, Request

from app.core.limits import LimitStore, RiskLimits

router = APIRouter(prefix="/settings", tags=["settings"])


def store_of(request: Request) -> LimitStore:
    found: LimitStore = request.app.state.limits
    return found


def apply(state: Any, limits: RiskLimits) -> None:
    """Hand the limits to the parts that hold their own copy: the coin order books."""
    if hasattr(state, "book"):
        state.book.max_imbalance = limits.book_imbalance


def changed(request: Request, before: RiskLimits, after: RiskLimits) -> None:
    request.app.state.audit.record(
        "settings-change",
        {"area": "limits", "before": before.model_dump(), "after": after.model_dump()},
    )
    apply(request.app.state, after)


@router.get("/limits")
def read_limits(request: Request) -> RiskLimits:
    """The limits every buy is checked against now."""
    return store_of(request).get()


@router.put("/limits")
def put_limits(request: Request, limits: RiskLimits) -> RiskLimits:
    """Save new limits; the change is written to the audit log with what it replaced."""
    store = store_of(request)
    before = store.get()
    saved = store.set(limits)
    changed(request, before, saved)
    return saved


@router.delete("/limits")
def reset_limits(request: Request) -> RiskLimits:
    """Go back to the limits in the environment."""
    store = store_of(request)
    before = store.get()
    after = store.reset()
    changed(request, before, after)
    return after
