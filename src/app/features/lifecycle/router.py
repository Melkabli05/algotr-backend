from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core import audit
from app.features.lifecycle.service import (
    ApprovalError,
    Event,
    Lifecycle,
    Stage,
    StrategyState,
)
from app.features.orders import trail

router = APIRouter(prefix="/lifecycle", tags=["lifecycle"])


def get_lifecycle(request: Request) -> Lifecycle:
    found: Lifecycle = request.app.state.lifecycle
    return found


Manager = Annotated[Lifecycle, Depends(get_lifecycle)]
Trail = Annotated[audit.AuditLog, Depends(trail.log_of)]


class Approval(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class StageChange(BaseModel):
    stage: Stage
    reason: str = Field(min_length=3, max_length=500)


@router.get("")
def read_strategies(manager: Manager) -> list[StrategyState]:
    """Every strategy's stage, the version it holds it under and its next gate."""
    return manager.states()


@router.get("/events")
def read_events(manager: Manager, limit: int = 100) -> list[Event]:
    """Promotions and demotions, newest first, with the gates each one passed."""
    return manager.events(limit)


@router.post("/evaluate")
def evaluate(manager: Manager) -> list[Event]:
    """Record today's decisions and apply every gate now instead of on the hour."""
    return manager.evaluate()


@router.put("/{name}")
def change_stage(manager: Manager, log: Trail, name: str, change: StageChange) -> StrategyState:
    """Move a strategy by hand; the reason is kept with the move and in the audit log."""
    try:
        state = manager.set_stage(name, change.stage, change.reason)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="No such strategy") from exc
    log.record("lifecycle", {"strategy": name, "stage": change.stage, "reason": change.reason})
    return state


@router.post("/{name}/approve")
def approve(manager: Manager, log: Trail, name: str, approval: Approval) -> StrategyState:
    """Approve a canary's move to live once it has passed every gate."""
    try:
        state = manager.approve(name, approval.reason)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="No such strategy") from exc
    except ApprovalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    log.record("lifecycle", {"strategy": name, "stage": state.stage, "reason": approval.reason})
    return state
