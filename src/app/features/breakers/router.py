from datetime import UTC, datetime
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.bridge import commands
from app.core import audit, breakers
from app.core.breakers import Breaker, KillSwitch, Scope
from app.core.database import transaction
from app.core.tables import breaker_events
from app.features.breakers.service import flatten
from app.features.orders import trail

router = APIRouter(prefix="/breakers", tags=["breakers"])
Trail = Annotated[audit.AuditLog, Depends(trail.log_of)]


class BreakerEvent(BaseModel):
    id: int
    time: datetime
    name: str
    level: str
    from_state: str
    to_state: str
    actor: str
    reason: str


class Overview(BaseModel):
    breakers: list[Breaker]
    switches: list[KillSwitch]
    events: list[BreakerEvent]
    stopped: str | None


class Reason(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class SwitchChange(BaseModel):
    engaged: bool
    reason: str = Field(min_length=3, max_length=500)


def scope_of(name: str) -> Scope:
    if name in (breakers.GLOBAL, breakers.ACCOUNT):
        return name  # type: ignore[return-value]
    if name.startswith("strategy:") and len(name) > len("strategy:"):
        return "strategy"
    raise HTTPException(
        status_code=422, detail="A kill switch is global, account or strategy:<name>"
    )


def engage(name: str, engaged: bool, reason: str, actor: str) -> KillSwitch:
    """Global and account switches halt and flatten at once; a strategy's pauses its targets."""
    scope = scope_of(name)
    now = datetime.now(UTC)
    with transaction() as connection:
        found = breakers.set_switch(connection, name, scope, engaged, reason, actor, now)
        if scope != "strategy" and engaged:
            flatten(connection, f"{name.title()} kill switch: {reason}", now, everything=True)
        if scope != "strategy" and not engaged and breakers.stopped(connection) is None:
            commands.set_halted(connection, False, None)
    return found


@router.get("")
def read_breakers(limit: int = 100) -> Overview:
    """Every breaker's state, every kill switch and the latest moves between states."""
    with transaction() as connection:
        rows = connection.execute(
            sa.select(breaker_events).order_by(breaker_events.c.id.desc()).limit(limit)
        )
        return Overview(
            breakers=sorted(breakers.read(connection).values(), key=lambda b: b.name),
            switches=sorted(breakers.switches(connection).values(), key=lambda s: s.name),
            events=[BreakerEvent.model_validate(row._mapping) for row in rows],
            stopped=breakers.stopped(connection),
        )


@router.post("/{name}/reset")
def reset_breaker(log: Trail, name: str, body: Reason) -> Breaker:
    """Close a breaker by hand: the only way back for the portfolio and system levels."""
    now = datetime.now(UTC)
    with transaction() as connection:
        try:
            found = breakers.reset(connection, name, body.reason, now)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="No such breaker") from exc
        if name == breakers.SYSTEM and breakers.stopped(connection) is None:
            commands.set_halted(connection, False, None)
    log.record("breaker", {"breaker": name, "action": "reset", "reason": body.reason})
    return found


@router.put("/switches/{name}")
def change_switch(log: Trail, name: str, body: SwitchChange) -> KillSwitch:
    """Engage or release a kill switch; its state survives a restart of anything."""
    found = engage(name, body.engaged, body.reason, "manual")
    log.record("kill_switch", {"switch": name, "engaged": body.engaged, "reason": body.reason})
    return found
