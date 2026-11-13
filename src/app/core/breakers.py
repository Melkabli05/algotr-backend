from datetime import datetime, timedelta
from typing import Literal

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from app.core.tables import breaker_events, breakers, kill_switches

Level = Literal["trade", "strategy", "portfolio", "system"]
State = Literal["closed", "open", "half_open"]
Scope = Literal["global", "account", "strategy"]
Actor = Literal["auto", "manual", "probe"]

MANUAL_ONLY: tuple[Level, ...] = ("portfolio", "system")
COOLDOWN = timedelta(hours=24)
SYSTEM = "system"
PORTFOLIO = "portfolio"
GLOBAL = "global"
ACCOUNT = "account"


def strategy_breaker(name: str) -> str:
    return f"strategy:{name}"


def strategy_switch(name: str) -> str:
    return f"strategy:{name}"


class Breaker(BaseModel):
    name: str
    level: Level
    state: State
    reason: str | None
    opened_at: datetime | None
    retry_at: datetime | None
    cooldown_seconds: int
    trips: int
    updated: datetime


class KillSwitch(BaseModel):
    name: str
    scope: Scope
    engaged: bool
    reason: str | None
    actor: str
    updated: datetime


def read(connection: Connection) -> dict[str, Breaker]:
    rows = connection.execute(sa.select(breakers))
    return {row.name: Breaker.model_validate(row._mapping) for row in rows}


def record(
    connection: Connection,
    name: str,
    level: Level,
    before: str,
    after: State,
    actor: Actor,
    reason: str,
    now: datetime,
) -> None:
    connection.execute(
        sa.insert(breaker_events).values(
            time=now,
            name=name,
            level=level,
            from_state=before,
            to_state=after,
            actor=actor,
            reason=reason,
        )
    )


def put(connection: Connection, name: str, level: Level, now: datetime, **values: object) -> None:
    statement = insert(breakers).values(name=name, level=level, updated=now, **values)
    connection.execute(
        statement.on_conflict_do_update(
            index_elements=[breakers.c.name], set_={"updated": now, **values}
        )
    )


def trip(
    connection: Connection,
    name: str,
    level: Level,
    reason: str,
    now: datetime,
    actor: Actor = "auto",
) -> Breaker:
    """Open a breaker. From half-open, a failed probe, the cooldown doubles; manual-only levels
    never retry on their own."""
    found = read(connection).get(name)
    before = "closed" if found is None else found.state
    if found is not None and found.state == "open":
        return found
    cooldown = COOLDOWN.total_seconds() if found is None else found.cooldown_seconds
    if before == "half_open":
        cooldown *= 2
    retry = None if level in MANUAL_ONLY else now + timedelta(seconds=cooldown)
    put(
        connection,
        name,
        level,
        now,
        state="open",
        reason=reason,
        opened_at=now,
        retry_at=retry,
        cooldown_seconds=int(cooldown),
        trips=(0 if found is None else found.trips) + 1,
    )
    record(connection, name, level, before, "open", actor, reason, now)
    return read(connection)[name]


def due(connection: Connection, now: datetime) -> list[Breaker]:
    """Open breakers whose cooldown is over move to half-open, where one probe decides."""
    moved = []
    for found in read(connection).values():
        if found.state != "open" or found.retry_at is None or found.retry_at > now:
            continue
        put(connection, found.name, found.level, now, state="half_open")
        record(
            connection, found.name, found.level, "open", "half_open", "auto", "Cooldown over", now
        )
        moved.append(read(connection)[found.name])
    return moved


def probe(connection: Connection, name: str, passed: bool, reason: str, now: datetime) -> Breaker:
    """The single probe of a half-open breaker: closed when it passes, open longer when not."""
    found = read(connection)[name]
    if found.state != "half_open":
        return found
    if not passed:
        return trip(connection, name, found.level, reason, now, actor="probe")
    put(connection, name, found.level, now, state="closed", reason=None, retry_at=None)
    record(connection, name, found.level, "half_open", "closed", "probe", reason, now)
    return read(connection)[name]


def reset(connection: Connection, name: str, reason: str, now: datetime) -> Breaker:
    """A person closes a breaker; the only way back for the portfolio and system levels."""
    found = read(connection).get(name)
    if found is None:
        raise LookupError(name)
    put(
        connection,
        name,
        found.level,
        now,
        state="closed",
        reason=None,
        retry_at=None,
        cooldown_seconds=int(COOLDOWN.total_seconds()),
    )
    record(connection, name, found.level, found.state, "closed", "manual", reason, now)
    return read(connection)[name]


def blocks(found: Breaker | None) -> bool:
    """An open breaker rejects everything; half-open lets its one probe through."""
    return found is not None and found.state == "open"


def switches(connection: Connection) -> dict[str, KillSwitch]:
    rows = connection.execute(sa.select(kill_switches))
    return {row.name: KillSwitch.model_validate(row._mapping) for row in rows}


def set_switch(
    connection: Connection,
    name: str,
    scope: Scope,
    engaged: bool,
    reason: str | None,
    actor: str,
    now: datetime,
) -> KillSwitch:
    """Engage or release one kill switch; its state survives every restart."""
    values = {"scope": scope, "engaged": engaged, "reason": reason, "actor": actor, "updated": now}
    statement = insert(kill_switches).values(name=name, **values)
    connection.execute(
        statement.on_conflict_do_update(index_elements=[kill_switches.c.name], set_=values)
    )
    return switches(connection)[name]


def stopped(connection: Connection) -> str | None:
    """Why all trading is stopped: the global switch first, then the account's, then the
    system breaker; None while trading may go on."""
    found = switches(connection)
    for name in (GLOBAL, ACCOUNT):
        switch = found.get(name)
        if switch is not None and switch.engaged:
            return f"{name.title()} kill switch: {switch.reason or 'engaged'}"
    system = read(connection).get(SYSTEM)
    if blocks(system) or (system is not None and system.state == "half_open"):
        return f"System breaker: {system.reason if system else ''}"
    return None


def paused(connection: Connection) -> dict[str, str]:
    """Strategies that may not send targets now, by breaker or by their own kill switch."""
    found: dict[str, str] = {}
    for name, breaker in read(connection).items():
        if breaker.level == "strategy" and blocks(breaker):
            found[name.removeprefix("strategy:")] = breaker.reason or "breaker open"
    for name, switch in switches(connection).items():
        if switch.scope == "strategy" and switch.engaged:
            found[name.removeprefix("strategy:")] = switch.reason or "kill switch"
    return found


def portfolio_cut(connection: Connection) -> bool:
    return blocks(read(connection).get(PORTFOLIO))
