"""One read of how every part of the app is doing, for the terminal's alert strip."""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core import sidecar
from app.core.broker import Trading
from app.features.compliance import service as compliance

Level = Literal["ok", "warn", "error"]
ORDER: dict[str, int] = {"ok": 0, "warn": 1, "error": 2}
STALE_AFTER = timedelta(hours=36)
SHOWN = 120

router = APIRouter(tags=["health"])


class Check(BaseModel):
    name: str
    level: Level
    detail: str


class Status(BaseModel):
    level: Level
    checks: list[Check]


def autopilot(automation: Any) -> Check:
    state = automation.status()
    last = state.runs[0] if state.runs else None
    if state.halted:
        return Check(name="Autopilot", level="error", detail=f"Halted: {state.halted}")
    if state.config.mode == "off":
        return Check(name="Autopilot", level="ok", detail="Off")
    if last is not None and last.status == "error":
        return Check(
            name="Autopilot", level="error", detail=f"Last run failed: {last.detail}"[:SHOWN]
        )
    return Check(
        name="Autopilot",
        level="ok",
        detail=f"{state.config.mode}, next run {state.next_run:%b %-d %H:%M} UTC"
        if state.next_run
        else state.config.mode,
    )


def audit_log(log: Any) -> Check:
    found = log.verify()
    if found.valid:
        return Check(name="Audit log", level="ok", detail=f"{found.entries} entries, chain intact")
    return Check(name="Audit log", level="error", detail=f"Chain broken at #{found.broken_at}")


def ledger(desk: Any) -> Check:
    state = desk.status()
    if state.error:
        return Check(name="Ledger", level="warn", detail=f"Sync failed: {state.error}"[:SHOWN])
    check = state.reconciliation
    if check is not None and not check.ok:
        return Check(
            name="Ledger", level="warn", detail=f"{len(check.differences)} differences from Alpaca"
        )
    return Check(name="Ledger", level="ok", detail=f"{state.activities} activities, matches Alpaca")


def book(desk: Any) -> Check:
    state = desk.status()
    if state.books and state.stale:
        return Check(name="Order book", level="warn", detail="The crypto stream has gone quiet")
    return Check(name="Order book", level="ok", detail=f"{len(state.books)} books")


def scanner(desk: Any, now: datetime) -> Check:
    state = desk.status()
    last = state.last
    if last is None:
        return Check(name="Scanner", level="warn", detail="No scan yet")
    if last.status == "error":
        return Check(name="Scanner", level="warn", detail=f"Refresh failed: {last.detail}"[:SHOWN])
    if now - last.started > STALE_AFTER:
        return Check(
            name="Scanner",
            level="warn",
            detail=f"Data is {(now - last.started) // timedelta(hours=1)} hours old",
        )
    problems = [f"{item.name}: {item.error}" for item in last.lists if item.error]
    problems += [
        message
        for message in (state.news.error, state.news.filings_error, state.news.sentiment.error)
        if message
    ]
    if problems:
        return Check(name="Scanner", level="warn", detail="; ".join(problems)[:SHOWN])
    return Check(name="Scanner", level="ok", detail=f"Session {last.session}")


def drift(monitor: Any) -> Check:
    cut = [stream.name for stream in monitor.status().streams if stream.until]
    if cut:
        return Check(name="Drift", level="warn", detail=f"Sized down: {', '.join(cut)}")
    return Check(name="Drift", level="ok", detail="No stream is sized down")


def agents(desk: Any) -> Check:
    state = desk.status()
    if not state.enabled:
        return Check(name="Agents", level="ok", detail="Off")
    if not sidecar.available():
        return Check(name="Agents", level="warn", detail="The FinRL sidecar is not set up")
    waiting = sum(agent.unexplained for agent in state.agents)
    if waiting:
        return Check(
            name="Agents", level="warn", detail=f"{waiting} decisions waiting to be explained"
        )
    return Check(
        name="Agents",
        level="ok",
        detail=f"Working on {state.busy}" if state.busy else "Followed and explained",
    )


def shariah(screen: Any, trading: Any) -> Check:
    """Held names that today's Shariah lists no longer approve; a fault here is itself a warning."""
    try:
        held = compliance.holdings(trading, screen)
    except Exception as exc:
        return Check(name="Shariah", level="warn", detail=f"Could not check: {exc}"[:SHOWN])
    failing = [item.symbol for item in held if not item.screening.allowed]
    if failing:
        return Check(
            name="Shariah", level="warn", detail=f"Not on the lists now: {', '.join(failing)}"
        )
    return Check(name="Shariah", level="ok", detail=f"{len(held)} holdings pass the screen")


def report(state: Any, now: datetime | None = None, trading: Any = None) -> Status:
    moment = now or datetime.now(UTC)
    checks = [
        autopilot(state.automation),
        audit_log(state.audit),
        ledger(state.ledger),
        book(state.book),
        scanner(state.scanner, moment),
        drift(state.drift),
        agents(state.agents),
    ]
    if trading is not None:
        checks.append(shariah(state.scanner, trading))
    worst = max((check.level for check in checks), key=lambda level: ORDER[level])
    return Status(level=worst, checks=checks)


@router.get("/status")
def read_status(request: Request, trading: Trading) -> Status:
    """How the autopilot, audit log, ledger, book, scanner, drift, agents and holdings are doing."""
    return report(request.app.state, trading=trading)


class Job(BaseModel):
    owner: str
    id: str
    next_run: datetime | None


OWNERS = (
    "automation",
    "scanner",
    "agents",
    "drift",
    "ledger",
    "book",
    "icebergs",
    "algos",
    "alerts",
    "risk_history",
    "lifecycle",
    "breakers",
    "features",
    "data_quality",
)


def jobs(state: Any) -> list[Job]:
    """Every scheduled background job and when it runs next."""
    found = []
    for owner in OWNERS:
        scheduler = getattr(getattr(state, owner, None), "scheduler", None)
        if scheduler is None or not scheduler.running:
            continue
        for job in scheduler.get_jobs():
            found.append(Job(owner=owner, id=job.id, next_run=job.next_run_time))
    return sorted(
        found,
        key=lambda job: (job.next_run is None, job.next_run or datetime.max.replace(tzinfo=UTC)),
    )


@router.get("/system/jobs")
def read_jobs(request: Request) -> list[Job]:
    """The background jobs of every part of the app, soonest first."""
    return jobs(request.app.state)
