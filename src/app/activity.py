from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter(tags=["activity"])

Level = Literal["info", "warn", "error"]
Kind = Literal["order", "fill", "run", "risk", "signal", "research", "system"]
TITLES = {
    "drift": "Drift detected",
    "review": "Pre-open news review",
    "xai": "Agent decisions explained",
    "ai": "Agent trained",
    "rl": "Agent trained",
    "lab": "Rule lab run",
    "finrl": "FinRL run",
    "ai-evaluation": "Agents judged by the gates",
    "scan-evaluation": "Scanner variants judged",
    "event-check": "Event-driven check",
    "scan": "Market scan",
    "ledger-adopt": "Ledger matched to Alpaca",
    "audit-note": "Audit note",
}
KINDS: dict[str, Kind] = {
    "order": "order",
    "run": "run",
    "drift": "risk",
    "review": "risk",
    "xai": "signal",
    "ai": "signal",
    "rl": "signal",
    "lab": "research",
    "finrl": "research",
    "ai-evaluation": "research",
    "scan-evaluation": "research",
    "event-check": "research",
    "scan": "research",
}


class Event(BaseModel):
    time: datetime
    kind: Kind
    level: Level
    title: str
    detail: str | None = None


def order_event(time: datetime, payload: dict[str, Any]) -> Event:
    request = payload.get("request") or {}
    words = [str(payload.get("action", "order")).capitalize()]
    words += [str(request[key]) for key in ("side", "qty", "symbol") if key in request]
    refused = payload.get("outcome") in ("refused", "failed")
    return Event(
        time=time,
        kind="risk" if refused else "order",
        level="warn" if refused else "info",
        title=" ".join(words),
        detail=payload.get("reason") if refused else payload.get("outcome"),
    )


def run_event(time: datetime, payload: dict[str, Any]) -> Event:
    run = payload.get("run") or {}
    status = run.get("status", "ok")
    level: Level = "error" if status in ("error", "halted") else "info"
    return Event(
        time=time,
        kind="run",
        level=level,
        title=f"Autopilot run {status}",
        detail=run.get("detail"),
    )


def from_entry(kind: str, time: datetime, payload: dict[str, Any]) -> Event:
    if kind == "order":
        return order_event(time, payload)
    if kind == "run":
        return run_event(time, payload)
    title = TITLES.get(kind, kind.replace("-", " ").capitalize())
    return Event(time=time, kind=KINDS.get(kind, "system"), level="info", title=title)


def collect(state: Any, since: datetime) -> list[Event]:
    """Audit log entries and ledger fills since `since`, newest first."""
    events = [
        from_entry(entry.kind, moment, entry.payload)
        for entry in state.audit.entries()
        if (moment := datetime.fromisoformat(entry.time)) >= since
    ]
    days = max(1, (datetime.now(UTC) - since).days + 1)
    for fill in state.ledger.fills(days=days, limit=500):
        if fill.time >= since:
            events.append(
                Event(
                    time=fill.time,
                    kind="fill",
                    level="info",
                    title=f"Filled {fill.side} {fill.qty:g} {fill.symbol} at {fill.price:,.2f}",
                )
            )
    return sorted(events, key=lambda event: event.time, reverse=True)


@router.get("/activity")
def read_activity(
    request: Request, hours: float = Query(default=24, gt=0, le=24 * 30)
) -> list[Event]:
    """Orders, fills, runs, risk events and research of the last `hours`, newest first."""
    return collect(request.app.state, datetime.now(UTC) - timedelta(hours=hours))
