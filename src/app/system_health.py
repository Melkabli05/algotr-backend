import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from threading import Lock
from typing import Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from app.core.broker import Trading
from app.core.database import transaction
from app.core.sidecar import available as sidecar_available
from app.features.orders.service import list_orders

router = APIRouter(prefix="/system", tags=["system"])

KEPT = 5000
WINDOW = timedelta(minutes=15)
PROBE_EVERY = timedelta(seconds=30)
FILES = {
    "Audit log": Path("data/audit.jsonl"),
    "Ledger": Path("data/ledger.sqlite"),
    "News": Path("data/scanner/news.sqlite"),
    "Daily bars": Path("data/scanner/bars.parquet"),
    "Order books": Path("data/book"),
}


class Timing:
    """The last few thousand requests: when, which route, how long and how they ended."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.rows: deque[tuple[float, str, float, int]] = deque(maxlen=KEPT)

    def add(self, route: str, ms: float, status: int) -> None:
        with self.lock:
            self.rows.append((time.time(), route, ms, status))

    def recent(self, seconds: float) -> list[tuple[float, str, float, int]]:
        cutoff = time.time() - seconds
        with self.lock:
            return [row for row in self.rows if row[0] >= cutoff]


async def time_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Record how long each API request took and how it ended."""
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    timing: Timing | None = getattr(request.app.state, "timing", None)
    if timing is not None and path not in ("/stream", "/health") and not path.startswith("/ui"):
        timing.add(path, (time.perf_counter() - started) * 1000, response.status_code)
    return response


class Probe:
    """A timed call to Alpaca's clock, at most every 30 seconds, with its recent results."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.results: deque[tuple[datetime, float | None, str | None]] = deque(maxlen=120)

    def check(self, trading: Any) -> None:
        now = datetime.now(UTC)
        with self.lock:
            if self.results and now - self.results[-1][0] < PROBE_EVERY:
                return
        started = time.perf_counter()
        try:
            trading.get_clock()
            found: tuple[datetime, float | None, str | None] = (
                now,
                (time.perf_counter() - started) * 1000,
                None,
            )
        except Exception as exc:
            found = (now, None, f"{type(exc).__name__}: {exc}"[:200])
        with self.lock:
            self.results.append(found)


class Latency(BaseModel):
    requests: int
    p50: float | None
    p95: float | None
    p99: float | None
    errors: int
    slowest: list[tuple[str, float]]


class Broker(BaseModel):
    last_ok: datetime | None
    last_ms: float | None
    error_rate: float | None
    last_error: str | None


class Storage(BaseModel):
    name: str
    bytes: int | None


class ModelHealth(BaseModel):
    algorithm: str
    trained: date | None
    next_retrain: date | None
    in_sample_sharpe: float | None


class Stage(BaseModel):
    name: str
    median: float | None
    worst: float | None
    unit: str
    samples: int


class Health(BaseModel):
    api: Latency
    broker: Broker
    storage: list[Storage]
    sidecar: bool
    busy: str | None
    models: list[ModelHealth]
    path: list[Stage]
    outcomes: dict[str, int]


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def latency(rows: list[tuple[float, str, float, int]]) -> Latency:
    times = [row[2] for row in rows]
    by_route: dict[str, list[float]] = {}
    for _, route, ms, _ in rows:
        by_route.setdefault(route, []).append(ms)
    slowest = sorted(
        ((route, median(values)) for route, values in by_route.items()), key=lambda item: -item[1]
    )
    return Latency(
        requests=len(rows),
        p50=percentile(times, 0.5),
        p95=percentile(times, 0.95),
        p99=percentile(times, 0.99),
        errors=sum(1 for row in rows if row[3] >= 500),
        slowest=slowest[:5],
    )


def size_of(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def next_month(day: date) -> date:
    return date(day.year + day.month // 12, day.month % 12 + 1, 1)


def stage(name: str, values: list[float], unit: str) -> Stage:
    return Stage(
        name=name,
        median=median(values) if values else None,
        worst=max(values) if values else None,
        unit=unit,
        samples=len(values),
    )


def order_path(orders: list[Any], entries: list[Any]) -> tuple[list[Stage], dict[str, int]]:
    """From submission to fill: our round trip, Alpaca's acceptance and the time to fill."""
    submits = [
        entry.payload
        for entry in entries
        if entry.kind == "order" and entry.payload.get("action") in ("submit", "algo", "iceberg")
    ]
    trips = [
        float(found["result"]["ms"])
        for found in submits
        if isinstance(found.get("result"), dict) and "ms" in found["result"]
    ]
    accepted = [
        (order.submitted_at - order.created_at).total_seconds() * 1000
        for order in orders
        if order.submitted_at and order.created_at
    ]
    filled = [
        (order.filled_at - order.submitted_at).total_seconds()
        for order in orders
        if order.filled_at and order.submitted_at
    ]
    outcomes: dict[str, int] = {}
    for found in submits:
        outcome = str(found.get("outcome"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return [
        stage("Risk checks and submission, round trip", trips, "ms"),
        stage("Accepted by Alpaca after creation", accepted, "ms"),
        stage("Filled after acceptance", filled, "s"),
    ], outcomes


@router.get("/health")
def read_health(request: Request, trading: Trading) -> Health:
    """Latency of this API, Alpaca's, storage, the sidecar and models, and the order path."""
    state: Any = request.app.state
    timing: Timing = state.timing
    probe: Probe = state.probe
    probe.check(trading)
    results = list(probe.results)
    good = [row for row in results if row[2] is None]
    bad = [row for row in results if row[2] is not None]
    agents = state.agents.status()
    models = [
        ModelHealth(
            algorithm=item.algorithm,
            trained=item.model.trained if item.model else None,
            next_retrain=next_month(item.model.trained) if item.model else None,
            in_sample_sharpe=item.model.in_sample_sharpe if item.model else None,
        )
        for item in agents.agents
    ]
    with transaction() as database:
        orders = list_orders(database, 100)
    path, outcomes = order_path(orders, state.audit.entries()[-1000:])
    return Health(
        api=latency(timing.recent(WINDOW.total_seconds())),
        broker=Broker(
            last_ok=good[-1][0] if good else None,
            last_ms=good[-1][1] if good else None,
            error_rate=len(bad) / len(results) if results else None,
            last_error=bad[-1][2] if bad else None,
        ),
        storage=[Storage(name=name, bytes=size_of(path)) for name, path in FILES.items()],
        sidecar=sidecar_available(),
        busy=agents.busy,
        models=models,
        path=path,
        outcomes=outcomes,
    )
