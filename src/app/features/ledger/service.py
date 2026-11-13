import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any, Literal, cast

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel
from requests import RequestException

from app.core.audit import AuditLog
from app.features.ledger import activities, purification, reconcile, schemas
from app.features.ledger.lots import ZERO, Book, Execution, Lot, Method, replay
from app.features.ledger.store import PATH, Adjustment, Store

logger = logging.getLogger(__name__)

JOB_ID = "ledger-sync"
EVERY = timedelta(minutes=5)
FIRST_RUN = timedelta(seconds=5)
LOOKBACK_DAYS = 3
FILL_JOB_ID = "ledger-fill"
FILL_DELAY = timedelta(seconds=3)
FILL_PATIENCE = timedelta(seconds=30)
FILLS_SHOWN = 500
REASON = "Adopted Alpaca's quantity"


class NothingToAdoptError(LookupError):
    """Raised when the ledger and the broker already agree on a symbol's quantity."""


class BrokerPosition(BaseModel):
    symbol: str
    qty: Decimal
    avg_entry_price: Decimal | None = None
    current_price: Decimal | None = None
    unrealized_pl: Decimal | None = None


def state_of(
    found: schemas.Difference | None, ours: Decimal, theirs: Decimal
) -> Literal["match", "quantity", "pending", "cost", "orphan", "zombie"]:
    if found is None:
        return "match"
    if found.kind == "quantity" and theirs == ZERO:
        return "orphan"
    if found.kind == "quantity" and ours == ZERO:
        return "zombie"
    return found.kind


def utc_today() -> date:
    return datetime.now(UTC).date()


def adopted_price(found: schemas.Difference, ours: Decimal, theirs: Decimal) -> Decimal:
    """The price that makes the pooled cost equal the broker's total for the quantity added."""
    paid = Decimal(str(found.broker_cost or 0))
    total = theirs * paid - ours * Decimal(str(found.ledger_cost or 0))
    return total / (theirs - ours) if total > ZERO else paid


def key_of(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def adjusted(item: Adjustment) -> Execution:
    kind: Literal["buy", "drop"] = "buy" if item.kind == "open" else "drop"
    return Execution(f"adjustment-{item.id}", item.symbol, kind, item.time, item.qty, item.price)


def number(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def unrealized_gain(lot: Lot, price: Decimal) -> Decimal:
    worth = lot.qty * price
    return worth - lot.cost if lot.side == "long" else lot.cost - worth


def percent(gain: Decimal, basis: Decimal) -> float | None:
    return float(gain / basis) if basis else None


class Ledger:
    """Lots and realized P&L rebuilt from Alpaca's fills, checked against the broker's positions."""

    def __init__(
        self, trading: TradingClient, path: Path = PATH, log: AuditLog | None = None
    ) -> None:
        self.trading = trading
        self.store = Store(path)
        self.log = log
        self.lock = Lock()
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.last: schemas.Reconciliation | None = None
        self.synced: datetime | None = None
        self.error: str | None = None
        self._waiting: datetime | None = None

    def sync(self) -> int:
        """Fetch the activity not seen yet; fees in coin post a day after their fills."""
        with self.lock:
            latest = self.store.last_day()
            if latest is None:
                rows = activities.fetch_all(self.trading)
            else:
                today = utc_today()
                first = min(today - timedelta(days=LOOKBACK_DAYS), latest - timedelta(days=1))
                span = (today - first).days + 2
                rows = [
                    row
                    for day in (first + timedelta(days=step) for step in range(span))
                    for row in activities.fetch_day(self.trading, day)
                ]
            added = self.store.add(rows)
            self.synced = datetime.now(UTC)
            return added

    def refresh(self, trigger: schemas.Trigger = "manual") -> schemas.Reconciliation:
        """Sync, rebuild and compare; a failure is kept for the status and raised."""
        try:
            self.sync()
        except (APIError, RequestException) as exc:
            self.error = str(exc)[:300]
            raise
        self.error = None
        return self.check(trigger)

    def comparison(self) -> schemas.Comparison:
        """Every symbol held by the ledger or by Alpaca, with quantities, costs and P&L."""
        rows = self.store.activities()
        book = replay(self.executions(rows), "average")
        ledger = {lot.symbol: lot for lot in book.lots}
        broker = {key_of(item.symbol): item for item in self.positions()}
        differences = {item.symbol: item for item in (self.last.differences if self.last else [])}
        compared = []
        for symbol in sorted({*ledger, *broker}):
            lot, position = ledger.get(symbol), broker.get(symbol)
            ours = lot.signed_qty if lot else ZERO
            theirs = position.qty if position else ZERO
            price = position.current_price if position else None
            found = differences.get(symbol)
            compared.append(
                schemas.Compared(
                    symbol=symbol,
                    state=state_of(found, ours, theirs),
                    ledger_qty=float(ours),
                    broker_qty=float(theirs),
                    delta=float(ours - theirs),
                    ledger_cost=number(lot.unit_cost) if lot else None,
                    broker_cost=number(position.avg_entry_price) if position else None,
                    ledger_pnl=number(unrealized_gain(lot, price)) if lot and price else None,
                    broker_pnl=number(position.unrealized_pl) if position else None,
                    note=found.note if found else "",
                )
            )
        newest = max(
            rows,
            key=lambda row: (
                row.transaction_time or row.created_at or datetime.min.replace(tzinfo=UTC)
            ),
            default=None,
        )
        cursor = (
            schemas.Cursor(id=newest.id, time=newest.transaction_time or newest.created_at)
            if newest
            else None
        )
        return schemas.Comparison(
            reconciliation=self.last, rows=compared, cursor=cursor, error=self.error
        )

    def check(self, trigger: schemas.Trigger | None = None) -> schemas.Reconciliation:
        rows = self.store.activities()
        book = replay(self.executions(rows), "average")
        held = [
            reconcile.Held(key_of(item.symbol), item.qty, item.avg_entry_price)
            for item in self.positions()
        ]
        now = datetime.now(UTC)
        found = reconcile.compare(
            book, held, activities.pending_fees(rows, now), activities.corporate_actions(rows)
        )
        self.last = schemas.Reconciliation(
            checked=now,
            synced=self.synced,
            activities=len(rows),
            ok=all(item.kind != "quantity" for item in found),
            trigger=trigger or (self.last.trigger if self.last else None),
            differences=[
                schemas.Difference(
                    symbol=item.symbol,
                    kind=item.kind,
                    ledger_qty=float(item.ledger_qty),
                    broker_qty=float(item.broker_qty),
                    ledger_cost=number(item.ledger_cost),
                    broker_cost=number(item.broker_cost),
                    note=item.note,
                )
                for item in found
            ],
        )
        return self.last

    def adopt(self, symbol: str) -> schemas.Reconciliation:
        """Make the ledger hold what Alpaca holds, with a lot or a write-off and an audit entry."""
        key = key_of(symbol)
        with self.lock:
            found = next(
                (
                    item
                    for item in (self.last.differences if self.last else [])
                    if item.symbol == key
                ),
                None,
            )
            if found is None or found.kind != "quantity":
                raise NothingToAdoptError(key)
            ours, theirs = Decimal(str(found.ledger_qty)), Decimal(str(found.broker_qty))
            delta = theirs - ours
            if delta > ZERO:
                self.store.adjust(key, "open", delta, adopted_price(found, ours, theirs), REASON)
            else:
                self.store.adjust(key, "drop", -delta, ZERO, REASON)
            if self.log is not None:
                self.log.record(
                    "ledger-adopt",
                    {"symbol": key, "ledger_qty": found.ledger_qty, "broker_qty": found.broker_qty},
                )
        return self.check()

    def executions(self, rows: Sequence[activities.Activity] | None = None) -> list[Execution]:
        found = activities.executions(self.store.activities() if rows is None else rows)
        return [*found, *(adjusted(item) for item in self.store.adjustments())]

    def book(self, method: Method) -> Book:
        """The lots by `method`, with the wash-sale rule applied to everything but coins."""
        rows = self.store.activities()
        coins = {row.key for row in rows if row.symbol and "/" in row.symbol}
        return replay(self.executions(rows), method, self.store.picks(), wash_exempt=coins)

    def positions(self) -> list[BrokerPosition]:
        return [BrokerPosition.model_validate(item) for item in self.trading.get_all_positions()]

    def marks(self) -> dict[str, Decimal]:
        return {
            key_of(item.symbol): item.current_price
            for item in self.positions()
            if item.current_price is not None
        }

    def lots(self, method: Method, symbol: str | None = None) -> list[schemas.LotRow]:
        marks, now = self.marks(), datetime.now(UTC)
        rows = []
        for lot in self.book(method).lots:
            if symbol and lot.symbol != key_of(symbol):
                continue
            price = marks.get(lot.symbol)
            gain = None if price is None else unrealized_gain(lot, price)
            rows.append(
                schemas.LotRow(
                    id=lot.id,
                    symbol=lot.symbol,
                    side=lot.side,
                    opened=lot.opened,
                    days=(now - lot.opened).days,
                    qty=float(lot.signed_qty),
                    original=float(lot.original),
                    unit_cost=float(lot.unit_cost),
                    basis=float(lot.cost),
                    wash=float(lot.wash),
                    price=number(price),
                    value=None if price is None else float(lot.signed_qty * price),
                    unrealized=number(gain),
                    unrealized_pct=None if gain is None else percent(gain, lot.cost),
                )
            )
        return rows

    def realized(
        self, method: Method, symbol: str | None = None, days: int | None = None
    ) -> schemas.Realized:
        since = datetime.now(UTC) - timedelta(days=days) if days else None
        rows = []
        for close in self.book(method).closes:
            if symbol and close.symbol != key_of(symbol):
                continue
            if since and close.closed < since:
                continue
            rows.append(
                schemas.CloseRow(
                    lot=close.lot,
                    symbol=close.symbol,
                    side=close.side,
                    opened=close.opened,
                    closed=close.closed,
                    days=(close.closed - close.opened).days,
                    order=close.order,
                    qty=float(close.qty),
                    basis=float(close.basis),
                    proceeds=float(close.proceeds),
                    gain=float(close.gain),
                    gain_pct=percent(close.gain, close.basis),
                    disallowed=float(close.disallowed),
                    reported=float(close.reported),
                )
            )
        rows.sort(key=lambda row: row.closed, reverse=True)
        return schemas.Realized(
            summary=schemas.RealizedSummary(
                count=len(rows),
                basis=sum(row.basis for row in rows),
                proceeds=sum(row.proceeds for row in rows),
                gain=sum(row.gain for row in rows),
                disallowed=sum(row.disallowed for row in rows),
                reported=sum(row.reported for row in rows),
            ),
            rows=rows,
        )

    def holdings(self, method: Method) -> list[schemas.Holding]:
        marks = self.marks()
        book = self.book(method)
        realized: dict[str, Decimal] = {}
        for close in book.closes:
            realized[close.symbol] = realized.get(close.symbol, ZERO) + close.gain
        held: dict[str, list[Lot]] = {}
        for lot in book.lots:
            held.setdefault(lot.symbol, []).append(lot)
        rows = []
        for symbol, parts in sorted(held.items()):
            qty = sum((lot.signed_qty for lot in parts), ZERO)
            basis = sum((lot.cost for lot in parts), ZERO)
            price = marks.get(symbol)
            gain = (
                None if price is None else sum((unrealized_gain(lot, price) for lot in parts), ZERO)
            )
            rows.append(
                schemas.Holding(
                    symbol=symbol,
                    qty=float(qty),
                    basis=float(basis),
                    average_cost=float(basis / abs(qty)),
                    price=number(price),
                    value=None if price is None else float(qty * price),
                    unrealized=number(gain),
                    unrealized_pct=None if gain is None else percent(gain, basis),
                    realized=float(realized.get(symbol, ZERO)),
                    lots=len(parts),
                )
            )
        return rows

    def fills(
        self, symbol: str | None = None, days: int | None = None, limit: int = FILLS_SHOWN
    ) -> list[schemas.FillRow]:
        """The stored fills, newest first, each with the coin fee that came with it."""
        rows = self.store.activities()
        fees = activities.coin_fees(rows)
        since = datetime.now(UTC) - timedelta(days=days) if days else None
        found = [
            fill
            for fill in rows
            if activities.is_fill(fill)
            and (symbol is None or fill.key == key_of(symbol))
            and (since is None or fill.moment >= since)
        ]
        found.sort(key=activities.moment_of, reverse=True)
        return [
            schemas.FillRow(
                id=fill.id,
                time=fill.moment,
                symbol=fill.symbol or "",
                side="buy" if fill.side == "buy" else "sell",
                qty=float(fill.qty or 0),
                price=float(fill.price or 0),
                value=float((fill.qty or 0) * (fill.price or 0)),
                partial=fill.fill == "partial_fill",
                order=fill.order_id or "",
                fee_qty=number(fees.get(fill.id)),
            )
            for fill in found[:limit]
        ]

    def set_picks(self, order: str, picks: schemas.Picks) -> schemas.Picks:
        merged: dict[str, Decimal] = {}
        for pick in picks.lots:
            merged[pick.lot] = merged.get(pick.lot, ZERO) + Decimal(str(pick.qty))
        self.store.set_picks(order, merged.items())
        return self.picks(order)

    def picks(self, order: str) -> schemas.Picks:
        found = self.store.picks().get(order, [])
        return schemas.Picks(lots=[schemas.Pick(lot=lot, qty=float(qty)) for lot, qty in found])

    def purification(self, year: int) -> purification.Statement:
        """The year's statement, with zakat on the account's value now when Alpaca answers."""
        try:
            raw = cast(dict[str, Any], self.trading.get_account())
            equity: Decimal | None = Decimal(str(raw["equity"]))
        except (APIError, RequestException, KeyError, TypeError):
            equity = None
        return purification.statement(self.store.activities(), self.store.ratios(), year, equity)

    def status(self) -> schemas.Status:
        return schemas.Status(
            activities=self.store.count(),
            synced=self.synced,
            error=self.error,
            reconciliation=self.last,
        )

    def nudge(self) -> None:
        """A fill arrived: sync in a few seconds, once for a burst of fills."""
        now = datetime.now(UTC)
        self._waiting = self._waiting or now
        delay = FILL_DELAY if now - self._waiting < FILL_PATIENCE else timedelta(0)
        self.scheduler.add_job(
            self._after_fills,
            "date",
            run_date=now + delay,
            id=FILL_JOB_ID,
            replace_existing=True,
            misfire_grace_time=60,
        )

    async def _after_fills(self) -> None:
        self._waiting = None
        await self._refresh("fill")

    async def _refresh(self, trigger: schemas.Trigger = "scheduled") -> None:
        try:
            await asyncio.to_thread(self.refresh, trigger)
        except (APIError, RequestException) as exc:
            logger.warning("The ledger could not sync, and will try again: %s", exc)

    def start(self) -> None:
        self.scheduler.add_job(
            self._refresh,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(UTC) + FIRST_RUN,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
