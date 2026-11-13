import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from uuid import uuid4

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel
from requests import RequestException

from app.core.errors import broker_message
from app.features.orders import service
from app.features.orders.schemas import Iceberg, IcebergRequest, Order, OrderRequest, Slice

logger = logging.getLogger(__name__)

PATH = Path("data/icebergs.json")
JOB_ID = "iceberg-slices"
EVERY = timedelta(seconds=3)
KEPT_FINISHED = 50
LEFT = 1e-9


class IcebergNotFoundError(LookupError):
    """Raised when an iceberg id is unknown."""


class State(BaseModel):
    icebergs: list[Iceberg] = []


class IcebergDesk:
    """Shows one slice of a larger limit order at a time and places the next as each one fills."""

    def __init__(self, trading: TradingClient, path: Path = PATH) -> None:
        self.trading = trading
        self.path = path
        self.lock = Lock()
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.state = self._load()

    def _load(self) -> State:
        try:
            return State.model_validate_json(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return State()

    def _save(self) -> None:
        working = [item for item in self.state.icebergs if item.status == "working"]
        finished = [item for item in self.state.icebergs if item.status != "working"]
        kept = [*working, *finished[-KEPT_FINISHED:]]
        self.state.icebergs = sorted(kept, key=lambda item: item.created)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.state.model_dump_json(indent=2))

    def icebergs(self) -> list[Iceberg]:
        with self.lock:
            return [item.model_copy(deep=True) for item in reversed(self.state.icebergs)]

    def place(self, request: IcebergRequest) -> Iceberg:
        """Show the first slice now; the rest is placed as each slice fills."""
        item = Iceberg(
            id=uuid4().hex[:12],
            created=datetime.now(UTC),
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            display_qty=request.display_qty,
            limit_price=request.limit_price,
            time_in_force=request.time_in_force,
        )
        with self.lock:
            self._show(item, request.display_qty)
            self.state.icebergs = [*self.state.icebergs, item]
            self._save()
            return item.model_copy(deep=True)

    def cancel(self, iceberg_id: str) -> Iceberg:
        """Stop an iceberg and cancel the slice on show; what already filled stays filled."""
        with self.lock:
            item = self._find(iceberg_id)
            if item.status == "working":
                self._cancel_slice(item)
                item.status, item.detail = "canceled", "Canceled"
                self._save()
            return item.model_copy(deep=True)

    def tick(self) -> None:
        with self.lock:
            changed = False
            for item in self.state.icebergs:
                try:
                    changed = self._advance(item) or changed
                except APIError as exc:
                    if exc.status_code in service.TRANSIENT:
                        logger.warning("Iceberg %s will retry: %s", item.id, exc)
                        continue
                    item.status, item.detail = "failed", broker_message(exc)
                    changed = True
                except RequestException as exc:
                    logger.warning("Iceberg %s will retry: %s", item.id, exc)
            if changed:
                self._save()

    def _find(self, iceberg_id: str) -> Iceberg:
        for item in self.state.icebergs:
            if item.id == iceberg_id:
                return item
        raise IcebergNotFoundError(iceberg_id)

    def _advance(self, item: Iceberg) -> bool:
        if not item.slices or (
            item.status != "working" and item.slices[-1].status not in service.LIVE
        ):
            return False
        current = item.slices[-1]
        before = current.model_copy()
        self._refresh(item, current)
        changed = current != before
        if item.status != "working" or current.status in service.LIVE:
            return changed
        if current.status == "filled":
            left = item.qty - item.filled_qty
            if left <= LEFT:
                item.status = "done"
            else:
                self._show(item, round(min(item.display_qty, left), 9))
            return True
        item.status = "expired" if current.status == "expired" else "canceled"
        item.detail = f"Slice {len(item.slices)} {current.status.replace('_', ' ')}"
        return True

    def _refresh(self, item: Iceberg, current: Slice) -> None:
        """Read the slice back; one replaced from the order table is followed to its new order."""
        order = Order.model_validate(self.trading.get_order_by_id(current.order))
        while order.status == "replaced" and order.replaced_by:
            current.order = order.replaced_by
            order = Order.model_validate(self.trading.get_order_by_id(current.order))
            item.limit_price = order.limit_price or item.limit_price
        current.status = order.status
        current.filled_qty = order.filled_qty
        current.average_price = order.filled_avg_price

    def _show(self, item: Iceberg, qty: float) -> None:
        key = f"iceberg-{item.id}-{len(item.slices) + 1}"
        request = OrderRequest.model_construct(
            symbol=item.symbol,
            side=item.side,
            type="limit",
            qty=qty,
            limit_price=item.limit_price,
            time_in_force=item.time_in_force,
        )
        order = service.submit_keyed(self.trading, request, key)
        item.slices.append(Slice(order=order.id, qty=qty, status=order.status))

    def _cancel_slice(self, item: Iceberg) -> None:
        if not item.slices or item.slices[-1].status not in service.LIVE:
            return
        try:
            self.trading.cancel_order_by_id(item.slices[-1].order)
        except APIError as exc:
            logger.info("Iceberg %s: the slice could not be canceled: %s", item.id, exc)
        self._refresh(item, item.slices[-1])

    async def _tick(self) -> None:
        await asyncio.to_thread(self.tick)

    def start(self) -> None:
        self.scheduler.add_job(
            self._tick,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
