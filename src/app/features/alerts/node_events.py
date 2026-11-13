import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.core import metrics

logger = logging.getLogger(__name__)

FILL_GAP = 0.02
RETRY_SECONDS = 5.0
REFUSED = ("rejected", "denied")

Raise = Callable[..., Any]


def classify(
    event: dict[str, Any], last_price: Callable[[str], float | None], seen: set[str]
) -> tuple[str, str, str | None] | None:
    """The alert a node event calls for at once, as (kind, detail, symbol), or None."""
    kind = event.get("type")
    if kind == "order" and event.get("status") in REFUSED:
        key = f"{event['client_order_id']}:{event['status']}"
        if key in seen:
            return None
        seen.add(key)
        symbol = str(event["instrument_id"]).rsplit(".", 1)[0]
        detail = (
            f"{symbol} {event['side']} {event['quantity']} {event['status']}: "
            f"{event.get('reason') or 'no reason given'}"
        )
        return "rejected", detail, symbol
    if kind == "fill":
        symbol = str(event["instrument_id"]).rsplit(".", 1)[0]
        last = last_price(symbol)
        price = float(event["price"])
        if last and abs(price - last) / last > FILL_GAP:
            gap = abs(price - last) / last
            return (
                "anomaly",
                f"{symbol} filled at {price:g}, {gap:.1%} from the last {last:g}",
                symbol,
            )
        return None
    if kind == "reconciliation" and event.get("severity") == "critical":
        return "reconcile", str(event.get("detail")), None
    if kind == "status" and event.get("name") == "node" and event.get("state") == "stopped":
        return "connection", "The trading node stopped", None
    return None


class NodeEventAlerts:
    """Alerts the moment the node reports a rejection, an odd fill, critical reconciliation or
    its own stop: it listens to the node's events rather than polling for them."""

    def __init__(
        self,
        listen: Callable[[], AsyncIterator[dict[str, Any]]],
        raise_alert: Raise,
        last_price: Callable[[str], float | None],
    ) -> None:
        self.listen = listen
        self.raise_alert = raise_alert
        self.last_price = last_price
        self.seen: set[str] = set()
        self.task: asyncio.Task[None] | None = None

    def handle(self, event: dict[str, Any]) -> None:
        if event.get("type") == "order" and event.get("status") in REFUSED:
            metrics.ORDER_REJECTIONS.inc()
        found = classify(event, self.last_price, self.seen)
        if found is not None:
            kind, detail, symbol = found
            self.raise_alert(kind, detail, symbol=symbol)

    async def run(self) -> None:
        while True:
            try:
                async for event in self.listen():
                    self.handle(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Lost the node's events, listening again: %s", exc)
            await asyncio.sleep(RETRY_SECONDS)

    def start(self) -> None:
        self.task = asyncio.get_running_loop().create_task(self.run())

    def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
