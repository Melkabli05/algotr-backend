from decimal import Decimal
from typing import Any, Self

from nautilus_trader.common import Cache, Clock
from nautilus_trader.model import ClientOrderId, OrderStatus
from nautilus_trader.trading import ExecutionAlgorithm

STOP_TOPIC = "commands.exec_algorithm.stop"
ENDED = "Schedule ended:"
CANCELED = "Canceled by command"
FINISHING = (ENDED, CANCELED)


def children(cache: Any, primary_id: ClientOrderId) -> list[Any]:
    """The orders an algorithm spawned from a primary, without the primary itself."""
    return [
        order
        for order in cache.orders_for_exec_spawn(primary_id)
        if order.client_order_id != primary_id
    ]


def open_children(cache: Any, primary_id: ClientOrderId) -> list[Any]:
    return [order for order in children(cache, primary_id) if not order.is_closed]


def is_algorithm_primary(order: Any) -> bool:
    return order.exec_algorithm_id is not None and order.exec_spawn_id == order.client_order_id


def filled(orders: list[Any]) -> tuple[Decimal, float | None]:
    """The quantity a family filled and its quantity-weighted average price."""
    quantity = sum((order.filled_qty.as_decimal() for order in orders), Decimal(0))
    if not quantity:
        return quantity, None
    value = sum(
        float(order.filled_qty.as_decimal()) * float(order.avg_px)
        for order in orders
        if order.filled_qty.as_double() > 0
    )
    return quantity, value / float(quantity)


def family_outcome(
    cache: Any, primary: Any, reason: str | None
) -> tuple[str, dict[str, Any] | None, str | None]:
    """An algorithm command finishes when its primary is closed and no child is still working."""
    family = cache.orders_for_exec_spawn(primary.client_order_id)
    quantity, average = filled(family)
    if not primary.is_closed or any(not order.is_closed for order in family):
        return "sent", None, None
    status = primary.status
    finishing = reason is not None and reason.startswith(FINISHING)
    if status == OrderStatus.DENIED and not finishing and not quantity:
        return "denied", None, reason
    if status == OrderStatus.REJECTED and not quantity:
        return "failed", None, reason
    result = {
        "status": "filled" if status == OrderStatus.FILLED else "ended",
        "filled_qty": float(quantity),
        "avg_px": average,
        "detail": reason if finishing else None,
    }
    return "done", result, None


class NodeAlgorithm(ExecutionAlgorithm):
    """An execution algorithm with its cache and clock, which exist once it is registered."""

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        return super().__new__(cls)

    @property
    def store(self) -> Cache:
        if self.cache is None:
            raise RuntimeError(f"{self.exec_algorithm_id} is not registered with a node")
        return self.cache

    @property
    def timer(self) -> Clock:
        if self.clock is None:
            raise RuntimeError(f"{self.exec_algorithm_id} is not registered with a node")
        return self.clock
