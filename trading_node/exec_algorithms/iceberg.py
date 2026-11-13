from typing import Any

from nautilus_trader.model import ClientOrderId, ExecAlgorithmId, OrderStatus, OrderType
from nautilus_trader.trading import ExecutionAlgorithmConfig

from trading_node.exec_algorithms.family import (
    CANCELED,
    ENDED,
    STOP_TOPIC,
    NodeAlgorithm,
    open_children,
)

ICEBERG = ExecAlgorithmId("ICEBERG")


def iceberg_params(display_qty: float) -> dict[str, str]:
    return {"display_qty": str(display_qty)}


class IcebergAlgorithm(NodeAlgorithm):
    """Shows one slice of a limit primary at a time and the next as each one fills.

    The last slice is the primary itself; a slice canceled or expired at the venue ends the order.
    """

    def __init__(self) -> None:
        super().__init__(ExecutionAlgorithmConfig(exec_algorithm_id=ICEBERG))
        self.display: dict[str, float] = {}
        self.stopping: set[str] = set()

    def on_start(self) -> None:
        self.subscribe_topic(STOP_TOPIC, self.on_stop_request)

    def on_order(self, order: Any) -> None:
        if order.order_type != OrderType.LIMIT:
            self.deny_order(order, "An iceberg works a limit order")
            return
        try:
            display = float((order.exec_algorithm_params or {})["display_qty"])
        except (KeyError, ValueError) as exc:
            self.deny_order(order, f"Bad iceberg: {exc}")
            return
        if display <= 0:
            self.deny_order(order, "Bad iceberg: display_qty must be positive")
            return
        self.display[str(order.client_order_id)] = display
        self.show(order)

    def show(self, primary: Any) -> None:
        display = self.display[str(primary.client_order_id)]
        if primary.quantity.as_double() <= display:
            self.submit_order(primary)
            return
        instrument = self.store.instrument(primary.instrument_id)
        if instrument is None:
            self.deny_order(primary, f"{primary.instrument_id} is not loaded")
            return
        child = self.spawn_limit(
            primary,
            instrument.make_qty(display),
            primary.price,
            time_in_force=primary.time_in_force,
        )
        self.submit_order(child)

    def primary_of(self, event: Any) -> Any:
        order = self.store.order(event.client_order_id)
        if order is None or order.exec_spawn_id is None:
            return None
        if order.client_order_id == order.exec_spawn_id:
            return None
        primary = self.store.order(order.exec_spawn_id)
        return None if primary is None or primary.status != OrderStatus.INITIALIZED else primary

    def on_order_filled(self, event: Any) -> None:
        primary = self.primary_of(event)
        child = self.store.order(event.client_order_id)
        if primary is None or child is None or not child.is_closed:
            return
        if str(primary.client_order_id) in self.stopping:
            return
        self.show(primary)

    def on_order_canceled(self, event: Any) -> None:
        self.slice_ended(event, "canceled")

    def on_order_expired(self, event: Any) -> None:
        self.slice_ended(event, "expired")

    def on_order_rejected(self, event: Any) -> None:
        self.slice_ended(event, f"rejected: {event.reason}")

    def slice_ended(self, event: Any, how: str) -> None:
        primary = self.primary_of(event)
        if primary is None:
            return
        key = str(primary.client_order_id)
        if key in self.stopping:
            self.stopping.discard(key)
            self.deny_order(primary, CANCELED)
            return
        self.deny_order(primary, f"{ENDED} a slice was {how}; the rest is not shown")

    def on_stop_request(self, message: object) -> None:
        key = str(message)
        primary = self.store.order(ClientOrderId(key))
        if primary is None or primary.is_closed or primary.exec_algorithm_id != ICEBERG:
            return
        if primary.status != OrderStatus.INITIALIZED:
            self.cancel_order(primary)
            return
        working = open_children(self.store, primary.client_order_id)
        if not working:
            self.deny_order(primary, CANCELED)
            return
        self.stopping.add(key)
        for child in working:
            self.cancel_order(child)
