import json
from dataclasses import dataclass
from typing import Any

from nautilus_trader.model import (
    ClientOrderId,
    ExecAlgorithmId,
    OrderStatus,
    OrderType,
    PriceType,
    TimeInForce,
)
from nautilus_trader.trading import ExecutionAlgorithmConfig

from app.features.orders.quotes import TopOfBook
from app.features.orders.schedule import next_slice
from trading_node.alpaca_adapter.parsing import is_crypto
from trading_node.exec_algorithms.family import (
    CANCELED,
    ENDED,
    STOP_TOPIC,
    NodeAlgorithm,
    open_children,
)

SCHEDULED = ExecAlgorithmId("SCHEDULE")
MIN_COIN_COST = 10.0


@dataclass
class Run:
    starts: list[int]
    weights: list[float]
    end: int
    tilt: float
    total: float


def schedule_params(intervals: list[tuple[int, float]], end: int, tilt: float) -> dict[str, str]:
    """A schedule as Nautilus carries it on the primary: string to string."""
    return {"schedule": json.dumps(intervals), "end": str(end), "tilt": str(tilt)}


def run_of(params: dict[str, str] | None, total: float) -> Run:
    if not params or "schedule" not in params or "end" not in params:
        raise ValueError("the order carries no schedule")
    intervals = json.loads(params["schedule"])
    if not intervals:
        raise ValueError("the schedule has no intervals")
    return Run(
        starts=[int(start) for start, _ in intervals],
        weights=[float(weight) for _, weight in intervals],
        end=int(params["end"]),
        tilt=float(params.get("tilt", "0")),
        total=total,
    )


def lean(quote: Any, side: str) -> float:
    if quote is None:
        return 0.0
    return TopOfBook.model_validate(
        {
            "bp": float(quote.bid_price),
            "ap": float(quote.ask_price),
            "bs": float(quote.bid_size),
            "as": float(quote.ask_size),
        }
    ).lean(side)


class ScheduledAlgorithm(NodeAlgorithm):
    """Works a primary as immediate-or-cancel slices on the API's TWAP, VWAP or shortfall schedule.

    A slice still open when the next interval starts is canceled and its quantity carried on; the
    last interval sends the primary itself for what is left; at the end the rest is not sent.
    """

    def __init__(self) -> None:
        super().__init__(ExecutionAlgorithmConfig(exec_algorithm_id=SCHEDULED))
        self.runs: dict[str, Run] = {}

    def on_start(self) -> None:
        self.subscribe_topic(STOP_TOPIC, self.on_stop_request)

    def on_order(self, order: Any) -> None:
        key = str(order.client_order_id)
        try:
            run = run_of(order.exec_algorithm_params, order.quantity.as_double())
        except (ValueError, TypeError, KeyError) as exc:
            self.deny_order(order, f"Bad schedule: {exc}")
            return
        self.runs[key] = run
        for index, start in enumerate(run.starts):
            self.timer.set_time_alert_ns(
                f"{key}:{index}", start, callback=self.on_interval, allow_past=True
            )
        self.timer.set_time_alert_ns(f"{key}:end", run.end, callback=self.on_end, allow_past=True)

    def primary(self, key: str) -> Any:
        order = self.store.order(ClientOrderId(key))
        return None if order is None or order.is_closed else order

    def on_interval(self, event: Any) -> None:
        key, index = str(event.name).rsplit(":", 1)
        run, primary = self.runs.get(key), self.primary(key)
        if run is None or primary is None or primary.status != OrderStatus.INITIALIZED:
            return
        working = open_children(self.store, primary.client_order_id)
        if working:
            for child in working:
                self.cancel_order(child)
            return
        position = int(index)
        instrument = self.store.instrument(primary.instrument_id)
        if instrument is None:
            return
        left = primary.quantity.as_double()
        side = primary.side.name.lower()
        tilt = run.tilt * lean(self.store.quote(primary.instrument_id), side)
        whole = not is_crypto(primary.instrument_id)
        size = next_slice(left, run.weights, position, whole=whole, lean=tilt)
        if not self.worth_sending(primary, size):
            return
        if position >= len(run.starts) - 1 or size >= left:
            self.submit_order(primary)
            return
        quantity = instrument.make_qty(size)
        if quantity.as_double() <= 0:
            return
        if primary.order_type == OrderType.LIMIT:
            child = self.spawn_limit(
                primary, quantity, primary.price, time_in_force=TimeInForce.IOC
            )
        else:
            child = self.spawn_market(primary, quantity, time_in_force=TimeInForce.IOC)
        self.submit_order(child)

    def worth_sending(self, primary: Any, size: float) -> bool:
        """A coin slice must cost Alpaca's minimum; a stock slice needs a whole share."""
        if size <= 0:
            return False
        if not is_crypto(primary.instrument_id):
            return True
        if primary.order_type == OrderType.LIMIT:
            price = primary.price
        else:
            price = self.store.price(primary.instrument_id, PriceType.MID) or self.store.price(
                primary.instrument_id, PriceType.LAST
            )
        return price is None or size * price.as_double() >= MIN_COIN_COST

    def on_end(self, event: Any) -> None:
        key = str(event.name).rsplit(":", 1)[0]
        primary = self.primary(key)
        run = self.runs.pop(key, None)
        if primary is None or run is None or primary.status != OrderStatus.INITIALIZED:
            return
        for child in open_children(self.store, primary.client_order_id):
            self.cancel_order(child)
        left = primary.quantity.as_double()
        self.deny_order(primary, f"{ENDED} {left:g} of {run.total:g} not filled")

    def on_stop_request(self, message: object) -> None:
        key = str(message)
        primary = self.primary(key)
        if primary is None or primary.exec_algorithm_id != SCHEDULED:
            return
        self.runs.pop(key, None)
        for child in open_children(self.store, primary.client_order_id):
            self.cancel_order(child)
        if primary.status == OrderStatus.INITIALIZED:
            self.deny_order(primary, CANCELED)
        else:
            self.cancel_order(primary)
