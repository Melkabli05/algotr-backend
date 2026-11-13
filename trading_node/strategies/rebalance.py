from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from nautilus_trader.model import ClientOrderId, InstrumentId

from trading_node.alpaca_adapter.parsing import instrument_id_of

PRICE_WAIT = timedelta(seconds=10)
SETTLE_WAIT = timedelta(seconds=20)
DEADLINE = timedelta(seconds=120)
Phase = Literal["loading", "selling", "buying"]


@dataclass
class Leg:
    """One order of a rebalance; `status` is set when the node decides it without Alpaca."""

    symbol: str
    side: str
    qty: Decimal
    limit_price: Decimal | None
    client_id: ClientOrderId
    status: str | None = None
    sent: bool = False

    @property
    def instrument_id(self) -> InstrumentId:
        return instrument_id_of(self.symbol)

    def payload(self, quantity: Decimal) -> dict[str, Any]:
        found: dict[str, Any] = {"symbol": self.symbol, "side": self.side, "qty": quantity}
        if self.limit_price is not None:
            found |= {"type": "limit", "limit_price": self.limit_price}
        return found


@dataclass
class Rebalance:
    """A day's planned orders as one command: stale orders canceled, sells first, then buys."""

    command: dict[str, Any]
    legs: list[Leg]
    cancel: list[InstrumentId]
    started: datetime
    phase: Phase = "loading"
    phase_started: datetime | None = None
    asked: set[str] = field(default_factory=set)

    @property
    def instrument_ids(self) -> set[InstrumentId]:
        return {leg.instrument_id for leg in self.legs}

    def side(self, side: str) -> list[Leg]:
        return [leg for leg in self.legs if leg.side == side]

    def waited(self, now: datetime, limit: timedelta) -> bool:
        return now - (self.phase_started or self.started) >= limit


def legs_of(payload: dict[str, Any], client_id: str) -> list[Leg]:
    """The planned orders, each under the command's client order ID and its own suffix."""
    legs, counts = [], {"sell": 0, "buy": 0}
    for item in payload["orders"]:
        side = item["side"]
        if side not in counts:
            raise ValueError(f"side {side!r}")
        counts[side] += 1
        limit = item.get("limit_price")
        legs.append(
            Leg(
                symbol=item["symbol"],
                side=side,
                qty=Decimal(str(item["qty"])),
                limit_price=None if limit is None else Decimal(str(limit)),
                client_id=ClientOrderId(f"{client_id}-{side[0].upper()}{counts[side]}"),
            )
        )
    return legs


def outcome(leg: Leg, order: Any, reason: str | None) -> dict[str, Any]:
    """What happened to one leg, in the shape the plan executor reports."""
    if order is None:
        status = leg.status or reason or "unknown"
        return {
            "symbol": leg.symbol,
            "side": leg.side,
            "qty": float(leg.qty),
            "order_id": None,
            "status": status,
        }
    status = order.status.name.lower()
    if reason and status in ("denied", "rejected"):
        status = f"{status}: {reason}"
    return {
        "symbol": leg.symbol,
        "side": leg.side,
        "qty": float(order.quantity.as_decimal()),
        "order_id": str(leg.client_id),
        "status": status,
        "filled_qty": float(order.filled_qty.as_decimal()),
        "avg_px": float(order.avg_px) if order.filled_qty.as_double() > 0 else None,
    }
