from collections.abc import Collection
from typing import Any
from uuid import UUID

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.requests import OrderRequest as BrokerOrder
from sqlalchemy.engine import Connection

from app.bridge import commands, node_state
from app.bridge.commands import Source
from app.bridge.node_state import NodeOrder
from app.features.orders import rules
from app.features.orders.schemas import (
    CancelAll,
    Cancellation,
    Order,
    OrderChange,
    OrderRequest,
    Rules,
)

OPEN_ORDER_LIMIT = 500
FINISHED = ("filled", "canceled", "expired", "replaced", "rejected")
LIVE = {
    "new",
    "accepted",
    "pending_new",
    "partially_filled",
    "held",
    "accepted_for_bidding",
    "pending_replace",
    "pending_cancel",
    "pending_review",
    "calculated",
}
TRANSIENT = {429, 500, 502, 503, 504}


STATUSES = {
    "initialized": "pending_new",
    "submitted": "pending_new",
    "released": "pending_new",
    "accepted": "new",
    "triggered": "new",
    "emulated": "held",
    "pending_update": "pending_replace",
    "denied": "rejected",
}
TYPES = {
    "stop_market": "stop",
    "trailing_stop_market": "trailing_stop",
    "market_to_limit": "market",
}
TIMES_IN_FORCE = {"at_the_open": "opg", "at_the_close": "cls"}
QUEUED = "Queued: the trading node is not running; it is sent when the node starts"
NODE_CANNOT_CHANGE = ("trail", "time_in_force")


class ChangeRefusedError(ValueError):
    """Raised when a change cannot be sent because Alpaca would refuse it."""


class OrderNotFoundError(LookupError):
    """Raised when the trading node has no order under that id."""


class UnsupportedOrderError(ValueError):
    """Raised when the trading node cannot build the order."""


class AlgoRefusedError(ValueError):
    """Raised when an algorithm order cannot be planned as asked."""


def number(value: object) -> float | None:
    return None if value is None else float(str(value))


def to_order(row: NodeOrder) -> Order:
    """The trading node's order in the platform's order shape; its id is the client order id."""
    kind = TYPES.get(row.order_type, row.order_type)
    priced = kind in ("limit", "stop_limit")
    return Order(
        id=row.client_order_id,
        client_order_id=row.client_order_id,
        symbol=row.symbol,
        side=row.side,
        type=kind,
        status=STATUSES.get(row.status, row.status),
        qty=float(row.quantity),
        filled_qty=float(row.filled_qty),
        filled_avg_price=number(row.avg_px),
        submitted_at=row.created,
        created_at=row.created,
        updated_at=row.updated,
        filled_at=row.updated if row.status == "filled" else None,
        time_in_force=TIMES_IN_FORCE.get(row.time_in_force, row.time_in_force),
        limit_price=number(row.price) if priced else None,
        stop_price=number(row.trigger_price),
        note=row.reason,
    )


def shown(row: NodeOrder) -> bool:
    """An algorithm's primary is shown only once it is sent itself; its slices always are."""
    return not row.is_primary or row.venue_order_id is not None


def list_orders(connection: Connection, limit: int) -> list[Order]:
    rows = node_state.orders(connection, limit)
    return [to_order(row) for row in rows if shown(row)]


def node_note(connection: Connection) -> str | None:
    return None if node_state.health(connection).up else QUEUED


def place(connection: Connection, payload: dict[str, Any], source: Source = "manual") -> Order:
    """Write the order as a command for the trading node; it reaches Alpaca only from there."""
    reason = commands.unsupported(payload)
    if reason is not None:
        raise UnsupportedOrderError(reason)
    command = commands.submit(connection, "submit", payload, source)
    return Order(
        id=command.client_order_id or str(command.id),
        client_order_id=command.client_order_id,
        symbol=payload["symbol"],
        side=payload["side"],
        type=payload.get("type", "market"),
        status="pending_new",
        qty=float(payload["qty"]),
        filled_qty=0.0,
        filled_avg_price=None,
        submitted_at=command.created,
        created_at=command.created,
        time_in_force=payload.get("time_in_force"),
        limit_price=payload.get("limit_price"),
        stop_price=payload.get("stop_price"),
        order_class=payload.get("order_class") or "simple",
        note=node_note(connection),
    )


def node_order(connection: Connection, order_id: str) -> NodeOrder:
    found = node_state.order(connection, order_id)
    if found is None:
        raise OrderNotFoundError(order_id)
    return found


def get_rules(symbol: str, qty: float) -> Rules:
    asset = rules.asset_class(symbol, qty)
    return Rules(
        asset=asset,
        time_in_force={kind: list(tif) for kind, tif in rules.TIME_IN_FORCE[asset].items()},
        trail_percent=rules.TRAIL_PERCENT,
    )


def broker_request(order: OrderRequest, client_order_id: str | None = None) -> BrokerOrder:
    """The Alpaca request for `order`; stock sells are closes, so nothing opens a short."""
    common: dict[str, Any] = {
        "symbol": order.symbol,
        "qty": order.qty,
        "side": OrderSide(order.side),
        "time_in_force": TimeInForce(order.time_in_force),
    }
    if client_order_id is not None:
        common["client_order_id"] = client_order_id
    if "/" not in order.symbol:
        common["position_intent"] = (
            PositionIntent.BUY_TO_OPEN if order.side == "buy" else PositionIntent.SELL_TO_CLOSE
        )
    if order.order_class != "simple":
        common["order_class"] = OrderClass(order.order_class)
    if order.take_profit is not None:
        common["take_profit"] = TakeProfitRequest(limit_price=order.take_profit)
    if order.stop_loss is not None:
        common["stop_loss"] = StopLossRequest(
            stop_price=order.stop_loss, limit_price=order.stop_loss_limit
        )
    match order.type:
        case "limit":
            return LimitOrderRequest(**common, limit_price=order.limit_price or order.take_profit)
        case "stop":
            return StopOrderRequest(**common, stop_price=order.stop_price)
        case "stop_limit":
            return StopLimitOrderRequest(
                **common, stop_price=order.stop_price, limit_price=order.limit_price
            )
        case "trailing_stop":
            return TrailingStopOrderRequest(
                **common, trail_price=order.trail_price, trail_percent=order.trail_percent
            )
        case _:
            return MarketOrderRequest(**common)


def place_order(trading: TradingClient, order: OrderRequest) -> Order:
    return Order.model_validate(trading.submit_order(broker_request(order)))


def submit_keyed(trading: TradingClient, order: OrderRequest, key: str) -> Order:
    """Send `order` under a client order id; one already sent under it is read back, not doubled."""
    try:
        raw = trading.submit_order(broker_request(order, key))
    except APIError as exc:
        if "client_order_id" not in str(exc):
            raise
        raw = trading.get_order_by_client_id(key)
    return Order.model_validate(raw)


def get_order(trading: TradingClient, order_id: UUID | str) -> Order:
    return Order.model_validate(trading.get_order_by_id(order_id))


def change_order(connection: Connection, order_id: str, change: OrderChange) -> Order:
    """Change an open order's price or quantity through the node; it keeps its id."""
    row = node_order(connection, order_id)
    current = to_order(row)
    reason = rules.change_problem(current, change)
    if reason is None and any(getattr(change, name) is not None for name in NODE_CANNOT_CHANGE):
        reason = "The trading node changes price and quantity; cancel and place anew for the rest"
    if reason is not None:
        raise ChangeRefusedError(reason)
    payload = {"client_order_id": order_id, **change.model_dump(exclude_none=True)}
    commands.submit(connection, "modify", payload)
    return current.model_copy(update={"status": "pending_replace", "note": node_note(connection)})


def cancel_order(connection: Connection, order_id: str) -> Cancellation:
    """Ask the node to cancel; an order already finished is reported, not an error."""
    current = to_order(node_order(connection, order_id))
    if current.status in FINISHED:
        return Cancellation(
            id=current.id,
            status=current.status,
            canceled=current.status == "canceled",
            detail=f"Already {current.status}",
        )
    commands.submit(connection, "cancel", {"client_order_id": order_id})
    return Cancellation(
        id=current.id,
        status="pending_cancel",
        canceled=True,
        detail=node_note(connection) or "Cancel sent to the trading node",
    )


def cancel_all(connection: Connection, source: Source = "manual") -> CancelAll:
    """One command stops every algorithm and cancels every open order; counts are as of now."""
    working = node_state.orders(connection, 1000, open_only=True)
    algos = [row for row in working if row.is_primary]
    icebergs = [row for row in algos if row.exec_algorithm == "ICEBERG"]
    orders = [row for row in working if not row.is_primary and row.is_open]
    commands.submit(connection, "cancel_all", {}, source)
    return CancelAll(
        canceled=len(orders),
        failed=[],
        icebergs=len(icebergs),
        algos=len(algos) - len(icebergs),
    )


def cancel_open_orders(connection: Connection, symbols: Collection[str]) -> int:
    """Cancel the open orders for `symbols` (crypto with or without the slash); returns how many."""
    wanted = {symbol.replace("/", "") for symbol in symbols}
    working = node_state.orders(connection, 1000, open_only=True)
    stale = [row for row in working if row.is_open and row.symbol.replace("/", "") in wanted]
    for row in stale:
        commands.submit(connection, "cancel", {"client_order_id": row.client_order_id})
    return len(stale)
