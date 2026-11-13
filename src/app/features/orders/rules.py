from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.features.orders.schemas import AlgoRequest, Order, OrderChange, OrderRequest

AssetClass = Literal["stock", "fractional", "crypto"]

TRAIL_PERCENT = (0.1, 25.0)
MIN_TICK = 0.01
MAX_DURATION = 390
MAX_SLICES = 100
MAX_TILT = 0.5
MAX_RISK = 100.0
MIN_GAP_SECONDS = 10
MIN_VWAP_MINUTES = 10
ALGOS = {"twap": "TWAP", "vwap": "VWAP", "shortfall": "Shortfall"}

DAY_GTC = ("day", "gtc")
EVERY = ("day", "gtc", "ioc", "fok")
COINS = ("gtc", "ioc")
IMMEDIATE = ("ioc",)

TIME_IN_FORCE: dict[AssetClass, dict[str, tuple[str, ...]]] = {
    "stock": {
        "market": EVERY,
        "limit": EVERY,
        "stop": DAY_GTC,
        "stop_limit": DAY_GTC,
        "trailing_stop": DAY_GTC,
        "bracket": DAY_GTC,
        "oto": DAY_GTC,
        "oco": DAY_GTC,
        "iceberg": DAY_GTC,
        "twap": IMMEDIATE,
        "vwap": IMMEDIATE,
        "shortfall": IMMEDIATE,
    },
    "fractional": dict.fromkeys(("market", "limit", "stop", "stop_limit", "iceberg"), ("day",)),
    "crypto": {
        "market": COINS,
        "limit": COINS,
        "stop_limit": COINS,
        "iceberg": ("gtc",),
        "twap": IMMEDIATE,
        "vwap": IMMEDIATE,
    },
}

KINDS = {
    "market": "Market",
    "limit": "Limit",
    "stop": "Stop",
    "stop_limit": "Stop-limit",
    "trailing_stop": "Trailing stop",
    "bracket": "Bracket",
    "oto": "OTO",
    "oco": "OCO",
    "iceberg": "Iceberg",
    "twap": "TWAP",
    "vwap": "VWAP",
    "shortfall": "Shortfall",
}
ASSETS: dict[AssetClass, str] = {
    "stock": "whole shares",
    "fractional": "fractional shares",
    "crypto": "crypto",
}
PRICE_FIELDS = {
    "market": (),
    "limit": ("limit_price",),
    "stop": ("stop_price",),
    "stop_limit": ("stop_price", "limit_price"),
    "trailing_stop": ("trail_price", "trail_percent"),
}
REQUIRED_FIELDS = {
    "limit": ("limit_price",),
    "stop": ("stop_price",),
    "stop_limit": ("stop_price", "limit_price"),
}
ENTRY_TYPES = ("market", "limit")


def asset_class(symbol: str, qty: float) -> AssetClass:
    if "/" in symbol:
        return "crypto"
    return "stock" if float(qty).is_integer() else "fractional"


def kind_of(order: "OrderRequest") -> str:
    return order.type if order.order_class == "simple" else order.order_class


def problem(order: "OrderRequest") -> str | None:
    """The first reason Alpaca would refuse `order`, or None; checked before anything is sent."""
    return (
        availability_problem(order)
        or exit_problem(order)
        or price_problem(order)
        or time_in_force_problem(order)
    )


def availability_problem(order: "OrderRequest") -> str | None:
    asset, kind = asset_class(order.symbol, order.qty), kind_of(order)
    if kind not in TIME_IN_FORCE[asset]:
        return f"{KINDS[kind]} orders are not available for {ASSETS[asset]}"
    return None


def price_problem(order: "OrderRequest") -> str | None:
    allowed = PRICE_FIELDS[order.type]
    required = () if order.order_class == "oco" else REQUIRED_FIELDS.get(order.type, ())
    for name in required:
        if getattr(order, name) is None:
            return f"{name} is required for {order.type} orders"
    for name in ("limit_price", "stop_price", "trail_price", "trail_percent"):
        if getattr(order, name) is not None and name not in allowed:
            return f"{name} does not apply to {order.type} orders"
    if order.type != "trailing_stop":
        return None
    if (order.trail_price is None) == (order.trail_percent is None):
        return "A trailing stop needs exactly one of trail_price and trail_percent"
    low, high = TRAIL_PERCENT
    if order.trail_percent is not None and not low <= order.trail_percent <= high:
        return f"trail_percent must be between {low:g} and {high:g}"
    return None


def exit_problem(order: "OrderRequest") -> str | None:
    profit, loss, loss_limit = order.take_profit, order.stop_loss, order.stop_loss_limit
    if order.order_class == "simple":
        if profit is not None or loss is not None or loss_limit is not None:
            return "take_profit and stop_loss need order_class bracket, oto or oco"
        return None
    if loss_limit is not None and loss is None:
        return "stop_loss_limit needs stop_loss"
    if loss_limit is not None and loss is not None and loss_limit > loss:
        return "stop_loss_limit cannot be above stop_loss"
    if order.order_class == "oco":
        return oco_problem(order)
    if order.side != "buy":
        return f"{KINDS[order.order_class]} orders open a position, so they must be buys"
    if order.type not in ENTRY_TYPES:
        return f"{KINDS[order.order_class]} orders enter with a market or limit order"
    if order.order_class == "bracket" and (profit is None or loss is None):
        return "A bracket needs take_profit and stop_loss"
    if order.order_class == "oto" and (profit is None) == (loss is None):
        return "An OTO order needs one of take_profit and stop_loss"
    return bracket_prices(order)


def oco_problem(order: "OrderRequest") -> str | None:
    if order.side != "sell":
        return "An OCO order closes a position, so it must be a sell"
    if order.type != "limit":
        return "An OCO order is type limit"
    if order.take_profit is None or order.stop_loss is None:
        return "An OCO order needs take_profit and stop_loss"
    if order.limit_price not in (None, order.take_profit):
        return "An OCO order's limit_price is its take_profit"
    if order.take_profit < order.stop_loss + MIN_TICK:
        return "take_profit must be above stop_loss"
    return None


def bracket_prices(order: "OrderRequest") -> str | None:
    profit, loss = order.take_profit, order.stop_loss
    if profit is not None and loss is not None and profit < loss + MIN_TICK:
        return "take_profit must be above stop_loss"
    entry = order.limit_price
    if entry is None:
        return None
    if profit is not None and profit < entry + MIN_TICK:
        return "take_profit must be above the entry price"
    if loss is not None and loss > entry - MIN_TICK:
        return "stop_loss must be below the entry price"
    return None


def time_in_force_problem(order: "OrderRequest") -> str | None:
    asset, kind = asset_class(order.symbol, order.qty), kind_of(order)
    allowed = TIME_IN_FORCE[asset][kind]
    if order.time_in_force not in allowed:
        return f"{KINDS[kind]} orders for {ASSETS[asset]} accept {', '.join(allowed)}"
    return None


def iceberg_problem(symbol: str, qty: float, time_in_force: str) -> str | None:
    asset = asset_class(symbol, qty)
    allowed = TIME_IN_FORCE[asset]["iceberg"]
    if time_in_force not in allowed:
        return f"Iceberg orders for {ASSETS[asset]} accept {', '.join(allowed)}"
    return None


CHANGEABLE = ("new", "partially_filled", "held")
WAITING = ("accepted", "pending_new", "pending_cancel", "pending_replace")
CHANGE_FIELDS = {
    "market": (),
    "limit": ("limit_price",),
    "stop": ("stop_price",),
    "stop_limit": ("stop_price", "limit_price"),
    "trailing_stop": ("trail",),
}


def change_problem(order: "Order", change: "OrderChange") -> str | None:
    """The first reason Alpaca would refuse `change` to `order` as it stands now, or None."""
    if order.status in WAITING:
        return f"The order is {order.status}; it can be changed once it is new"
    if order.status not in CHANGEABLE:
        return f"The order is {order.status}, so it can no longer be changed"
    given = change.model_dump(exclude_none=True)
    allowed = (*CHANGE_FIELDS.get(order.type, ()), "qty", "time_in_force")
    for name in given:
        if name not in allowed:
            return f"{name} does not apply to a {order.type.replace('_', ' ')} order"
    return qty_change_problem(order, change) or change_time_in_force_problem(order, change)


def qty_change_problem(order: "Order", change: "OrderChange") -> str | None:
    if change.qty is None:
        return None
    symbol = order.symbol
    if "/" not in symbol and not float(order.qty or 0).is_integer():
        return "The quantity of a fractional order cannot change"
    if "/" not in symbol and not float(change.qty).is_integer():
        return "A stock order can only change to a whole number of shares"
    if change.qty <= order.filled_qty:
        return f"The quantity must stay above the {order.filled_qty:g} already filled"
    return None


def change_time_in_force_problem(order: "Order", change: "OrderChange") -> str | None:
    if change.time_in_force is None:
        return None
    asset = asset_class(order.symbol, order.qty or 0)
    kind = order.order_class if order.order_class != "simple" else order.type
    allowed = TIME_IN_FORCE[asset].get(kind, ())
    if change.time_in_force not in allowed:
        return f"{KINDS.get(kind, kind)} orders for {ASSETS[asset]} accept {', '.join(allowed)}"
    return None


def algo_problem(order: "AlgoRequest") -> str | None:
    """The first reason an algorithm order could not run as asked, or None."""
    asset = asset_class(order.symbol, order.qty)
    name = ALGOS[order.algo]
    if asset == "fractional":
        return f"A {name} order trades whole shares"
    if asset == "crypto" and order.algo == "shortfall":
        return "Implementation shortfall is for stocks"
    if order.tilt and order.algo != "twap":
        return "Only a TWAP follows the quote imbalance"
    if order.algo == "vwap":
        return vwap_problem(order)
    return slices_problem(order, asset, name)


def vwap_problem(order: "AlgoRequest") -> str | None:
    if order.slices is not None:
        return "A VWAP follows the volume profile; leave slices out"
    if order.duration < MIN_VWAP_MINUTES:
        return f"A VWAP needs at least {MIN_VWAP_MINUTES} minutes"
    return None


def slices_problem(order: "AlgoRequest", asset: AssetClass, name: str) -> str | None:
    if order.slices is None:
        return f"A {name} needs slices"
    if asset == "stock" and order.slices > order.qty:
        return f"Each slice needs a share; use at most {int(order.qty)} slices"
    if order.duration * 60 / order.slices < MIN_GAP_SECONDS:
        most = int(order.duration * 60 / MIN_GAP_SECONDS)
        return f"Slices must be {MIN_GAP_SECONDS} seconds apart or more; use at most {most}"
    return None
