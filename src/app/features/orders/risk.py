import logging
from typing import Any, cast

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoLatestTradeRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from pydantic import BaseModel
from requests import RequestException

from app.core.book import BookGuard
from app.features.orders.schemas import Order, OrderRequest

logger = logging.getLogger(__name__)

OPEN_ORDER_LIMIT = 500
TOLERANCE = 0.01


class RiskRefusedError(ValueError):
    """Raised when an order breaks a rule of this account rather than one of Alpaca's."""


class Funds(BaseModel):
    cash: float


def entry_price(order: OrderRequest) -> float | None:
    """The price per unit an order names, if it does: a market or trailing order names none."""
    return order.limit_price or order.stop_price


def last_price(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbol: str,
    feed: DataFeed,
) -> float | None:
    try:
        if "/" in symbol:
            found = crypto.get_crypto_latest_trade(
                CryptoLatestTradeRequest(symbol_or_symbols=symbol)
            )
        else:
            found = stocks.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed)
            )
        return float(cast(dict[str, Any], found)[symbol]["p"])
    except (APIError, RequestException, KeyError, TypeError, ValueError) as exc:
        logger.warning("The last price of %s could not be read: %s", symbol, exc)
        return None


def reserved(trading: TradingClient) -> float:
    """What the open buy orders with a price will spend."""
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=OPEN_ORDER_LIMIT)
    total = 0.0
    for raw in trading.get_orders(request):
        item = Order.model_validate(raw)
        price = item.limit_price or item.stop_price
        if item.side == "buy" and price and item.qty:
            total += (item.qty - item.filled_qty) * price
    return total


def check_cash(
    trading: TradingClient,
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    *,
    feed: DataFeed,
    symbol: str,
    side: str,
    qty: float,
    price: float | None,
) -> None:
    """Refuse a buy that costs more than the cash left after the open buy orders: no margin.

    A buy whose cost cannot be priced is refused too, since it could not be shown to fit.
    """
    if side != "buy":
        return
    unit = price or last_price(stocks, crypto, symbol, feed)
    if unit is None:
        raise RiskRefusedError(
            f"The price of {symbol} could not be read, so this buy cannot be checked against "
            "the cash; give it a limit price or try again"
        )
    cost = qty * unit
    left = Funds.model_validate(trading.get_account()).cash - reserved(trading)
    if cost > left + TOLERANCE:
        raise RiskRefusedError(
            f"A buy costing ${cost:,.2f} is more than the ${max(left, 0):,.2f} of cash "
            "left after open buy orders; this account does not borrow"
        )


def check_book(guard: BookGuard, order: OrderRequest) -> None:
    """Refuse a market order on a coin that its order book says would cost too much."""
    if order.type != "market" or "/" not in order.symbol:
        return
    reason = guard.refusal(order.symbol, order.side, order.qty)
    if reason is not None:
        raise RiskRefusedError(reason)
