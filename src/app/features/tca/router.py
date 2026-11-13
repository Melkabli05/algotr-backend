from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from alpaca.common.enums import Sort
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from fastapi import APIRouter, Query, Request

from app.core.broker import CryptoData, StockData, Trading
from app.features.orders.schemas import Order
from app.features.tca.service import Report, report

router = APIRouter(prefix="/tca", tags=["tca"])
ORDERS = 500


@router.get("")
def read_tca(
    request: Request,
    trading: Trading,
    stocks: StockData,
    crypto: CryptoData,
    days: Annotated[int, Query(ge=1, le=90)] = 30,
) -> Report:
    """Every filled order against its decision, arrival, TWAP and day VWAP prices, with totals."""
    state: Any = request.app.state
    since = datetime.now(UTC) - timedelta(days=days)
    found = trading.get_orders(
        GetOrdersRequest(
            status=QueryOrderStatus.ALL, after=since, limit=ORDERS, direction=Sort.DESC
        )
    )
    orders = {item.id: item for item in (Order.model_validate(raw) for raw in found)}
    decided = {
        executed.order_id: executed.price
        for run in state.automation.state.runs
        for executed in run.executed
        if executed.order_id and executed.price
    }
    algos = {item.id: item.algo for item in state.algos.algos()}
    fills = state.ledger.fills(days=days, limit=5000)
    return report(fills, orders, decided, algos, stocks, crypto, days)
