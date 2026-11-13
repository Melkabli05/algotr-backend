import logging
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, cast
from uuid import UUID
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from requests import RequestException

from app.features.orders.schemas import Order
from app.features.systematic.catalog import STRATEGIES, is_crypto
from app.features.systematic.schemas import (
    ExecutedOrder,
    ExecutionReport,
    ExecutionSummary,
    PlannedOrder,
    RunRecord,
)
from app.quant import history

logger = logging.getLogger(__name__)

DONE = ("filled", "canceled", "expired", "rejected", "replaced")
NOT_SENT = ("rejected", "denied", "skipped")
CRYPTO_TAKER_BPS = 25.0
BPS = 1e4
SETTLE_DAYS = 7
EXCHANGE = ZoneInfo("America/New_York")

Asset = Literal["stock", "crypto"]
OpenPrice = Callable[[str, date], float | None]


def asset_of(symbol: str) -> Asset:
    return "crypto" if "/" in symbol else "stock"


def assumed_bps(asset: Asset) -> float:
    """The most the backtest charges per trade for this asset class (`fee_bps` of its rules)."""
    fees = [spec.fee_bps for spec in STRATEGIES.values() if is_crypto(spec) == (asset == "crypto")]
    return max(fees)


def with_prices(
    done: Iterable[ExecutedOrder], planned: Iterable[PlannedOrder]
) -> list[ExecutedOrder]:
    """The orders sent, each with the decision price and limit of the planned order it came from."""
    prices: dict[tuple[str, str], PlannedOrder] = {(o.symbol, o.side): o for o in planned}
    found = []
    for item in done:
        plan = prices.get((item.symbol, item.side))
        found.append(
            item
            if plan is None
            else item.model_copy(update={"price": plan.price, "limit_price": plan.limit_price})
        )
    return found


def fetch(trading: TradingClient, order_id: str) -> Order:
    """The order by Alpaca's id, or by the client id Nautilus gave it."""
    try:
        UUID(order_id)
    except ValueError:
        return Order.model_validate(trading.get_order_by_client_id(order_id))
    return Order.model_validate(trading.get_order_by_id(order_id))


def signed_bps(side: str, filled: float, reference: float) -> float:
    """How far a fill is from a reference price, positive when it cost money."""
    sign = 1.0 if side == "buy" else -1.0
    return sign * (filled - reference) / reference * BPS


def bar_open(stocks: StockHistoricalDataClient, crypto: CryptoHistoricalDataClient) -> OpenPrice:
    """Reads a stock's opening price on a session from Alpaca's adjusted daily bars."""

    def open_on(symbol: str, day: date) -> float | None:
        start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        rows = history.bars(stocks, crypto, symbol, start, start + timedelta(days=1))
        for row in rows:
            if str(row["t"])[:10] == day.isoformat():
                return float(row["o"])
        return None

    return open_on


def settled(item: ExecutedOrder, order: Order, open_on: OpenPrice) -> ExecutedOrder | None:
    """The order with its fill and costs; None while it is working or its open is unknown."""
    if order.status not in DONE:
        return None
    update: dict[str, Any] = {"status": order.status, "filled_qty": order.filled_qty}
    price = order.filled_avg_price
    if not order.filled_qty or price is None:
        return item.model_copy(update={**update, "settled": True})
    stamp = order.filled_at or order.submitted_at
    if asset_of(item.symbol) == "stock":
        if stamp is None:
            return None
        model = open_on(item.symbol, stamp.astimezone(EXCHANGE).date())
        if model is None:
            return None
        fee = 0.0
    else:
        model, fee = item.price, CRYPTO_TAKER_BPS
    update |= {"filled_price": price, "filled_at": stamp, "model_price": model, "settled": True}
    if model:
        update["cost_bps"] = signed_bps(item.side, price, model) + fee
    if item.price:
        update["shortfall_bps"] = signed_bps(item.side, price, item.price) + fee
    return item.model_copy(update=update)


def settle(
    trading: TradingClient, runs: list[RunRecord], open_on: OpenPrice, now: datetime
) -> bool:
    """Fill in the outcome of every recent order that has finished; True when one changed.

    An order that could not be read is tried again on the next call.
    """
    changed = False
    for run in runs:
        if now - run.time > timedelta(days=SETTLE_DAYS):
            continue
        for index, item in enumerate(run.executed):
            pending = not item.settled and item.order_id and not item.status.startswith(NOT_SENT)
            if not pending:
                continue
            try:
                found = settled(item, fetch(trading, cast(str, item.order_id)), open_on)
            except (APIError, RequestException, KeyError, ValueError) as exc:
                logger.warning("Could not read order %s: %s", item.order_id, exc)
                continue
            if found is not None:
                run.executed[index] = found
                changed = True
    return changed


def summaries(runs: Iterable[RunRecord]) -> list[ExecutionSummary]:
    """Notional-weighted costs per asset class over every settled fill on record."""
    found = []
    fills = [item for run in runs for item in run.executed if item.filled_price is not None]
    for asset in ("stock", "crypto"):
        mine = [item for item in fills if asset_of(item.symbol) == asset]
        if not mine:
            continue
        weights = [(item.filled_qty or 0) * (item.filled_price or 0) for item in mine]
        total = sum(weights)
        found.append(
            ExecutionSummary(
                asset=asset,
                orders=len(mine),
                notional=total,
                cost_bps=weighted([i.cost_bps for i in mine], weights),
                shortfall_bps=weighted([i.shortfall_bps for i in mine], weights),
                assumed_bps=assumed_bps(asset),
            )
        )
    return found


def weighted(values: list[float | None], weights: list[float]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights, strict=True) if v is not None and w > 0]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def report(runs: Iterable[RunRecord]) -> ExecutionReport:
    runs = list(runs)
    fills = [item for run in runs for item in run.executed if item.settled]
    return ExecutionReport(fills=fills, summaries=summaries(runs))
