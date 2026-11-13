from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal, cast

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from cachetools import TTLCache, cached
from pydantic import BaseModel

BPS = 1e4
Source = Literal["autopilot", "algo", "iceberg", "manual"]
BARS: TTLCache[tuple[str, date], list[dict[str, Any]]] = TTLCache(maxsize=256, ttl=6 * 3600)
DAYS: TTLCache[tuple[str, date], float | None] = TTLCache(maxsize=256, ttl=6 * 3600)


class Trade(BaseModel):
    """One order's fills against the prices it could have had."""

    order: str
    time: datetime
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    notional: float
    source: Source
    algo: str | None
    decision: float | None
    arrival: float | None
    fill: float
    vwap: float | None
    twap: float | None
    seconds_to_fill: float | None
    timing_bps: float | None
    impact_bps: float | None
    shortfall_bps: float | None
    vs_vwap_bps: float | None
    vs_twap_bps: float | None
    fee_bps: float | None


class Summary(BaseModel):
    name: str
    trades: int
    notional: float
    timing_bps: float | None
    impact_bps: float | None
    shortfall_bps: float | None
    vs_vwap_bps: float | None
    vs_twap_bps: float | None
    fee_bps: float | None


class Report(BaseModel):
    days: int
    total: Summary
    sources: list[Summary]
    algorithms: list[Summary]
    symbols: list[Summary]
    trades: list[Trade]
    notes: list[str]


@dataclass
class Filled:
    order: str
    symbol: str
    side: str
    qty: float
    paid: float
    first: datetime
    last: datetime
    fee_qty: float


def cost(side: str, price: float | None, benchmark: float | None) -> float | None:
    """Basis points paid against a benchmark: positive costs money on either side."""
    if price is None or not benchmark:
        return None
    gap = (price - benchmark) / benchmark * BPS
    return gap if side == "buy" else -gap


def grouped(fills: Iterable[Any]) -> list[Filled]:
    orders: dict[str, Filled] = {}
    for fill in fills:
        key = fill.order or fill.id
        found = orders.get(key)
        if found is None:
            orders[key] = Filled(
                key,
                fill.symbol,
                fill.side,
                fill.qty,
                fill.qty * fill.price,
                fill.time,
                fill.time,
                fill.fee_qty or 0.0,
            )
            continue
        found.qty += fill.qty
        found.paid += fill.qty * fill.price
        found.first = min(found.first, fill.time)
        found.last = max(found.last, fill.time)
        found.fee_qty += fill.fee_qty or 0.0
    return list(orders.values())


def source_of(
    client_id: str | None, order: str, decided: Mapping[str, float]
) -> tuple[Source, str | None]:
    if order in decided:
        return "autopilot", None
    if client_id and client_id.startswith("algo-"):
        return "algo", client_id.split("-")[1]
    if client_id and client_id.startswith("iceberg-"):
        return "iceberg", None
    return "manual", None


def pair(symbol: str) -> str:
    if "/" in symbol or not symbol.endswith("USD") or len(symbol) <= 3:
        return symbol
    return f"{symbol.removesuffix('USD')}/USD"


@cached(BARS, key=lambda stocks, crypto, symbol, day: (symbol, day))
def minute_bars(
    stocks: StockHistoricalDataClient, crypto: CryptoHistoricalDataClient, symbol: str, day: date
) -> list[dict[str, Any]]:
    start = datetime.combine(day, time(), tzinfo=UTC)
    end = start + timedelta(days=1)
    if "/" in symbol:
        found = crypto.get_crypto_bars(
            CryptoBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end
            )
        )
    else:
        found = stocks.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=min(end, datetime.now(UTC) - timedelta(minutes=16)),
                adjustment=Adjustment.RAW,
                feed=DataFeed.SIP,
            )
        )
    return cast(dict[str, list[dict[str, Any]]], found).get(symbol, [])


@cached(DAYS, key=lambda stocks, crypto, symbol, day: (symbol, day))
def day_vwap(
    stocks: StockHistoricalDataClient, crypto: CryptoHistoricalDataClient, symbol: str, day: date
) -> float | None:
    rows = minute_bars(stocks, crypto, symbol, day)
    volume = sum(float(row["v"]) for row in rows)
    if not volume:
        return None
    return sum(float(row.get("vw") or row["c"]) * float(row["v"]) for row in rows) / volume


def moment(row: Mapping[str, Any]) -> datetime:
    value = row["t"]
    return (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )


def arrival_and_twap(
    rows: list[dict[str, Any]], submitted: datetime, filled: datetime
) -> tuple[float | None, float | None]:
    """The open of the minute the order arrived in, and the mean close over its life."""
    minute = submitted.replace(second=0, microsecond=0)
    arrival = next((float(row["o"]) for row in rows if moment(row) >= minute), None)
    life = [float(row["c"]) for row in rows if minute <= moment(row) <= filled]
    return arrival, (sum(life) / len(life) if life else None)


def weighted(trades: list[Trade], name: str) -> Summary:
    notional = sum(trade.notional for trade in trades)

    def mean(field: str) -> float | None:
        pairs = [(getattr(trade, field), trade.notional) for trade in trades]
        known = [(value, weight) for value, weight in pairs if value is not None]
        total = sum(weight for _, weight in known)
        return sum(value * weight for value, weight in known) / total if total else None

    return Summary(
        name=name,
        trades=len(trades),
        notional=notional,
        timing_bps=mean("timing_bps"),
        impact_bps=mean("impact_bps"),
        shortfall_bps=mean("shortfall_bps"),
        vs_vwap_bps=mean("vs_vwap_bps"),
        vs_twap_bps=mean("vs_twap_bps"),
        fee_bps=mean("fee_bps"),
    )


def by(trades: list[Trade], key: Any) -> list[Summary]:
    groups: dict[str, list[Trade]] = {}
    for trade in trades:
        name = key(trade)
        if name is not None:
            groups.setdefault(name, []).append(trade)
    return sorted(
        (weighted(items, name) for name, items in groups.items()), key=lambda item: -item.notional
    )


def report(
    fills: Iterable[Any],
    orders: Mapping[str, Any],
    decided: Mapping[str, float],
    algos: Mapping[str, str],
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    days: int,
) -> Report:
    trades = []
    for item in grouped(fills):
        order = orders.get(item.order)
        symbol = pair(item.symbol)
        submitted = (order.submitted_at or order.created_at) if order else None
        source, algo_id = source_of(order.client_order_id if order else None, item.order, decided)
        fill = item.paid / item.qty if item.qty else 0.0
        arrival = twap = vwap = None
        if submitted is not None:
            try:
                rows = minute_bars(stocks, crypto, symbol, submitted.astimezone(UTC).date())
                arrival, twap = arrival_and_twap(rows, submitted, item.last)
                vwap = day_vwap(stocks, crypto, symbol, submitted.astimezone(UTC).date())
            except Exception:
                arrival = twap = vwap = None
        decision = decided.get(item.order)
        trades.append(
            Trade(
                order=item.order,
                time=item.first,
                symbol=symbol,
                side="buy" if item.side == "buy" else "sell",
                qty=item.qty,
                notional=item.paid,
                source=source,
                algo=algos.get(algo_id or "") if algo_id else None,
                decision=decision,
                arrival=arrival,
                fill=fill,
                vwap=vwap,
                twap=twap,
                seconds_to_fill=(item.last - submitted).total_seconds() if submitted else None,
                timing_bps=cost(item.side, arrival, decision),
                impact_bps=cost(item.side, fill, arrival),
                shortfall_bps=cost(item.side, fill, decision),
                vs_vwap_bps=cost(item.side, fill, vwap),
                vs_twap_bps=cost(item.side, fill, twap),
                fee_bps=item.fee_qty / item.qty * BPS if item.qty and item.fee_qty else None,
            )
        )
    trades.sort(key=lambda trade: trade.time, reverse=True)
    return Report(
        days=days,
        total=weighted(trades, "All trades"),
        sources=by(trades, lambda trade: trade.source),
        algorithms=by(trades, lambda trade: trade.algo),
        symbols=by(trades, lambda trade: trade.symbol)[:20],
        trades=trades,
        notes=[
            "Positive basis points cost money, for buys and sells alike.",
            "Arrival is the open of the minute the order reached Alpaca; decision is the "
            "autopilot's price when it planned the trade, so timing cost is the night between.",
            "Alpaca does not report the venue of a fill, so there is no venue breakdown.",
            "Stock trades are commission-free; coin fees are Alpaca's, taken in the coin.",
        ],
    )
