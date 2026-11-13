from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, cast

from alpaca.trading.client import TradingClient
from pydantic import BaseModel, Field

from app.features.ledger.lots import Execution

PAGE = 100
CORPORATE_ACTIONS = ("SPLIT", "SPIN", "MA", "NC", "SC", "REORG", "SSP", "SSO")
FEE_RATE = Decimal("0.005")
FEE_DAYS = 3


class Activity(BaseModel):
    """One row of Alpaca's account activity: a fill, a fee in coin, or something else."""

    id: str
    activity_type: str
    transaction_time: datetime | None = None
    created_at: datetime | None = None
    day: date | None = Field(default=None, alias="date")
    symbol: str | None = None
    side: str | None = None
    qty: Decimal | None = None
    price: Decimal | None = None
    order_id: str | None = None
    net_amount: Decimal | None = None
    fill: str | None = Field(default=None, alias="type")

    @property
    def moment(self) -> datetime:
        if self.transaction_time is not None:
            return self.transaction_time
        if self.created_at is not None:
            return self.created_at
        return datetime.combine(self.day or date.min, time(), tzinfo=UTC)

    @property
    def key(self) -> str:
        return (self.symbol or "").replace("/", "")


def fetch_all(trading: TradingClient) -> list[dict[str, Any]]:
    """Every activity, oldest first, paged by the id of the last row."""
    rows: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        params: dict[str, Any] = {"page_size": PAGE, "direction": "asc"}
        if token:
            params["page_token"] = token
        batch = cast(list[dict[str, Any]], trading.get("/account/activities", params))
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        token = batch[-1]["id"]


def fetch_day(trading: TradingClient, day: date) -> list[dict[str, Any]]:
    """The activities dated `day`; fees in coin are dated the day after their fills."""
    params = {"date": day.isoformat(), "direction": "asc"}
    return cast(list[dict[str, Any]], trading.get("/account/activities", params))


def coin_fees(rows: Iterable[Activity]) -> dict[str, Decimal]:
    """Each coin fee on the latest buy fill before it: the fee posts seconds after its fill."""
    rows = list(rows)
    buys: dict[str, list[Activity]] = defaultdict(list)
    for fill in sorted((a for a in rows if is_fill(a) and a.side == "buy"), key=moment_of):
        buys[fill.key].append(fill)
    moments = {key: [fill.moment for fill in found] for key, found in buys.items()}
    fees: dict[str, Decimal] = defaultdict(Decimal)
    for fee in (a for a in rows if a.activity_type == "CFEE" and a.symbol and a.qty):
        found = bisect_right(moments.get(fee.key, []), fee.moment) - 1
        if found >= 0 and fee.qty is not None:
            fees[buys[fee.key][found].id] += abs(fee.qty)
    return dict(fees)


def is_fill(row: Activity) -> bool:
    return row.activity_type == "FILL" and bool(row.order_id and row.qty and row.price)


def moment_of(row: Activity) -> datetime:
    return row.moment


def executions(activities: Iterable[Activity]) -> list[Execution]:
    """Fills grouped by order, each buy net of the coin fee that came with it."""
    rows = list(activities)
    fees = coin_fees(rows)
    grouped: dict[str, list[Activity]] = defaultdict(list)
    for fill in sorted((a for a in rows if is_fill(a)), key=moment_of):
        grouped[fill.order_id or ""].append(fill)
    result = []
    for order_id, group in grouped.items():
        total = sum((f.qty or Decimal(0) for f in group), Decimal(0))
        paid = sum(((f.qty or Decimal(0)) * (f.price or Decimal(0)) for f in group), Decimal(0))
        side = group[0].side
        in_kind = sum((fees.get(fill.id, Decimal(0)) for fill in group), Decimal(0))
        result.append(
            Execution(
                order=order_id,
                symbol=group[0].key,
                kind="buy" if side == "buy" else "sell",
                time=group[0].moment,
                qty=total - in_kind if side == "buy" else total,
                price=paid / total,
                paid=paid if side == "buy" else None,
            )
        )
    return result


def pending_fees(rows: Iterable[Activity], now: datetime) -> dict[str, Decimal]:
    """The most coin fee that recent buys can still owe: Alpaca posts the fee row a day later."""
    owed: dict[str, Decimal] = defaultdict(Decimal)
    for fill in rows:
        coin_buy = (
            fill.activity_type == "FILL" and fill.side == "buy" and "/" in (fill.symbol or "")
        )
        if coin_buy and fill.qty and now - fill.moment < timedelta(days=FEE_DAYS):
            owed[fill.key] += fill.qty * FEE_RATE
    return dict(owed)


def corporate_actions(rows: Iterable[Activity]) -> dict[str, str]:
    """The latest corporate action on record for each symbol."""
    found: dict[str, str] = {}
    for row in sorted(rows, key=lambda activity: activity.moment):
        if row.activity_type in CORPORATE_ACTIONS and row.symbol:
            found[row.key] = row.activity_type
    return found
