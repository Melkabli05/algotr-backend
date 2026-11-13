from datetime import date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest
from pydantic import BaseModel

from app.features.scanner.news import MAX_TAGS_FOR_VETO, NewsFeed, NewsStore

EASTERN = ZoneInfo("America/New_York")
LEAD = timedelta(minutes=45)
ORDER_LIMIT = 500


class Cancelled(BaseModel):
    symbol: str
    order_id: str
    label: str
    headline: str


class Review(BaseModel):
    time: datetime
    opens: datetime | None = None
    checked: int = 0
    cancelled: list[Cancelled] = []


def opening(trading: TradingClient, day: date) -> datetime | None:
    """When the US market opens on `day`, from Alpaca's calendar, or None on a closed day."""
    found = cast(list[dict[str, Any]], trading.get_calendar(GetCalendarRequest(start=day, end=day)))
    if not found or found[0]["date"] != day.isoformat():
        return None
    return datetime.combine(day, time.fromisoformat(found[0]["open"]), tzinfo=EASTERN)


def due(trading: TradingClient, now: datetime, reviewed: date | None) -> bool:
    """Whether the market opens within 45 minutes and today's review has not run."""
    today = now.astimezone(EASTERN).date()
    if reviewed == today:
        return False
    opens = opening(trading, today)
    return opens is not None and opens - LEAD <= now < opens


def run(trading: TradingClient, feed: NewsFeed, store: NewsStore, now: datetime) -> Review:
    """Cancel queued stock buys whose company has had a serious setback since they were placed."""
    queued = cast(
        list[dict[str, Any]],
        trading.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, side=OrderSide.BUY, limit=ORDER_LIMIT)
        ),
    )
    stocks = [order for order in queued if "/" not in order["symbol"]]
    review = Review(time=now, opens=opening(trading, now.astimezone(EASTERN).date()))
    review.checked = len(stocks)
    symbols = sorted({order["symbol"] for order in stocks})
    if not symbols:
        return review
    feed.poll(symbols, now)
    for order in stocks:
        since = datetime.fromisoformat(order["created_at"])
        recent = store.of([order["symbol"]], since).get(order["symbol"], [])
        setbacks = [h for h in recent if h.event.veto and h.tags <= MAX_TAGS_FOR_VETO]
        if setbacks:
            trading.cancel_order_by_id(order["id"])
            review.cancelled.append(
                Cancelled(
                    symbol=order["symbol"],
                    order_id=order["id"],
                    label=setbacks[0].event.label,
                    headline=setbacks[0].headline,
                )
            )
    return review
