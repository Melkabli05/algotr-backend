from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Timeframe = Literal["1Min", "5Min", "15Min", "1Hour", "1Day"]


class Bar(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Clock(BaseModel):
    is_open: bool
    next_open: datetime
    next_close: datetime


class DataSource(BaseModel):
    feed: Literal["iex", "sip"]


class Quote(BaseModel):
    type: Literal["quote"] = "quote"
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    time: datetime


class Trade(BaseModel):
    type: Literal["trade"] = "trade"
    symbol: str
    price: float
    size: float
    time: datetime


class OrderUpdate(BaseModel):
    type: Literal["order_update"] = "order_update"
    event: str
    order_id: str
    symbol: str
    side: str
    status: str
    qty: float | None
    filled_qty: float
    filled_avg_price: float | None
    time: datetime
    execution_id: str | None = None
    price: float | None = None
    fill_qty: float | None = None
    position_qty: float | None = None


FILL_EVENTS = ("fill", "partial_fill")
StreamEvent = Quote | Trade | OrderUpdate
