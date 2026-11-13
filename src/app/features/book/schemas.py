from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.features.book.metrics import Pressure


class Level(BaseModel):
    price: float
    size: float
    total: float


class Reading(BaseModel):
    depth: int
    value: float | None
    pressure: Pressure | None


class Depth(BaseModel):
    bps: int
    bid: float
    ask: float
    bid_value: float
    ask_value: float


class FlowView(BaseModel):
    time: datetime
    ofi: float
    ofi_60: float
    windows: int


class FlowPoint(BaseModel):
    time: datetime
    span: float
    ofi: float


class BookView(BaseModel):
    symbol: str
    time: datetime
    mid: float
    spread: float
    spread_bps: float
    bids: list[Level]
    asks: list[Level]
    imbalance_levels: list[Reading]
    imbalance_bands: list[Reading]
    flow: FlowView | None
    spread_usual_bps: float | None
    spread_ratio: float | None
    depth: list[Depth]


class Impact(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    mid: float
    filled: float
    average: float | None
    slippage_bps: float | None
    levels: int
    complete: bool
    spread_bps: float
    spread_ratio: float | None
    max_qty: float | None
    advice: Literal["market", "limit"]
    reason: str | None


class BookState(BaseModel):
    symbol: str
    ready: bool
    bid_levels: int
    ask_levels: int


class BookStatus(BaseModel):
    feed_age: float
    stale: bool
    books: list[BookState]


class Point(BaseModel):
    time: datetime
    mid: float
    spread_bps: float
    imbalance: list[Reading]
