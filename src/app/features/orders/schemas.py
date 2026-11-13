from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from app.core.types import Symbol
from app.features.orders import rules

OrderType = Literal["market", "limit", "stop", "stop_limit", "trailing_stop"]
OrderClass = Literal["simple", "bracket", "oto", "oco"]
TimeInForce = Literal["day", "gtc", "ioc", "fok"]


class Order(BaseModel):
    id: str
    symbol: str
    side: str
    type: str
    status: str
    qty: float | None
    filled_qty: float
    filled_avg_price: float | None
    submitted_at: datetime | None
    filled_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    canceled_at: datetime | None = None
    expired_at: datetime | None = None
    failed_at: datetime | None = None
    replaced_at: datetime | None = None
    client_order_id: str | None = None
    order_class: str = "simple"
    time_in_force: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    trail_price: float | None = None
    trail_percent: float | None = None
    hwm: float | None = None
    replaces: str | None = None
    replaced_by: str | None = None
    note: str | None = None

    @field_validator("order_class", mode="before")
    @classmethod
    def simple_when_blank(cls, value: object) -> object:
        return value or "simple"


class OrderRequest(BaseModel):
    symbol: Symbol
    side: Literal["buy", "sell"]
    type: OrderType
    qty: float = Field(gt=0)
    time_in_force: TimeInForce = "day"
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    trail_price: float | None = Field(default=None, gt=0)
    trail_percent: float | None = Field(default=None, gt=0)
    order_class: OrderClass = "simple"
    take_profit: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    stop_loss_limit: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def alpaca_would_accept_it(self) -> Self:
        reason = rules.problem(self)
        if reason is not None:
            raise ValueError(reason)
        return self


class OrderChange(BaseModel):
    """What to change on an open order; `trail` is the new dollars or percent of a trailing stop."""

    qty: float | None = Field(default=None, gt=0)
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    trail: float | None = Field(default=None, gt=0)
    time_in_force: TimeInForce | None = None

    @model_validator(mode="after")
    def changes_something(self) -> Self:
        if not self.model_dump(exclude_none=True):
            raise ValueError(
                "Give at least one of qty, limit_price, stop_price, trail, time_in_force"
            )
        return self


class Cancellation(BaseModel):
    id: str
    status: str
    canceled: bool
    detail: str | None = None


class CanceledOrder(BaseModel):
    id: str
    status: int


class CancelAll(BaseModel):
    canceled: int
    failed: list[CanceledOrder]
    icebergs: int
    algos: int = 0


class Rules(BaseModel):
    asset: str
    time_in_force: dict[str, list[str]]
    trail_percent: tuple[float, float]


class IcebergRequest(BaseModel):
    symbol: Symbol
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0)
    display_qty: float = Field(gt=0)
    limit_price: float = Field(gt=0)
    time_in_force: Literal["day", "gtc"] = "gtc"

    @model_validator(mode="after")
    def every_slice_is_a_limit_order_alpaca_takes(self) -> Self:
        if self.display_qty >= self.qty:
            raise ValueError("display_qty must be below qty; a limit order shows all of it")
        rest = round(self.qty % self.display_qty, 9)
        sizes = [self.display_qty, *([rest] if rest else [])]
        reasons = [rules.problem(self.slice(size)) for size in sizes]
        reasons.append(rules.iceberg_problem(self.symbol, self.qty, self.time_in_force))
        reason = next((found for found in reasons if found), None)
        if reason is not None:
            raise ValueError(reason)
        return self

    def slice(self, qty: float) -> OrderRequest:
        return OrderRequest.model_construct(
            symbol=self.symbol,
            side=self.side,
            type="limit",
            qty=qty,
            limit_price=self.limit_price,
            time_in_force=self.time_in_force,
        )


class Slice(BaseModel):
    order: str
    qty: float
    status: str = "new"
    filled_qty: float = 0.0
    average_price: float | None = None


Algorithm = Literal["twap", "vwap", "shortfall"]


class AlgoRequest(BaseModel):
    algo: Algorithm
    symbol: Symbol
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0)
    duration: int = Field(ge=1, le=rules.MAX_DURATION)
    slices: int | None = Field(default=None, ge=2, le=rules.MAX_SLICES)
    limit_price: float | None = Field(default=None, gt=0)
    tilt: float = Field(default=0.0, ge=0, le=rules.MAX_TILT)
    risk_aversion: float = Field(default=1.0, ge=0, le=rules.MAX_RISK)

    @model_validator(mode="after")
    def the_schedule_can_be_run(self) -> Self:
        reason = rules.algo_problem(self)
        if reason is not None:
            raise ValueError(reason)
        return self


class Interval(BaseModel):
    start: datetime
    weight: float


class PlannedSlice(BaseModel):
    due: datetime
    qty: float


class AlgoPlan(BaseModel):
    algo: str
    start: datetime
    end: datetime
    arrival_price: float | None
    intervals: list[Interval]
    slices: list[PlannedSlice]
    notes: dict[str, float | str] = {}


class Child(BaseModel):
    order: str
    interval: int
    qty: float
    status: str = "new"
    filled_qty: float = 0.0
    average_price: float | None = None
    cancel_sent: bool = False


class Algo(BaseModel):
    id: str
    created: datetime
    algo: Algorithm
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    duration: int
    limit_price: float | None = None
    tilt: float = 0.0
    risk_aversion: float = 1.0
    start: datetime
    end: datetime
    arrival_price: float | None = None
    intervals: list[Interval]
    next_interval: int = 0
    children: list[Child] = []
    status: Literal["working", "done", "ended", "canceled", "failed"] = "working"
    detail: str | None = None
    notes: dict[str, float | str] = {}

    @computed_field
    @property
    def filled_qty(self) -> float:
        return sum(child.filled_qty for child in self.children)

    @computed_field
    @property
    def average_price(self) -> float | None:
        paid = sum(child.filled_qty * (child.average_price or 0.0) for child in self.children)
        return paid / self.filled_qty if self.filled_qty else None

    @computed_field
    @property
    def slippage_bps(self) -> float | None:
        """The average fill against the arrival price in basis points; positive costs money."""
        if self.average_price is None or not self.arrival_price:
            return None
        gap = (self.average_price - self.arrival_price) / self.arrival_price * 1e4
        return gap if self.side == "buy" else -gap


class Iceberg(BaseModel):
    id: str
    created: datetime
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    display_qty: float
    limit_price: float
    time_in_force: Literal["day", "gtc"]
    status: Literal["working", "done", "canceled", "expired", "failed"] = "working"
    detail: str | None = None
    slices: list[Slice] = []

    @computed_field
    @property
    def filled_qty(self) -> float:
        return sum(item.filled_qty for item in self.slices)

    @computed_field
    @property
    def average_price(self) -> float | None:
        paid = sum(item.filled_qty * (item.average_price or 0.0) for item in self.slices)
        return paid / self.filled_qty if self.filled_qty else None
