from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Method = Literal["fifo", "average", "specific"]
Side = Literal["long", "short"]


class LotRow(BaseModel):
    id: str
    symbol: str
    side: Side
    opened: datetime
    days: int
    qty: float
    original: float
    unit_cost: float
    basis: float
    wash: float = 0.0
    price: float | None
    value: float | None
    unrealized: float | None
    unrealized_pct: float | None


class CloseRow(BaseModel):
    lot: str | None
    symbol: str
    side: Side
    opened: datetime
    closed: datetime
    days: int
    order: str
    qty: float
    basis: float
    proceeds: float
    gain: float
    gain_pct: float | None
    disallowed: float = 0.0
    reported: float = 0.0


class RealizedSummary(BaseModel):
    """`gain` is the economic result; `reported` is it less losses the wash-sale rule disallows."""

    count: int
    basis: float
    proceeds: float
    gain: float
    disallowed: float = 0.0
    reported: float = 0.0


class Realized(BaseModel):
    summary: RealizedSummary
    rows: list[CloseRow]


class Holding(BaseModel):
    symbol: str
    qty: float
    basis: float
    average_cost: float
    price: float | None
    value: float | None
    unrealized: float | None
    unrealized_pct: float | None
    realized: float
    lots: int


class FillRow(BaseModel):
    id: str
    time: datetime
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    price: float
    value: float
    partial: bool
    order: str
    fee_qty: float | None


class Difference(BaseModel):
    symbol: str
    kind: Literal["quantity", "pending", "cost"]
    ledger_qty: float
    broker_qty: float
    ledger_cost: float | None
    broker_cost: float | None
    note: str


Trigger = Literal["scheduled", "fill", "manual"]


class Reconciliation(BaseModel):
    checked: datetime
    synced: datetime | None
    activities: int
    ok: bool
    differences: list[Difference]
    trigger: Trigger | None = None


class Compared(BaseModel):
    """One symbol in the ledger, at Alpaca, or both, side by side."""

    symbol: str
    state: Literal["match", "quantity", "pending", "cost", "orphan", "zombie"]
    ledger_qty: float
    broker_qty: float
    delta: float
    ledger_cost: float | None
    broker_cost: float | None
    ledger_pnl: float | None
    broker_pnl: float | None
    note: str


class Cursor(BaseModel):
    """The newest Alpaca activity stored: where the next sync carries on from."""

    id: str
    time: datetime | None


class Comparison(BaseModel):
    reconciliation: Reconciliation | None
    rows: list[Compared]
    cursor: Cursor | None
    error: str | None


class Status(BaseModel):
    activities: int
    synced: datetime | None
    error: str | None
    reconciliation: Reconciliation | None


class Pick(BaseModel):
    lot: str
    qty: float = Field(gt=0)


class Picks(BaseModel):
    lots: list[Pick]
