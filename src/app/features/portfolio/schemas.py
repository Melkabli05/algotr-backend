from datetime import datetime
from typing import Literal

from pydantic import BaseModel, computed_field

Period = Literal["1D", "1W", "1M", "3M", "1A"]


class Account(BaseModel):
    status: str
    currency: str
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    trading_blocked: bool
    paper: bool

    @computed_field
    @property
    def day_pl(self) -> float:
        return self.equity - self.last_equity

    @computed_field
    @property
    def day_pl_pct(self) -> float:
        return self.day_pl / self.last_equity * 100 if self.last_equity else 0.0


class Position(BaseModel):
    symbol: str
    asset_class: str | None = None
    side: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    cost_basis: float | None = None
    unrealized_plpc: float | None = None
    unrealized_intraday_pl: float | None = None
    qty_available: float | None = None


class EquityPoint(BaseModel):
    time: datetime
    equity: float


class PortfolioHistory(BaseModel):
    timestamp: list[int]
    equity: list[float | None]
