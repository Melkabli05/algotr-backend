from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Var(BaseModel):
    confidence: float
    horizon: int
    historical: float
    parametric: float
    shortfall: float
    dollars: float


class Exposure(BaseModel):
    name: str
    value: float
    share: float


class Correlation(BaseModel):
    symbols: list[str]
    matrix: list[list[float]]


class Limit(BaseModel):
    name: str
    value: float | None
    threshold: float
    used: float | None
    unit: str
    scope: Literal["account", "order"] = "account"
    last_triggered: datetime | None


class RiskEvent(BaseModel):
    time: datetime
    severity: str
    title: str
    detail: str | None = None


class RiskSummary(BaseModel):
    equity: float
    days: int
    var: list[Var]
    sectors: list[Exposure]
    strategies: list[Exposure]
    concentration: list[Exposure]
    correlation: Correlation
    limits: list[Limit]
    events: list[RiskEvent]
