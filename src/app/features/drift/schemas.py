from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

Kind = Literal["mean", "volatility"]
Action = Literal["off", "log", "act"]


class DriftEvent(BaseModel):
    stream: str
    time: date
    kind: Kind
    adverse: bool
    distance: float
    p_value: float
    window: int
    scale: float | None = None
    retrain: str | None = None


class StreamStatus(BaseModel):
    name: str
    kind: str
    monitored: bool
    note: str = ""
    days: int
    first: date | None = None
    last: date | None = None
    distance: float | None = None
    scale: float = 1.0
    until: date | None = None
    events: list[DriftEvent] = []


class DriftStatus(BaseModel):
    action: Action
    threshold: float
    significance: float
    scale: float
    hold_days: int
    baseline_days: int
    started: date | None
    checked: datetime | None
    streams: list[StreamStatus]
    recorded: list[DriftEvent]
