from datetime import date
from typing import Protocol

from pydantic import BaseModel, computed_field


class Holding(BaseModel):
    symbol: str
    weight: float
    price: float
    name: str | None = None
    blocked: str | None = None


class Criterion(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Sleeve(BaseModel):
    """The stocks a ranking would hold now, and whether its record allows trading them."""

    variant: str
    session: date
    holdings: list[Holding]
    criteria: list[Criterion]
    problems: list[str] = []

    @computed_field
    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.criteria)


class SleeveProvider(Protocol):
    def sleeve(self, variant: str) -> Sleeve | None:
        """The holdings under `variant`, or None while there is nothing to hold."""
