from typing import Literal, Protocol

from pydantic import BaseModel

SHARIAH_FUNDS = ("SPUS", "HLAL", "UMMA", "SPSK")
KillState = Literal["normal", "watch", "reduce", "defensive", "flatten"]
BUYS_STOPPED: tuple[KillState, ...] = ("defensive", "flatten")


class Screening(BaseModel):
    """Whether a symbol may be bought under the Shariah screen, and on what authority."""

    symbol: str
    allowed: bool
    reason: str


class ShariahScreen(Protocol):
    def screen(self, symbol: str) -> Screening:
        """The verdict for `symbol` from the screening lists in force today."""

    def listed(self) -> list[str]:
        """Every symbol on a current screening list."""


class KillSwitch(Protocol):
    def kill_state(self) -> KillState:
        """How far the autopilot's losses have escalated, from normal to flatten."""


class Sectors(Protocol):
    def sector_of(self, symbol: str) -> str:
        """The symbol's GICS sector, or Fund, Crypto or Unclassified."""
