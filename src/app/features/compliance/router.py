from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.core.broker import Trading
from app.core.compliance import Screening, ShariahScreen
from app.core.types import Symbol
from app.features.compliance import service
from app.features.compliance.service import HeldScreening

router = APIRouter(prefix="/compliance", tags=["compliance"])


def get_screen(request: Request) -> ShariahScreen:
    screen: ShariahScreen = request.app.state.scanner
    return screen


Screen = Annotated[ShariahScreen, Depends(get_screen)]


@router.get("/screen/{symbol:path}")
def read_screening(screen: Screen, symbol: Symbol) -> Screening:
    """Whether `symbol` may be bought under the Shariah screen, and why."""
    return screen.screen(symbol)


@router.get("/holdings")
def read_holdings(trading: Trading, screen: Screen) -> list[HeldScreening]:
    """Every position with today's verdict, so a name that left the lists can be sold."""
    return service.holdings(trading, screen)


@router.get("/universe")
def read_universe(screen: Screen) -> list[Screening]:
    """Every name on a current Shariah list with its verdict."""
    return [screen.screen(symbol) for symbol in screen.listed()]
