from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status

from app.core.broker import Trading
from app.core.settings import Settings, get_settings
from app.core.types import Symbol
from app.features.portfolio import service
from app.features.portfolio.schemas import Account, EquityPoint, Period, Position

router = APIRouter(tags=["portfolio"])


@router.get("/account")
def read_account(trading: Trading, settings: Annotated[Settings, Depends(get_settings)]) -> Account:
    """Account summary; a successful response also means Alpaca is reachable and authenticated."""
    return service.get_account(trading, settings.alpaca_paper)


@router.get("/positions")
def read_positions(trading: Trading) -> list[Position]:
    return service.list_positions(trading)


@router.delete("/positions/{symbol:path}", status_code=status.HTTP_204_NO_CONTENT)
def delete_position(trading: Trading, symbol: Annotated[Symbol, Path()]) -> Response:
    service.close_position(trading, symbol)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/portfolio/history")
def read_equity_history(trading: Trading, period: Period = "1M") -> list[EquityPoint]:
    return service.get_equity_history(trading, period)
