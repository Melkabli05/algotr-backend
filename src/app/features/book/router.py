from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.types import Symbol
from app.features.book.book import LEVELS
from app.features.book.schemas import BookStatus, BookView, FlowPoint, Impact, Point
from app.features.book.service import BookDesk, BookNotFoundError, BookNotReadyError

router = APIRouter(prefix="/book", tags=["book"])


def get_book(request: Request) -> BookDesk:
    desk: BookDesk = request.app.state.book
    return desk


Desk = Annotated[BookDesk, Depends(get_book)]


def absent(symbol: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"There is no order book for {symbol}: Alpaca streams one for coins only",
    )


@router.get("/status")
def read_status(desk: Desk) -> BookStatus:
    """The books being followed, whether each is ready, and how long the stream has been quiet."""
    return desk.status()


@router.get("/view")
def read_view(
    desk: Desk, symbol: Symbol, levels: Annotated[int, Query(ge=1, le=LEVELS)] = 10
) -> BookView:
    """The best levels of a coin's book with the running total of size, and its spread."""
    try:
        return desk.view(symbol, levels)
    except BookNotFoundError as exc:
        raise absent(symbol) from exc
    except BookNotReadyError as exc:
        raise HTTPException(status_code=503, detail=f"The book of {symbol} is not ready") from exc


@router.get("/history")
def read_history(
    desk: Desk, symbol: Symbol, minutes: Annotated[int, Query(ge=1, le=1440)] = 30
) -> list[Point]:
    """The mid price and spread of a coin once a second over the last `minutes`."""
    try:
        return desk.history(symbol, minutes)
    except BookNotFoundError as exc:
        raise absent(symbol) from exc


@router.get("/flow")
def read_flow(
    desk: Desk, symbol: Symbol, minutes: Annotated[int, Query(ge=1, le=1440)] = 30
) -> list[FlowPoint]:
    """The order flow imbalance of a coin in windows of about 5 seconds over the last `minutes`."""
    try:
        return desk.flows(symbol, minutes)
    except BookNotFoundError as exc:
        raise absent(symbol) from exc


@router.get("/impact")
def read_impact(
    desk: Desk,
    symbol: Symbol,
    side: Literal["buy", "sell"],
    qty: Annotated[float, Query(gt=0)],
) -> Impact:
    """What a market order of `qty` would pay walking the book, and whether to use a limit."""
    try:
        return desk.impact(symbol, side, qty)
    except BookNotFoundError as exc:
        raise absent(symbol) from exc
    except BookNotReadyError as exc:
        raise HTTPException(status_code=503, detail=f"The book of {symbol} is not ready") from exc
