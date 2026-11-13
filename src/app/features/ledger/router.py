from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import Field

from app.core.types import Symbol
from app.features.ledger import purification
from app.features.ledger.lots import Method
from app.features.ledger.schemas import (
    Comparison,
    FillRow,
    Holding,
    LotRow,
    Picks,
    Realized,
    Reconciliation,
    Status,
)
from app.features.ledger.service import Ledger, NothingToAdoptError

router = APIRouter(prefix="/ledger", tags=["ledger"])


def get_ledger(request: Request) -> Ledger:
    ledger: Ledger = request.app.state.ledger
    return ledger


Books = Annotated[Ledger, Depends(get_ledger)]


@router.get("/status")
def read_status(ledger: Books) -> Status:
    """How much activity is stored, when it was last synced and the last reconciliation."""
    return ledger.status()


@router.post("/sync")
def run_sync(ledger: Books) -> Reconciliation:
    """Fetch new activity from Alpaca, rebuild the lots and compare them with the positions."""
    return ledger.refresh()


@router.get("/comparison")
def read_comparison(ledger: Books) -> Comparison:
    """Every position in the ledger and at Alpaca side by side, and where the last sync reached."""
    return ledger.comparison()


@router.get("/reconciliation")
def read_reconciliation(ledger: Books) -> Reconciliation | None:
    return ledger.last


@router.post("/adopt/{symbol:path}")
def adopt(ledger: Books, symbol: Symbol) -> Reconciliation:
    """Add the lot or write-off that makes the ledger hold what Alpaca holds in `symbol`."""
    try:
        return ledger.adopt(symbol)
    except NothingToAdoptError as exc:
        raise HTTPException(status_code=409, detail="The quantities already agree") from exc


@router.get("/positions")
def read_positions(ledger: Books, method: Method = "fifo") -> list[Holding]:
    """Open positions with cost basis and unrealized and realized P&L by the chosen method."""
    return ledger.holdings(method)


@router.get("/lots")
def read_lots(ledger: Books, method: Method = "fifo", symbol: Symbol | None = None) -> list[LotRow]:
    """Open lots, each marked at the broker's current price."""
    return ledger.lots(method, symbol)


@router.get("/realized")
def read_realized(
    ledger: Books,
    method: Method = "fifo",
    symbol: Symbol | None = None,
    days: int | None = None,
) -> Realized:
    """Closed lots, newest first, with their gain and the totals."""
    return ledger.realized(method, symbol, days)


@router.get("/fills")
def read_fills(
    ledger: Books,
    symbol: Symbol | None = None,
    days: int | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> list[FillRow]:
    """The stored fills, newest first, with the coin fee that came with each."""
    return ledger.fills(symbol, days, limit)


@router.get("/picks/{order_id}")
def read_picks(ledger: Books, order_id: str) -> Picks:
    return ledger.picks(order_id)


@router.put("/picks/{order_id}")
def put_picks(ledger: Books, order_id: str, picks: Picks) -> Picks:
    """Choose the lots a sell order closes, for the specific-lot method; the rest close FIFO."""
    return ledger.set_picks(order_id, picks)


@router.delete("/picks/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_picks(ledger: Books, order_id: str) -> Response:
    ledger.set_picks(order_id, Picks(lots=[]))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


Ratio = Annotated[Decimal, Field(ge=0, le=1)]


@router.get("/purification")
def read_purification(
    ledger: Books, year: Annotated[int | None, Query(ge=2000, le=2100)] = None
) -> purification.Statement:
    """The year's dividends and interest, what they owe to charity, and a zakat estimate."""
    return ledger.purification(year or datetime.now(UTC).year)


@router.get("/purification/ratios")
def read_ratios(ledger: Books) -> dict[str, Decimal]:
    """Every purification ratio entered."""
    return ledger.store.ratios()


@router.put("/purification/ratios")
def put_ratios(ledger: Books, ratios: dict[str, Ratio | None]) -> dict[str, Decimal]:
    """Set the share of each holding's dividends to purify (0.012 for 1.2%); null removes one."""
    ledger.store.set_ratios({key.upper().replace("/", ""): value for key, value in ratios.items()})
    return ledger.store.ratios()
