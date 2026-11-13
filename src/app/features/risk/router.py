from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.broker import CryptoData, StockData, Trading
from app.core.settings import get_settings
from app.features.orders import risk as order_risk
from app.features.portfolio.schemas import Position
from app.features.risk.history import NearMiss, Snapshot, near_misses
from app.features.risk.schemas import RiskSummary
from app.features.risk.summary import summarize

router = APIRouter(prefix="/risk", tags=["risk"])

# main.py imports this router under its feature name.
risk_router = router


class WhatIf(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0)
    price: float | None = Field(default=None, gt=0)


class Change(BaseModel):
    name: str
    before: float | None
    after: float | None
    unit: str
    cap: float | None = None


class WhatIfResult(BaseModel):
    sector: str
    changes: list[Change]


@router.get("/summary")
def read_summary(
    request: Request, trading: Trading, stocks: StockData, crypto: CryptoData
) -> RiskSummary:
    """Value at risk, exposure by sector, strategy and position, correlation, limits and events."""
    return summarize(request.app.state, trading, stocks, crypto)


@router.get("/history")
def read_history(
    request: Request, hours: Annotated[float, Query(gt=0, le=24 * 14)] = 24
) -> list[Snapshot]:
    """The five-minute snapshots of limit use and value at risk over the last `hours`."""
    recorder: Any = getattr(request.app.state, "risk_history", None)
    if recorder is None:
        return []
    found: list[Snapshot] = recorder.history(datetime.now(UTC) - timedelta(hours=hours))
    return found


@router.get("/near-misses")
def read_near_misses(
    request: Request, days: Annotated[int, Query(ge=1, le=90)] = 7
) -> list[NearMiss]:
    """Orders that passed while using 80% or more of a limit."""
    return near_misses(request.app.state.audit.entries(), datetime.now(UTC) - timedelta(days=days))


@router.post("/what-if")
def what_if(
    request: Request, body: WhatIf, trading: Trading, stocks: StockData, crypto: CryptoData
) -> WhatIfResult:
    """Value at risk, the symbol's sector, its position share and correlation, before and after."""
    price = body.price or order_risk.last_price(
        stocks, crypto, body.symbol, get_settings().data_feed
    )
    if price is None:
        raise HTTPException(status_code=422, detail=f"No price for {body.symbol}")
    value = body.qty * price * (1 if body.side == "buy" else -1)
    extra = Position(
        symbol=body.symbol.replace("/", "") if "/" in body.symbol else body.symbol,
        asset_class="crypto" if "/" in body.symbol else "us_equity",
        side="long",
        qty=body.qty,
        avg_entry_price=price,
        current_price=price,
        market_value=value,
        unrealized_pl=0.0,
    )
    state = request.app.state
    before = summarize(state, trading, stocks, crypto)
    after = summarize(state, trading, stocks, crypto, extra)
    sector = state.sectors.sector_of(body.symbol)

    def var(found: RiskSummary) -> float | None:
        return next(
            (row.dollars for row in found.var if row.confidence == 0.95 and row.horizon == 1),
            None,
        )

    def share(found: RiskSummary, name: str) -> float | None:
        return next((item.share for item in found.sectors if item.name == name), 0.0)

    def used(found: RiskSummary, name: str) -> float | None:
        return next((item.value for item in found.limits if item.name == name), None)

    def cap(name: str) -> float | None:
        return next((item.threshold for item in before.limits if item.name == name), None)

    return WhatIfResult(
        sector=sector,
        changes=[
            Change(
                name="Value at risk, 95%, 1 day",
                before=var(before),
                after=var(after),
                unit="dollars",
            ),
            Change(
                name=f"Sector, {sector}",
                before=share(before, sector),
                after=share(after, sector),
                unit="share",
                cap=cap("Sector"),
            ),
            Change(
                name="Largest position",
                before=used(before, "Position size"),
                after=used(after, "Position size"),
                unit="share",
                cap=cap("Position size"),
            ),
            Change(
                name="Highest correlation",
                before=used(before, "Correlation"),
                after=used(after, "Correlation"),
                unit="ratio",
                cap=cap("Correlation"),
            ),
        ],
    )
