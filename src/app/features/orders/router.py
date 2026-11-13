from contextlib import suppress
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from app.core import audit
from app.core.book import BookGuard
from app.core.broker import (
    CryptoData,
    Feed,
    StockData,
    Trading,
    get_crypto_data_client,
    get_data_feed,
    get_stock_data_client,
    get_trading_client,
)
from app.core.database import transaction
from app.core.limits import LimitStore
from app.core.settings import get_settings
from app.core.types import Symbol
from app.features.orders import estimate, gate, risk, rules, service, trail
from app.features.orders.algo import AlgoDesk, AlgoNotFoundError
from app.features.orders.iceberg import IcebergDesk, IcebergNotFoundError
from app.features.orders.schemas import (
    Algo,
    AlgoPlan,
    AlgoRequest,
    CancelAll,
    Cancellation,
    Iceberg,
    IcebergRequest,
    Order,
    OrderChange,
    OrderRequest,
    Rules,
)

router = APIRouter(prefix="/orders", tags=["orders"])


def get_icebergs(request: Request) -> IcebergDesk:
    desk: IcebergDesk = request.app.state.icebergs
    return desk


def get_algos(request: Request) -> AlgoDesk:
    desk: AlgoDesk = request.app.state.algos
    return desk


def get_guard(request: Request) -> BookGuard:
    guard: BookGuard = request.app.state.book
    return guard


def get_gate(request: Request) -> gate.PreTrade:
    """The pre-trade checks, built from the settings and the app's screen and kill switch."""
    settings = get_settings()
    found: LimitStore = getattr(request.app.state, "limits", None) or LimitStore(settings)
    limits = found.get()
    return gate.PreTrade(
        get_trading_client(),
        get_stock_data_client(),
        get_crypto_data_client(),
        get_data_feed(settings),
        gate.Limits(
            position_share=limits.position_share,
            daily_loss=limits.daily_loss,
            adv_share=limits.adv_share,
            spread_bps=limits.spread_bps,
            sector_share=limits.sector_share,
        ),
        request.app.state.scanner,
        request.app.state.automation,
        getattr(request.app.state, "sectors", None),
    )


Icebergs = Annotated[IcebergDesk, Depends(get_icebergs)]
Algos = Annotated[AlgoDesk, Depends(get_algos)]
Guard = Annotated[BookGuard, Depends(get_guard)]
Trail = Annotated[audit.AuditLog, Depends(trail.log_of)]
Gate = Annotated[gate.PreTrade, Depends(get_gate)]


class Preview(BaseModel):
    """What an order would cost and leave, and whether every check before the broker passes."""

    allowed: bool
    reason: str | None
    price: float | None
    cost: float | None
    position_value: float | None
    position_share: float | None


class LimitsView(BaseModel):
    limits: gate.Limits
    kill_state: str


def refuse(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@router.get("")
def read_orders(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[Order]:
    """The orders as the trading node keeps them."""
    with transaction() as connection:
        return service.list_orders(connection, limit)


@router.get("/limits")
def read_limits(checks: Gate) -> LimitsView:
    """The limits every buy is checked against, and the kill switch's step."""
    return LimitsView(limits=checks.limits, kill_state=checks.kill.kill_state())


@router.post("/check")
def check_order(
    trading: Trading,
    stocks: StockData,
    crypto: CryptoData,
    feed: Feed,
    guard: Guard,
    checks: Gate,
    order: OrderRequest,
) -> Preview:
    """Run every check an order would face, without sending it."""
    price = risk.entry_price(order) or risk.last_price(stocks, crypto, order.symbol, feed)
    cost = None if price is None else order.qty * price
    held: float | None = None
    equity: float | None = None
    with suppress(risk.RiskRefusedError):
        held, (equity, _) = checks.held_value(order.symbol), checks.equity()
    after = None
    if held is not None and cost is not None:
        after = held + cost if order.side == "buy" else max(held - cost, 0.0)
    reason = rules.problem(order)
    if reason is None:
        try:
            checks.check(order.symbol, order.side, order.qty, risk.entry_price(order))
            risk.check_book(guard, order)
            risk.check_cash(
                trading,
                stocks,
                crypto,
                feed=feed,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=risk.entry_price(order),
            )
        except risk.RiskRefusedError as exc:
            reason = str(exc)
    return Preview(
        allowed=reason is None,
        reason=reason,
        price=price,
        cost=cost,
        position_value=after,
        position_share=None if after is None or not equity else after / equity,
    )


@router.get("/rules")
def read_rules(symbol: Symbol, qty: Annotated[float, Query(gt=0)] = 1) -> Rules:
    """The order types and time in force available for this symbol and quantity."""
    return service.get_rules(symbol, qty)


@router.patch("/{order_id}")
def change_order(log: Trail, order_id: str, change: OrderChange) -> Order:
    """Change an open order's price or quantity; the trading node sends it and the id stays."""
    request = {"id": order_id, **change.model_dump(exclude_none=True)}

    def send() -> Order:
        with transaction() as connection:
            return service.change_order(connection, order_id, change)

    try:
        return trail.audited(
            log, "change", request, send, lambda order: {"id": order.id, "status": order.status}
        )
    except service.ChangeRefusedError as exc:
        raise refuse(exc) from exc
    except service.OrderNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No such order") from exc


@router.delete("/{order_id}")
def delete_order(log: Trail, order_id: str) -> Cancellation:
    """Cancel an order; one that is already filled or canceled is reported, not an error."""

    def send() -> Cancellation:
        with transaction() as connection:
            return service.cancel_order(connection, order_id)

    try:
        return trail.audited(
            log, "cancel", {"id": order_id}, send, lambda result: result.model_dump()
        )
    except service.OrderNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No such order") from exc


@router.delete("")
def delete_orders(desk: Icebergs, algos: Algos, log: Trail) -> CancelAll:
    """Stop every working iceberg and algorithm order, then cancel every open order."""

    def everything() -> CancelAll:
        stopped = [desk.cancel(item.id) for item in desk.icebergs() if item.status == "working"]
        halted = [algos.cancel(item.id) for item in algos.algos() if item.status == "working"]
        with transaction() as connection:
            result = service.cancel_all(connection)
        result.icebergs += len(stopped)
        result.algos += len(halted)
        return result

    return trail.audited(log, "cancel-all", {}, everything, lambda result: result.model_dump())


@router.post("", status_code=status.HTTP_201_CREATED)
def create_order(
    trading: Trading,
    stocks: StockData,
    crypto: CryptoData,
    feed: Feed,
    guard: Guard,
    checks: Gate,
    log: Trail,
    order: OrderRequest,
) -> Order:
    """Queue an order for the trading node; every attempt, accepted or refused, is audited."""

    usage: dict[str, float] = {}

    def submit() -> Order:
        usage.update(
            checks.check(order.symbol, order.side, order.qty, risk.entry_price(order)) or {}
        )
        risk.check_book(guard, order)
        risk.check_cash(
            trading,
            stocks,
            crypto,
            feed=feed,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=risk.entry_price(order),
        )
        with transaction() as connection:
            return service.place(connection, order.model_dump(exclude_none=True))

    try:
        return trail.audited(
            log,
            "submit",
            order.model_dump(exclude_none=True),
            submit,
            lambda placed: {
                "id": placed.id,
                "status": placed.status,
                "usage": {name: round(value, 4) for name, value in usage.items()},
            },
        )
    except (risk.RiskRefusedError, service.UnsupportedOrderError) as exc:
        raise refuse(exc) from exc


@router.post("/estimate")
def estimate_order(
    request: Request,
    stocks: StockData,
    crypto: CryptoData,
    feed: Feed,
    order: OrderRequest,
) -> estimate.Estimate:
    """Expected spread, impact and total cost of the order, or a resting limit's chance to trade."""
    price = risk.last_price(stocks, crypto, order.symbol, feed)
    if price is None:
        raise HTTPException(status_code=422, detail=f"No price for {order.symbol}")
    limit = order.limit_price if order.type in ("limit", "stop_limit") else None
    return estimate.estimate(
        stocks,
        crypto,
        getattr(request.app.state, "book", None),
        order.symbol,
        order.side,
        order.qty,
        price,
        limit,
        order.time_in_force == "day",
    )


@router.post("/iceberg", status_code=status.HTTP_201_CREATED)
def create_iceberg(
    trading: Trading,
    stocks: StockData,
    crypto: CryptoData,
    feed: Feed,
    desk: Icebergs,
    checks: Gate,
    log: Trail,
    iceberg: IcebergRequest,
) -> Iceberg:
    """Show part of a larger limit order and place the next part as each one fills."""

    def start() -> Iceberg:
        checks.check(iceberg.symbol, iceberg.side, iceberg.qty, iceberg.limit_price)
        risk.check_cash(
            trading,
            stocks,
            crypto,
            feed=feed,
            symbol=iceberg.symbol,
            side=iceberg.side,
            qty=iceberg.qty,
            price=iceberg.limit_price,
        )
        return desk.place(iceberg)

    try:
        return trail.audited(
            log,
            "iceberg",
            iceberg.model_dump(),
            start,
            lambda placed: {"id": placed.id, "status": placed.status},
        )
    except risk.RiskRefusedError as exc:
        raise refuse(exc) from exc


@router.get("/iceberg")
def read_icebergs(desk: Icebergs) -> list[Iceberg]:
    return desk.icebergs()


@router.delete("/iceberg/{iceberg_id}")
def cancel_iceberg(desk: Icebergs, log: Trail, iceberg_id: str) -> Iceberg:
    try:
        return trail.audited(
            log,
            "iceberg-cancel",
            {"id": iceberg_id},
            lambda: desk.cancel(iceberg_id),
            lambda found: {"id": found.id, "status": found.status},
        )
    except IcebergNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No such iceberg order") from exc


@router.post("/algo/preview")
def preview_algo(algos: Algos, order: AlgoRequest) -> AlgoPlan:
    """The slices an algorithm order would run on; nothing is placed."""
    try:
        return algos.plan(order)
    except service.AlgoRefusedError as exc:
        raise refuse(exc) from exc


@router.post("/algo", status_code=status.HTTP_201_CREATED)
def create_algo(
    trading: Trading,
    stocks: StockData,
    crypto: CryptoData,
    feed: Feed,
    algos: Algos,
    checks: Gate,
    log: Trail,
    order: AlgoRequest,
) -> Algo:
    """Run an order as slices over a time; every attempt goes to the audit log."""

    def start() -> Algo:
        checks.check(order.symbol, order.side, order.qty, order.limit_price)
        risk.check_cash(
            trading,
            stocks,
            crypto,
            feed=feed,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=order.limit_price,
        )
        return algos.place(order)

    try:
        return trail.audited(
            log,
            "algo",
            order.model_dump(exclude_none=True),
            start,
            lambda placed: {"id": placed.id, "status": placed.status},
        )
    except (risk.RiskRefusedError, service.AlgoRefusedError) as exc:
        raise refuse(exc) from exc


@router.get("/algo")
def read_algos(algos: Algos) -> list[Algo]:
    return algos.algos()


@router.delete("/algo/{algo_id}")
def cancel_algo(algos: Algos, log: Trail, algo_id: str) -> Algo:
    """Stop an algorithm order and cancel its child on show; what filled stays filled."""
    try:
        return trail.audited(
            log,
            "algo-cancel",
            {"id": algo_id},
            lambda: algos.cancel(algo_id),
            lambda found: {"id": found.id, "status": found.status},
        )
    except AlgoNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No such algorithm order") from exc
