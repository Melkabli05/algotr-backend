from collections.abc import Collection, Mapping
from decimal import ROUND_FLOOR, Decimal
from typing import NamedTuple

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient

from app.core.agents import ENSEMBLE
from app.core.sleeve import Sleeve
from app.features.portfolio import service as portfolio
from app.features.portfolio.schemas import Account, Position
from app.features.systematic import backtest, signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import (
    Plan,
    PlannedOrder,
    SleeveConfig,
    SleevePlan,
    StrategyBook,
    StrategyName,
    Target,
)
from app.quant import allocation

MIN_TRADE_VALUE = 10.0
MIN_EXIT_VALUE = 1.0
CASH_RESERVE = 0.001
RISK_WINDOW = 252
RISK_MIN = 60
REBALANCE_BAND = 0.05
QTY_STEP = Decimal("0.000001")
NEW_NAMES_PER_RUN = 3
GAP_LIMIT = 0.02
CENT = Decimal("0.01")
SUB_DOLLAR_TICK = Decimal("0.0001")


class PlanError(ValueError):
    """Raised when the account cannot be planned against, for example when trading is blocked."""


class AgentPart(NamedTuple):
    """The agents' ensemble as one more strategy: its weights, its assets' prices, its returns."""

    weights: Mapping[str, float]
    closes: pd.DataFrame
    returns: pd.Series


class SleeveRequest(NamedTuple):
    """The sleeve's settings, what the scanner offers now, and the symbols it bought earlier."""

    config: SleeveConfig
    offered: Sleeve | None
    held: Collection[str]


def allowed(config: SleeveConfig, offered: Sleeve | None) -> bool:
    """Whether the sleeve may place orders: trading is on and its record passed or is overridden."""
    return config.mode == "trade" and offered is not None and (offered.passed or config.override)


def plain(symbol: str) -> str:
    return symbol.replace("/", "")


def floor_qty(value: float) -> float:
    return float(Decimal(str(value)).quantize(QTY_STEP, rounding=ROUND_FLOOR))


def is_stock(symbol: str) -> bool:
    return "/" not in symbol


def buy_limit(symbol: str, price: float) -> float | None:
    """The most a stock buy may pay: `GAP_LIMIT` above the decision price, on Alpaca's tick.

    Stock orders wait for the next open, so an overnight gap would otherwise buy more than the
    cash covers on an account that lends. A coin fills at once, at about the decision price.
    """
    if not is_stock(symbol):
        return None
    tick = CENT if price >= 1 else SUB_DOLLAR_TICK
    return float(Decimal(str(price * (1 + GAP_LIMIT))).quantize(tick, rounding=ROUND_FLOOR))


def proceeds(order: PlannedOrder) -> float:
    """The least a sell can bring in: a stock sell fills at the open, up to `GAP_LIMIT` lower."""
    return order.notional * (1 - GAP_LIMIT) if is_stock(order.symbol) else order.notional


def cost(order: PlannedOrder) -> float:
    """The most a buy can cost: its limit for a stock, its decision price for a coin."""
    return order.qty * (order.limit_price or order.notional / order.qty)


def account_for(trading: TradingClient, paper: bool) -> Account:
    account = portfolio.get_account(trading, paper)
    if account.trading_blocked or account.status != "ACTIVE":
        raise PlanError(f"Account is {account.status}; trading blocked: {account.trading_blocked}")
    return account


def fit_to_cash(orders: list[PlannedOrder], cash: float) -> list[PlannedOrder]:
    """Scale the buys down so the plan never spends more than the cash and what its sells bring in.

    The account is a margin account that would lend the difference; this one is cash only. Buys
    count at the most they can cost and sells at the least they can bring in (see `buy_limit`).
    """
    buys = sum(cost(order) for order in orders if order.side == "buy")
    sells = sum(proceeds(order) for order in orders if order.side == "sell")
    room = max(cash + sells, 0.0) * (1 - CASH_RESERVE)
    if buys <= room:
        return orders
    factor = room / buys
    fitted = []
    for order in orders:
        if order.side == "sell":
            fitted.append(order)
            continue
        price = order.notional / order.qty
        qty = floor_qty(order.qty * factor)
        if qty:
            fitted.append(order.model_copy(update={"qty": qty, "notional": qty * price}))
    return fitted


def strategy_returns(name: str, frame: pd.DataFrame) -> pd.Series:
    """The strategy's recent daily returns, or its universe's when the rule has not moved.

    A rule that sat in cash for the whole window shows no risk, but it carries its universe's
    as soon as it buys, so that is what it is sized by.
    """
    universe = frame.pct_change().mean(axis=1).dropna()
    try:
        own = backtest.daily_returns(frame, STRATEGIES[name])
    except (IndexError, ValueError, KeyError):
        return universe
    own = own.tail(RISK_WINDOW)
    return own if len(own) >= RISK_MIN and float(own.std()) > 0 else universe


def risk_shares(
    closes: Mapping[str, pd.DataFrame], extra: Mapping[str, pd.Series] | None = None
) -> dict[str, float]:
    """Each strategy's share of capital, so each adds the same risk to the whole (risk parity).

    Equal risk contribution over the covariance of the strategies' own returns: a calm ETF
    sleeve is not swamped by a volatile crypto one, and two that move together share a budget.
    Returns are compared on the dates they share, a coin's weekend folded into Monday.
    """
    streams = {name: strategy_returns(name, frame) for name, frame in closes.items()}
    streams.update(extra or {})
    names = list(streams)
    if len(names) < 2:
        return dict.fromkeys(names, 1.0)
    growth = pd.concat(
        {name: (1 + series).cumprod() for name, series in streams.items()},
        axis=1,
        join="inner",
        sort=True,
    )
    returns = growth.pct_change().dropna().tail(RISK_WINDOW)
    if len(returns) < RISK_MIN:
        return {name: 1 / len(names) for name in names}
    covariance = returns.cov().to_numpy()
    if not (np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) > 0)):
        return {name: 1 / len(names) for name in names}
    weights = allocation.equal_risk(covariance)
    return {name: float(weight) for name, weight in zip(names, weights, strict=True)}


def build_plan(
    trading: TradingClient,
    closes: dict[StrategyName, pd.DataFrame],
    allocation: float,
    paper: bool,
    managed: Collection[str] = (),
    flat: bool = False,
    scanner: SleeveRequest | None = None,
    scales: Mapping[str, float] | None = None,
    agent: AgentPart | None = None,
) -> Plan:
    """Orders that move the account to the strategies' combined targets.

    `allocation` of equity is split between the strategies by risk (see `risk_shares`). Symbols
    in `managed` that no active strategy wants any more are sold, and `flat` targets everything
    at zero. When the scanner's sleeve trades, it takes its share of the allocation first.
    `scales` cuts a strategy's positions (the drift monitor's reduction); the rest stays in cash.
    """
    account = account_for(trading, paper)
    held = {plain(p.symbol): p for p in portfolio.list_positions(trading)}
    capital = account.equity * allocation
    trades = scanner is not None and not flat and allowed(scanner.config, scanner.offered)
    own = capital * (1 - scanner.config.share) if scanner is not None and trades else capital
    targets: dict[str, float] = {}
    last_close: dict[str, float] = {}
    signal_weights: dict[str, dict[str, float]] = {}
    frames: dict[str, pd.DataFrame] = {str(rule): frame for rule, frame in closes.items()}
    shares = risk_shares(frames, {ENSEMBLE: agent.returns} if agent else None)
    names = [*closes, *([ENSEMBLE] if agent else [])]
    cuts = {name: scales.get(name, 1.0) for name in names} if scales else {}
    books: dict[str, StrategyBook] = {}
    parts: list[tuple[str, pd.DataFrame, pd.Series]] = []
    for rule, frame in closes.items():
        parts.append((rule, frame, signals.weights(frame, STRATEGIES[rule]).iloc[-1]))
    if agent is not None:
        chosen = pd.Series({symbol: agent.weights.get(symbol, 0.0) for symbol in agent.closes})
        parts.append((ENSEMBLE, agent.closes, chosen))
    for name, frame, wanted in parts:
        latest = wanted * 0 if flat else wanted
        signal_weights[name] = {symbol: float(latest[symbol]) for symbol in frame}
        mine: dict[str, float] = {}
        for symbol in frame:
            weight = shares[name] * cuts.get(name, 1.0) * float(latest[symbol])
            mine[symbol] = own * weight
            targets[symbol] = targets.get(symbol, 0.0) + own * weight
            last_close[symbol] = float(frame[symbol].iloc[-1])
        books[name] = StrategyBook(
            capital=own * shares[name],
            targets=mine,
            prices={symbol: last_close[symbol] for symbol in frame},
        )
    sleeve = capital / max(1, len(targets))
    for symbol in managed:
        targets.setdefault(symbol, 0.0)
    rows, planned = [], []
    for symbol, target in targets.items():
        position = held.get(plain(symbol))
        if position is None and symbol not in last_close:
            continue
        rows.append(
            Target(
                symbol=symbol,
                weight=target / capital if capital else 0.0,
                target_value=target,
                current_value=position.market_value if position else 0.0,
            )
        )
        order = order_for(symbol, target, position, last_close.get(symbol, 0.0), sleeve)
        if order:
            planned.append(order)
    part = None if scanner is None else plan_sleeve(scanner, capital, held, set(targets), flat)
    if part is not None and not part.preview:
        rows.extend(part.targets)
        planned.extend(part.orders)
    return Plan(
        strategies=list(closes),
        allocation=allocation,
        equity=account.equity,
        flat=flat,
        signals=signal_weights,
        targets=rows,
        orders=fit_to_cash(planned, account.cash),
        sleeve=part,
        scales={name: scale for name, scale in cuts.items() if scale != 1.0},
        books=books,
    )


def sleeve_targets(
    offered: Sleeve, capital: float, held: dict[str, Position], taken: Collection[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Dollars to hold in each offered name, and its price.

    A blocked name gets no new buy, and at most `NEW_NAMES_PER_RUN` names the account does not
    hold yet are started.
    """
    targets: dict[str, float] = {}
    prices: dict[str, float] = {}
    new = 0
    for item in offered.holdings:
        if item.symbol in taken:
            continue
        position = held.get(plain(item.symbol))
        value = position.market_value if position else 0.0
        target = capital * item.weight
        if item.blocked:
            target = min(target, value)
        elif position is None:
            if new < NEW_NAMES_PER_RUN:
                new += 1
            else:
                target = 0.0
        targets[item.symbol], prices[item.symbol] = target, item.price
    return targets, prices


def plan_sleeve(
    request: SleeveRequest,
    capital: float,
    held: dict[str, Position],
    taken: Collection[str],
    flat: bool,
) -> SleevePlan:
    """The sleeve's targets and orders.

    Off, or flat after a drawdown halt, it sells what it bought earlier. In shadow it only shows
    what it would do. Trading, it holds the offered names; a stale scan or a failed record
    freezes it in place and says why, and it never sells for that reason alone.
    """
    config, offered = request.config, request.offered
    carry = allowed(config, offered)
    problems = list(offered.problems) if offered else []
    if config.mode != "off" and offered is None:
        problems.append("There is no scan to take holdings from")
    if config.mode == "trade" and offered is not None and not carry:
        problems.append("The promotion checks have not passed")
    preview = config.mode == "shadow" and not flat
    sleeve_capital = capital * config.share
    prices: dict[str, float] = {}
    if flat or config.mode == "off":
        targets = dict.fromkeys(request.held, 0.0)
    elif offered is not None and (preview or carry):
        targets, prices = sleeve_targets(offered, sleeve_capital, held, taken)
        if carry:
            targets.update({symbol: 0.0 for symbol in request.held if symbol not in targets})
    else:
        targets = {}
    acting = preview or flat or config.mode == "off" or (carry and not problems)
    average = sleeve_capital / max(1, len(targets))
    rows: list[Target] = []
    planned: list[PlannedOrder] = []
    for symbol, target in targets.items():
        position = held.get(plain(symbol))
        if position is None and target == 0:
            continue
        rows.append(
            Target(
                symbol=symbol,
                weight=target / capital if capital else 0.0,
                target_value=target,
                current_value=position.market_value if position else 0.0,
                source="sleeve",
            )
        )
        order = order_for(symbol, target, position, prices.get(symbol, 0.0), average)
        if order and acting:
            planned.append(order.model_copy(update={"source": "sleeve"}))
    trading = acting and not preview
    book = (
        StrategyBook(
            capital=sleeve_capital if carry else 0.0,
            targets={symbol: target for symbol, target in targets.items() if carry},
            prices=prices,
        )
        if trading
        else None
    )
    return SleevePlan(
        book=book,
        mode=config.mode,
        variant=config.variant,
        share=config.share,
        session=offered.session if offered else None,
        preview=preview,
        override=config.override,
        passed=bool(offered and offered.passed),
        criteria=offered.criteria if offered else [],
        problems=problems,
        targets=rows,
        orders=planned,
    )


def order_for(
    symbol: str, target: float, position: Position | None, last_close: float, sleeve: float
) -> PlannedOrder | None:
    current = position.market_value if position else 0.0
    delta = target - current
    if position and target == 0:
        if current < MIN_EXIT_VALUE:
            return None
        return PlannedOrder(
            symbol=symbol,
            side="sell",
            qty=position.qty,
            notional=current,
            price=position.current_price,
        )
    if abs(delta) < max(MIN_TRADE_VALUE, REBALANCE_BAND * sleeve):
        return None
    if delta > 0:
        price = position.current_price if position else last_close
        qty = floor_qty(delta / price)
        if not qty:
            return None
        return PlannedOrder(
            symbol=symbol,
            side="buy",
            qty=qty,
            notional=qty * price,
            price=price,
            limit_price=buy_limit(symbol, price),
        )
    assert position is not None
    qty = min(floor_qty(-delta / position.current_price), position.qty)
    return PlannedOrder(
        symbol=symbol,
        side="sell",
        qty=qty,
        notional=qty * position.current_price,
        price=position.current_price,
    )
