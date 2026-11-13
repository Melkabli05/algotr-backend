from typing import Any

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from app.core.limits import LimitStore
from app.core.settings import get_settings
from app.features.portfolio import service as portfolio
from app.features.portfolio.schemas import Position
from app.features.risk import service
from app.features.risk.schemas import Correlation, RiskSummary
from app.quant import history, risk


def merged(positions: list[Position], extra: Position | None) -> list[Position]:
    """The positions with `extra` added to its symbol's, or as a new one."""
    if extra is None:
        return positions
    found = [p for p in positions if p.symbol == extra.symbol]
    if not found:
        return [*positions, extra]
    return [
        p.model_copy(update={"market_value": p.market_value + extra.market_value})
        if p.symbol == extra.symbol
        else p
        for p in positions
    ]


def summarize(
    state: Any,
    trading: TradingClient,
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    extra: Position | None = None,
) -> RiskSummary:
    """Value at risk, exposure by sector, strategy and position, correlation, limits and events."""
    settings = get_settings()
    found: LimitStore = getattr(state, "limits", None) or LimitStore(settings)
    limits_now = found.get()
    account = portfolio.get_account(trading, settings.alpaca_paper)
    positions = merged(
        [p for p in portfolio.list_positions(trading) if abs(p.market_value) >= 1], extra
    )
    equity = account.equity
    symbols = [service.pair(position) for position in positions]
    weights = pd.Series({service.pair(p): p.market_value / equity for p in positions}, dtype=float)
    returns = pd.DataFrame()
    if symbols:
        closes = pd.concat(
            [history.cached_closes(stocks, crypto, [symbol]) for symbol in symbols],
            axis=1,
            sort=True,
        )
        returns = closes.ffill().pct_change().dropna(how="all").tail(service.WINDOW).fillna(0.0)
    daily = risk.portfolio_returns(returns, weights) if symbols else pd.Series(dtype=float)
    automation = state.automation
    entries = state.audit.entries()
    triggered = service.last_triggered(entries)
    sectors = service.sector_exposure(positions, state.sectors.sector_of, equity)
    stock_sectors = [item for item in sectors if item.name not in ("Fund", "Crypto")]
    limits = automation.state.config
    shares = [abs(p.market_value) / equity for p in positions] if equity else []
    rows = [
        service.limit(
            "Daily loss",
            max(0.0, -account.day_pl / account.last_equity) if account.last_equity else None,
            limits_now.daily_loss,
            "share",
            triggered,
        ),
        service.limit(
            "Autopilot drawdown",
            automation.state.book.drawdown,
            limits.max_drawdown,
            "share",
            triggered,
        ),
        service.limit(
            "Position size",
            max(shares, default=0.0),
            limits_now.position_share,
            "share",
            triggered,
        ),
        service.limit(
            "Order size against daily volume",
            None,
            limits_now.adv_share,
            "share",
            triggered,
            "order",
        ),
        service.limit(
            "Sector",
            max((item.share for item in stock_sectors), default=0.0),
            limits_now.sector_share,
            "share",
            triggered,
        ),
        service.limit(
            "Correlation",
            service.highest_pair(returns, weights) if len(symbols) > 1 else None,
            limits_now.correlation,
            "ratio",
            triggered,
        ),
        service.limit("Gross exposure", sum(shares), 1.0, "share", triggered),
        service.limit("Spread", None, limits_now.spread_bps, "bps", triggered, "order"),
        service.limit(
            "Book imbalance", None, limits_now.book_imbalance, "ratio", triggered, "order"
        ),
    ]
    drift = state.drift.status().recorded if hasattr(state, "drift") else []
    return RiskSummary(
        equity=equity,
        days=len(daily),
        var=service.var_table(daily, equity) if len(daily) > 5 else [],
        sectors=sectors,
        strategies=service.strategy_exposure(positions, automation.state.books, equity),
        concentration=service.grouped({p.symbol: abs(p.market_value) for p in positions}, equity)[
            :10
        ],
        correlation=service.correlation(returns, weights)
        if symbols
        else Correlation(symbols=[], matrix=[]),
        limits=rows,
        events=service.events(entries, drift)[:200],
    )
