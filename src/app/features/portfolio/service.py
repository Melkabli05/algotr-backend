from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

from app.features.portfolio.schemas import (
    Account,
    EquityPoint,
    Period,
    PortfolioHistory,
    Position,
)

TIMEFRAMES: dict[Period, str] = {"1D": "5Min", "1W": "1H", "1M": "1D", "3M": "1D", "1A": "1D"}


def get_account(trading: TradingClient, paper: bool) -> Account:
    return Account(**dict(trading.get_account()), paper=paper)


def list_positions(trading: TradingClient) -> list[Position]:
    return [Position.model_validate(position) for position in trading.get_all_positions()]


def close_position(trading: TradingClient, symbol: str) -> None:
    trading.close_position(symbol)


def get_equity_history(trading: TradingClient, period: Period) -> list[EquityPoint]:
    """Equity over time; points before the account existed (no equity) are dropped."""
    history = PortfolioHistory.model_validate(
        trading.get_portfolio_history(
            GetPortfolioHistoryRequest(period=period, timeframe=TIMEFRAMES[period])
        )
    )
    return [
        EquityPoint(time=timestamp, equity=equity)
        for timestamp, equity in zip(history.timestamp, history.equity, strict=True)
        if equity
    ]
