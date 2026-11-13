from alpaca.trading.client import TradingClient
from pydantic import BaseModel

from app.core.compliance import Screening, ShariahScreen
from app.features.portfolio import service as portfolio


class HeldScreening(BaseModel):
    symbol: str
    market_value: float
    screening: Screening


def holdings(trading: TradingClient, screen: ShariahScreen) -> list[HeldScreening]:
    """Every position with today's verdict: a name that dropped off the lists shows here."""
    return [
        HeldScreening(
            symbol=position.symbol,
            market_value=position.market_value,
            screening=screen.screen(symbol_of(position.symbol, position.asset_class)),
        )
        for position in portfolio.list_positions(trading)
    ]


def symbol_of(held: str, asset_class: str | None) -> str:
    """Alpaca reports a coin position as BTCUSD; the screen reads BTC/USD."""
    if asset_class == "crypto" and "/" not in held and held.endswith("USD"):
        return f"{held.removesuffix('USD')}/USD"
    return held
