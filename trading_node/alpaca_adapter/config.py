from decimal import Decimal
from typing import Any

from nautilus_trader.config import DataClientConfig, ExecutionClientConfig


class AlpacaDataClientConfig(DataClientConfig):
    """Alpaca market data client; `feed` is the stock data feed (`iex` is free)."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        feed: str = "iex",
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.feed = feed


class AlpacaExecClientConfig(ExecutionClientConfig):
    """Alpaca execution client; crypto fees default to Alpaca's first volume tier."""

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        crypto_maker_fee: Decimal = Decimal("0.0015"),
        crypto_taker_fee: Decimal = Decimal("0.0025"),
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.crypto_maker_fee = crypto_maker_fee
        self.crypto_taker_fee = crypto_taker_fee
