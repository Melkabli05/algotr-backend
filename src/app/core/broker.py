from functools import lru_cache
from typing import Annotated

from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.trading.client import TradingClient
from fastapi import Depends
from requests import PreparedRequest, Response
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from app.core.settings import Settings, get_settings

DATA_TIMEOUT = 60.0
DATA_RETRIES = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=(502, 503),
    allowed_methods=("GET",),
    raise_on_status=False,
)


class TimeoutAdapter(HTTPAdapter):
    """Gives every request a timeout, which `requests` never does on its own."""

    def send(
        self,
        request: PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float | None, float | None] | None = None,
        verify: bool | str = True,
        cert: str | tuple[str, str] | None = None,
        proxies: dict[str, str] | None = None,
    ) -> Response:
        return super().send(request, stream, timeout or DATA_TIMEOUT, verify, cert, proxies)


def hardened[C: StockHistoricalDataClient | CryptoHistoricalDataClient | NewsClient](
    client: C,
) -> C:
    """A market-data client that times out and retries dropped connections; its calls only read."""
    client._session.mount("https://", TimeoutAdapter(max_retries=DATA_RETRIES))
    return client


@lru_cache
def get_trading_client() -> TradingClient:
    settings = get_settings()
    return TradingClient(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=settings.alpaca_paper,
        raw_data=True,
    )


Trading = Annotated[TradingClient, Depends(get_trading_client)]


@lru_cache
def get_stock_data_client() -> StockHistoricalDataClient:
    settings = get_settings()
    return hardened(
        StockHistoricalDataClient(
            settings.alpaca_api_key, settings.alpaca_api_secret, raw_data=True
        )
    )


@lru_cache
def get_crypto_data_client() -> CryptoHistoricalDataClient:
    settings = get_settings()
    return hardened(
        CryptoHistoricalDataClient(
            settings.alpaca_api_key, settings.alpaca_api_secret, raw_data=True
        )
    )


@lru_cache
def get_news_client() -> NewsClient:
    settings = get_settings()
    return hardened(NewsClient(settings.alpaca_api_key, settings.alpaca_api_secret, raw_data=True))


def get_data_feed(settings: Annotated[Settings, Depends(get_settings)]) -> DataFeed:
    return settings.data_feed


StockData = Annotated[StockHistoricalDataClient, Depends(get_stock_data_client)]
CryptoData = Annotated[CryptoHistoricalDataClient, Depends(get_crypto_data_client)]
Feed = Annotated[DataFeed, Depends(get_data_feed)]
