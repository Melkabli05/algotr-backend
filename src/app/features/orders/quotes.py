import logging
from typing import Any, cast

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest, StockLatestQuoteRequest
from pydantic import BaseModel, Field
from requests import RequestException

logger = logging.getLogger(__name__)


class TopOfBook(BaseModel):
    """The best bid and ask with their sizes (IEX's book alone on the free plan)."""

    bid: float = Field(alias="bp")
    ask: float = Field(alias="ap")
    bid_size: float = Field(alias="bs")
    ask_size: float = Field(alias="as")

    @property
    def mid(self) -> float | None:
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else None

    def lean(self, side: str) -> float:
        """-1 to 1: how far the quote sizes favour trading on `side` now."""
        total = self.bid_size + self.ask_size
        if total <= 0:
            return 0.0
        own = self.bid_size if side == "buy" else self.ask_size
        return 2 * own / total - 1


def top_of_book(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    symbol: str,
    feed: DataFeed,
) -> TopOfBook | None:
    try:
        if "/" in symbol:
            found = crypto.get_crypto_latest_quote(
                CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            )
        else:
            found = stocks.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=feed)
            )
        return TopOfBook.model_validate(cast(dict[str, Any], found)[symbol])
    except (APIError, RequestException, KeyError, TypeError, ValueError) as exc:
        logger.warning("The quote of %s could not be read: %s", symbol, exc)
        return None
