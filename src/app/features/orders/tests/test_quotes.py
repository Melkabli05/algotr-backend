from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.core.tests.support import api_error
from app.features.orders import quotes
from app.features.orders.quotes import TopOfBook

BOOK = {"bp": 9.99, "ap": 10.01, "bs": 600, "as": 400}


def book(**fields: float) -> TopOfBook:
    return TopOfBook.model_validate({**BOOK, **fields})


def clients() -> tuple[MagicMock, MagicMock]:
    stocks: MagicMock = create_autospec(StockHistoricalDataClient, instance=True)
    crypto: MagicMock = create_autospec(CryptoHistoricalDataClient, instance=True)
    return stocks, crypto


def test_the_mid_is_between_the_best_prices_and_missing_without_both() -> None:
    assert book().mid == pytest.approx(10.0)
    assert book(bp=0).mid is None


def test_more_size_on_our_side_of_the_book_leans_towards_trading() -> None:
    assert book().lean("buy") == pytest.approx(0.2)
    assert book().lean("sell") == pytest.approx(-0.2)
    assert book(bs=0, **{"as": 0}).lean("buy") == 0.0


def test_a_stock_is_read_on_the_chosen_feed() -> None:
    stocks, crypto = clients()
    stocks.get_stock_latest_quote.return_value = {"F": BOOK}
    assert quotes.top_of_book(stocks, crypto, "F", DataFeed.SIP) == book()
    assert stocks.get_stock_latest_quote.call_args.args[0].feed == DataFeed.SIP
    crypto.get_crypto_latest_quote.assert_not_called()


def test_a_coin_is_read_from_the_crypto_client() -> None:
    stocks, crypto = clients()
    crypto.get_crypto_latest_quote.return_value = {"BTC/USD": BOOK}
    assert quotes.top_of_book(stocks, crypto, "BTC/USD", DataFeed.IEX) == book()
    stocks.get_stock_latest_quote.assert_not_called()


def test_a_quote_that_cannot_be_read_is_none_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    stocks, crypto = clients()
    stocks.get_stock_latest_quote.side_effect = api_error("no data", 404)
    assert quotes.top_of_book(stocks, crypto, "F", DataFeed.IEX) is None
    assert "The quote of F could not be read" in caplog.text
