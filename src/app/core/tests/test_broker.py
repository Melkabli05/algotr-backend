from collections.abc import Iterator

import pytest
from requests import PreparedRequest
from requests.adapters import HTTPAdapter

from app.core import broker
from app.core.settings import Settings

URL = "https://data.alpaca.markets/v2/stocks/bars"


@pytest.fixture
def factories(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    fake = Settings(alpaca_api_key="key", alpaca_api_secret="secret", alpaca_paper=True)
    monkeypatch.setattr(broker, "get_settings", lambda: fake)
    cached = (
        broker.get_stock_data_client,
        broker.get_crypto_data_client,
        broker.get_trading_client,
    )
    for factory in cached:
        factory.cache_clear()
    yield
    for factory in cached:
        factory.cache_clear()


@pytest.mark.parametrize("name", ["get_stock_data_client", "get_crypto_data_client"])
def test_market_data_clients_time_out_and_retry_reads(factories: None, name: str) -> None:
    adapter = getattr(broker, name)()._session.get_adapter(URL)
    assert isinstance(adapter, broker.TimeoutAdapter)
    assert adapter.max_retries.total == 3
    assert list(adapter.max_retries.allowed_methods or ()) == ["GET"]


def test_the_trading_client_is_left_alone_because_a_retried_order_could_be_placed_twice(
    factories: None,
) -> None:
    adapter = broker.get_trading_client()._session.get_adapter("https://paper-api.alpaca.markets")
    assert not isinstance(adapter, broker.TimeoutAdapter)


def test_every_request_gets_a_timeout_unless_the_caller_set_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def record(self: HTTPAdapter, request: PreparedRequest, *args: object) -> None:
        seen.append(args[1])

    monkeypatch.setattr(HTTPAdapter, "send", record)
    adapter = broker.TimeoutAdapter()
    adapter.send(PreparedRequest())
    adapter.send(PreparedRequest(), timeout=5)
    assert seen == [broker.DATA_TIMEOUT, 5]
