import gc
from collections.abc import Iterator
from typing import Any

import pytest
from alpaca.data.enums import DataFeed
from nautilus_trader.model import InstrumentId

from app.core.settings import Settings
from trading_node import session
from trading_node.alpaca_adapter.config import AlpacaDataClientConfig

SETTINGS = Settings(alpaca_api_key="key", alpaca_api_secret="secret")
BTC = InstrumentId.from_str("BTC/USD.ALPACA")


@pytest.fixture(autouse=True)
def collect_node_cycles() -> Iterator[None]:
    yield
    gc.collect()


def test_a_node_is_built_from_the_settings_for_the_given_instruments() -> None:
    node = session.build_node(SETTINGS, [BTC])
    assert not node.is_running
    assert str(node.trader_id) == "ALGOTR-001"


@pytest.mark.parametrize("feed", [DataFeed.IEX, DataFeed.SIP])
def test_the_data_client_reads_the_chosen_feed(
    monkeypatch: pytest.MonkeyPatch, feed: DataFeed
) -> None:
    made: list[AlpacaDataClientConfig] = []

    def spy(**fields: Any) -> AlpacaDataClientConfig:
        made.append(AlpacaDataClientConfig(**fields))
        return made[-1]

    monkeypatch.setattr(session, "AlpacaDataClientConfig", spy)
    settings = Settings(alpaca_api_key="key", alpaca_api_secret="secret", data_feed=feed)
    session.build_node(settings, [BTC])
    assert [config.feed for config in made] == [feed.value]
