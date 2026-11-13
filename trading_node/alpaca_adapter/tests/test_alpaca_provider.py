import asyncio
from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.trading.client import TradingClient
from nautilus_trader.common import Clock
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.model import CurrencyPair, Equity, InstrumentId

from trading_node.alpaca_adapter.providers import AlpacaInstrumentProvider
from trading_node.alpaca_adapter.tests import support

BTC = InstrumentId.from_str("BTC/USD.ALPACA")
AAPL = InstrumentId.from_str("AAPL.ALPACA")


@pytest.fixture
def trading() -> MagicMock:
    mock: MagicMock = create_autospec(TradingClient, instance=True)
    assets = {
        "AAPL": support.asset("AAPL"),
        "BTC/USD": support.asset("BTC/USD"),
        "DEAD": support.asset("DEAD", tradable=False),
    }
    mock.get_all_assets.return_value = list(assets.values())
    mock.get_asset.side_effect = lambda symbol: assets[symbol]
    return mock


def provider(
    trading: MagicMock, config: InstrumentProviderConfig | None = None
) -> AlpacaInstrumentProvider:
    return AlpacaInstrumentProvider(trading, Clock.new_test(), config)


def test_load_all_keeps_tradable_equities_and_crypto(trading: MagicMock) -> None:
    loaded = provider(trading)
    asyncio.run(loaded.load_all_async())
    assert set(loaded.get_all()) == {AAPL, BTC}
    assert isinstance(loaded.find(AAPL), Equity)
    assert isinstance(loaded.find(BTC), CurrencyPair)


def test_load_all_can_be_limited_to_an_asset_class(trading: MagicMock) -> None:
    asyncio.run(provider(trading).load_all_async({"asset_class": "crypto"}))
    request = trading.get_all_assets.call_args.args[0]
    assert request.asset_class.value == "crypto"
    assert request.status.value == "active"


def test_load_ids_fetches_each_asset_by_symbol(trading: MagicMock) -> None:
    loaded = provider(trading)
    asyncio.run(loaded.load_ids_async([BTC, AAPL]))
    assert set(loaded.get_all()) == {AAPL, BTC}
    trading.get_all_assets.assert_not_called()
    assert sorted(call.args[0] for call in trading.get_asset.call_args_list) == ["AAPL", "BTC/USD"]


def test_an_unknown_symbol_is_skipped_not_fatal(trading: MagicMock) -> None:
    trading.get_asset.side_effect = [support.asset("AAPL"), support.api_error(404, "not found")]
    loaded = provider(trading)
    asyncio.run(loaded.load_ids_async([AAPL, InstrumentId.from_str("NOPE.ALPACA")]))
    assert set(loaded.get_all()) == {AAPL}


def test_instruments_are_stamped_with_the_clock_time(trading: MagicMock) -> None:
    clock = Clock.new_test()
    clock.set_time(123)
    loaded = AlpacaInstrumentProvider(trading, clock)
    asyncio.run(loaded.load_ids_async([AAPL]))
    instrument = loaded.find(AAPL)
    assert isinstance(instrument, Equity)
    assert instrument.ts_init == 123


def test_initialize_loads_only_the_configured_ids(trading: MagicMock) -> None:
    config = InstrumentProviderConfig(load_ids=["BTC/USD.ALPACA"])
    loaded = provider(trading, config)
    asyncio.run(loaded.initialize())
    assert set(loaded.get_all()) == {BTC}


def test_initialize_with_load_all_fetches_the_whole_universe(trading: MagicMock) -> None:
    loaded = provider(trading, InstrumentProviderConfig(load_all=True))
    asyncio.run(loaded.initialize())
    assert loaded.count == 2


def test_load_async_skips_instruments_already_loaded(trading: MagicMock) -> None:
    loaded = provider(trading)
    asyncio.run(loaded.load_async(AAPL))
    asyncio.run(loaded.load_async(AAPL))
    assert trading.get_asset.call_count == 1
