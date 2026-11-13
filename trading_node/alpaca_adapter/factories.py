from alpaca.trading.client import TradingClient
from nautilus_trader.common import Clock
from nautilus_trader.config import DataClientConfig, ExecutionClientConfig
from nautilus_trader.live import ClientCache
from nautilus_trader.live.clients import DataClientFactory, ExecutionClientFactory
from nautilus_trader.model import TraderId

from trading_node.alpaca_adapter.config import AlpacaDataClientConfig, AlpacaExecClientConfig
from trading_node.alpaca_adapter.data_client import AlpacaDataClient
from trading_node.alpaca_adapter.execution_client import AlpacaExecutionClient
from trading_node.alpaca_adapter.providers import AlpacaInstrumentProvider


class AlpacaDataClientFactory(DataClientFactory):
    """Builds the Alpaca data client; nothing connects until the node starts it."""

    @staticmethod
    def create(
        *, name: str, config: DataClientConfig, cache: ClientCache, clock: Clock
    ) -> AlpacaDataClient:
        if not isinstance(config, AlpacaDataClientConfig):
            raise TypeError("Expected AlpacaDataClientConfig")
        trading = TradingClient(config.api_key, config.api_secret, paper=config.paper)
        provider = AlpacaInstrumentProvider(trading, clock, config.instrument_provider)
        return AlpacaDataClient(
            name=name,
            config=config,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
        )


class AlpacaExecutionClientFactory(ExecutionClientFactory):
    """Builds the Alpaca execution client; nothing connects until the node starts it."""

    @staticmethod
    def create(
        *,
        name: str,
        config: ExecutionClientConfig,
        cache: ClientCache,
        clock: Clock,
        trader_id: TraderId,
    ) -> AlpacaExecutionClient:
        if not isinstance(config, AlpacaExecClientConfig):
            raise TypeError("Expected AlpacaExecClientConfig")
        trading = TradingClient(config.api_key, config.api_secret, paper=config.paper)
        provider = AlpacaInstrumentProvider(trading, clock, config.instrument_provider)
        return AlpacaExecutionClient(
            name=name,
            config=config,
            cache=cache,
            clock=clock,
            trader_id=trader_id,
            instrument_provider=provider,
            trading=trading,
        )
