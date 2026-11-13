from collections.abc import Sequence

from nautilus_trader.common import Environment
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live import LiveNode
from nautilus_trader.model import InstrumentId, TraderId

from app.core.settings import Settings
from trading_node.alpaca_adapter.config import AlpacaDataClientConfig, AlpacaExecClientConfig
from trading_node.alpaca_adapter.constants import ALPACA
from trading_node.alpaca_adapter.factories import (
    AlpacaDataClientFactory,
    AlpacaExecutionClientFactory,
)


def build_node(settings: Settings, instrument_ids: Sequence[InstrumentId]) -> LiveNode:
    """A live node on the Alpaca account that preloads the given instruments."""
    provider = InstrumentProviderConfig(load_ids=[str(i) for i in instrument_ids])
    key, secret, paper = settings.alpaca_api_key, settings.alpaca_api_secret, settings.alpaca_paper
    return (
        LiveNode.builder("ALGOTR-PAPER", TraderId("ALGOTR-001"), Environment.LIVE)
        .add_data_client(
            ALPACA,
            AlpacaDataClientFactory,
            AlpacaDataClientConfig(
                api_key=key,
                api_secret=secret,
                paper=paper,
                feed=settings.data_feed.value,
                instrument_provider=provider,
            ),
        )
        .add_exec_client(
            ALPACA,
            AlpacaExecutionClientFactory,
            AlpacaExecClientConfig(
                api_key=key, api_secret=secret, paper=paper, instrument_provider=provider
            ),
        )
        .build()
    )
