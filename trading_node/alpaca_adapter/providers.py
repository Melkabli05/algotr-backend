import asyncio
from collections.abc import Iterable
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.models import Asset
from alpaca.trading.requests import GetAssetsRequest
from nautilus_trader.common import Clock
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live.providers import InstrumentProvider
from nautilus_trader.model import CurrencyPair, Equity, InstrumentId

from trading_node.alpaca_adapter.parsing import AlpacaInstrument, parse_instrument, symbol_of
from trading_node.alpaca_adapter.sdk import call, call_list


class AlpacaInstrumentProvider(InstrumentProvider):
    """Loads Alpaca equities and crypto pairs; `filters` may hold an `asset_class`."""

    def __init__(
        self,
        trading: TradingClient,
        clock: Clock,
        config: InstrumentProviderConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._trading = trading
        self._clock = clock

    def find_instrument(self, instrument_id: InstrumentId) -> AlpacaInstrument | None:
        instrument = self.find(instrument_id)
        return instrument if isinstance(instrument, Equity | CurrencyPair) else None

    async def load_all_async(self, filters: dict[str, Any] | None = None) -> None:
        asset_class = (filters or {}).get("asset_class")
        request = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass(asset_class) if asset_class else None,
        )
        self._add_assets(await call_list(self._trading.get_all_assets, Asset, request))

    async def load_ids_async(
        self,
        instrument_ids: list[InstrumentId],
        filters: dict[str, Any] | None = None,
    ) -> None:
        assets = await asyncio.gather(*(self._fetch(symbol_of(i)) for i in instrument_ids))
        self._add_assets(asset for asset in assets if asset is not None)

    async def _fetch(self, symbol: str) -> Asset | None:
        try:
            return await call(self._trading.get_asset, Asset, symbol)
        except APIError as exc:
            self._log.warning(f"Could not load {symbol}: {exc}")
            return None

    def _add_assets(self, assets: Iterable[Asset]) -> None:
        ts_init = self._clock.timestamp_ns()
        for asset in assets:
            instrument = parse_instrument(asset, ts_init)
            if instrument is not None:
                self.add(instrument)
