import logging
from threading import Condition

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from cachetools import TTLCache, cached
from requests import RequestException

from app.core.drift import ReturnStream
from app.features.systematic import backtest, signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import MAX_YEARS, ResearchResult, StrategySpec
from app.quant import history, proxy
from app.quant.history import CACHE_SECONDS, NotEnoughHistoryError

logger = logging.getLogger(__name__)

RESULTS: TTLCache[tuple[str, int, float, int, float | None], ResearchResult] = TTLCache(
    maxsize=64, ttl=CACHE_SECONDS
)


def clear() -> None:
    backtested.cache_clear()
    history.cached_prices.cache_clear()


def recent(closes: pd.DataFrame, spec: StrategySpec, years: int) -> pd.DataFrame:
    """The last `years` of evaluation plus the warm-up the rules need before it."""
    days = round(years * spec.periods_per_year) + signals.warmup_days(spec)
    return closes.iloc[-days:]


@cached(
    RESULTS,
    key=lambda stocks, crypto, spec, years, fee, trials, spread: (
        spec.name,
        years,
        fee,
        trials,
        spread,
    ),
    condition=Condition(),
)
def backtested(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    spec: StrategySpec,
    years: int,
    fee: float,
    trials: int,
    spread: float | None,
) -> ResearchResult:
    prices = history.cached_prices(stocks, crypto, spec.universe)
    closes = recent(prices.closes, spec, years)
    market = proxy.market() if spec.proxy else None
    return backtest.run(closes, spec, fee, prices.opens.loc[closes.index], trials, spread, market)


def research(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    spec: StrategySpec,
    years: int = MAX_YEARS,
    fee_bps: float | None = None,
    trials: int | None = None,
    spread: float | None = None,
) -> ResearchResult:
    """A cached backtest of `spec` on the last `years` of daily bars, by default all there are."""
    fee = spec.fee_bps if fee_bps is None else fee_bps
    found = STRATEGIES.trials() if trials is None else trials
    return backtested(stocks, crypto, spec, years, fee, found, spread)


class StrategyReturns:
    """The daily returns of every strategy's rule, which the drift monitor watches."""

    def __init__(
        self, stocks: StockHistoricalDataClient, crypto: CryptoHistoricalDataClient
    ) -> None:
        self.stocks = stocks
        self.crypto = crypto

    def streams(self) -> list[ReturnStream]:
        found = []
        for spec in STRATEGIES.values():
            try:
                prices = history.cached_prices(self.stocks, self.crypto, spec.universe)
            except (NotEnoughHistoryError, APIError, RequestException) as exc:
                logger.warning("No returns of %s to watch: %s", spec.name, exc)
                continue
            returns = backtest.daily_returns(prices.closes, spec, opens=prices.opens)
            found.append(ReturnStream(spec.name, "strategy", returns))
        return found
