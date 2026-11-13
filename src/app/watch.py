from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from app.core.broker import get_crypto_data_client, get_stock_data_client, get_trading_client
from app.core.settings import get_settings
from app.features.portfolio import service as portfolio
from app.features.risk.history import Snapshot, snapshot_of
from app.features.risk.summary import summarize
from app.freshness import connected
from app.health import jobs
from app.quant import history

KEYS_REFUSED = (
    "Alpaca refuses the API keys (401 unauthorized): create new paper keys in Alpaca's "
    "dashboard and set ALPACA_API_KEY and ALPACA_API_SECRET in .env"
)
QUOTE_SILENCE = timedelta(minutes=2)
OVERDUE = timedelta(minutes=10)


def keys_refused(state: Any) -> bool:
    """Whether Alpaca's last answer to the clock probe was a refusal of the keys."""
    probe = getattr(state, "probe", None)
    if probe is None:
        return False
    probe.check(get_trading_client())
    if not probe.results:
        return False
    error = (probe.results[-1][2] or "").lower()
    return "unauthorized" in error or "401" in error


def connection_problems(state: Any, now: datetime) -> list[str]:
    """Streams that are down, and coins whose quotes stopped arriving while data streams.

    Refused keys are the one problem then: every stream fails because of them.
    """
    if keys_refused(state):
        return [KEYS_REFUSED]
    found = [
        f"The {item.name} stream is disconnected"
        for item in state.stream_health.report()
        if item.connects and not connected(item)
    ]
    if state.streams.data_paused():
        return found
    for symbol in get_settings().stream_symbols:
        if "/" not in symbol:
            continue
        seen = state.hub.seen.get(("quote", symbol))
        if seen is not None and now - seen > QUOTE_SILENCE:
            found.append(f"No {symbol} quotes for {(now - seen).total_seconds():.0f}s")
    return found


def daily_returns() -> list[float]:
    """Daily returns of the account's value over the last three months, today last."""
    points = portfolio.get_equity_history(get_trading_client(), "3M")
    values = [point.equity for point in points if point.equity > 0]
    return [after / before - 1 for before, after in zip(values, values[1:], strict=False)]


def overdue_jobs(state: Any, now: datetime) -> list[str]:
    return [
        f"job {job.owner}/{job.id}"
        for job in jobs(state)
        if job.next_run is not None and job.next_run < now - OVERDUE
    ]


def risk_snapshot(state: Any) -> Snapshot:
    summary = summarize(
        state, get_trading_client(), get_stock_data_client(), get_crypto_data_client()
    )
    return snapshot_of(summary, state.automation.kill_state(), datetime.now(UTC))


def stale_feeds(state: Any, now: datetime) -> list[str]:
    """Quote feeds gone stale while their market is open."""
    from app.features.breakers.service import stale_quotes
    from app.freshness import freshness, is_open

    clock = get_trading_client().get_clock()
    return stale_quotes(freshness(state, is_open(clock), now))


def strategy_returns(name: str, since: datetime) -> tuple[pd.Series, int] | None:
    """A lifecycle strategy's daily returns since it entered its stage."""
    from app.features.systematic import backtest
    from app.features.systematic.catalog import STRATEGIES

    spec = STRATEGIES.get(name)
    if spec is None:
        return None
    prices = history.cached_prices(get_stock_data_client(), get_crypto_data_client(), spec.universe)
    returns = backtest.daily_returns(prices.closes, spec, opens=prices.opens)
    return returns.loc[pd.DatetimeIndex(returns.index).date >= since.date()], spec.periods_per_year


def equity_series() -> pd.Series:
    points = portfolio.get_equity_history(get_trading_client(), "3M")
    return pd.Series([point.equity for point in points if point.equity > 0], dtype=float)


def platform_symbol(symbol: str, asset_class: str | None) -> str:
    """Alpaca writes a held coin as BTCUSD; the platform and its price history as BTC/USD."""
    if asset_class == "crypto" and "/" not in symbol and symbol.endswith("USD"):
        return f"{symbol[:-3]}/USD"
    return symbol


def holdings_closes(sector_of: Any) -> pd.DataFrame:
    """Recent closes of the held names the portfolio's correlation test applies to."""
    from app.features.breakers.service import holdings_frame

    positions = portfolio.list_positions(get_trading_client())
    symbols = [platform_symbol(p.symbol, p.asset_class) for p in positions]
    if len(symbols) < 2:
        return pd.DataFrame()
    closes = history.cached_closes(get_stock_data_client(), get_crypto_data_client(), symbols)
    recent = closes.iloc[-90:]
    return holdings_frame({str(c): recent[c].dropna().tolist() for c in recent}, sector_of)


def drift_scores(monitor: Any) -> dict[str, float]:
    """The drift monitor's latest distance per watched stream, for the metrics."""
    return {
        stream.name: float(stream.distance)
        for stream in monitor.status().streams
        if stream.distance is not None
    }
