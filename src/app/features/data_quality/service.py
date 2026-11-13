import asyncio
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core import data_quality
from app.core.database import transaction
from app.quant import history

logger = logging.getLogger(__name__)

JOB_ID = "data-quality"
EVERY = timedelta(hours=1)
REFERENCE_DAYS = 30
Reference = Callable[[str, date, date], pd.Series | None]


def eodhd_reference(api_key: str, client: Any = None) -> Reference:
    """EODHD's adjusted daily closes, the second source Alpaca's closes are checked against."""
    from eodhd import APIClient

    api = client or APIClient(api_key)

    def closes(symbol: str, start: date, end: date) -> pd.Series | None:
        code = f"{symbol.replace('/', '-')}.CC" if "/" in symbol else f"{symbol}.US"
        try:
            rows = api.get_eod_historical_stock_market_data(
                symbol=code, period="d", from_date=start.isoformat(), to_date=end.isoformat()
            )
        except Exception as exc:
            logger.warning("EODHD did not answer for %s: %s", symbol, exc)
            return None
        if not isinstance(rows, list) or not rows:
            return None
        frame = pd.DataFrame(rows)
        column = "adjusted_close" if "adjusted_close" in frame else "close"
        return pd.Series(
            frame[column].astype(float).to_numpy(), index=pd.to_datetime(frame["date"])
        )

    return closes


class DataQualityJob:
    """The validation stage: each symbol's finished daily bars, fetched alone so a coin keeps
    its weekends, checked for gaps, staleness,
    anomalies and the second source, every finding recorded and every bar stored with its
    flags for the backtests."""

    def __init__(
        self,
        symbols: Callable[[], Iterable[str]],
        prices: Callable[[list[str]], history.Prices],
        reference: Reference | None = None,
    ) -> None:
        self.symbols = symbols
        self.prices = prices
        self.reference = reference
        self.scheduler = AsyncIOScheduler(timezone=UTC)

    def run(self, now: datetime | None = None) -> int:
        moment = now or datetime.now(UTC)
        names = sorted(set(self.symbols()))
        if not names:
            return 0
        written = 0
        with transaction() as connection:
            for symbol in names:
                prices = self.prices([symbol])
                if symbol not in prices.closes:
                    continue
                closes = prices.closes[symbol].dropna()
                other = None
                if self.reference is not None and not closes.empty:
                    end = closes.index[-1].date()
                    other = self.reference(symbol, end - timedelta(days=REFERENCE_DAYS), end)
                flags, findings = data_quality.validate(closes, symbol, moment, other)
                written += data_quality.record(connection, "api", findings)
                opens = prices.opens.get(symbol)
                data_quality.store_bars(connection, symbol, closes, opens, flags, "alpaca")
        return written

    async def _run(self) -> None:
        try:
            await asyncio.to_thread(self.run)
        except Exception:
            logger.exception("The data quality check failed")

    def start(self) -> None:
        self.scheduler.add_job(
            self._run,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(UTC),
            misfire_grace_time=None,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
