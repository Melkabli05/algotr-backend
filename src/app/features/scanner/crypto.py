import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, cast

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from pydantic import BaseModel

from app.features.scanner import bars
from app.features.scanner.store import BarStore

logger = logging.getLogger(__name__)
FILE = "crypto.parquet"
QUOTE = "USD"
HISTORY = timedelta(days=1460)
MIN_DAYS = 365
TREND_DAYS = 200
GAP_DAYS = 2
WINDOWS = {"week": 7, "month": 28, "quarter": 91}
NOT_RANKED = frozenset({"USDT", "USDC", "USDG", "PAXG"})


class Dropped(BaseModel):
    coin: str
    reason: str


class CryptoRow(BaseModel):
    rank: int
    coin: str
    price: float
    week: float | None
    month: float | None
    quarter: float | None
    trend: float | None
    above: bool
    days: int


class CryptoScan(BaseModel):
    as_of: date
    rows: list[CryptoRow]
    dropped: list[Dropped]


class CryptoStatus(BaseModel):
    enabled: bool = False
    coins: int = 0
    as_of: date | None = None
    refreshed: datetime | None = None
    error: str | None = None


def coins_of(names: Sequence[str]) -> list[str]:
    """The configured coins as codes: upper case, without the quote currency, each one once."""
    codes = (name.strip().upper().removesuffix(f"/{QUOTE}") for name in names)
    return list(dict.fromkeys(code for code in codes if code))


def number(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


class CryptoDesk:
    """Daily bars of a list of coins and the figures to compare them by; nothing here trades."""

    def __init__(
        self, client: CryptoHistoricalDataClient, coins: Sequence[str], root: Path
    ) -> None:
        self.client = client
        self.coins = coins_of(coins)
        self.store = BarStore(root / FILE)
        self.lock = Lock()
        self.error: str | None = None

    @property
    def requested(self) -> list[str]:
        return [coin for coin in self.coins if coin not in NOT_RANKED]

    def status(self) -> CryptoStatus:
        panel = self.store.read()
        stored = self.store.path.exists()
        return CryptoStatus(
            enabled=True,
            coins=len(self.coins),
            as_of=None if panel.empty else panel["date"].max().date(),
            refreshed=datetime.fromtimestamp(self.store.path.stat().st_mtime, UTC)
            if stored
            else None,
            error=self.error,
        )

    def stale(self, now: datetime) -> bool:
        """Whether the last complete UTC day is not in the panel yet."""
        panel = self.store.read()
        return bool(panel.empty or panel["date"].max().date() < now.date() - timedelta(days=1))

    def refresh(self, now: datetime) -> None:
        """Replace the panel with every coin's daily history; today's open bar is left out."""
        found = cast(
            dict[str, list[dict[str, Any]]],
            self.client.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=[f"{coin}/{QUOTE}" for coin in self.requested],
                    timeframe=TimeFrame.Day,
                    start=now - HISTORY,
                )
            ),
        )
        frame = bars.to_frame(found)
        frame["symbol"] = frame["symbol"].str.removesuffix(f"/{QUOTE}")
        frame = frame[frame["date"] < pd.Timestamp(now.date())]
        if frame.empty:
            raise RuntimeError("Alpaca returned no daily crypto bars")
        self.store.write(frame.sort_values(["symbol", "date"], ignore_index=True))

    def refresh_if_stale(self, now: datetime) -> None:
        """Bring the panel up to yesterday; a failure is kept for the status and tried again."""
        if not self.requested or not self.lock.acquire(blocking=False):
            return
        try:
            if self.stale(now):
                self.refresh(now)
            self.error = None
        except Exception as exc:
            logger.exception("Crypto refresh failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.lock.release()

    def scan(self) -> CryptoScan | None:
        """Coins with a year of history by 4 week return, and why the others are left out."""
        panel = self.store.read()
        if panel.empty:
            return None
        days = panel.groupby("symbol").size()
        kept = [coin for coin in self.requested if days.get(coin, 0) >= MIN_DAYS]
        dropped = [
            Dropped(coin=coin, reason=self.reason(coin, int(days.get(coin, 0))))
            for coin in self.coins
            if coin not in kept
        ]
        closes = (
            panel.pivot(index="date", columns="symbol", values="close")
            .asfreq("D")
            .ffill(limit=GAP_DAYS)[kept]
        )
        last = closes.iloc[-1]
        average = closes.rolling(TREND_DAYS).mean().iloc[-1]
        returns = {name: last / closes.iloc[-1 - span] - 1 for name, span in WINDOWS.items()}
        table = pd.DataFrame({**returns, "price": last, "trend": last / average - 1})
        table = table.sort_values("month", ascending=False, na_position="last")
        rows = [
            CryptoRow(
                rank=rank,
                coin=str(coin),
                price=float(item["price"]),
                week=number(item["week"]),
                month=number(item["month"]),
                quarter=number(item["quarter"]),
                trend=number(item["trend"]),
                above=bool(item["price"] > average[coin]),
                days=int(days[coin]),
            )
            for rank, (coin, item) in enumerate(table.iterrows(), start=1)
        ]
        return CryptoScan(as_of=closes.index[-1].date(), rows=rows, dropped=dropped)

    @staticmethod
    def reason(coin: str, days: int) -> str:
        if coin in NOT_RANKED:
            return "Stablecoin or gold token"
        if days == 0:
            return "No daily bars from Alpaca"
        return f"{days} days of history, {MIN_DAYS} needed"
