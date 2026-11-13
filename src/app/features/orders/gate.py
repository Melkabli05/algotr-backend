import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from pydantic import BaseModel
from requests import RequestException

from app.core.compliance import BUYS_STOPPED, KillSwitch, Sectors, ShariahScreen
from app.features.orders import risk
from app.quant import history

logger = logging.getLogger(__name__)

ADV_DAYS = 20
ADV_LOOKBACK = timedelta(days=40)
BPS = 1e4
UNCAPPED = ("Fund", "Crypto", "Unclassified")


class Limits(BaseModel):
    """The account's own limits on a buy, on top of the cash and book checks."""

    position_share: float
    daily_loss: float
    adv_share: float
    spread_bps: float
    sector_share: float = 1.0


class PreTrade:
    """Every check a buy passes before it reaches the broker; a sell is never stopped here.

    In order: the Shariah screen, the autopilot's kill switch, the day's loss, the size of the
    position it would leave, its share of the stock's daily trading, and the quoted spread of
    a market order. Each check that cannot read what it needs refuses, so a buy never passes
    for want of data.
    """

    def __init__(
        self,
        trading: TradingClient,
        stocks: StockHistoricalDataClient,
        crypto: CryptoHistoricalDataClient,
        feed: DataFeed,
        limits: Limits,
        screen: ShariahScreen,
        kill: KillSwitch,
        sectors: Sectors | None = None,
    ) -> None:
        self.trading = trading
        self.stocks = stocks
        self.crypto = crypto
        self.feed = feed
        self.limits = limits
        self.screen = screen
        self.kill = kill
        self.sectors = sectors

    def check(self, symbol: str, side: str, qty: float, price: float | None) -> dict[str, float]:
        """Refuse a buy past any limit; for one that passes, how much of each limit it uses."""
        usage: dict[str, float] = {}
        if side != "buy":
            return usage
        verdict = self.screen.screen(symbol)
        if not verdict.allowed:
            raise risk.RiskRefusedError(f"{symbol} fails the Shariah screen: {verdict.reason}")
        state = self.kill.kill_state()
        if state in BUYS_STOPPED:
            raise risk.RiskRefusedError(
                f"The kill switch is at {state}: new buys are stopped until losses recover"
            )
        equity, last = self.equity()
        if last > 0:
            usage["Daily loss"] = max(0.0, (last - equity) / last) / self.limits.daily_loss
        if last > 0 and (equity - last) / last <= -self.limits.daily_loss:
            raise risk.RiskRefusedError(
                f"The account is down {(last - equity) / last:.1%} today, past the "
                f"{self.limits.daily_loss:.0%} daily loss limit; buys resume tomorrow"
            )
        unit = price or risk.last_price(self.stocks, self.crypto, symbol, self.feed)
        if unit is None:
            raise risk.RiskRefusedError(f"The price of {symbol} could not be read")
        cost = qty * unit
        held = self.held_value(symbol)
        if equity > 0:
            usage["Position size"] = (held + cost) / (self.limits.position_share * equity)
        if held + cost > self.limits.position_share * equity:
            raise risk.RiskRefusedError(
                f"{symbol} would be ${held + cost:,.0f}, more than the "
                f"{self.limits.position_share:.0%} of account value one position may be"
            )
        if "/" in symbol:
            return usage
        sector = self.check_sector(symbol, cost, equity)
        if sector is not None:
            usage["Sector"] = sector
        volume = self.daily_volume(symbol)
        usage["Daily volume"] = cost / (self.limits.adv_share * volume)
        if cost > self.limits.adv_share * volume:
            raise risk.RiskRefusedError(
                f"${cost:,.0f} is more than {self.limits.adv_share:.1%} of the "
                f"${volume:,.0f} {symbol} trades on an average day"
            )
        if price is None:
            spread = self.spread_bps(symbol)
            usage["Spread"] = spread / self.limits.spread_bps
            if spread > self.limits.spread_bps:
                raise risk.RiskRefusedError(
                    f"The spread of {symbol} is {spread:.0f} bps, over the "
                    f"{self.limits.spread_bps:.0f} bps a market buy may cross; use a limit order"
                )
        return usage

    def check_sector(self, symbol: str, cost: float, equity: float) -> float | None:
        """Refuse a stock buy that would leave its GICS sector above its share of the account."""
        if self.sectors is None:
            return None
        sector = self.sectors.sector_of(symbol)
        if sector in UNCAPPED:
            return None
        held = sum(
            abs(float(item["market_value"]))
            for item in cast(list[dict[str, Any]], self.trading.get_all_positions())
            if item.get("asset_class") != "crypto"
            and self.sectors.sector_of(str(item["symbol"])) == sector
        )
        if held + cost > self.limits.sector_share * equity:
            raise risk.RiskRefusedError(
                f"{sector} would be ${held + cost:,.0f}, more than the "
                f"{self.limits.sector_share:.0%} of account value one sector may be"
            )
        return (held + cost) / (self.limits.sector_share * equity) if equity > 0 else None

    def equity(self) -> tuple[float, float]:
        raw = cast(dict[str, Any], self.trading.get_account())
        try:
            return float(raw["equity"]), float(raw["last_equity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise risk.RiskRefusedError("The account value could not be read") from exc

    def held_value(self, symbol: str) -> float:
        try:
            found = cast(dict[str, Any], self.trading.get_open_position(symbol.replace("/", "")))
        except APIError as exc:
            if exc.status_code == 404:
                return 0.0
            raise risk.RiskRefusedError(f"The position in {symbol} could not be read") from exc
        return float(found["market_value"])

    def daily_volume(self, symbol: str) -> float:
        """The stock's average dollar volume over its last `ADV_DAYS` sessions, all venues.

        Consolidated (SIP) bars, which the free plan serves once they are 16 minutes old; IEX's
        own volume is a few percent of the market's and would make every limit tiny.
        """
        now = datetime.now(UTC)
        try:
            found = cast(
                dict[str, list[dict[str, Any]]],
                self.stocks.get_stock_bars(
                    StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Day,
                        start=now - ADV_LOOKBACK,
                        end=now - history.SIP_DELAY,
                        feed=DataFeed.SIP,
                    )
                ),
            )
            rows = found.get(symbol, [])[-ADV_DAYS:]
        except (APIError, RequestException) as exc:
            raise risk.RiskRefusedError(f"The daily volume of {symbol} could not be read") from exc
        if not rows:
            raise risk.RiskRefusedError(f"{symbol} has no recent trading to size against")
        return sum(float(row["c"]) * float(row["v"]) for row in rows) / len(rows)

    def spread_bps(self, symbol: str) -> float:
        try:
            found = cast(
                dict[str, dict[str, Any]],
                self.stocks.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.feed)
                ),
            )
            bid, ask = float(found[symbol]["bp"]), float(found[symbol]["ap"])
        except (APIError, RequestException, KeyError, TypeError, ValueError) as exc:
            raise risk.RiskRefusedError(f"The quote of {symbol} could not be read") from exc
        if bid <= 0 or ask <= bid:
            raise risk.RiskRefusedError(f"{symbol} has no two-sided quote now; use a limit order")
        return (ask - bid) / ((ask + bid) / 2) * BPS
