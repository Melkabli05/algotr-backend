import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from app.features.scanner.lists import TICKER

EXCHANGES = frozenset(
    exchange.value
    for exchange in (
        AssetExchange.NASDAQ,
        AssetExchange.NYSE,
        AssetExchange.ARCA,
        AssetExchange.AMEX,
        AssetExchange.BATS,
    )
)
NAMES = "names.json"
MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 20_000_000.0
MIN_SESSIONS = 15
SESSIONS = 20


def assets(trading: TradingClient) -> list[dict[str, Any]]:
    found = trading.get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    )
    return cast(list[dict[str, Any]], found)


def save_names(folder: Path, found: list[dict[str, Any]], tradable: set[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    names = {asset["symbol"]: asset["name"] for asset in found if asset["symbol"] in tradable}
    (folder / NAMES).write_text(json.dumps(names))


def load_names(folder: Path) -> dict[str, str]:
    try:
        names: dict[str, str] = json.loads((folder / NAMES).read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return names


def tradable(found: list[dict[str, Any]]) -> set[str]:
    """Plain tickers of tradable stocks on the listed exchanges; OTC and odd ids are left out."""
    return {
        asset["symbol"]
        for asset in found
        if asset["tradable"]
        and asset["exchange"] in EXCHANGES
        and TICKER.fullmatch(asset["symbol"])
    }


def liquid(recent: pd.DataFrame) -> set[str]:
    """Symbols above the price and dollar-volume floors, measured on their last 20 sessions."""
    last = recent.sort_values(["symbol", "date"]).groupby("symbol").tail(SESSIONS)
    stats = (
        last.assign(dollars=last["close"] * last["volume"])
        .groupby("symbol")
        .agg(price=("close", "last"), dollars=("dollars", "mean"), sessions=("date", "count"))
    )
    passing = stats[
        (stats["sessions"] >= MIN_SESSIONS)
        & (stats["price"] >= MIN_PRICE)
        & (stats["dollars"] >= MIN_DOLLAR_VOLUME)
    ]
    return set(passing.index)
