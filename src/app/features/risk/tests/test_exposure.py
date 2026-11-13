from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from app.features.portfolio.schemas import Position
from app.features.risk import sectors, service

CSV = "Symbol,Security,GICS Sector\nAAPL,Apple,Information Technology\nBRK-B,Berkshire,Financials\n"


def held(symbol: str, value: float, asset_class: str = "us_equity") -> Position:
    return Position(
        symbol=symbol,
        asset_class=asset_class,
        side="long",
        qty=1,
        avg_entry_price=value,
        current_price=value,
        market_value=value,
        unrealized_pl=0,
    )


def client_serving(text: str, status: int = 200) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status, text=text)))


def test_sectors_come_from_the_file_with_funds_and_coins_by_kind(tmp_path: Path) -> None:
    found = sectors.SectorMap(tmp_path / "sectors.csv", client_serving(CSV))
    assert found.sector_of("AAPL") == sectors.UNKNOWN
    assert found.refresh() is True
    assert found.sector_of("aapl") == "Information Technology"
    assert found.sector_of("BRK.B") == "Financials"
    assert found.sector_of("SPUS") == sectors.FUND
    assert found.sector_of("BTC/USD") == sectors.CRYPTO
    assert found.sector_of("ZZZZ") == sectors.UNKNOWN
    assert found.refresh() is False


def test_a_bad_download_keeps_the_last_good_file(tmp_path: Path) -> None:
    path = tmp_path / "sectors.csv"
    path.write_text(CSV)
    import os

    os.utime(path, (0, 0))
    assert sectors.SectorMap(path, client_serving("not,a,csv")).refresh() is False
    assert sectors.SectorMap(path).sector_of("AAPL") == "Information Technology"


def test_exposure_by_sector_and_strategy_with_the_rest_held_by_hand() -> None:
    positions = [held("AAPL", 600), held("SPUS", 300), held("BTCUSD", 100, "crypto")]
    lookup = {"AAPL": "Information Technology", "SPUS": "Fund", "BTC/USD": "Crypto"}.__getitem__
    by_sector = service.sector_exposure(positions, lookup, 1000)
    assert [(item.name, item.share) for item in by_sector] == [
        ("Information Technology", 0.6),
        ("Fund", 0.3),
        ("Crypto", 0.1),
    ]
    books = {"halal-etf-trend": NS(holdings={"SPUS": NS(qty=2, price=140.0)})}
    by_strategy = service.strategy_exposure(positions, books, 1000)
    names = {item.name: item.value for item in by_strategy}
    assert names == {"halal-etf-trend": 600, "By hand": 400}


def test_the_highest_correlation_is_between_material_holdings_only() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, 300)
    returns = pd.DataFrame(
        {"A": base, "B": base + rng.normal(0, 0.001, 300), "C": rng.normal(0, 0.01, 300)}
    )
    weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.01})
    assert service.highest_pair(returns, weights) == pytest.approx(0.99, abs=0.01)
    assert service.highest_pair(returns, pd.Series({"A": 0.5, "C": 0.01})) is None


def entry(kind: str, payload: dict[str, Any], time: str = "2026-11-06T15:00:00+00:00") -> Any:
    return NS(kind=kind, payload=payload, time=time)


def test_refusals_say_which_limit_last_fired_and_become_events() -> None:
    entries = [
        entry(
            "order",
            {
                "outcome": "refused",
                "reason": "SPUS would be $46,264, more than the 25% of account value "
                "one position may be",
                "request": {"side": "buy", "symbol": "SPUS"},
            },
        ),
        entry("order", {"outcome": "placed"}),
        entry(
            "run", {"run": {"status": "halted", "detail": "drawdown"}}, "2026-11-06T16:00:00+00:00"
        ),
    ]
    triggered = service.last_triggered(entries)
    assert set(triggered) == {"Position size"}
    drift = [
        NS(
            adverse=True,
            time=date(2026, 11, 1),
            stream="ai-ppo",
            kind="mean",
            distance=0.3,
            p_value=0.0001,
        )
    ]
    found = service.events(entries, drift)
    assert [event.severity for event in found] == ["critical", "warning", "info"]
    limit = service.limit("Position size", 0.48, 0.25, "share", triggered)
    assert limit.used == pytest.approx(1.92) and limit.last_triggered == datetime(
        2026, 11, 6, 15, tzinfo=UTC
    )
