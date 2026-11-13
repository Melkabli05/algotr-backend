from datetime import date, timedelta
from pathlib import Path

from app.features.scanner import lists, screening
from app.features.scanner.lists import Snapshot

TODAY = date(2026, 11, 6)
SPUS = Snapshot(name="SPUS", as_of=TODAY - timedelta(days=2), symbols=["AAPL", "MSFT"])
OLD = Snapshot(name="HLAL", as_of=TODAY - lists.STALE_AFTER - timedelta(days=1), symbols=["XOM"])


def test_a_stock_on_a_current_list_is_allowed_with_the_list_named() -> None:
    found = screening.screen("AAPL", [SPUS], [], TODAY)
    assert found.allowed and found.reason == "held by SPUS (2026-11-04)"


def test_a_stock_on_no_list_is_refused() -> None:
    found = screening.screen("JPM", [SPUS], [], TODAY)
    assert not found.allowed and found.reason == "not held by SPUS (2026-11-04)"


def test_a_stale_list_approves_nothing() -> None:
    assert not screening.screen("XOM", [SPUS, OLD], [], TODAY).allowed
    found = screening.screen("XOM", [OLD], [], TODAY)
    assert not found.allowed and "no Shariah list is current" in found.reason


def test_the_screened_funds_themselves_are_allowed_without_a_list() -> None:
    assert screening.screen("spsk", [], [], TODAY).allowed


def test_a_coin_is_allowed_only_when_listed() -> None:
    assert screening.screen("BTC/USD", [], ["BTC", "ETH"], TODAY).allowed
    assert not screening.screen("DOGE/USD", [], ["BTC"], TODAY).allowed


def test_the_scanner_screens_against_its_current_list_on_disk(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from app.features.scanner.tests.support import NAMES, listed_scanner, noisy_market, spus_csv

    today = datetime.now(UTC).date()
    scanner = listed_scanner(tmp_path, noisy_market())
    folder = tmp_path / "scanner" / "lists"
    folder.mkdir(parents=True)
    (folder / f"SPUS-{today.isoformat()}.csv").write_text(
        spus_csv(NAMES, as_of=today.strftime("%m/%d/%Y"))
    )
    assert scanner.listed() == NAMES
    assert scanner.screen("AAA").allowed
    assert not scanner.screen("ZZZ").allowed
