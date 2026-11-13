import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.tests.support import api_error
from app.features.scanner import crypto
from app.features.scanner import service as module
from app.features.scanner.crypto import CryptoDesk
from app.features.scanner.router import get_scanner
from app.features.scanner.service import Scanner
from app.features.scanner.tests.support import NOW, Market, crypto_client, trading_with
from app.main import app

TODAY = NOW.date()
YESTERDAY = TODAY - timedelta(days=1)
HISTORY = {
    "AAA": [100.0 + day for day in range(400)],
    "BBB": [1000.0 - day for day in range(400)],
    "SHORT": [10.0 + day for day in range(100)],
    "USDT": [1.0] * 400,
}
COINS = ["AAA", "bbb", "SHORT", "USDT", "MISSING", "AAA/USD"]


@pytest.fixture
def client_for_coins() -> MagicMock:
    return crypto_client(HISTORY, TODAY)


@pytest.fixture
def desk(tmp_path: Path, client_for_coins: MagicMock) -> CryptoDesk:
    return CryptoDesk(client_for_coins, COINS, tmp_path)


def test_coins_are_codes_in_upper_case_without_the_quote_and_listed_once() -> None:
    assert crypto.coins_of([" btc ", "ETH/USD", "btc", "", "sol"]) == ["BTC", "ETH", "SOL"]


def test_a_refresh_stores_every_requested_coin_except_today_which_is_still_open(
    desk: CryptoDesk, client_for_coins: MagicMock
) -> None:
    desk.refresh(NOW)
    request = client_for_coins.get_crypto_bars.call_args.args[0]
    panel = desk.store.read()
    assert request.symbol_or_symbols == ["AAA/USD", "BBB/USD", "SHORT/USD", "MISSING/USD"]
    assert set(panel["symbol"]) == {"AAA", "BBB", "SHORT"}
    assert panel["date"].max().date() == YESTERDAY


def test_the_figures_are_returns_over_7_28_and_91_days_and_the_200_day_trend(
    desk: CryptoDesk,
) -> None:
    desk.refresh(NOW)
    result = desk.scan()
    assert result is not None and result.as_of == YESTERDAY
    first, second = result.rows
    assert (first.rank, first.coin, second.rank, second.coin) == (1, "AAA", 2, "BBB")
    last = 498.0
    assert first.price == last
    assert first.week == pytest.approx(last / 491 - 1)
    assert first.month == pytest.approx(last / 470 - 1)
    assert first.quarter == pytest.approx(last / 407 - 1)
    assert first.trend == pytest.approx(last / 398.5 - 1)
    assert first.above and not second.above
    assert first.days == 399 and second.month is not None and second.month < 0


def test_a_coin_is_dropped_with_the_reason_instead_of_being_ranked(desk: CryptoDesk) -> None:
    desk.refresh(NOW)
    result = desk.scan()
    assert result is not None
    assert [(item.coin, item.reason) for item in result.dropped] == [
        ("SHORT", "99 days of history, 365 needed"),
        ("USDT", "Stablecoin or gold token"),
        ("MISSING", "No daily bars from Alpaca"),
    ]


def test_there_is_no_scan_before_the_first_refresh(desk: CryptoDesk) -> None:
    assert desk.scan() is None
    assert desk.status().as_of is None and desk.status().refreshed is None


def test_the_panel_is_stale_until_the_last_complete_day_is_in_it(desk: CryptoDesk) -> None:
    assert desk.stale(NOW)
    desk.refresh(NOW)
    assert not desk.stale(NOW)
    assert not desk.stale(NOW + timedelta(hours=8))
    assert desk.stale(NOW + timedelta(days=1))


def test_a_fresh_panel_is_not_fetched_again(desk: CryptoDesk, client_for_coins: MagicMock) -> None:
    desk.refresh_if_stale(NOW)
    desk.refresh_if_stale(NOW)
    assert client_for_coins.get_crypto_bars.call_count == 1
    assert desk.error is None


def test_a_failed_refresh_keeps_the_old_panel_and_the_error_until_it_works_again(
    desk: CryptoDesk, client_for_coins: MagicMock
) -> None:
    desk.refresh_if_stale(NOW)
    client_for_coins.get_crypto_bars.side_effect = api_error("service unavailable")
    desk.refresh_if_stale(NOW + timedelta(days=1))
    assert desk.error is not None and "service unavailable" in desk.error
    assert desk.scan() is not None and desk.status().error == desk.error
    client_for_coins.get_crypto_bars.side_effect = crypto_client(HISTORY, TODAY).get_crypto_bars
    desk.refresh_if_stale(NOW + timedelta(days=1))
    assert desk.error is None


def test_an_empty_answer_never_replaces_the_stored_history(
    desk: CryptoDesk, client_for_coins: MagicMock
) -> None:
    desk.refresh(NOW)
    client_for_coins.get_crypto_bars.side_effect = None
    client_for_coins.get_crypto_bars.return_value = {}
    with pytest.raises(RuntimeError, match="no daily crypto bars"):
        desk.refresh(NOW + timedelta(days=1))
    assert desk.scan() is not None


def test_only_stablecoins_listed_means_nothing_to_fetch(tmp_path: Path) -> None:
    client = crypto_client(HISTORY, TODAY)
    CryptoDesk(client, ["USDT", "PAXG"], tmp_path).refresh_if_stale(NOW)
    client.get_crypto_bars.assert_not_called()


def scanner_with(tmp_path: Path, desk: CryptoDesk | None) -> Scanner:
    return Scanner(trading_with([]), Market().client(), root=tmp_path / "scanner", crypto=desk)


def test_the_status_and_the_scan_come_from_the_desk(tmp_path: Path) -> None:
    desk = CryptoDesk(crypto_client(HISTORY, TODAY), COINS, tmp_path / "scanner")
    scanner = scanner_with(tmp_path, desk)
    assert scanner.crypto_scan() is None and scanner.status().crypto.as_of is None
    desk.refresh(NOW)
    status = scanner.status().crypto
    assert (status.enabled, status.coins, status.as_of) == (True, 5, YESTERDAY)
    scan = scanner.crypto_scan()
    assert scan is not None and [row.coin for row in scan.rows] == ["AAA", "BBB"]


def test_without_a_desk_crypto_is_off(tmp_path: Path) -> None:
    scanner = scanner_with(tmp_path, None)
    assert scanner.crypto_scan() is None and not scanner.status().crypto.enabled


def test_the_panel_is_checked_every_hour_and_first_right_after_a_start(tmp_path: Path) -> None:
    async def scenario() -> tuple[datetime | None, datetime | None]:
        desk = CryptoDesk(crypto_client(HISTORY, datetime.now(UTC).date()), COINS, tmp_path)
        scanner = scanner_with(tmp_path, desk)
        scanner.start()
        job = scanner.scheduler.get_job(module.CRYPTO_JOB_ID)
        found = None if job is None else job.next_run_time
        await asyncio.sleep(module.CATCH_UP_DELAY.total_seconds() + 1)
        scanner.stop()
        return found, desk.status().refreshed

    first, refreshed = asyncio.run(scenario())
    assert first is not None and first - datetime.now(UTC) <= module.CATCH_UP_DELAY
    assert refreshed is not None


def test_the_route_serves_the_scan_or_null(client: TestClient, tmp_path: Path) -> None:
    desk = CryptoDesk(crypto_client(HISTORY, TODAY), COINS, tmp_path / "scanner")
    built = scanner_with(tmp_path, desk)
    app.dependency_overrides[get_scanner] = lambda: built
    assert client.get("/scanner/crypto").json() is None
    desk.refresh(NOW)
    body = client.get("/scanner/crypto").json()
    assert body["as_of"] == YESTERDAY.isoformat()
    assert [row["coin"] for row in body["rows"]] == ["AAA", "BBB"]
    assert body["dropped"][0] == {"coin": "SHORT", "reason": "99 days of history, 365 needed"}
    assert date.fromisoformat(body["as_of"]) == YESTERDAY
