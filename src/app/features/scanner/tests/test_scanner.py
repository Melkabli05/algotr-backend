import asyncio
from collections.abc import Collection
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from alpaca.data.requests import StockBarsRequest

from app.core.tests.support import api_error, until
from app.features.scanner import rank
from app.features.scanner import refresh as refreshing
from app.features.scanner import scan as analysis
from app.features.scanner import service as module
from app.features.scanner.schemas import Progress, Refresh
from app.features.scanner.service import AlreadyRunningError, Scanner
from app.features.scanner.tests.support import (
    SOURCE,
    URL,
    Market,
    asset,
    last_business_day,
    listed_scanner,
    noisy_market,
    sessions,
    spus_csv,
    trading_with,
    website,
    yesterday,
)
from app.quant.history import NotEnoughHistoryError

ASSETS = [
    asset("AAA"),
    asset("BBB"),
    asset("CCC"),
    asset("DDD"),
    asset("EEE", exchange="OTC"),
]


@pytest.fixture
def market() -> Market:
    found = Market()
    found.add("AAA")
    found.add("BBB")
    found.add("CCC", volume=1_000)
    found.add("DDD", price=5.0, volume=10_000_000)
    found.add("EEE")
    found.through = sessions(last_business_day(yesterday()), 4)[0]
    return found


def build(
    tmp_path: Path,
    market: Market,
    listed: list[str] | None = None,
    assets: list[dict[str, object]] | None = None,
    files: dict[str, str] | None = None,
) -> Scanner:
    names = ["AAA", "CCC", "GONE"] if listed is None else listed
    return Scanner(
        trading_with(assets or ASSETS),
        market.client(),
        root=tmp_path / "scanner",
        sources=[SOURCE],
        http=website({URL: spus_csv(names)} if files is None else files),
    )


def client_of(scanner: Scanner) -> MagicMock:
    return cast(MagicMock, scanner.stocks)


def panel_of(scanner: Scanner) -> pd.DataFrame:
    return scanner.store.read()


def test_a_cold_build_holds_the_liquid_and_the_listed_names(tmp_path: Path, market: Market) -> None:
    scanner = build(tmp_path, market)
    record = scanner.refresh()
    assert record.status == "ok" and record.detail is None
    frame = panel_of(scanner)
    assert set(frame["symbol"]) == {"AAA", "BBB", "CCC"}
    assert frame.groupby("symbol").size().min() > 1500
    assert record.session == market.through
    assert record.rows == len(frame)
    funnel = record.funnel
    assert funnel is not None
    assert (funnel.assets, funnel.tradable, funnel.liquid) == (5, 4, 2)
    assert (funnel.listed, funnel.listed_liquid, funnel.panel) == (2, 1, 3)


def test_the_funnel_shows_each_list_and_the_names_alpaca_does_not_know(
    tmp_path: Path, market: Market
) -> None:
    record = build(tmp_path, market).refresh()
    [spus] = record.lists
    assert (spus.name, spus.symbols, spus.unmatched) == ("SPUS", 3, ["GONE"])
    assert spus.stale is False and spus.error is None


def test_a_later_night_adds_only_the_new_sessions(tmp_path: Path, market: Market) -> None:
    scanner = build(tmp_path, market)
    scanner.refresh()
    before = panel_of(scanner)
    market.through = last_business_day(yesterday())
    market.requests.clear()
    record = scanner.refresh()
    after = panel_of(scanner)
    assert market.history_requests() == []
    assert record.rebuilt == []
    assert record.session == market.through
    assert len(after) == len(before) + 3 * 3
    assert not after.duplicated(["symbol", "date"]).any()


def test_a_split_rebuilds_that_symbol_and_no_other(tmp_path: Path, market: Market) -> None:
    scanner = build(tmp_path, market)
    scanner.refresh()
    market.through = last_business_day(yesterday())
    market.split("AAA", 2, on=sessions(market.through, 5)[0])
    market.requests.clear()
    record = scanner.refresh()
    assert record.rebuilt == ["AAA"]
    assert market.history_requests() == [["AAA"]]
    stored = panel_of(scanner).query("symbol == 'AAA'").set_index("date")["close"]
    served = pd.Series(
        {pd.Timestamp(row["t"][:10]): row["c"] for row in market.served("AAA")}
    ).reindex(stored.index)
    pd.testing.assert_series_equal(stored, served, check_names=False, check_freq=False)


def test_a_symbol_alpaca_refuses_does_not_stop_the_refresh(tmp_path: Path, market: Market) -> None:
    market.invalid = {"BAD"}
    record = build(tmp_path, market, assets=[*ASSETS, asset("BAD")]).refresh()
    assert record.status == "ok"
    assert record.rejected == ["BAD"]
    assert record.funnel is not None and record.funnel.panel == 3


def test_a_list_older_than_a_week_gives_no_trading_names(tmp_path: Path, market: Market) -> None:
    old = (datetime.now(UTC) - timedelta(days=20)).strftime("%m/%d/%Y")
    record = build(tmp_path, market, files={URL: spus_csv(["AAA"], as_of=old)}).refresh()
    assert record.funnel is not None and record.funnel.listed == 0
    assert record.lists[0].stale is True and record.lists[0].age_days == 20


def test_the_last_good_list_is_used_when_the_download_fails(tmp_path: Path, market: Market) -> None:
    folder = tmp_path / "scanner" / "lists"
    folder.mkdir(parents=True)
    fresh = (datetime.now(UTC) - timedelta(days=1)).strftime("%m/%d/%Y")
    (folder / f"SPUS-{yesterday().isoformat()}.csv").write_text(spus_csv(["AAA"], as_of=fresh))
    record = build(tmp_path, market, files={}).refresh()
    assert record.funnel is not None and record.funnel.listed == 1
    assert record.lists[0].error is not None and record.lists[0].error.startswith("download:")


def test_a_cold_build_that_fails_midway_keeps_what_it_saved_and_resumes(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refreshing, "HISTORY_CHUNK", 1)
    monkeypatch.setattr(refreshing, "HISTORY_BATCH", 1)
    scanner = build(tmp_path, market)
    real = market.get_stock_bars
    deep = {"calls": 0}

    def flaky(request: StockBarsRequest) -> dict[str, list[dict[str, Any]]]:
        if request.start and request.start.replace(tzinfo=None) < datetime.now() - timedelta(
            days=100
        ):
            deep["calls"] += 1
            if deep["calls"] == 2:
                raise api_error("connection reset")
        return real(request)

    client_of(scanner).get_stock_bars.side_effect = flaky
    failed = scanner.refresh()
    assert failed.status == "error"
    assert set(panel_of(scanner)["symbol"]) == {"AAA"}

    client_of(scanner).get_stock_bars.side_effect = real
    market.requests.clear()
    resumed = scanner.refresh()
    assert resumed.status == "ok"
    assert set(panel_of(scanner)["symbol"]) == {"AAA", "BBB", "CCC"}
    assert sorted(symbol for batch in market.history_requests() for symbol in batch) == [
        "BBB",
        "CCC",
    ]


def test_a_failed_refresh_is_recorded_and_the_old_panel_is_kept(
    tmp_path: Path, market: Market
) -> None:
    scanner = build(tmp_path, market)
    scanner.refresh()
    before = panel_of(scanner)
    market.through = last_business_day(yesterday())
    client_of(scanner).get_stock_bars.side_effect = api_error("forbidden")
    record = scanner.refresh()
    assert record.status == "error" and "APIError" in (record.detail or "")
    pd.testing.assert_frame_equal(panel_of(scanner), before)
    assert Scanner(scanner.trading, scanner.stocks, root=scanner.root).state.last == record


def test_the_state_survives_a_restart(tmp_path: Path, market: Market) -> None:
    scanner = build(tmp_path, market)
    record = scanner.refresh()
    restarted = Scanner(scanner.trading, scanner.stocks, root=scanner.root)
    assert restarted.state.last == record
    assert restarted.status().last == record


def test_a_second_refresh_is_refused_while_one_is_running(tmp_path: Path, market: Market) -> None:
    scanner = build(tmp_path, market)
    scanner.progress = Progress(stage="History", done=1, total=2)
    with scanner.lock:
        assert scanner.status().running == Progress(stage="History", done=1, total=2)
        with pytest.raises(AlreadyRunningError):
            scanner.refresh()
        with pytest.raises(AlreadyRunningError):
            scanner.request_run()
    assert scanner.status().running is None


def test_progress_is_visible_while_the_refresh_runs(tmp_path: Path, market: Market) -> None:
    scanner = build(tmp_path, market)
    seen: list[Progress | None] = []
    real = market.get_stock_bars

    def spy(request: StockBarsRequest) -> dict[str, list[dict[str, Any]]]:
        seen.append(scanner.status().running)
        return real(request)

    client_of(scanner).get_stock_bars.side_effect = spy
    scanner.refresh()
    assert None not in seen
    assert {progress.stage for progress in seen if progress} == {"Recent bars", "History"}
    assert scanner.status().running is None


def test_refreshing_twice_in_a_night_changes_nothing(tmp_path: Path, market: Market) -> None:
    scanner = build(tmp_path, market)
    first = scanner.refresh()
    second = scanner.refresh()
    assert second.rows == first.rows
    assert second.rebuilt == []


def test_the_nightly_job_runs_at_one_past_midnight_utc(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", time.max)

    async def scenario() -> tuple[str, object]:
        scanner = build(tmp_path, market)
        scanner.start()
        job = scanner.scheduler.get_job(module.JOB_ID)
        upcoming = scanner.status().next_run
        scanner.stop()
        return str(job.trigger), upcoming

    trigger, upcoming = asyncio.run(scenario())
    assert "hour='0'" in trigger and "minute='1'" in trigger
    assert isinstance(upcoming, datetime) and (upcoming.hour, upcoming.minute) == (0, 1)


def test_a_missed_refresh_is_caught_up_on_start(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    monkeypatch.setattr(module, "CATCH_UP_DELAY", timedelta(milliseconds=50))

    async def scenario() -> Refresh | None:
        scanner = build(tmp_path, market)
        scanner.start()
        await until(lambda: scanner.state.last is not None)
        scanner.stop()
        return scanner.state.last

    last = asyncio.run(scenario())
    assert last is not None and last.status == "ok"


def test_nothing_is_caught_up_when_tonights_refresh_already_ran(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    scanner = build(tmp_path, market)
    assert scanner._missed_today() is True
    now = datetime.now(UTC)
    scanner.state.last = Refresh(started=now, seconds=1, status="ok")
    assert scanner._missed_today() is False
    scanner.state.last = Refresh(started=now, seconds=1, status="error", detail="boom")
    assert scanner._missed_today() is True
    scanner.state.last = Refresh(started=now - timedelta(days=1), seconds=1, status="ok")
    assert scanner._missed_today() is True


def test_nothing_is_caught_up_before_the_nightly_time(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", time.max)
    assert build(tmp_path, market)._missed_today() is False


def test_a_refresh_that_failed_offline_is_repeated_by_the_catch_up_check(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    scanner = build(tmp_path, market)
    client = client_of(scanner)
    working = client.get_stock_bars.side_effect

    def status() -> str | None:
        return scanner.state.last.status if scanner.state.last else None

    client.get_stock_bars.side_effect = ConnectionError("offline")
    asyncio.run(scanner._caught_up())
    assert status() == "error"
    client.get_stock_bars.side_effect = working
    asyncio.run(scanner._caught_up())
    assert status() == "ok"


def test_the_catch_up_check_does_nothing_once_today_has_succeeded(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    scanner = build(tmp_path, market)
    asyncio.run(scanner._caught_up())
    first = scanner.state.last
    asyncio.run(scanner._caught_up())
    assert first is not None and scanner.state.last is first


def test_a_refresh_that_keeps_failing_is_given_up_for_the_day(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    scanner = build(tmp_path, market)
    client_of(scanner).get_stock_bars.side_effect = ConnectionError("offline")
    for _ in range(module.MAX_FAILURES):
        assert scanner._missed_today() is True
        scanner.refresh()
    assert scanner._missed_today() is False


def test_the_catch_up_check_is_scheduled_every_ten_minutes(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", time.max)

    async def scenario() -> object:
        scanner = build(tmp_path, market)
        scanner.start()
        job = scanner.scheduler.get_job(module.CATCH_UP_JOB_ID)
        scanner.stop()
        return job.trigger.interval

    assert asyncio.run(scenario()) == module.CATCH_UP_EVERY


def test_a_refresh_can_be_requested_and_runs_in_the_background(
    tmp_path: Path, market: Market, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", time.max)

    async def scenario() -> Refresh | None:
        scanner = build(tmp_path, market)
        scanner.start()
        scanner.request_run()
        await until(lambda: scanner.state.last is not None)
        scanner.stop()
        return scanner.state.last

    last = asyncio.run(scenario())
    assert last is not None and last.status == "ok"


def test_a_scheduled_refresh_that_overlaps_another_is_skipped_quietly(
    tmp_path: Path, market: Market
) -> None:
    scanner = build(tmp_path, market)
    with scanner.lock:
        asyncio.run(scanner._scheduled())
    assert scanner.state.last is None


def test_the_variants_are_judged_on_the_stored_panel_and_the_result_is_audited(
    tmp_path: Path,
) -> None:
    market = noisy_market()
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    scanner = build(tmp_path, market, listed=names, assets=[asset(name) for name in names])
    scanner.refresh()
    result = scanner.evaluate()
    assert result.symbols == 6 and result.trials == len(rank.VARIANTS)
    assert result.lists[0].startswith("SPUS ")
    [entry] = [entry for entry in scanner.audit.entries() if entry.kind == "scan-evaluation"]
    assert entry.payload["result"]["trials"] == len(rank.VARIANTS)
    assert entry.payload["manifest"]["data"]["close"]["symbols"] == names
    assert scanner.audit.verify().valid


def test_there_is_nothing_to_judge_before_the_first_refresh(tmp_path: Path, market: Market) -> None:
    with pytest.raises(NotEnoughHistoryError):
        build(tmp_path, market).evaluate()


def test_a_refresh_logs_the_days_picks_once(tmp_path: Path) -> None:
    market = noisy_market()
    scanner = listed_scanner(tmp_path, market)
    record = scanner.refresh()
    again = scanner.refresh()
    entries = [entry for entry in scanner.audit.entries() if entry.kind == "scan"]
    assert len(entries) == 1
    assert record.scanned == again.scanned == market.through
    assert entries[0].payload["scan"]["session"] == market.through.isoformat()
    assert "candidates" not in entries[0].payload["scan"]
    assert scanner.audit.verify().valid


def test_nothing_is_logged_when_no_list_is_current(tmp_path: Path) -> None:
    old = (datetime.now(UTC) - timedelta(days=30)).strftime("%m/%d/%Y")
    scanner = listed_scanner(tmp_path, noisy_market(), as_of=old)
    record = scanner.refresh()
    assert record.status == "ok" and record.scanned is None
    assert record.detail == "No current Shariah list, so no scan was logged"
    assert not [entry for entry in scanner.audit.entries() if entry.kind == "scan"]


def test_a_new_session_adds_a_second_scan_and_the_first_gets_its_forward_returns(
    tmp_path: Path,
) -> None:
    market = noisy_market()
    market.through = sessions(last_business_day(yesterday()), 4)[0]
    scanner = listed_scanner(tmp_path, market)
    scanner.refresh()
    assert all(value is None for row in scanner.shadow().rows for value in row.returns.values())
    market.through = last_business_day(yesterday())
    scanner.refresh()
    report = scanner.shadow()
    assert len([e for e in scanner.audit.entries() if e.kind == "scan"]) == 2
    first = [row for row in report.rows if row.session == sessions(market.through, 4)[0]]
    assert first and all(row.returns[1] is not None for row in first)
    assert all(row.returns[21] is None for row in first)


def test_the_latest_scan_ranks_the_market_and_names_the_stocks(tmp_path: Path) -> None:
    scanner = listed_scanner(tmp_path, noisy_market())
    assert scanner.latest() is None
    scanner.refresh()
    result, last = scanner.latest(), scanner.state.last
    assert result is not None and last is not None
    assert result.session == last.session
    assert result.candidates and result.candidates[0].name == "AAA Inc"
    assert result.funnel.listed == 6
    explanation = scanner.explain("AAA")
    assert explanation is not None and explanation.symbol == "AAA"


def test_names_that_arrive_later_are_picked_up_without_a_new_panel(tmp_path: Path) -> None:
    scanner = listed_scanner(tmp_path, noisy_market())
    scanner.refresh()
    (scanner.root / "names.json").write_text('{"AAA": "Renamed Inc"}')
    top = scanner.latest()
    assert top is not None
    named = {c.symbol: c.name for c in top.candidates}
    assert named["AAA"] == "Renamed Inc"


def test_the_market_is_analysed_once_until_the_panel_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    market = noisy_market()
    market.through = sessions(last_business_day(yesterday()), 4)[0]
    scanner = listed_scanner(tmp_path, market)
    calls: list[int] = []
    real = analysis.analyse

    def counted(panel: pd.DataFrame, members: Collection[str]) -> analysis.Analysis:
        calls.append(len(panel))
        return real(panel, members)

    monkeypatch.setattr(analysis, "analyse", counted)
    scanner.refresh()
    scanner.latest()
    scanner.explain("AAA")
    scanner.latest()
    assert len(calls) == 1
    market.through = last_business_day(yesterday())
    scanner.refresh()
    scanner.latest()
    assert len(calls) == 2
