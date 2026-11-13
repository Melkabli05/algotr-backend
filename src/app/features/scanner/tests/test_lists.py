from datetime import date, timedelta
from pathlib import Path

import pytest

from app.features.scanner import lists
from app.features.scanner.lists import ListFormatError, Snapshot, Source
from app.features.scanner.tests.support import spus_csv, website

URL = "https://lists.test/SPUS.csv"
SOURCE = Source(name="SPUS", url=URL)
TODAY = date(2026, 11, 3)


def test_the_rows_that_are_not_securities_are_dropped() -> None:
    snapshot = lists.parse("SPUS", spus_csv(["AAPL", "MSFT"]), TODAY)
    assert snapshot.symbols == ["AAPL", "MSFT"]
    assert snapshot.as_of == date(2026, 11, 2)


def test_share_classes_are_written_the_way_alpaca_writes_them() -> None:
    assert lists.parse("SPUS", spus_csv(["BRK/B", "AAPL"]), TODAY).symbols == ["AAPL", "BRK.B"]


def test_another_publishers_column_names_work_and_the_date_falls_back() -> None:
    text = "Ticker,Name,Weight\nAAPL,Apple,5%\nmsft ,Microsoft,4%\n"
    snapshot = lists.parse("HLAL", text, TODAY)
    assert snapshot.symbols == ["AAPL", "MSFT"]
    assert snapshot.as_of == TODAY


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<html><head><title>Internet Security by Zscaler</title></head></html>",
        "<html>\n<body>\n<p>blocked, by policy</p>\n<script>var a = {b: 1, c: 2};</script>\n",
        "Name,Weight\nApple,5%\n",
        "Ticker,Name\nCash&Other,Cash\n003654100CVR,Abiomed\n",
    ],
)
def test_a_file_that_is_not_a_holdings_list_is_refused(text: str) -> None:
    with pytest.raises(ListFormatError):
        lists.parse("SPUS", text, TODAY)


def test_a_download_is_kept_as_a_dated_copy(folder: Path) -> None:
    client = website({URL: spus_csv(["AAPL", "MSFT"])})
    snapshot = lists.download(SOURCE, folder, client, TODAY)
    assert snapshot.symbols == ["AAPL", "MSFT"]
    assert (folder / "SPUS-2026-11-02.csv").read_text() == spus_csv(["AAPL", "MSFT"])
    assert lists.newest(SOURCE, folder) == (snapshot, [])


def test_a_download_that_lost_most_of_the_names_is_refused(folder: Path) -> None:
    names = [f"S{letter}{other}" for letter in "ABCDEFGHIJ" for other in "ABCDEFGHIJ"]
    lists.download(SOURCE, folder, website({URL: spus_csv(names)}), TODAY)
    truncated = spus_csv(names[:10], as_of="11/03/2026")
    with pytest.raises(ListFormatError, match="10 tickers against 100"):
        lists.download(SOURCE, folder, website({URL: truncated}), TODAY)
    assert not (folder / "SPUS-2026-11-03.csv").exists()


def test_the_last_good_copy_is_used_when_the_download_fails(folder: Path) -> None:
    lists.download(SOURCE, folder, website({URL: spus_csv(["AAPL"])}), TODAY)
    snapshot, problems = lists.collect(SOURCE, folder, website({}), TODAY)
    assert snapshot is not None and snapshot.symbols == ["AAPL"]
    assert len(problems) == 1 and problems[0].startswith("download:")


def test_a_list_with_no_download_address_reads_the_file_that_was_put_there(folder: Path) -> None:
    folder.mkdir()
    (folder / "HLAL.csv").write_text("Symbol,Weight\nAAPL,1%\nNVDA,1%\n")
    snapshot, problems = lists.collect(Source(name="HLAL"), folder, website({}), TODAY)
    assert snapshot is not None and snapshot.symbols == ["AAPL", "NVDA"]
    assert problems == []


def test_a_broken_hand_made_file_is_reported_and_the_dated_copy_still_serves(folder: Path) -> None:
    lists.download(SOURCE, folder, website({URL: spus_csv(["AAPL"])}), TODAY)
    (folder / "SPUS.csv").write_text("nothing useful here\n")
    snapshot, problems = lists.newest(SOURCE, folder)
    assert snapshot is not None and snapshot.symbols == ["AAPL"]
    assert problems == ["SPUS.csv: SPUS: no ticker column"]


def test_a_missing_list_is_reported_not_raised(folder: Path) -> None:
    snapshot, problems = lists.collect(Source(name="HLAL"), folder, None, TODAY)
    assert snapshot is None
    assert lists.health("HLAL", snapshot, problems, set(), TODAY).error == "no file"


def test_a_list_is_stale_after_a_week() -> None:
    snapshot = Snapshot(name="SPUS", as_of=TODAY - timedelta(days=7), symbols=["AAPL"])
    older = Snapshot(name="SPUS", as_of=TODAY - timedelta(days=8), symbols=["AAPL", "ZZZ"])
    assert lists.health("SPUS", snapshot, [], {"AAPL"}, TODAY).stale is False
    health = lists.health("SPUS", older, [], {"AAPL"}, TODAY)
    assert (health.stale, health.age_days, health.unmatched) == (True, 8, ["ZZZ"])


def test_only_fresh_lists_and_names_alpaca_trades_count_as_listed() -> None:
    fresh = Snapshot(name="SPUS", as_of=TODAY, symbols=["AAPL", "GONE"])
    stale = Snapshot(name="HLAL", as_of=TODAY - timedelta(days=30), symbols=["MSFT"])
    assert lists.listed([fresh, stale], {"AAPL", "MSFT"}, TODAY) == {"AAPL"}
