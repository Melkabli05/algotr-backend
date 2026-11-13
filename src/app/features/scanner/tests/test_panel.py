from datetime import date
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from app.features.scanner import bars
from app.features.scanner.store import BarStore, combine, diverged
from app.features.scanner.tests.support import bar, last_business_day, sessions, yesterday

END = last_business_day(yesterday())


def panel(closes: dict[str, list[float]], end: date = END) -> pd.DataFrame:
    return bars.to_frame(
        {
            symbol: [
                bar(day, close, 1000.0)
                for day, close in zip(sessions(end, len(series)), series, strict=True)
            ]
            for symbol, series in closes.items()
        }
    )


def test_the_panel_survives_a_round_trip_and_leaves_no_scratch_file(tmp_path: Path) -> None:
    store = BarStore(tmp_path / "data" / "bars.parquet")
    assert store.read().empty and list(store.read().columns) == bars.COLUMNS
    written = panel({"AAA": [10, 11, 12], "BBB": [20, 21, 22]})
    store.write(written)
    pdt.assert_frame_equal(store.read(), written)
    assert [path.name for path in store.path.parent.iterdir()] == ["bars.parquet"]


def test_a_matching_window_leaves_every_symbol_alone() -> None:
    stored = panel({"AAA": [10.0] * 30, "BBB": [20.0] * 30})
    assert diverged(stored, stored.groupby("symbol").tail(10)) == set()


def test_a_split_marks_only_the_symbol_whose_prices_moved() -> None:
    stored = panel({"AAA": [10.0] * 30, "BBB": [20.0] * 30})
    recent = panel({"AAA": [5.0] * 10, "BBB": [20.0] * 10})
    assert diverged(stored, recent) == {"AAA"}


def test_a_dividend_sized_change_counts_but_rounding_noise_does_not() -> None:
    stored = panel({"AAA": [100.0] * 30, "BBB": [100.0] * 30})
    recent = panel({"AAA": [99.5] * 10, "BBB": [100.0 + 1e-7] * 10})
    assert diverged(stored, recent) == {"AAA"}


def test_a_gap_between_the_stored_history_and_the_window_counts() -> None:
    stored = panel({"AAA": [10.0] * 5}, end=sessions(END, 40)[0])
    recent = panel({"AAA": [10.0] * 10, "NEW": [5.0] * 10})
    assert diverged(stored, recent) == {"AAA"}


def test_symbols_in_only_one_of_the_two_are_not_diverged() -> None:
    stored = panel({"GONE": [10.0] * 30, "AAA": [10.0] * 30})
    recent = panel({"AAA": [10.0] * 10, "NEW": [5.0] * 10})
    assert diverged(stored, recent) == set()


def test_nothing_to_compare_means_nothing_diverged() -> None:
    assert diverged(bars.empty(), panel({"AAA": [10.0]})) == set()
    assert diverged(panel({"AAA": [10.0]}), bars.empty()) == set()


def test_new_sessions_extend_each_symbol_without_repeating_old_ones() -> None:
    stored = panel({"AAA": [10.0] * 30}, end=sessions(END, 4)[0])
    recent = panel({"AAA": [10.0] * 10})
    merged = combine(stored, recent, bars.empty())
    assert merged["date"].is_unique
    assert merged["date"].max() == pd.Timestamp(END)
    assert len(merged) == 30 + 3


def test_running_the_same_night_twice_changes_nothing() -> None:
    stored = panel({"AAA": [10.0] * 30, "BBB": [20.0] * 30}, end=sessions(END, 4)[0])
    recent = panel({"AAA": [10.0] * 10, "BBB": [20.0] * 10})
    once = combine(stored, recent, bars.empty())
    pdt.assert_frame_equal(combine(once, recent, bars.empty()), once)


def test_a_stale_symbol_is_replaced_by_its_new_history() -> None:
    stored = panel({"AAA": [10.0] * 30, "BBB": [20.0] * 30})
    history = panel({"AAA": [5.0] * 40})
    recent = panel({"AAA": [5.0] * 10, "BBB": [20.0] * 10})
    merged = combine(stored, recent, history, stale={"AAA"})
    assert merged.loc[merged["symbol"] == "AAA", "close"].eq(5.0).all()
    assert (merged["symbol"] == "AAA").sum() == 40
    assert (merged["symbol"] == "BBB").sum() == 30


def test_a_stale_symbol_that_could_not_be_refetched_is_dropped_not_kept_wrong() -> None:
    stored = panel({"AAA": [10.0] * 30, "BBB": [20.0] * 30})
    recent = panel({"AAA": [5.0] * 10, "BBB": [20.0] * 10})
    merged = combine(stored, recent, bars.empty(), stale={"AAA"})
    assert set(merged["symbol"]) == {"BBB"}


def test_symbols_never_admitted_are_not_picked_up_from_the_window() -> None:
    stored = panel({"AAA": [10.0] * 30})
    recent = panel({"AAA": [10.0] * 10, "PENNY": [1.0] * 10})
    assert set(combine(stored, recent, bars.empty())["symbol"]) == {"AAA"}


def test_a_new_symbol_arrives_with_its_whole_history() -> None:
    history = panel({"NEW": [3.0] * 50})
    merged = combine(bars.empty(), panel({"NEW": [3.0] * 10}), history)
    assert len(merged) == 50
