from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core import audit
from app.core.drift import ReturnStream
from app.features.drift import monitor
from app.features.drift.monitor import DriftMonitor, Rules
from app.features.drift.tests.support import BASE, START, Source, series, steps

RULES = Rules()


def first_live_day() -> date:
    day: date = pd.bdate_range(START, periods=BASE + 1)[BASE].date()
    return day


def test_a_stream_that_is_too_short_is_not_monitored() -> None:
    status = monitor.scan(series(base=300, days=100), RULES)
    assert not status.monitored and status.note == "Needs 560 days; has 400"
    assert status.events == [] and status.days == 400


def test_an_empty_stream_has_nothing_to_watch() -> None:
    empty = ReturnStream("none", "strategy", pd.Series(dtype=float))
    status = monitor.scan(empty, RULES)
    assert not status.monitored and status.note == "No returns yet" and status.first is None


def test_a_baseline_that_never_moves_cannot_be_measured() -> None:
    index = pd.bdate_range(START, periods=700)
    flat = ReturnStream("cash", "agent", pd.Series(0.0, index=index))
    status = monitor.scan(flat, RULES)
    assert not status.monitored and status.note == "The baseline never moves"


def test_a_stream_that_does_not_change_has_no_events() -> None:
    status = monitor.scan(series(), RULES)
    assert status.monitored and status.events == []
    assert status.distance is not None and status.distance < 0.3


def test_a_jump_in_volatility_is_an_adverse_event_found_within_a_hundred_days() -> None:
    status = monitor.scan(series((0.0005, 0.02)), RULES)
    [event, *_] = [item for item in status.events if item.kind == "volatility"]
    assert event.adverse and event.distance >= RULES.threshold and event.p_value < 0.001
    assert first_live_day() <= event.time <= first_live_day() + timedelta(days=140)
    assert status.distance is not None and status.distance > 0.5


def test_a_fall_in_volatility_is_found_but_is_not_adverse() -> None:
    status = monitor.scan(series((0.0005, 0.004)), RULES)
    found = [item for item in status.events if item.kind == "volatility"]
    assert found and not any(item.adverse for item in found)


def test_a_collapse_of_the_mean_is_an_adverse_event() -> None:
    status = monitor.scan(series((-0.01, 0.01)), RULES)
    found = [item for item in status.events if item.kind == "mean"]
    assert found and all(item.adverse for item in found)


def test_the_events_do_not_depend_on_the_scale_of_the_returns() -> None:
    plain = series((0.0005, 0.02))
    scaled = ReturnStream(plain.name, plain.kind, plain.returns * 100)
    first = [(item.time, item.kind) for item in monitor.scan(plain, RULES).events]
    second = [(item.time, item.kind) for item in monitor.scan(scaled, RULES).events]
    assert first and first == second


def test_a_change_inside_the_baseline_is_part_of_normal_and_not_an_event() -> None:
    rng = np.random.default_rng(2)
    values = np.concatenate(
        [rng.normal(0, 0.01, 250), rng.normal(0, 0.03, 250), rng.normal(0, 0.03, 150)]
    )
    index = pd.bdate_range(START, periods=len(values))
    status = monitor.scan(ReturnStream("s", "strategy", pd.Series(values, index=index)), RULES)
    assert all(event.time >= first_live_day() for event in status.events)


def test_the_hold_runs_from_the_latest_adverse_event_not_the_first(
    tmp_path: Path, today: list[date]
) -> None:
    found = DriftMonitor(RULES, None, tmp_path / "drift.json", lambda: today[0])
    found.sources = [Source(steps([(150, 2.0), (150, 1.0), (150, 3.0)]))]
    found.refresh()
    days = adverse_days(found)
    assert len(days) >= 2 and days[-1] - days[0] > timedelta(days=RULES.hold_days)
    today[0] = days[-1] + timedelta(days=5)
    found.refresh()
    assert found.scale("halal-etf-trend") == 0.5
    assert found.status().streams[0].until == days[-1] + timedelta(days=RULES.hold_days)


def test_a_change_must_be_both_large_and_unlikely_to_be_noise() -> None:
    rng = np.random.default_rng(3)
    baseline = rng.normal(0, 1, 500)
    shifted = rng.normal(0.5, 1, 60)
    same = rng.normal(0, 1, 60)
    day = date(2023, 1, 2)
    assert monitor.judged("s", "mean", day, shifted, baseline, RULES) is not None
    assert monitor.judged("s", "mean", day, same, baseline, RULES) is None
    assert monitor.judged("s", "mean", day, shifted, baseline, Rules(threshold=2.0)) is None
    assert monitor.judged("s", "mean", day, shifted, baseline, Rules(significance=1e-12)) is None


def test_a_window_of_a_few_days_is_not_judged() -> None:
    baseline = np.random.default_rng(3).normal(0, 1, 500)
    assert monitor.judged("s", "mean", date(2023, 1, 2), np.full(5, 3.0), baseline, RULES) is None


@pytest.fixture
def today() -> list[date]:
    return [first_live_day()]


@pytest.fixture
def desk(tmp_path: Path, today: list[date]) -> DriftMonitor:
    found = DriftMonitor(
        RULES, audit.AuditLog(tmp_path / "audit.jsonl"), tmp_path / "drift.json", lambda: today[0]
    )
    found.sources = [Source(series((0.0005, 0.02)))]
    return found


def adverse_days(found: DriftMonitor) -> list[date]:
    return sorted(event.time for event in found.state.recorded if event.adverse)


def test_an_adverse_event_cuts_the_size_for_the_hold_and_then_restores_it(
    desk: DriftMonitor, today: list[date]
) -> None:
    desk.refresh()
    last = adverse_days(desk)[-1]
    today[0] = last + timedelta(days=10)
    desk.refresh()
    assert desk.scale("halal-etf-trend") == 0.5
    today[0] = last + timedelta(days=RULES.hold_days + 1)
    desk.refresh()
    assert desk.scale("halal-etf-trend") == 1.0
    assert desk.status().streams[0].until is None


def test_the_cut_runs_to_the_end_of_the_hold_counted_from_the_last_adverse_event(
    desk: DriftMonitor, today: list[date]
) -> None:
    desk.refresh()
    last = adverse_days(desk)[-1]
    today[0] = last + timedelta(days=RULES.hold_days)
    desk.refresh()
    stream = desk.status().streams[0]
    assert stream.scale == 0.5 and stream.until == last + timedelta(days=RULES.hold_days)


def test_the_log_action_records_events_but_never_cuts(tmp_path: Path, today: list[date]) -> None:
    found = DriftMonitor(Rules(action="log"), None, tmp_path / "drift.json", lambda: today[0])
    found.sources = [Source(series((0.0005, 0.02)))]
    found.retrain = lambda name: pytest.fail("a logged event must not retrain")
    fresh = found.refresh()
    assert fresh and all(event.scale is None and event.retrain is None for event in fresh)
    today[0] = adverse_days(found)[-1] + timedelta(days=1)
    found.refresh()
    assert found.scale("halal-etf-trend") == 1.0


def test_the_off_action_does_not_even_scan(tmp_path: Path) -> None:
    source = Source(series((0.0005, 0.02)))
    found = DriftMonitor(Rules(action="off"), None, tmp_path / "drift.json")
    found.sources = [source]
    assert found.refresh() == [] and source.calls == 0
    assert found.status().streams == [] and found.scale("halal-etf-trend") == 1.0


def test_an_event_goes_to_the_audit_log_and_asks_for_a_retrain(
    desk: DriftMonitor, tmp_path: Path
) -> None:
    asked: list[str] = []

    def schedule(name: str) -> str:
        asked.append(name)
        return "scheduled"

    desk.retrain = schedule
    fresh = desk.refresh()
    adverse = [event for event in fresh if event.adverse]
    assert adverse and asked == ["halal-etf-trend"] * len(adverse)
    assert all(event.retrain == "scheduled" and event.scale == 0.5 for event in adverse)
    entries = audit.AuditLog(tmp_path / "audit.jsonl").entries()
    assert len(entries) == len(fresh) and {entry.kind for entry in entries} == {"drift"}
    assert entries[0].payload["event"]["stream"] == "halal-etf-trend"


def test_a_favourable_event_is_recorded_without_a_cut_or_a_retrain(
    tmp_path: Path, today: list[date]
) -> None:
    found = DriftMonitor(RULES, None, tmp_path / "drift.json", lambda: today[0])
    found.sources = [Source(series((0.0005, 0.004)))]
    found.retrain = lambda name: pytest.fail("a favourable event must not retrain")
    fresh = found.refresh()
    assert fresh and not any(event.adverse for event in fresh)
    assert all(event.scale is None and event.retrain is None for event in fresh)


def test_an_event_is_recorded_once(desk: DriftMonitor) -> None:
    first = desk.refresh()
    assert first and desk.refresh() == []
    assert len(desk.state.recorded) == len(first)


def test_events_before_the_monitor_started_are_history_only(
    desk: DriftMonitor, today: list[date]
) -> None:
    today[0] = first_live_day() + timedelta(days=150)
    assert desk.refresh() == [] or all(event.time >= today[0] for event in desk.refresh())
    stream = desk.status().streams[0]
    assert stream.events and desk.state.started == today[0]
    assert all(event.time >= today[0] for event in desk.state.recorded)


def test_a_failed_retrain_is_noted_and_the_cut_stands(
    desk: DriftMonitor, today: list[date]
) -> None:
    def broken(name: str) -> str:
        raise RuntimeError("the sidecar is not set up")

    desk.retrain = broken
    fresh = desk.refresh()
    adverse = [event for event in fresh if event.adverse]
    assert adverse and all(event.retrain == "could not be scheduled" for event in adverse)
    today[0] = adverse_days(desk)[-1] + timedelta(days=1)
    desk.refresh()
    assert desk.scale("halal-etf-trend") == 0.5


def test_the_scale_of_a_stream_nobody_watches_is_one(desk: DriftMonitor) -> None:
    assert desk.scale("btc-trend-200") == 1.0


def test_the_check_runs_once_a_day_whoever_asks(tmp_path: Path, today: list[date]) -> None:
    source = Source(series())
    found = DriftMonitor(RULES, None, tmp_path / "drift.json", lambda: today[0])
    found.sources = [source]
    found.scale("halal-etf-trend")
    found.status()
    found.scale("halal-etf-trend")
    assert source.calls == 1
    today[0] += timedelta(days=1)
    found.status()
    assert source.calls == 2


def test_a_failing_source_leaves_the_last_answer_standing(
    desk: DriftMonitor, today: list[date], caplog: pytest.LogCaptureFixture
) -> None:
    desk.refresh()
    last = adverse_days(desk)[-1]
    today[0] = last + timedelta(days=1)
    desk.refresh()
    assert desk.scale("halal-etf-trend") == 0.5

    class Broken:
        def streams(self) -> list[ReturnStream]:
            raise RuntimeError("Alpaca is down")

    desk.sources = [Broken()]
    today[0] += timedelta(days=1)
    assert desk.scale("halal-etf-trend") == 0.5
    assert "The drift check failed" in caplog.text


def test_what_was_recorded_survives_a_restart(desk: DriftMonitor, tmp_path: Path) -> None:
    fresh = desk.refresh()
    again = DriftMonitor(RULES, None, tmp_path / "drift.json")
    assert [event.time for event in again.state.recorded] == [event.time for event in fresh]
    assert again.state.started == desk.state.started


def test_a_corrupt_state_file_starts_afresh(tmp_path: Path) -> None:
    (tmp_path / "drift.json").write_text("not json")
    assert DriftMonitor(RULES, None, tmp_path / "drift.json").state.recorded == []


def test_the_status_reports_the_rules_the_streams_and_the_recorded_events(
    desk: DriftMonitor,
) -> None:
    desk.refresh()
    status = desk.status()
    assert (status.action, status.threshold, status.scale, status.hold_days) == (
        "act",
        0.1,
        0.5,
        30,
    )
    assert status.baseline_days == 500 and status.started == first_live_day()
    [stream] = status.streams
    assert stream.name == "halal-etf-trend" and stream.monitored and stream.events
    assert [event.time for event in status.recorded] == sorted(
        (event.time for event in desk.state.recorded), reverse=True
    )
