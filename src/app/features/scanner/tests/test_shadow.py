from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.core import audit
from app.features.scanner import bars, shadow
from app.features.scanner.schemas import Candidate, Scan, ScanFunnel
from app.features.scanner.tests.support import bar, last_business_day, sessions, yesterday

END = last_business_day(yesterday())
DAYS = sessions(END, 30)
PRICE: dict[str, Callable[[int], float]] = {
    "A": lambda i: 100.0 + i,
    "B": lambda i: 50.0 - 0.5 * i,
    "C": lambda i: 10.0 + 0.1 * i,
    "SPUS": lambda i: 30.0 + 0.2 * i,
}
WEIGHTS = {"A": 0.6, "B": 0.4}
BENCHMARK = ["A", "B", "C"]
LABEL = "mom_12_1 top 10"


def panel() -> pd.DataFrame:
    return bars.to_frame(
        {
            symbol: [bar(day, price(index), 1000.0) for index, day in enumerate(DAYS)]
            for symbol, price in PRICE.items()
        }
    )


def made(session: date, weights: dict[str, float] | None = None) -> Scan:
    return Scan(
        session=session,
        variant=LABEL,
        funnel=ScanFunnel(liquid=3, listed=3, trending=3, eligible=3, listed_eligible=3, picked=2),
        candidates=[
            Candidate(
                rank=1,
                symbol="A",
                score=1.0,
                mom_12_1=0.1,
                mom_6_1=0.1,
                clenow_90=0.1,
                price=1.0,
                dollar_volume=1.0,
                lists=[],
                picked=True,
            )
        ],
        picks={LABEL: weights or WEIGHTS, "mom_6_1 top 10": {"C": 0.1}},
        benchmark=BENCHMARK,
        stale_lists=[],
    )


def moved(symbol: str, start: int, horizon: int) -> float:
    return float(PRICE[symbol](start + horizon) / PRICE[symbol](start) - 1)


@pytest.fixture
def log(tmp_path: Path) -> audit.AuditLog:
    found = audit.AuditLog(tmp_path / "audit.jsonl")
    for start in (2, 10, 20):
        shadow.record(found, made(DAYS[start]), {"git": None})
    return found


def test_the_day_is_logged_once_without_the_candidate_table(tmp_path: Path) -> None:
    found = audit.AuditLog(tmp_path / "audit.jsonl")
    assert shadow.record(found, made(DAYS[2]), {"git": None}) is not None
    assert shadow.record(found, made(DAYS[2]), {"git": None}) is None
    [entry] = shadow.logged(found)
    assert "candidates" not in entry.payload["scan"]
    assert entry.payload["scan"]["picks"][LABEL] == WEIGHTS
    assert found.verify().valid


def test_forward_returns_match_a_hand_calculation_on_three_dates(log: audit.AuditLog) -> None:
    report = shadow.shadow(log, panel())
    rows = {(row.session, row.variant): row for row in report.rows}
    for start in (2, 10, 20):
        row = rows[(DAYS[start], LABEL)]
        assert row.picks == 2
        for horizon in shadow.HORIZONS:
            if start + horizon >= len(DAYS):
                assert row.returns[horizon] is None
                assert row.excess_equal_weight[horizon] is None
                assert row.excess_spus[horizon] is None
                continue
            mine = 0.6 * moved("A", start, horizon) + 0.4 * moved("B", start, horizon)
            basket = sum(moved(s, start, horizon) for s in BENCHMARK) / 3
            assert row.returns[horizon] == pytest.approx(mine, abs=1e-12)
            assert row.excess_equal_weight[horizon] == pytest.approx(mine - basket, abs=1e-12)
            assert row.excess_spus[horizon] == pytest.approx(
                mine - moved("SPUS", start, horizon), abs=1e-12
            )


def test_a_horizon_that_has_not_happened_yet_stays_empty(log: audit.AuditLog) -> None:
    report = shadow.shadow(log, panel())
    summary = {item.variant: item for item in report.summary}[LABEL]
    assert summary.scans == 3
    assert summary.completed == {1: 3, 5: 3, 21: 1}
    expected = sum(0.6 * moved("A", s, 5) + 0.4 * moved("B", s, 5) for s in (2, 10, 20)) / 3
    assert summary.mean_return[5] == pytest.approx(expected, abs=1e-12)
    assert report.horizons == [1, 5, 21]


def test_every_variant_logged_is_followed(log: audit.AuditLog) -> None:
    report = shadow.shadow(log, panel())
    assert {item.variant for item in report.summary} == {LABEL, "mom_6_1 top 10"}
    other = [row for row in report.rows if row.variant == "mom_6_1 top 10"]
    assert all(row.picks == 1 for row in other)


def test_without_scans_there_is_nothing_to_report(tmp_path: Path) -> None:
    report = shadow.shadow(audit.AuditLog(tmp_path / "audit.jsonl"), panel())
    assert report.rows == [] and report.summary == []


def test_a_scan_on_a_day_missing_from_the_prices_is_skipped_quietly(tmp_path: Path) -> None:
    found = audit.AuditLog(tmp_path / "audit.jsonl")
    shadow.record(found, made(date(2001, 1, 2)), {"git": None})
    report = shadow.shadow(found, panel())
    assert all(value is None for row in report.rows for value in row.returns.values())
