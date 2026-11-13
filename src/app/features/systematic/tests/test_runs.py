from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from pydantic import ValidationError

from app.core import audit
from app.features.systematic import lab, runs
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.tests.support import fake_history

STOCKS = create_autospec(StockHistoricalDataClient, instance=True)
CRYPTO = create_autospec(CryptoHistoricalDataClient, instance=True)


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    return fake_history(monkeypatch)


def request(**fields: object) -> runs.BacktestRequest:
    return runs.BacktestRequest.model_validate(fields)


def test_a_sweep_makes_one_variant_per_point_of_the_grid() -> None:
    found = runs.variants(
        request(
            family="halal-etf-rotation",
            window={"start": 63, "stop": 126, "step": 63},
            top_n={"start": 1, "stop": 2},
        )
    )
    assert [runs.label_of(spec) for spec in found] == [
        "63-day, top 1",
        "63-day, top 2",
        "126-day, top 1",
        "126-day, top 2",
    ]


@pytest.mark.parametrize(
    "fields",
    [
        {"family": "crypto-donchian"},
        {"family": "halal-etf-trend", "top_n": {"start": 1, "stop": 2}},
        {"family": "halal-etf-rotation", "top_n": {"start": 1, "stop": 4}},
        {"family": "halal-etf-trend", "window": {"start": 10, "stop": 500}},
        {"family": "halal-etf-rotation", "event": True},
    ],
)
def test_requests_the_rules_cannot_run_are_refused(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        request(**fields)


def test_a_run_judges_every_variant_and_counts_them_as_trials(
    tmp_path: Path, market: dict[str, str]
) -> None:
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    before = lab.search(log).trials
    run = runs.BacktestRun(
        id="x",
        created=datetime.now(UTC),
        request=request(
            family="halal-etf-trend", years=3, window={"start": 150, "stop": 300, "step": 50}
        ),
    )
    done = runs.compute(run, STOCKS, CRYPTO, log)
    assert done.status == "complete"
    assert [variant.label for variant in done.variants] == [
        "150-day average",
        "200-day average",
        "250-day average",
        "300-day average",
    ]
    assert done.trials == before + 1
    assert done.best in {variant.label for variant in done.variants}
    assert len(done.folds) == runs.FOLDS
    assert done.trades and all(trade.days >= 1 for trade in done.trades)
    assert done.heatmap is not None and done.heatmap.windows == [150, 200, 250, 300]
    assert {item.name for item in done.bias} >= {"Look-ahead", "Multiple testing", "Survivorship"}
    assert [entry.kind for entry in log.entries()] == ["sweep"]


def test_the_desk_runs_in_the_background_and_keeps_the_result(
    tmp_path: Path, market: dict[str, str]
) -> None:
    desk = runs.BacktestDesk(
        STOCKS, CRYPTO, audit.AuditLog(tmp_path / "audit.jsonl"), tmp_path / "runs.json"
    )
    queued = desk.submit(
        request(family="halal-etf-trend", years=3, window={"start": 200, "stop": 200})
    )
    assert queued.status == "queued"
    desk.pool.shutdown(wait=True)
    found = desk.get(queued.id)
    assert found is not None and found.status == "complete"
    again = runs.BacktestDesk(
        STOCKS, CRYPTO, audit.AuditLog(tmp_path / "audit.jsonl"), tmp_path / "runs.json"
    )
    assert [item.id for item in again.runs()] == [queued.id]


def test_a_run_that_breaks_is_kept_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object) -> runs.BacktestRun:
        raise RuntimeError("no data")

    monkeypatch.setattr(runs, "compute", broken)
    desk = runs.BacktestDesk(
        STOCKS, CRYPTO, audit.AuditLog(tmp_path / "audit.jsonl"), tmp_path / "runs.json"
    )
    queued = desk.submit(request(family="halal-etf-trend"))
    desk.pool.shutdown(wait=True)
    found = desk.get(queued.id)
    assert found is not None and (found.status, found.detail) == ("failed", "RuntimeError: no data")
    assert STRATEGIES["halal-etf-trend"].window == 200
