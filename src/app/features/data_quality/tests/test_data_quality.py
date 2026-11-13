from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.core import data_quality
from app.core.data_quality import CROSS_SOURCE, Finding
from app.core.feature_store import ANOMALOUS, GAP, STALE
from app.core.tables import daily_bars
from app.core.tables import data_quality as findings
from app.features.data_quality.service import DataQualityJob, eodhd_reference
from app.quant import history

NOW = datetime(2026, 11, 20, 23, tzinfo=UTC)


def stock(days: pd.DatetimeIndex, start: float = 50.0) -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(start * np.cumprod(1 + rng.normal(0, 0.005, len(days))), index=days)


def sessions(start: str, end: str) -> pd.DatetimeIndex:
    return data_quality.expected_sessions(
        "SPUS", date.fromisoformat(start), date.fromisoformat(end)
    )


def test_a_holiday_is_no_gap_but_a_missing_session_is() -> None:
    days = sessions("2026-08-03", "2026-11-20")
    assert pd.Timestamp("2026-09-07") not in days
    closes = stock(days).drop(pd.Timestamp("2026-10-14"))

    flags, found = data_quality.validate(closes, "SPUS", NOW)

    assert flags[pd.Timestamp("2026-10-14")] == GAP
    assert [(f.session, f.reason_code) for f in found] == [(date(2026, 10, 14), "gap")]


def test_a_last_bar_behind_the_last_session_is_stale_and_coins_trade_every_day() -> None:
    days = sessions("2026-08-03", "2026-11-18")
    _, found = data_quality.validate(stock(days), "SPUS", NOW)
    assert [f.reason_code for f in found] == ["stale"]
    coin_days = pd.date_range("2026-08-01", "2026-11-19")
    _, coin = data_quality.validate(stock(coin_days, 80_000), "BTC/USD", NOW)
    assert coin == []


def test_impossible_and_extreme_moves_are_anomalous() -> None:
    days = sessions("2026-06-01", "2026-11-20")
    closes = stock(days)
    closes[pd.Timestamp("2026-10-01")] *= 1.6
    closes[pd.Timestamp("2026-11-02")] = -1.0
    flags, found = data_quality.validate(closes, "SPUS", NOW)
    assert flags[pd.Timestamp("2026-10-01")] & ANOMALOUS
    assert flags[pd.Timestamp("2026-11-02")] & ANOMALOUS
    assert {f.reason_code for f in found} == {"anomalous"}


def test_a_close_the_second_source_disagrees_with_is_flagged() -> None:
    days = sessions("2026-10-01", "2026-11-20")
    closes = stock(days)
    other = closes.copy()
    other[pd.Timestamp("2026-11-05")] *= 1.02
    flags, found = data_quality.validate(closes, "SPUS", NOW, other)
    assert flags[pd.Timestamp("2026-11-05")] == CROSS_SOURCE
    assert found[0].reason_code == "cross_source" and "differ 1.96%" in found[0].detail


def test_flagged_bars_become_missing() -> None:
    days = sessions("2026-11-02", "2026-11-06")
    frame = pd.DataFrame({"SPUS": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=days)
    flags = {"SPUS": pd.Series([0, ANOMALOUS, 0, 0, STALE], index=days)}
    assert data_quality.mask(frame, flags)["SPUS"].isna().tolist() == [
        False,
        True,
        False,
        False,
        True,
    ]


def test_findings_are_kept_once_and_chained_so_an_edit_is_caught(database: Engine) -> None:
    found = [
        Finding("SPUS", date(2026, 11, 5), CROSS_SOURCE, "differ 2%"),
        Finding("HLAL", date(2026, 11, 6), GAP, "No daily bar"),
    ]
    with database.begin() as connection:
        assert data_quality.record(connection, "api", found) == 2
        assert data_quality.record(connection, "api", found) == 0
        assert data_quality.verify(connection)
        connection.execute(
            sa.update(findings).where(findings.c.symbol == "SPUS").values(detail="fine")
        )
        assert not data_quality.verify(connection)


class Eodhd:
    def get_eod_historical_stock_market_data(self, **kwargs: object) -> list[dict[str, object]]:
        return [{"date": "2026-11-19", "adjusted_close": 60.0}, {"date": "2026-11-20", "close": 1}]


def test_the_job_checks_against_eodhd_and_stores_each_bar_with_its_flags(
    database: Engine,
) -> None:
    days = sessions("2026-10-01", "2026-11-20")
    closes = pd.DataFrame({"SPUS": stock(days)})
    closes.loc[pd.Timestamp("2026-11-19"), "SPUS"] = 50.0
    job = DataQualityJob(
        symbols=lambda: ["SPUS"],
        prices=lambda names: history.Prices(closes, closes),
        reference=eodhd_reference("key", Eodhd()),
    )

    assert job.run(NOW) == 1

    with database.connect() as connection:
        bars = {row.session: row.quality_flags for row in connection.execute(sa.select(daily_bars))}
    assert bars[date(2026, 11, 19)] == CROSS_SOURCE
    assert len(bars) == len(days)


def test_the_findings_are_served_with_the_chain_verified(
    client: TestClient, database: Engine
) -> None:
    with database.begin() as connection:
        data_quality.record(connection, "node", [Finding("SPUS", date(2026, 11, 5), GAP, "gap")])
    body = client.get("/data-quality").json()
    assert body["verified"] and body["findings"][0]["reason_code"] == "gap"


def test_a_coin_checked_beside_stocks_keeps_its_weekends(database: Engine) -> None:
    coin_days = pd.date_range("2026-08-01", "2026-11-19")
    stock_days = sessions("2026-08-03", "2026-11-20")
    frames = {"BTC/USD": stock(coin_days, 80_000), "SPUS": stock(stock_days)}
    asked: list[list[str]] = []

    def prices(names: list[str]) -> history.Prices:
        asked.append(names)
        frame = pd.DataFrame({name: frames[name] for name in names})
        return history.Prices(frame, frame)

    job = DataQualityJob(symbols=lambda: ["BTC/USD", "SPUS"], prices=prices)

    assert job.run(NOW) == 0
    assert asked == [["BTC/USD"], ["SPUS"]]
