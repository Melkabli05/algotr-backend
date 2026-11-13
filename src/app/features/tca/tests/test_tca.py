from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.features.tca.service import Trade, arrival_and_twap, cost, grouped, source_of, weighted

START = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)


def test_cost_is_positive_when_it_costs_money_either_way() -> None:
    assert cost("buy", 101, 100) == pytest.approx(100)
    assert cost("sell", 99, 100) == pytest.approx(100)
    assert cost("sell", 101, 100) == pytest.approx(-100)
    assert cost("buy", None, 100) is None


def test_fills_of_one_order_are_pooled() -> None:
    fills = [
        SimpleNamespace(
            order="a", id="1", symbol="X", side="buy", qty=1, price=10, time=START, fee_qty=None
        ),
        SimpleNamespace(
            order="a",
            id="2",
            symbol="X",
            side="buy",
            qty=3,
            price=12,
            time=START + timedelta(minutes=2),
            fee_qty=None,
        ),
    ]
    [found] = grouped(fills)
    assert (found.qty, found.paid / found.qty) == (4, 11.5)
    assert found.last - found.first == timedelta(minutes=2)


def test_arrival_is_the_open_of_the_minute_and_twap_the_mean_close_over_the_life() -> None:
    rows = [
        {"t": START + timedelta(minutes=step), "o": 100 + step, "c": 100.5 + step, "v": 10}
        for step in range(5)
    ]
    arrival, twap = arrival_and_twap(
        rows, START + timedelta(minutes=1, seconds=20), START + timedelta(minutes=3)
    )
    assert arrival == 101
    assert twap == pytest.approx((101.5 + 102.5 + 103.5) / 3)


def test_sources_come_from_the_client_id_and_the_autopilot_record() -> None:
    assert source_of("algo-ab12-3", "o1", {}) == ("algo", "ab12")
    assert source_of("iceberg-x-1", "o1", {}) == ("iceberg", None)
    assert source_of(None, "o1", {"o1": 50.0}) == ("autopilot", None)
    assert source_of("web", "o1", {}) == ("manual", None)


def test_totals_weight_by_notional_and_skip_unknowns() -> None:
    def trade(notional: float, impact: float | None) -> Trade:
        return Trade(
            order="o",
            time=START,
            symbol="X",
            side="buy",
            qty=1,
            notional=notional,
            source="manual",
            algo=None,
            decision=None,
            arrival=None,
            fill=1,
            vwap=None,
            twap=None,
            seconds_to_fill=None,
            timing_bps=None,
            impact_bps=impact,
            shortfall_bps=None,
            vs_vwap_bps=None,
            vs_twap_bps=None,
            fee_bps=None,
        )

    found = weighted([trade(100, 10), trade(300, 2), trade(50, None)], "all")
    assert found.impact_bps == pytest.approx((100 * 10 + 300 * 2) / 400)
    assert found.notional == 450
