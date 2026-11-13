import numpy as np
import pandas as pd
import pytest

from trading_node.actors.arbiter import managed_part
from trading_node.arbiter import (
    Limits,
    StrategyTarget,
    arbitrate,
    combine,
    constrain,
    net_orders,
)

LIMITS = Limits(sector_share=0.4, correlation=0.8)
SECTORS = {"AAPL": "Information Technology", "MSFT": "Information Technology", "XOM": "Energy"}


def sector_of(symbol: str) -> str:
    return SECTORS.get(symbol, "Fund")


def target(name: str, allocation: float, stage: str = "live", **weights: float) -> StrategyTarget:
    return StrategyTarget(
        name, stage, allocation, {k.replace("_", "/"): v for k, v in weights.items()}
    )


def test_targets_combine_weighted_by_each_strategys_allocation() -> None:
    found = combine([target("a", 0.3, AAPL=0.1), target("b", 0.7, AAPL=0.0)])
    assert found == pytest.approx({"AAPL": 0.03})


def test_opposing_wishes_net_out_into_one_order() -> None:
    wishes = [target("a", 0.5, AAPL=0.4), target("b", 0.5, AAPL=0.0, XOM=0.2)]
    result = arbitrate(wishes, LIMITS, sector_of, 10_000, {"AAPL": 30}, {"AAPL": 100, "XOM": 50})

    assert result.weights == pytest.approx({"AAPL": 0.2, "XOM": 0.1})
    assert result.orders == [
        {"symbol": "AAPL", "side": "sell", "qty": 10.0},
        {"symbol": "XOM", "side": "buy", "qty": 20.0},
    ]


def test_shadow_strategies_never_reach_the_portfolio() -> None:
    wishes = [target("a", 1.0, "shadow", AAPL=1.0), target("b", 0.5, "canary", XOM=1.0)]
    result = arbitrate(wishes, LIMITS, sector_of, 10_000, {}, {"AAPL": 100, "XOM": 50})
    assert result.weights == {"XOM": 0.4}


def test_a_sector_is_cut_to_its_cap_but_funds_and_coins_are_not() -> None:
    weights, notes = constrain(
        {"AAPL": 0.3, "MSFT": 0.3, "SPUS": 0.3, "BTC/USD": 0.1}, LIMITS, sector_of
    )
    assert weights["AAPL"] + weights["MSFT"] == pytest.approx(0.4)
    assert (weights["SPUS"], weights["BTC/USD"]) == (0.3, 0.1)
    assert notes == ["Information Technology cut to 40% of the portfolio"]


def test_stocks_that_move_together_are_capped_together() -> None:
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 100)
    returns = pd.DataFrame(
        {
            "AAPL": common,
            "XOM": common + rng.normal(0, 0.001, 100),
            "SPUS": rng.normal(0, 0.01, 100),
        }
    )
    weights, notes = constrain({"AAPL": 0.3, "XOM": 0.3, "SPUS": 0.4}, LIMITS, sector_of, returns)
    assert weights["AAPL"] + weights["XOM"] == pytest.approx(0.4)
    assert weights["SPUS"] == 0.4
    assert "AAPL, XOM move together" in notes[0]


def test_gross_exposure_never_exceeds_the_account() -> None:
    weights, notes = constrain({"SPUS": 0.8, "HLAL": 0.8}, LIMITS, sector_of)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert notes[0] == "Gross exposure cut from 160% to 100%"


def test_net_orders_round_to_whole_shares_and_coin_steps_and_skip_dust() -> None:
    orders = net_orders(
        {"SPUS": 0.5, "BTC/USD": 0.25, "HLAL": 0.0},
        10_000,
        {"HLAL": 3, "SPUS": 99},
        {"SPUS": 50.3, "BTC/USD": 80_000, "HLAL": 40},
    )
    assert orders == [
        {"symbol": "HLAL", "side": "sell", "qty": 3},
        {"symbol": "BTC/USD", "side": "buy", "qty": 0.03125},
    ]


def test_the_portfolio_breaker_halves_every_position() -> None:
    wishes = [target("a", 1.0, SPUS=0.6)]
    result = arbitrate(wishes, LIMITS, sector_of, 10_000, {}, {"SPUS": 50}, cut=0.5)
    assert result.weights == pytest.approx({"SPUS": 0.3})
    assert result.notes[-1] == "Portfolio breaker open: every position at 50%"


def test_a_position_placed_by_hand_is_left_alone() -> None:
    wanted = target("trend", 1.0, SPUS=1.0)
    positions = {"AAPL": 10.0, "SPUS": 825.0, "HLAL": 4.0}

    held = managed_part(positions, {"HLAL"}, [wanted])
    result = arbitrate([wanted], LIMITS, sector_of, 50_000.0, held, {"SPUS": 50.0, "HLAL": 50.0})

    assert held == {"SPUS": 825.0, "HLAL": 4.0}
    assert {order["symbol"] for order in result.orders} == {"SPUS", "HLAL"}
