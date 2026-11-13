from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.features.systematic import rebalance
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import Plan, PlannedOrder, StrategyName
from app.features.systematic.tests.support import downtrend, position, trading_with, uptrend


def frames(*names: StrategyName, down: bool = False) -> dict[StrategyName, pd.DataFrame]:
    build = downtrend if down else uptrend
    return {name: build(STRATEGIES[name].universe) for name in names}


def plan_for(
    trading: MagicMock,
    closes: dict[StrategyName, pd.DataFrame] | None = None,
    allocation: float = 0.1,
    **kwargs: Any,
) -> Plan:
    closes = frames("btc-trend-200") if closes is None else closes
    return rebalance.build_plan(trading, closes, allocation, True, **kwargs)


def test_a_flat_account_buys_the_strategys_capital_when_the_signal_is_long() -> None:
    closes = frames("btc-trend-200")
    plan = plan_for(trading_with(), closes)
    [order] = plan.orders
    assert (order.symbol, order.side) == ("BTC/USD", "buy")
    assert order.notional == pytest.approx(10_000, rel=1e-3)
    assert order.qty == pytest.approx(
        10_000 / closes["btc-trend-200"]["BTC/USD"].iloc[-1], rel=1e-3
    )
    assert plan.targets[0].target_value == pytest.approx(10_000)
    assert plan.targets[0].weight == pytest.approx(1.0)
    assert plan.signals == {"btc-trend-200": {"BTC/USD": 1.0}}
    assert plan.strategies == ["btc-trend-200"]


def test_the_allocation_is_split_by_risk_between_strategies_and_equally_within_one() -> None:
    plan = plan_for(trading_with(), frames("btc-trend-200", "halal-etf-trend"), allocation=0.4)
    values = {t.symbol: t.target_value for t in plan.targets}
    assert sum(values.values()) == pytest.approx(40_000)
    assert values["SPUS"] == pytest.approx(values["HLAL"])
    assert len(plan.orders) == 3


def test_a_calmer_strategy_gets_a_larger_share_of_the_capital() -> None:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=200)
    calm = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 200)))}, index=index)
    wild = pd.DataFrame({"B": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))}, index=index)

    shares = rebalance.risk_shares({"halal-etf-trend": calm, "btc-trend-200": wild})

    assert sum(shares.values()) == pytest.approx(1)
    assert shares["halal-etf-trend"] > 3 * shares["btc-trend-200"]


def test_a_single_strategy_gets_all_of_the_capital() -> None:
    closes = {"halal-etf-trend": uptrend(["SPUS", "HLAL"])}
    assert rebalance.risk_shares(closes) == {"halal-etf-trend": 1.0}


def test_flat_prices_fall_back_to_an_equal_split() -> None:
    flat = pd.DataFrame({"X": [10.0] * 100}, index=pd.date_range("2024-01-01", periods=100))
    shares = rebalance.risk_shares({"halal-etf-trend": flat, "btc-trend-200": flat})
    assert shares == {"halal-etf-trend": 0.5, "btc-trend-200": 0.5}


def test_strategies_that_share_a_symbol_have_their_targets_added() -> None:
    rotation = STRATEGIES["halal-etf-rotation"].universe
    closes = {
        "halal-etf-trend": uptrend(["SPUS", "HLAL"]),
        "halal-etf-rotation": pd.concat(
            [uptrend(["SPUS", "HLAL"]), downtrend(["UMMA", "SPSK"])], axis=1
        )[rotation],
    }
    plan = plan_for(trading_with(), closes, allocation=0.2)
    values = {t.symbol: t.target_value for t in plan.targets}
    assert values["SPUS"] == pytest.approx(5_000 + 5_000)
    assert values["UMMA"] == 0.0
    assert {o.symbol for o in plan.orders} == {"SPUS", "HLAL"}


def test_a_flat_signal_sells_the_whole_position_exactly() -> None:
    held = [position("BTCUSD", 0.123456789, 80_000)]
    plan = plan_for(trading_with(held), frames("btc-trend-200", down=True))
    [order] = plan.orders
    assert (order.symbol, order.side, order.qty) == ("BTC/USD", "sell", 0.123456789)


def test_a_leftover_worth_under_a_dollar_is_not_sold_because_alpaca_would_refuse_it() -> None:
    dust = [position("BTCUSD", 9e-9, 78_000)]
    assert plan_for(trading_with(dust), frames("btc-trend-200", down=True)).orders == []


def test_buys_are_scaled_to_the_cash_so_the_plan_never_borrows() -> None:
    closes = frames("halal-etf-trend")
    plan = plan_for(trading_with(cash="30000"), closes, allocation=0.99)
    buys = [order for order in plan.orders if order.side == "buy"]
    assert len(buys) == 2
    assert sum(order.notional for order in buys) <= 30_000
    assert sum(order.notional for order in buys) > 29_000
    assert buys[0].notional == pytest.approx(buys[1].notional, rel=0.01)
    assert all(order.qty > 0 for order in buys)


def test_what_a_sell_brings_in_can_be_spent_on_the_buys() -> None:
    closes = frames("btc-trend-200", "halal-etf-trend")
    held = [position("BTCUSD", 1.0, 20_000)]
    plan = plan_for(trading_with(held, cash="1000"), closes, allocation=0.5, managed=["BTC/USD"])
    sold = sum(order.notional for order in plan.orders if order.side == "sell")
    bought = sum(order.notional for order in plan.orders if order.side == "buy")
    assert bought <= (1_000 + sold) and bought > 0


def test_plenty_of_cash_leaves_the_orders_as_the_targets_ask() -> None:
    plan = plan_for(trading_with(cash="100000"), frames("btc-trend-200"))
    [order] = plan.orders
    assert order.notional == pytest.approx(10_000, rel=1e-3)


def test_a_position_on_target_places_no_order() -> None:
    closes = frames("btc-trend-200")
    price = float(closes["btc-trend-200"]["BTC/USD"].iloc[-1])
    plan = plan_for(trading_with([position("BTCUSD", 10_000 / price, price)]), closes)
    assert plan.orders == []
    assert plan.targets[0].current_value == pytest.approx(10_000)


def test_drift_inside_the_band_is_left_alone_and_outside_it_is_trimmed() -> None:
    closes = frames("btc-trend-200")
    price = float(closes["btc-trend-200"]["BTC/USD"].iloc[-1])
    near = plan_for(trading_with([position("BTCUSD", 10_300 / price, price)]), closes)
    far = plan_for(trading_with([position("BTCUSD", 11_000 / price, price)]), closes)
    assert near.orders == []
    [trim] = far.orders
    assert trim.side == "sell"
    assert trim.notional == pytest.approx(1_000, rel=1e-3)


def test_dust_below_the_minimum_trade_is_ignored() -> None:
    assert plan_for(trading_with(), allocation=0.00005).orders == []


def test_quantities_are_cut_to_six_decimals_without_shaving_the_ones_already_there() -> None:
    assert rebalance.floor_qty(8.2) == 8.2
    assert rebalance.floor_qty(16.634759) == 16.634759
    assert rebalance.floor_qty(12.3456789) == 12.345678
    assert rebalance.floor_qty(0.0000009) == 0.0


def test_other_symbols_in_the_account_are_never_touched() -> None:
    plan = plan_for(trading_with([position("AAPL", 10, 200)]))
    assert {order.symbol for order in plan.orders} == {"BTC/USD"}


def test_a_symbol_the_automation_bought_is_sold_when_no_strategy_wants_it_any_more() -> None:
    held = [position("HLAL", 20, 50), position("AAPL", 10, 200)]
    plan = plan_for(trading_with(held), frames("btc-trend-200"), managed=["HLAL", "BTC/USD"])
    sells = [o for o in plan.orders if o.side == "sell"]
    assert [(o.symbol, o.qty) for o in sells] == [("HLAL", 20.0)]
    assert any(t.symbol == "HLAL" and t.target_value == 0 for t in plan.targets)


def test_a_managed_symbol_that_is_not_held_is_ignored() -> None:
    plan = plan_for(trading_with(), frames("btc-trend-200"), managed=["HLAL"])
    assert [t.symbol for t in plan.targets] == ["BTC/USD"]


def test_with_no_strategies_only_the_managed_positions_are_sold() -> None:
    plan = plan_for(trading_with([position("HLAL", 20, 50)]), {}, managed=["HLAL"])
    assert [(o.symbol, o.side) for o in plan.orders] == [("HLAL", "sell")]
    assert plan.strategies == []


def test_flat_targets_everything_at_zero_even_in_an_uptrend() -> None:
    held = [position("BTCUSD", 0.5, 80_000)]
    plan = plan_for(trading_with(held), flat=True)
    assert plan.flat is True
    assert [(o.symbol, o.side, o.qty) for o in plan.orders] == [("BTC/USD", "sell", 0.5)]
    assert plan.signals == {"btc-trend-200": {"BTC/USD": 0.0}}


@pytest.mark.parametrize("account", [{"trading_blocked": True}, {"status": "ACCOUNT_UPDATED"}])
def test_a_blocked_account_cannot_be_planned_against(account: dict[str, object]) -> None:
    with pytest.raises(rebalance.PlanError, match="blocked"):
        plan_for(trading_with(**account))


def test_a_scale_cuts_a_strategys_targets_and_the_rest_stays_in_cash() -> None:
    plan = plan_for(trading_with(), frames("btc-trend-200"), scales={"btc-trend-200": 0.5})
    [order] = plan.orders
    assert plan.targets[0].target_value == pytest.approx(5_000)
    assert order.notional == pytest.approx(5_000, rel=1e-3)
    assert plan.scales == {"btc-trend-200": 0.5}
    assert plan.signals == {"btc-trend-200": {"BTC/USD": 1.0}}


def test_a_scale_of_one_changes_nothing_and_is_not_reported() -> None:
    plan = plan_for(trading_with(), frames("btc-trend-200"), scales={"btc-trend-200": 1.0})
    assert plan.targets[0].target_value == pytest.approx(10_000) and plan.scales == {}


def test_a_scale_applies_to_the_strategy_it_names_only() -> None:
    two = frames("btc-trend-200", "halal-etf-trend")
    plain = {t.symbol: t.target_value for t in plan_for(trading_with(), two, 0.4).targets}
    cut = plan_for(trading_with(), two, 0.4, scales={"halal-etf-trend": 0.5})
    values = {t.symbol: t.target_value for t in cut.targets}
    assert values["BTC/USD"] == pytest.approx(plain["BTC/USD"])
    assert values["SPUS"] == pytest.approx(plain["SPUS"] / 2)
    assert values["HLAL"] == pytest.approx(plain["HLAL"] / 2)
    assert cut.scales == {"halal-etf-trend": 0.5}


def test_a_cut_position_is_sold_down_to_the_smaller_target() -> None:
    closes = frames("btc-trend-200")
    price = closes["btc-trend-200"]["BTC/USD"].iloc[-1]
    held = [position("BTCUSD", 10_000 / price, price)]
    plan = plan_for(trading_with(held), closes, scales={"btc-trend-200": 0.5})
    [order] = plan.orders
    assert order.side == "sell" and order.notional == pytest.approx(5_000, rel=1e-3)


def test_a_stock_buy_is_capped_at_the_gap_limit_above_its_decision_price() -> None:
    closes = frames("halal-etf-trend")
    plan = plan_for(trading_with(cash="100000"), closes)
    for order in plan.orders:
        price = closes["halal-etf-trend"][order.symbol].iloc[-1]
        assert order.price == pytest.approx(price)
        assert order.limit_price == pytest.approx(price * 1.02, abs=0.01)
        assert order.limit_price <= price * 1.02


def test_a_coin_buy_is_a_market_order_with_no_cap() -> None:
    [order] = plan_for(trading_with(cash="100000"), frames("btc-trend-200")).orders
    assert order.limit_price is None
    assert order.price is not None


def test_buys_fit_the_cash_even_if_every_stock_fills_at_its_cap() -> None:
    plan = plan_for(trading_with(cash="30000"), frames("halal-etf-trend"), allocation=0.99)
    worst = sum(o.qty * (o.limit_price or 0) for o in plan.orders if o.side == "buy")
    assert worst <= 30_000


def test_a_stock_sell_counts_at_the_gap_limit_below_its_price_when_fitting_buys() -> None:
    orders = [
        PlannedOrder(symbol="SPUS", side="sell", qty=100, notional=10_000, price=100),
        PlannedOrder(symbol="HLAL", side="buy", qty=200, notional=10_000, price=50, limit_price=51),
    ]
    fitted = rebalance.fit_to_cash(orders, cash=0)
    [buy] = [o for o in fitted if o.side == "buy"]
    assert buy.qty * 51 <= 10_000 * 0.98


def test_a_sub_dollar_stock_is_capped_on_a_hundredth_of_a_cent() -> None:
    assert rebalance.buy_limit("ABC", 0.5) == 0.51
    assert rebalance.buy_limit("ABC", 0.12345) == 0.1259
    assert rebalance.buy_limit("BTC/USD", 80_000) is None


def test_a_strategy_in_cash_all_window_is_sized_by_its_universe() -> None:
    held = rebalance.strategy_returns("halal-etf-trend", downtrend(["SPUS", "HLAL"], 600))
    assert float(held.std()) > 0


def test_strategies_that_move_together_share_their_risk_budget() -> None:
    rng = np.random.default_rng(5)
    index = pd.date_range("2024-01-01", periods=300, freq="B")
    common = rng.normal(0, 0.01, 300)
    twin = pd.DataFrame({"A": 100 * np.exp(np.cumsum(common))}, index=index)
    other = pd.DataFrame({"B": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))}, index=index)
    shares = rebalance.risk_shares(
        {"halal-etf-trend": twin, "halal-etf-rotation": twin, "btc-trend-200": other}
    )
    assert shares["btc-trend-200"] > shares["halal-etf-trend"] + 0.15
    assert sum(shares.values()) == pytest.approx(1)
