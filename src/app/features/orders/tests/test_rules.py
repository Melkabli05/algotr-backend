from typing import Any

import pytest
from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)
from pydantic import ValidationError

from app.features.orders import rules, service
from app.features.orders.schemas import AlgoRequest, OrderRequest

BRACKET = {"order_class": "bracket", "take_profit": 12.0, "stop_loss": 8.0}


def order(**fields: Any) -> OrderRequest:
    body = {"symbol": "F", "side": "buy", "type": "market", "qty": 1, **fields}
    return OrderRequest.model_validate(body)


def refusal(**fields: Any) -> str:
    with pytest.raises(ValidationError) as caught:
        order(**fields)
    return str(caught.value)


@pytest.mark.parametrize("tif", ["day", "gtc", "ioc", "fok"])
def test_whole_share_market_and_limit_orders_take_every_time_in_force(tif: str) -> None:
    order(type="market", time_in_force=tif)
    order(type="limit", limit_price=10, time_in_force=tif)


@pytest.mark.parametrize("tif", ["day", "gtc"])
def test_stop_stop_limit_and_trailing_orders_take_day_and_gtc(tif: str) -> None:
    order(type="stop", stop_price=12, time_in_force=tif)
    order(type="stop_limit", stop_price=12, limit_price=12.5, time_in_force=tif)
    order(type="trailing_stop", trail_percent=5, time_in_force=tif)
    order(type="trailing_stop", trail_price=0.5, time_in_force=tif)


@pytest.mark.parametrize("tif", ["ioc", "fok"])
def test_immediate_time_in_force_is_refused_for_orders_that_wait(tif: str) -> None:
    assert "accept day, gtc" in refusal(type="stop", stop_price=12, time_in_force=tif)
    assert "accept day, gtc" in refusal(
        type="stop_limit", stop_price=12, limit_price=13, time_in_force=tif
    )
    assert "accept day, gtc" in refusal(type="trailing_stop", trail_percent=5, time_in_force=tif)


@pytest.mark.parametrize("tif", ["ioc", "fok"])
def test_a_bracket_takes_day_and_gtc_only(tif: str) -> None:
    assert "Bracket orders for whole shares accept day, gtc" in refusal(
        type="limit", limit_price=10, time_in_force=tif, **BRACKET
    )


def test_fractional_shares_take_day_orders_of_the_simple_kinds_only() -> None:
    order(qty=1.5, type="limit", limit_price=10)
    order(qty=1.5, type="stop", stop_price=12)
    assert "accept day" in refusal(qty=1.5, type="limit", limit_price=10, time_in_force="gtc")
    assert "not available for fractional shares" in refusal(
        qty=1.5, type="trailing_stop", trail_percent=5
    )
    assert "Bracket orders are not available for fractional shares" in refusal(
        qty=1.5, type="limit", limit_price=10, **BRACKET
    )


def test_crypto_takes_gtc_and_ioc_on_market_limit_and_stop_limit_only() -> None:
    coin = {"symbol": "BTC/USD", "qty": 0.01}
    order(**coin, type="market", time_in_force="ioc")
    order(**coin, type="limit", limit_price=50000, time_in_force="gtc")
    order(**coin, type="stop_limit", stop_price=90000, limit_price=91000, time_in_force="gtc")
    assert "accept gtc, ioc" in refusal(**coin, type="limit", limit_price=50000)
    assert "Stop orders are not available for crypto" in refusal(
        **coin, type="stop", stop_price=90000, time_in_force="gtc"
    )
    assert "Trailing stop orders are not available for crypto" in refusal(
        **coin, type="trailing_stop", trail_percent=5, time_in_force="gtc"
    )
    assert "Bracket orders are not available for crypto" in refusal(
        **coin, type="limit", limit_price=50000, time_in_force="gtc", **BRACKET
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"type": "limit"}, "limit_price is required for limit orders"),
        ({"type": "stop"}, "stop_price is required for stop orders"),
        ({"type": "stop_limit", "stop_price": 12}, "limit_price is required"),
        ({"type": "market", "limit_price": 10}, "limit_price does not apply to market orders"),
        ({"type": "limit", "limit_price": 10, "stop_price": 9}, "stop_price does not apply"),
        ({"type": "trailing_stop"}, "exactly one of trail_price and trail_percent"),
        (
            {"type": "trailing_stop", "trail_price": 1, "trail_percent": 5},
            "exactly one of trail_price and trail_percent",
        ),
        ({"type": "trailing_stop", "trail_percent": 0.05}, "between 0.1 and 25"),
        ({"type": "trailing_stop", "trail_percent": 30}, "between 0.1 and 25"),
        ({"take_profit": 12}, "need order_class bracket, oto or oco"),
        ({"stop_loss_limit": 7}, "need order_class bracket, oto or oco"),
    ],
)
def test_prices_must_fit_the_order_type(fields: dict[str, Any], message: str) -> None:
    assert message in refusal(**fields)


def test_a_bracket_is_a_buy_with_both_exits_around_the_entry() -> None:
    placed = order(type="limit", limit_price=10, **BRACKET)
    assert (placed.order_class, placed.take_profit, placed.stop_loss) == ("bracket", 12, 8)
    order(type="market", **BRACKET)
    order(type="limit", limit_price=10, stop_loss_limit=7.5, **BRACKET)
    assert "must be buys" in refusal(side="sell", type="market", **BRACKET)
    assert "needs take_profit and stop_loss" in refusal(
        type="market", order_class="bracket", take_profit=12
    )
    assert "take_profit must be above stop_loss" in refusal(
        type="market", order_class="bracket", take_profit=8, stop_loss=12
    )
    assert "take_profit must be above the entry price" in refusal(
        type="limit", limit_price=13, **BRACKET
    )
    assert "stop_loss must be below the entry price" in refusal(
        type="limit", limit_price=7, **BRACKET
    )
    assert "market or limit order" in refusal(type="stop", stop_price=11, **BRACKET)
    assert "stop_loss_limit cannot be above stop_loss" in refusal(
        type="market", stop_loss_limit=9, **BRACKET
    )
    assert "stop_loss_limit needs stop_loss" in refusal(
        type="market", order_class="oto", take_profit=12, stop_loss_limit=7
    )


def test_an_oto_order_has_exactly_one_exit() -> None:
    order(type="limit", limit_price=10, order_class="oto", stop_loss=8)
    order(type="market", order_class="oto", take_profit=12)
    assert "one of take_profit and stop_loss" in refusal(type="market", order_class="oto")
    assert "one of take_profit and stop_loss" in refusal(
        type="market", order_class="oto", take_profit=12, stop_loss=8
    )


def test_an_oco_order_is_a_sell_that_closes_a_position() -> None:
    placed = order(side="sell", type="limit", order_class="oco", take_profit=12, stop_loss=8)
    assert placed.limit_price is None
    order(side="sell", type="limit", order_class="oco", take_profit=12, stop_loss=8, limit_price=12)
    assert "must be a sell" in refusal(type="limit", order_class="oco", take_profit=12, stop_loss=8)
    assert "is type limit" in refusal(
        side="sell", type="market", order_class="oco", take_profit=12, stop_loss=8
    )
    assert "needs take_profit and stop_loss" in refusal(
        side="sell", type="limit", order_class="oco", take_profit=12
    )
    assert "limit_price is its take_profit" in refusal(
        side="sell", type="limit", order_class="oco", take_profit=12, stop_loss=8, limit_price=13
    )
    assert "take_profit must be above stop_loss" in refusal(
        side="sell", type="limit", order_class="oco", take_profit=8, stop_loss=12
    )


def test_asset_classes_follow_the_symbol_and_the_quantity() -> None:
    assert rules.asset_class("AAPL", 3) == "stock"
    assert rules.asset_class("AAPL", 3.5) == "fractional"
    assert rules.asset_class("BTC/USD", 3) == "crypto"


def test_every_kind_the_table_lists_has_a_name_for_messages() -> None:
    kinds = {kind for table in rules.TIME_IN_FORCE.values() for kind in table}
    assert kinds <= set(rules.KINDS)


def test_a_market_order_is_built_for_alpaca_with_the_position_intent() -> None:
    request = service.broker_request(order(side="buy", type="market", time_in_force="ioc"))
    assert isinstance(request, MarketOrderRequest)
    assert request.position_intent == PositionIntent.BUY_TO_OPEN
    assert (request.side, request.time_in_force) == (OrderSide.BUY, TimeInForce.IOC)
    assert request.order_class is None


def test_every_stock_sell_is_a_close_so_alpaca_refuses_to_short() -> None:
    sell = service.broker_request(order(side="sell", type="limit", limit_price=10))
    assert sell.position_intent == PositionIntent.SELL_TO_CLOSE
    coin = service.broker_request(
        order(symbol="BTC/USD", qty=0.01, side="sell", type="market", time_in_force="gtc")
    )
    assert coin.position_intent is None


def test_each_order_type_becomes_its_alpaca_request() -> None:
    stop = service.broker_request(order(type="stop", stop_price=12, time_in_force="gtc"))
    assert isinstance(stop, StopOrderRequest) and stop.stop_price == 12
    both = service.broker_request(order(type="stop_limit", stop_price=12, limit_price=12.5))
    assert isinstance(both, StopLimitOrderRequest)
    assert (both.stop_price, both.limit_price) == (12, 12.5)
    trail = service.broker_request(order(type="trailing_stop", trail_percent=5))
    assert isinstance(trail, TrailingStopOrderRequest)
    assert (trail.trail_percent, trail.trail_price) == (5, None)


def test_a_bracket_carries_both_exit_legs() -> None:
    request = service.broker_request(
        order(type="limit", limit_price=10, stop_loss_limit=7.5, **BRACKET)
    )
    assert isinstance(request, LimitOrderRequest)
    assert request.order_class == OrderClass.BRACKET
    assert request.take_profit is not None and request.take_profit.limit_price == 12
    assert request.stop_loss is not None
    assert (request.stop_loss.stop_price, request.stop_loss.limit_price) == (8, 7.5)


def test_an_oco_order_rests_at_its_take_profit() -> None:
    request = service.broker_request(
        order(side="sell", type="limit", order_class="oco", take_profit=12, stop_loss=8)
    )
    assert isinstance(request, LimitOrderRequest)
    assert request.order_class == OrderClass.OCO
    assert request.limit_price == 12
    assert request.position_intent == PositionIntent.SELL_TO_CLOSE


def test_the_rules_are_served_for_one_symbol_and_quantity() -> None:
    coin = service.get_rules("BTC/USD", 1)
    assert (coin.asset, coin.time_in_force["limit"]) == ("crypto", ["gtc", "ioc"])
    assert "stop" not in coin.time_in_force
    shares = service.get_rules("F", 10)
    assert (shares.asset, shares.time_in_force["stop"]) == ("stock", ["day", "gtc"])
    assert service.get_rules("F", 1.5).asset == "fractional"
    assert shares.trail_percent == (0.1, 25.0)


def algo(**fields: Any) -> AlgoRequest:
    body = {"algo": "twap", "symbol": "F", "side": "buy", "qty": 10, "duration": 10, "slices": 5}
    return AlgoRequest.model_validate({**body, **fields})


def algo_refusal(**fields: Any) -> str:
    with pytest.raises(ValidationError) as caught:
        algo(**fields)
    return str(caught.value)


def test_an_algorithm_order_trades_whole_shares() -> None:
    assert "A TWAP order trades whole shares" in algo_refusal(qty=1.5)


def test_a_twap_needs_slices() -> None:
    assert "A TWAP needs slices" in algo_refusal(slices=None)


def test_every_slice_of_a_stock_needs_a_share() -> None:
    assert "use at most 3 slices" in algo_refusal(qty=3, slices=5)


def test_slices_cannot_be_closer_than_ten_seconds() -> None:
    assert "use at most 6" in algo_refusal(duration=1, slices=7)
    assert algo(duration=1, slices=6).slices == 6


def test_a_coin_order_may_have_more_slices_than_whole_units() -> None:
    assert algo(symbol="BTC/USD", qty=0.2, slices=5).qty == 0.2


def test_the_ticket_is_offered_a_twap_for_whole_shares_and_coins_but_not_part_shares() -> None:
    assert service.get_rules("F", 10).time_in_force["twap"] == ["ioc"]
    assert service.get_rules("BTC/USD", 1).time_in_force["twap"] == ["ioc"]
    assert "twap" not in service.get_rules("F", 1.5).time_in_force


def test_a_vwap_takes_no_slices_and_at_least_ten_minutes() -> None:
    assert "leave slices out" in algo_refusal(algo="vwap")
    assert "needs at least 10 minutes" in algo_refusal(algo="vwap", slices=None, duration=9)
    assert algo(algo="vwap", slices=None, duration=10).duration == 10


def test_only_a_twap_follows_the_quote_imbalance() -> None:
    assert "Only a TWAP follows" in algo_refusal(algo="vwap", slices=None, tilt=0.2)


def test_the_ticket_is_offered_a_vwap_for_whole_shares_and_coins() -> None:
    assert service.get_rules("F", 10).time_in_force["vwap"] == ["ioc"]
    assert service.get_rules("BTC/USD", 1).time_in_force["vwap"] == ["ioc"]
    assert "vwap" not in service.get_rules("F", 1.5).time_in_force


def test_a_shortfall_is_for_stocks_and_needs_slices() -> None:
    assert "for stocks" in algo_refusal(algo="shortfall", symbol="BTC/USD", qty=0.2)
    assert "A Shortfall needs slices" in algo_refusal(algo="shortfall", slices=None)
    assert "use at most 3 slices" in algo_refusal(algo="shortfall", qty=3, slices=5)
    assert algo(algo="shortfall", risk_aversion=2.0).risk_aversion == 2.0


def test_risk_aversion_is_between_none_and_a_hundred() -> None:
    assert "risk_aversion" in algo_refusal(algo="shortfall", risk_aversion=-1)
    assert "risk_aversion" in algo_refusal(algo="shortfall", risk_aversion=101)
    assert algo(algo="shortfall", risk_aversion=0).risk_aversion == 0


def test_the_ticket_is_offered_a_shortfall_for_whole_share_stocks_only() -> None:
    assert service.get_rules("F", 10).time_in_force["shortfall"] == ["ioc"]
    assert "shortfall" not in service.get_rules("BTC/USD", 1).time_in_force
    assert "shortfall" not in service.get_rules("F", 1.5).time_in_force
