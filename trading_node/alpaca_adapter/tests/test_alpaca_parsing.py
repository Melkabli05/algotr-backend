from decimal import Decimal
from typing import Any

import pytest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading import enums as alpaca_enums
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)
from nautilus_trader.common import Clock, OrderFactory
from nautilus_trader.model import (
    AccountId,
    BarType,
    ContingencyType,
    Currency,
    CurrencyPair,
    Equity,
    InstrumentId,
    LimitOrder,
    LiquiditySide,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    Price,
    Quantity,
    StrategyId,
    TimeInForce,
    TraderId,
    TrailingOffsetType,
    TriggerType,
)

from trading_node.alpaca_adapter import parsing
from trading_node.alpaca_adapter.tests import support

ACCOUNT_ID = AccountId("ALPACA-PAPER-001")
BTC = parsing.parse_instrument(support.asset("BTC/USD"), support.NOW_NS)
AAPL = parsing.parse_instrument(support.asset("AAPL"), support.NOW_NS)
BTC_ID = InstrumentId.from_str("BTC/USD.ALPACA")
FACTORY = OrderFactory(TraderId("T-001"), StrategyId("S-001"), Clock.new_test())


def crypto() -> CurrencyPair:
    assert isinstance(BTC, CurrencyPair)
    return BTC


def equity() -> Equity:
    assert isinstance(AAPL, Equity)
    return AAPL


@pytest.mark.parametrize(
    ("value", "places"),
    [("0.01", 2), (0.000000001, 9), ("1", 0), ("100", 0), (0.5, 1), (Decimal("1E-4"), 4)],
)
def test_decimal_places(value: Any, places: int) -> None:
    assert parsing.decimal_places(value) == places


def test_plain_never_uses_scientific_notation() -> None:
    assert parsing.plain(0.000000001) == "0.000000001"
    assert parsing.plain("12.50") == "12.50"


@pytest.mark.parametrize(
    ("symbol", "asset_class", "expected"),
    [
        ("BTCUSD", alpaca_enums.AssetClass.CRYPTO, "BTC/USD"),
        ("SHIBUSDC", alpaca_enums.AssetClass.CRYPTO, "SHIB/USDC"),
        ("ETHUSDT", alpaca_enums.AssetClass.CRYPTO, "ETH/USDT"),
        ("ETHBTC", alpaca_enums.AssetClass.CRYPTO, "ETH/BTC"),
        ("BTC/USD", alpaca_enums.AssetClass.CRYPTO, "BTC/USD"),
        ("AAPL", alpaca_enums.AssetClass.US_EQUITY, "AAPL"),
    ],
)
def test_normalize_symbol(symbol: str, asset_class: alpaca_enums.AssetClass, expected: str) -> None:
    assert parsing.normalize_symbol(symbol, asset_class) == expected


def test_instrument_ids_round_trip() -> None:
    assert parsing.instrument_id_of("BTC/USD") == BTC_ID
    assert parsing.symbol_of(BTC_ID) == "BTC/USD"
    assert parsing.is_crypto(BTC_ID)
    assert not parsing.is_crypto(parsing.instrument_id_of("AAPL"))


def test_crypto_asset_becomes_a_currency_pair_with_alpacas_own_precision() -> None:
    pair = crypto()
    assert pair.id == BTC_ID
    assert (pair.base_currency.code, pair.quote_currency.code) == ("BTC", "USD")
    assert (pair.price_precision, pair.size_precision) == (9, 9)
    assert pair.min_quantity == Quantity.from_str("0.000012372")


def test_equity_asset_becomes_a_whole_share_equity_in_usd() -> None:
    stock = equity()
    assert stock.id == InstrumentId.from_str("AAPL.ALPACA")
    assert stock.price_precision == 2
    assert stock.price_increment == Price.from_str("0.01")
    assert stock.make_qty(3).as_decimal() == Decimal(3)


def test_untradable_and_unsupported_assets_are_skipped() -> None:
    assert parsing.parse_instrument(support.asset("DEAD", tradable=False), 1) is None
    assert parsing.parse_instrument(support.asset("AAPL", **{"class": "us_option"}), 1) is None
    assert parsing.parse_instrument(support.asset("BTC/USD", price_increment=None), 1) is None


def test_quote_uses_the_instrument_precision() -> None:
    tick = parsing.parse_quote(support.quote(), crypto(), 5)
    assert tick.instrument_id == BTC_ID
    assert tick.bid_price.as_decimal() == Decimal("81000.1")
    assert tick.ask_size.as_decimal() == Decimal("2.25")
    assert (tick.ts_event, tick.ts_init) == (support.NOW_NS, 5)


def test_a_book_reset_clears_and_adds_every_level_as_one_snapshot() -> None:
    from nautilus_trader.model import BookAction, RecordFlag

    book = support.book([(81000.1, 1.5), (80999.0, 2.0)], [(81001.9, 0.75)], reset=True)
    deltas = parsing.parse_book(book, crypto(), 5)

    actions = [delta.action for delta in deltas.deltas]
    assert actions == [BookAction.CLEAR, BookAction.ADD, BookAction.ADD, BookAction.ADD]
    assert deltas.is_snapshot
    assert deltas.deltas[-1].flags & RecordFlag.F_LAST.value
    prices = [delta.order.price.as_decimal() for delta in deltas.deltas[1:]]
    assert prices == [Decimal("81000.1"), Decimal("80999.0"), Decimal("81001.9")]


def test_a_book_update_changes_levels_and_a_zero_size_removes_one() -> None:
    from nautilus_trader.model import BookAction, OrderSide

    book = support.book([(81000.1, 0.0)], [(81001.9, 3.0)], reset=False)
    deltas = parsing.parse_book(book, crypto(), 5)

    found = [(delta.action, delta.order.side) for delta in deltas.deltas]
    assert found == [(BookAction.DELETE, OrderSide.BUY), (BookAction.UPDATE, OrderSide.SELL)]
    assert not deltas.is_snapshot


def test_trade_has_no_aggressor_and_keeps_the_venue_trade_id() -> None:
    tick = parsing.parse_trade(support.trade(), crypto(), 5)
    assert str(tick.trade_id) == "42"
    assert tick.price.as_decimal() == Decimal("81000.5")
    assert tick.size.as_decimal() == Decimal("0.01")


def test_bar_keeps_the_requested_bar_type() -> None:
    bar_type = BarType.from_str("BTC/USD.ALPACA-1-MINUTE-LAST-EXTERNAL")
    bar = parsing.parse_bar(support.bar(), bar_type, crypto(), 5)
    assert bar.bar_type == bar_type
    assert (bar.open.as_decimal(), bar.close.as_decimal()) == (
        Decimal("80990.0"),
        Decimal("81000.5"),
    )
    assert bar.volume.as_decimal() == Decimal("12.5")


@pytest.mark.parametrize(
    ("spec", "timeframe"),
    [
        ("1-MINUTE", TimeFrame(1, TimeFrameUnit.Minute)),
        ("15-MINUTE", TimeFrame(15, TimeFrameUnit.Minute)),
        ("4-HOUR", TimeFrame(4, TimeFrameUnit.Hour)),
        ("1-DAY", TimeFrame(1, TimeFrameUnit.Day)),
    ],
)
def test_bar_types_map_to_alpaca_timeframes(spec: str, timeframe: TimeFrame) -> None:
    bar_type = BarType.from_str(f"BTC/USD.ALPACA-{spec}-LAST-EXTERNAL")
    result = parsing.to_timeframe(bar_type)
    assert (result.amount, result.unit) == (timeframe.amount, timeframe.unit)


def test_unsupported_bar_aggregations_are_rejected() -> None:
    with pytest.raises(ValueError, match="minute, hour and day"):
        parsing.to_timeframe(BarType.from_str("BTC/USD.ALPACA-100-TICK-LAST-INTERNAL"))


def test_market_order_becomes_a_market_request() -> None:
    nautilus_order = FACTORY.market(
        BTC_ID, OrderSide.BUY, Quantity.from_str("0.5"), TimeInForce.GTC
    )
    request = parsing.to_order_request(nautilus_order)
    assert isinstance(request, MarketOrderRequest)
    assert request.symbol == "BTC/USD"
    assert request.qty == 0.5
    assert request.side == alpaca_enums.OrderSide.BUY
    assert request.time_in_force == alpaca_enums.TimeInForce.GTC
    assert request.client_order_id == str(nautilus_order.client_order_id)


def test_a_stock_sell_only_closes_and_a_coin_order_carries_no_intent() -> None:
    aapl = equity().id
    sell = parsing.to_order_request(FACTORY.market(aapl, OrderSide.SELL, Quantity.from_int(1)))
    buy = parsing.to_order_request(FACTORY.market(aapl, OrderSide.BUY, Quantity.from_int(1)))
    coin = parsing.to_order_request(FACTORY.market(BTC_ID, OrderSide.SELL, Quantity.from_str("1")))

    assert sell.position_intent == alpaca_enums.PositionIntent.SELL_TO_CLOSE
    assert buy.position_intent == alpaca_enums.PositionIntent.BUY_TO_OPEN
    assert coin.position_intent is None


def test_limit_order_becomes_a_limit_request() -> None:
    nautilus_order = FACTORY.limit(
        BTC_ID, OrderSide.SELL, Quantity.from_str("1"), Price.from_str("90000.5"), TimeInForce.DAY
    )
    request = parsing.to_order_request(nautilus_order)
    assert isinstance(request, LimitOrderRequest)
    assert request.limit_price == 90000.5
    assert request.side == alpaca_enums.OrderSide.SELL


def test_unsupported_orders_are_refused_with_a_reason() -> None:
    touched = FACTORY.market_if_touched(
        BTC_ID, OrderSide.SELL, Quantity.from_str("1"), Price.from_str("1")
    )
    with pytest.raises(parsing.UnsupportedOrderError, match="order type"):
        parsing.to_order_request(touched)
    price = Price.from_str("1")
    gtd = FACTORY.limit(
        BTC_ID, OrderSide.BUY, Quantity.from_str("1"), price, TimeInForce.GTD, expire_time=10**18
    )
    with pytest.raises(parsing.UnsupportedOrderError, match="time in force"):
        parsing.to_order_request(gtd)


def test_stocks_are_commission_free() -> None:
    fee = parsing.commission(
        equity(),
        OrderSide.BUY,
        Quantity.from_int(10),
        Price.from_str("100.00"),
        LiquiditySide.TAKER,
        Decimal("0.0015"),
        Decimal("0.0025"),
    )
    assert (fee.as_decimal(), fee.currency.code) == (Decimal(0), "USD")


@pytest.mark.parametrize(
    ("side", "liquidity", "amount", "currency"),
    [
        (OrderSide.BUY, LiquiditySide.TAKER, "0.00125", "BTC"),
        (OrderSide.BUY, LiquiditySide.MAKER, "0.00075", "BTC"),
        (OrderSide.SELL, LiquiditySide.TAKER, "125.00", "USD"),
    ],
)
def test_crypto_fees_are_charged_in_the_asset_received(
    side: OrderSide, liquidity: LiquiditySide, amount: str, currency: str
) -> None:
    fee = parsing.commission(
        crypto(),
        side,
        Quantity.from_str("0.5"),
        Price.from_str("100000"),
        liquidity,
        Decimal("0.0015"),
        Decimal("0.0025"),
    )
    assert fee.currency.code == currency
    assert fee.as_decimal() == Decimal(amount)


def test_buy_fees_round_up_so_nautilus_never_holds_more_than_alpaca() -> None:
    fee = parsing.commission(
        crypto(),
        OrderSide.BUY,
        Quantity.from_str("0.00024605"),
        Price.from_str("81000"),
        LiquiditySide.TAKER,
        Decimal("0.0015"),
        Decimal("0.0025"),
    )
    assert fee.as_decimal() == Decimal("0.00000062")


def test_fees_already_at_the_currency_precision_are_unchanged() -> None:
    fee = parsing.commission(
        crypto(),
        OrderSide.BUY,
        Quantity.from_str("0.001"),
        Price.from_str("81000"),
        LiquiditySide.TAKER,
        Decimal("0.0015"),
        Decimal("0.0025"),
    )
    assert fee.as_decimal() == Decimal("0.0000025")


def test_order_status_report_for_a_working_limit_order() -> None:
    report = parsing.parse_order_status_report(
        support.order(), account_id=ACCOUNT_ID, instrument=crypto(), ts_init=9
    )
    assert report is not None
    assert str(report.venue_order_id) == str(support.ORDER_ID)
    assert str(report.client_order_id) == "O-1"
    assert (report.order_type, report.order_side) == (OrderType.LIMIT, OrderSide.BUY)
    assert report.order_status == OrderStatus.ACCEPTED
    assert report.price == Price.from_str("80000.000000000")
    assert report.avg_px is None


def test_a_stop_order_report_names_the_price_that_triggers_it() -> None:
    stop = support.order(order_type="stop_limit", type="stop_limit", stop_price="79000")
    report = parsing.parse_order_status_report(
        stop, account_id=ACCOUNT_ID, instrument=crypto(), ts_init=9
    )
    assert report is not None
    assert report.order_type == OrderType.STOP_LIMIT
    assert report.trigger_type == TriggerType.LAST_PRICE


def test_order_status_report_for_a_partial_fill_carries_the_average_price() -> None:
    filled = support.order(status="partially_filled", filled_qty="0.2", filled_avg_price="79950.5")
    report = parsing.parse_order_status_report(
        filled, account_id=ACCOUNT_ID, instrument=crypto(), ts_init=9
    )
    assert report is not None
    assert report.order_status == OrderStatus.PARTIALLY_FILLED
    assert report.filled_qty.as_decimal() == Decimal("0.2")
    assert report.avg_px == Decimal("79950.5")


@pytest.mark.parametrize(
    ("alpaca_status", "expected"),
    [
        ("pending_new", OrderStatus.SUBMITTED),
        ("accepted", OrderStatus.ACCEPTED),
        ("filled", OrderStatus.FILLED),
        ("canceled", OrderStatus.CANCELED),
        ("expired", OrderStatus.EXPIRED),
        ("rejected", OrderStatus.REJECTED),
        ("pending_cancel", OrderStatus.PENDING_CANCEL),
        ("pending_replace", OrderStatus.PENDING_UPDATE),
        ("replaced", OrderStatus.CANCELED),
    ],
)
def test_every_alpaca_status_maps_to_a_nautilus_status(
    alpaca_status: str, expected: OrderStatus
) -> None:
    report = parsing.parse_order_status_report(
        support.order(status=alpaca_status), account_id=ACCOUNT_ID, instrument=crypto(), ts_init=9
    )
    assert report is not None
    assert report.order_status == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"side": None},
        {"qty": None},
    ],
)
def test_orders_nautilus_cannot_represent_yield_no_report(overrides: dict[str, Any]) -> None:
    report = parsing.parse_order_status_report(
        support.order(**overrides), account_id=ACCOUNT_ID, instrument=crypto(), ts_init=9
    )
    assert report is None


def filled_order(**overrides: Any) -> Any:
    fields = {
        "status": "filled",
        "filled_qty": "0.25",
        "filled_avg_price": "81000.5",
        "filled_at": support.NOW,
        **overrides,
    }
    return support.order(**fields)


def test_fill_report_is_one_aggregate_fill_per_filled_order() -> None:
    report = parsing.parse_fill_report(
        filled_order(),
        account_id=ACCOUNT_ID,
        instrument=crypto(),
        maker_fee=Decimal("0.0015"),
        taker_fee=Decimal("0.0025"),
        ts_init=9,
    )
    assert report is not None
    assert str(report.trade_id) == str(support.ORDER_ID)
    assert str(report.venue_order_id) == str(support.ORDER_ID)
    assert report.last_qty.as_decimal() == Decimal("0.25")
    assert report.last_px.as_decimal() == Decimal("81000.5")
    assert report.commission.currency.code == "BTC"
    assert report.commission.as_decimal() == Decimal("0.000625")
    assert report.ts_event == support.NOW_NS


@pytest.mark.parametrize(
    "overrides",
    [{"filled_qty": "0"}, {"filled_qty": None}, {"filled_avg_price": None}, {"side": None}],
)
def test_orders_without_fills_yield_no_fill_report(overrides: dict[str, Any]) -> None:
    report = parsing.parse_fill_report(
        filled_order(**overrides),
        account_id=ACCOUNT_ID,
        instrument=crypto(),
        maker_fee=Decimal(0),
        taker_fee=Decimal(0),
        ts_init=9,
    )
    assert report is None


@pytest.mark.parametrize(
    ("side", "expected"), [("long", PositionSide.LONG), ("short", PositionSide.SHORT)]
)
def test_position_report_side_and_quantity(side: str, expected: PositionSide) -> None:
    report = parsing.parse_position_report(
        support.position(side=side, qty="-0.25" if side == "short" else "0.25"),
        account_id=ACCOUNT_ID,
        instrument=crypto(),
        ts_init=9,
    )
    assert report is not None
    assert report.position_side == expected
    assert report.quantity.as_decimal() == Decimal("0.25")
    assert report.avg_px_open == Decimal("81000.5")


def test_balances_reserve_cash_for_open_orders_and_list_crypto_holdings() -> None:
    usd, btc = parsing.parse_balances(
        support.account(), [support.position(), support.position("AAPL")]
    )
    assert (usd.total.as_decimal(), usd.free.as_decimal(), usd.locked.as_decimal()) == (
        Decimal(100000),
        Decimal(99000),
        Decimal(1000),
    )
    assert btc.total.currency == Currency.from_str("BTC")
    assert (btc.total.as_decimal(), btc.free.as_decimal(), btc.locked.as_decimal()) == (
        Decimal("0.25"),
        Decimal("0.25"),
        Decimal(0),
    )


def test_balances_never_report_negative_free_cash() -> None:
    [usd] = parsing.parse_balances(support.account(non_marginable_buying_power="-5"), [])
    assert usd.free.as_decimal() == Decimal(0)
    assert usd.locked.as_decimal() == Decimal(100000)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0.25", Decimal("0.25")), (1, Decimal(1)), (0.000997498, Decimal("0.000997498"))],
)
def test_exact_quantities_survive_a_crypto_pairs_precision(value: Any, expected: Decimal) -> None:
    quantity = parsing.exact_qty(crypto(), value)
    assert quantity is not None
    assert quantity.as_decimal() == expected


@pytest.mark.parametrize("value", [0.25, 1.5, "0.4"])
def test_fractional_shares_are_not_exact_for_a_whole_share_equity(value: Any) -> None:
    assert parsing.exact_qty(equity(), value) is None


def test_whole_shares_are_exact_for_an_equity() -> None:
    quantity = parsing.exact_qty(equity(), "10")
    assert quantity is not None
    assert quantity.as_decimal() == Decimal(10)


def test_market_data_sizes_round_instead_of_failing() -> None:
    assert parsing.rounded_qty(equity(), 0.4).as_decimal() == Decimal(0)
    assert parsing.rounded_qty(equity(), 1.5).as_decimal() == Decimal(2)


def test_a_fractional_share_position_yields_no_report() -> None:
    fractional = support.position("AAPL", qty="0.25")
    whole = support.position("AAPL", qty="10")
    kwargs: dict[str, Any] = {"account_id": ACCOUNT_ID, "instrument": equity(), "ts_init": 9}
    assert parsing.parse_position_report(fractional, **kwargs) is None
    report = parsing.parse_position_report(whole, **kwargs)
    assert report is not None
    assert report.quantity.as_decimal() == Decimal(10)


def test_fractional_orders_and_fills_for_whole_share_equities_are_skipped() -> None:
    order = support.order(
        "AAPL",
        qty="0.5",
        filled_qty="0.5",
        filled_avg_price="100",
        status="filled",
        filled_at=support.NOW,
    )
    kwargs: dict[str, Any] = {"account_id": ACCOUNT_ID, "instrument": equity(), "ts_init": 9}
    assert parsing.parse_order_status_report(order, **kwargs) is None
    fill = parsing.parse_fill_report(order, maker_fee=Decimal(0), taker_fee=Decimal(0), **kwargs)
    assert fill is None


def test_no_alpaca_order_status_is_left_without_a_nautilus_status() -> None:
    assert set(parsing.ORDER_STATUSES) == set(alpaca_enums.OrderStatus)


def test_stops_and_trailing_stops_go_to_alpaca_with_their_prices() -> None:
    quantity, price = Quantity.from_str("1"), Price.from_str("50000")
    stop = parsing.to_order_request(FACTORY.stop_market(BTC_ID, OrderSide.SELL, quantity, price))
    assert isinstance(stop, StopOrderRequest) and stop.stop_price == Decimal("50000")
    trigger = Price.from_str("49000")
    stop_limit = parsing.to_order_request(
        FACTORY.stop_limit(BTC_ID, OrderSide.SELL, quantity, price, trigger)
    )
    assert isinstance(stop_limit, StopLimitOrderRequest)
    assert (stop_limit.stop_price, stop_limit.limit_price) == (49000, 50000)
    trail = parsing.to_order_request(
        FACTORY.trailing_stop_market(
            BTC_ID,
            OrderSide.SELL,
            quantity,
            trailing_offset=Decimal(150),
            trailing_offset_type=TrailingOffsetType.BASIS_POINTS,
        )
    )
    assert isinstance(trail, TrailingStopOrderRequest) and trail.trail_percent == Decimal("1.5")


def test_stop_and_trailing_reports_carry_their_trigger_and_trail() -> None:
    stop = parsing.parse_order_status_report(
        support.order(order_type="stop", type="stop", stop_price="50000", limit_price=None),
        account_id=ACCOUNT_ID,
        instrument=crypto(),
        ts_init=9,
    )
    assert stop is not None and stop.order_type == OrderType.STOP_MARKET
    assert stop.trigger_price == Price.from_str("50000.00")
    trailing = parsing.parse_order_status_report(
        support.order(
            order_type="trailing_stop", type="trailing_stop", trail_percent="1.5", limit_price=None
        ),
        account_id=ACCOUNT_ID,
        instrument=crypto(),
        ts_init=9,
    )
    assert trailing is not None and trailing.trailing_offset == Decimal("150")
    assert trailing.trailing_offset_type == TrailingOffsetType.BASIS_POINTS
    opening = parsing.parse_order_status_report(
        support.order(time_in_force="opg"), account_id=ACCOUNT_ID, instrument=crypto(), ts_init=9
    )
    assert opening is not None and opening.time_in_force == TimeInForce.AT_THE_OPEN


def bracket(entry_type: OrderType = OrderType.MARKET) -> list[Any]:
    found = FACTORY.bracket(
        BTC_ID,
        OrderSide.BUY,
        Quantity.from_str("1"),
        entry_order_type=entry_type,
        entry_price=Price.from_str("50000") if entry_type == OrderType.LIMIT else None,
        tp_price=Price.from_str("60000"),
        sl_trigger_price=Price.from_str("45000"),
    )
    return list(found.orders) if hasattr(found, "orders") else found


def test_a_bracket_is_one_alpaca_request_with_both_legs() -> None:
    orders = bracket()
    found = parsing.to_list_request(orders)
    request = found.request
    assert request.order_class == "bracket"
    assert request.take_profit is not None and request.take_profit.limit_price == 60000
    assert request.stop_loss is not None and request.stop_loss.stop_price == 45000
    assert found.parent.contingency_type == ContingencyType.OTO
    assert found.take_profit is not None and found.stop_loss is not None
    assert request.client_order_id == str(found.parent.client_order_id)


def test_a_limit_entry_bracket_keeps_its_limit() -> None:
    found = parsing.to_list_request(bracket(OrderType.LIMIT))
    assert isinstance(found.request, LimitOrderRequest) and found.request.limit_price == 50000


def test_two_exits_without_an_entry_are_an_oco() -> None:
    exits = [order for order in bracket() if order.contingency_type != ContingencyType.OTO]
    found = parsing.to_list_request(exits)
    assert found.request.order_class == "oco"
    assert isinstance(found.parent, LimitOrder) and isinstance(found.request, LimitOrderRequest)
    assert found.request.limit_price == 60000
    assert found.take_profit is None and found.stop_loss is not None
