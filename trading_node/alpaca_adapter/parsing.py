from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_UP, Decimal
from typing import Any, Final

from alpaca.data.models import Bar as AlpacaBar
from alpaca.data.models import Orderbook as AlpacaBook
from alpaca.data.models import Quote as AlpacaQuote
from alpaca.data.models import Trade as AlpacaTrade
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading import enums as alpaca_enums
from alpaca.trading.enums import OrderClass
from alpaca.trading.models import Asset, Order, Position, TradeAccount
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from nautilus_trader.core import dt_to_unix_nanos
from nautilus_trader.model import (
    AccountBalance,
    AccountId,
    AggressorSide,
    Bar,
    BarAggregation,
    BarType,
    BookAction,
    BookOrder,
    ClientOrderId,
    ContingencyType,
    Currency,
    CurrencyPair,
    Equity,
    FillReport,
    InstrumentId,
    LimitOrder,
    LiquiditySide,
    MarketOrder,
    Money,
    OrderBookDelta,
    OrderBookDeltas,
    OrderSide,
    OrderStatus,
    OrderStatusReport,
    OrderType,
    PositionSide,
    PositionStatusReport,
    Price,
    Quantity,
    QuoteTick,
    RecordFlag,
    StopLimitOrder,
    StopMarketOrder,
    Symbol,
    TimeInForce,
    TradeId,
    TradeTick,
    TrailingOffsetType,
    TrailingStopMarketOrder,
    TriggerType,
    VenueOrderId,
)

from trading_node.alpaca_adapter.constants import ALPACA_VENUE

type AlpacaInstrument = Equity | CurrencyPair

USD: Final[Currency] = Currency.from_str("USD")
CRYPTO_QUOTE_CURRENCIES: Final[tuple[str, ...]] = ("USDC", "USDT", "USD", "BTC")
EQUITY_PRICE_PRECISION: Final[int] = 2
EQUITY_PRICE_INCREMENT: Final[Price] = Price.from_str("0.01")

ORDER_SIDES: Final[dict[alpaca_enums.OrderSide, OrderSide]] = {
    alpaca_enums.OrderSide.BUY: OrderSide.BUY,
    alpaca_enums.OrderSide.SELL: OrderSide.SELL,
}
ALPACA_ORDER_SIDES: Final[dict[OrderSide, alpaca_enums.OrderSide]] = {
    nautilus: alpaca for alpaca, nautilus in ORDER_SIDES.items()
}
ORDER_TYPES: Final[dict[alpaca_enums.OrderType, OrderType]] = {
    alpaca_enums.OrderType.MARKET: OrderType.MARKET,
    alpaca_enums.OrderType.LIMIT: OrderType.LIMIT,
    alpaca_enums.OrderType.STOP: OrderType.STOP_MARKET,
    alpaca_enums.OrderType.STOP_LIMIT: OrderType.STOP_LIMIT,
    alpaca_enums.OrderType.TRAILING_STOP: OrderType.TRAILING_STOP_MARKET,
}
TIME_IN_FORCE: Final[dict[alpaca_enums.TimeInForce, TimeInForce]] = {
    alpaca_enums.TimeInForce.DAY: TimeInForce.DAY,
    alpaca_enums.TimeInForce.GTC: TimeInForce.GTC,
    alpaca_enums.TimeInForce.IOC: TimeInForce.IOC,
    alpaca_enums.TimeInForce.FOK: TimeInForce.FOK,
    alpaca_enums.TimeInForce.OPG: TimeInForce.AT_THE_OPEN,
    alpaca_enums.TimeInForce.CLS: TimeInForce.AT_THE_CLOSE,
}
ALPACA_TIME_IN_FORCE: Final[dict[TimeInForce, alpaca_enums.TimeInForce]] = {
    nautilus: alpaca for alpaca, nautilus in TIME_IN_FORCE.items()
}
ORDER_STATUSES: Final[dict[alpaca_enums.OrderStatus, OrderStatus]] = {
    alpaca_enums.OrderStatus.PENDING_NEW: OrderStatus.SUBMITTED,
    alpaca_enums.OrderStatus.NEW: OrderStatus.ACCEPTED,
    alpaca_enums.OrderStatus.ACCEPTED: OrderStatus.ACCEPTED,
    alpaca_enums.OrderStatus.ACCEPTED_FOR_BIDDING: OrderStatus.ACCEPTED,
    alpaca_enums.OrderStatus.PENDING_REVIEW: OrderStatus.ACCEPTED,
    alpaca_enums.OrderStatus.HELD: OrderStatus.ACCEPTED,
    alpaca_enums.OrderStatus.CALCULATED: OrderStatus.ACCEPTED,
    alpaca_enums.OrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
    alpaca_enums.OrderStatus.FILLED: OrderStatus.FILLED,
    alpaca_enums.OrderStatus.CANCELED: OrderStatus.CANCELED,
    alpaca_enums.OrderStatus.DONE_FOR_DAY: OrderStatus.CANCELED,
    alpaca_enums.OrderStatus.EXPIRED: OrderStatus.EXPIRED,
    alpaca_enums.OrderStatus.REPLACED: OrderStatus.CANCELED,
    alpaca_enums.OrderStatus.REJECTED: OrderStatus.REJECTED,
    alpaca_enums.OrderStatus.SUSPENDED: OrderStatus.REJECTED,
    alpaca_enums.OrderStatus.STOPPED: OrderStatus.REJECTED,
    alpaca_enums.OrderStatus.PENDING_CANCEL: OrderStatus.PENDING_CANCEL,
    alpaca_enums.OrderStatus.PENDING_REPLACE: OrderStatus.PENDING_UPDATE,
}
BAR_UNITS: Final[dict[BarAggregation, TimeFrameUnit]] = {
    BarAggregation.MINUTE: TimeFrameUnit.Minute,
    BarAggregation.HOUR: TimeFrameUnit.Hour,
    BarAggregation.DAY: TimeFrameUnit.Day,
}


class UnsupportedOrderError(ValueError):
    """Raised when an order cannot be expressed as an Alpaca market or limit order."""


def decimal_places(value: float | str | Decimal) -> int:
    exponent = Decimal(str(value)).normalize().as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def plain(value: float | str | Decimal) -> str:
    return format(Decimal(str(value)), "f")


def normalize_symbol(symbol: str, asset_class: alpaca_enums.AssetClass) -> str:
    """Return crypto symbols in `BASE/QUOTE` form; positions report them as `BASEQUOTE`."""
    if asset_class != alpaca_enums.AssetClass.CRYPTO or "/" in symbol:
        return symbol
    quote = next(code for code in CRYPTO_QUOTE_CURRENCIES if symbol.endswith(code))
    return f"{symbol.removesuffix(quote)}/{quote}"


def rounded_qty(instrument: AlpacaInstrument, value: float) -> Quantity:
    """Market-data sizes may round to the instrument's precision."""
    return Quantity(value, instrument.size_precision)


def exact_qty(instrument: AlpacaInstrument, value: float | str) -> Quantity | None:
    """Accounting quantities must survive the instrument's precision unchanged, or be skipped."""
    quantity = Quantity(float(value), instrument.size_precision)
    return quantity if quantity.as_decimal() == Decimal(str(float(value))) else None


def instrument_id_of(symbol: str) -> InstrumentId:
    return InstrumentId(Symbol(symbol), ALPACA_VENUE)


def symbol_of(instrument_id: InstrumentId) -> str:
    return instrument_id.symbol.value


def is_crypto(instrument_id: InstrumentId) -> bool:
    return "/" in symbol_of(instrument_id)


def parse_instrument(asset: Asset, ts_init: int) -> AlpacaInstrument | None:
    """Build an Equity or crypto CurrencyPair; other asset classes are not supported."""
    if not asset.tradable:
        return None
    instrument_id = instrument_id_of(asset.symbol)
    if asset.asset_class == alpaca_enums.AssetClass.US_EQUITY:
        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(asset.symbol),
            currency=USD,
            price_precision=EQUITY_PRICE_PRECISION,
            price_increment=EQUITY_PRICE_INCREMENT,
            lot_size=Quantity.from_int(1),
            ts_event=ts_init,
            ts_init=ts_init,
        )
    if asset.asset_class == alpaca_enums.AssetClass.CRYPTO:
        price_step, size_step, min_size = (
            asset.price_increment,
            asset.min_trade_increment,
            asset.min_order_size,
        )
        if price_step is None or size_step is None or min_size is None:
            return None
        base, quote = asset.symbol.split("/")
        return CurrencyPair(
            instrument_id=instrument_id,
            raw_symbol=Symbol(asset.symbol),
            base_currency=Currency.from_str(base),
            quote_currency=Currency.from_str(quote),
            price_precision=decimal_places(price_step),
            size_precision=decimal_places(size_step),
            price_increment=Price.from_str(plain(price_step)),
            size_increment=Quantity.from_str(plain(size_step)),
            min_quantity=Quantity.from_str(plain(min_size)),
            ts_event=ts_init,
            ts_init=ts_init,
        )
    return None


def parse_quote(quote: AlpacaQuote, instrument: AlpacaInstrument, ts_init: int) -> QuoteTick:
    return QuoteTick(
        instrument_id=instrument.id,
        bid_price=instrument.make_price(quote.bid_price),
        ask_price=instrument.make_price(quote.ask_price),
        bid_size=rounded_qty(instrument, quote.bid_size),
        ask_size=rounded_qty(instrument, quote.ask_size),
        ts_event=dt_to_unix_nanos(quote.timestamp),
        ts_init=ts_init,
    )


def parse_book(book: AlpacaBook, instrument: AlpacaInstrument, ts_init: int) -> OrderBookDeltas:
    """Alpaca's book message as deltas: a reset clears and adds, else levels update or go."""
    ts_event = dt_to_unix_nanos(book.timestamp)
    levels = [(OrderSide.BUY, level) for level in book.bids]
    levels += [(OrderSide.SELL, level) for level in book.asks]
    flags = RecordFlag.F_SNAPSHOT.value if book.reset else 0
    deltas = []
    if book.reset:
        deltas.append(OrderBookDelta.clear(instrument.id, 0, ts_event, ts_init))
    for index, (side, level) in enumerate(levels):
        last = RecordFlag.F_LAST.value if index == len(levels) - 1 else 0
        if book.reset:
            action = BookAction.ADD
        else:
            action = BookAction.DELETE if level.size == 0 else BookAction.UPDATE
        order = BookOrder(
            side, instrument.make_price(level.price), rounded_qty(instrument, level.size), 0
        )
        deltas.append(
            OrderBookDelta(instrument.id, action, order, flags | last, 0, ts_event, ts_init)
        )
    return OrderBookDeltas(instrument.id, deltas)


def parse_trade(trade: AlpacaTrade, instrument: AlpacaInstrument, ts_init: int) -> TradeTick:
    return TradeTick(
        instrument_id=instrument.id,
        price=instrument.make_price(trade.price),
        size=rounded_qty(instrument, trade.size),
        aggressor_side=AggressorSide.NO_AGGRESSOR,
        trade_id=TradeId(str(trade.id)),
        ts_event=dt_to_unix_nanos(trade.timestamp),
        ts_init=ts_init,
    )


def parse_bar(bar: AlpacaBar, bar_type: BarType, instrument: AlpacaInstrument, ts_init: int) -> Bar:
    return Bar(
        bar_type=bar_type,
        open=instrument.make_price(bar.open),
        high=instrument.make_price(bar.high),
        low=instrument.make_price(bar.low),
        close=instrument.make_price(bar.close),
        volume=rounded_qty(instrument, bar.volume),
        ts_event=dt_to_unix_nanos(bar.timestamp),
        ts_init=ts_init,
    )


def to_timeframe(bar_type: BarType) -> TimeFrame:
    spec = bar_type.spec
    if spec.aggregation not in BAR_UNITS:
        raise ValueError(f"Alpaca supports minute, hour and day bars, not {spec.aggregation}")
    return TimeFrame(spec.step, BAR_UNITS[spec.aggregation])


AlpacaRequest = (
    MarketOrderRequest
    | LimitOrderRequest
    | StopOrderRequest
    | StopLimitOrderRequest
    | TrailingStopOrderRequest
)


def trailing(order: Any) -> dict[str, Decimal]:
    """Alpaca trails by a price or a percent; Nautilus by price or basis points."""
    offset = Decimal(str(order.trailing_offset))
    if order.trailing_offset_type == TrailingOffsetType.BASIS_POINTS:
        return {"trail_percent": offset / 100}
    if order.trailing_offset_type == TrailingOffsetType.PRICE:
        return {"trail_price": offset}
    raise UnsupportedOrderError(f"Alpaca cannot trail by {order.trailing_offset_type}")


@dataclass(frozen=True)
class ListRequest:
    """One Alpaca request for a Nautilus order list, and which order each leg stands for."""

    request: MarketOrderRequest | LimitOrderRequest
    parent: Any
    take_profit: Any | None
    stop_loss: Any | None


def is_exit_stop(order: Any) -> bool:
    return isinstance(order, StopMarketOrder | StopLimitOrder)


def to_list_request(orders: list[Any]) -> ListRequest:
    """A bracket, OTO or OCO list as Alpaca's single request with take-profit and stop-loss legs."""
    entries = [o for o in orders if o.contingency_type == ContingencyType.OTO]
    exits = [o for o in orders if o not in entries]
    take_profit = next((o for o in exits if isinstance(o, LimitOrder)), None)
    stop_loss = next((o for o in exits if is_exit_stop(o)), None)
    if len(entries) > 1 or len(exits) != len([o for o in (take_profit, stop_loss) if o]):
        raise UnsupportedOrderError("Alpaca takes one entry with a take-profit and a stop-loss")
    legs: dict[str, Any] = {}
    if take_profit is not None:
        legs["take_profit"] = TakeProfitRequest(limit_price=take_profit.price.as_decimal())
    if stop_loss is not None:
        legs["stop_loss"] = StopLossRequest(
            stop_price=stop_loss.trigger_price.as_decimal(),
            limit_price=stop_loss.price.as_decimal()
            if isinstance(stop_loss, StopLimitOrder)
            else None,
        )
    if entries:
        [entry] = entries
        if not isinstance(entry, MarketOrder | LimitOrder):
            raise UnsupportedOrderError("An Alpaca bracket enters with a market or limit order")
        kind = OrderClass.BRACKET if take_profit and stop_loss else OrderClass.OTO
        base = to_order_request(entry)
        fields = base.model_dump(exclude_none=True)
        entry_request: MarketOrderRequest | LimitOrderRequest = (
            LimitOrderRequest(**fields, order_class=kind, **legs)
            if isinstance(entry, LimitOrder)
            else MarketOrderRequest(**fields, order_class=kind, **legs)
        )
        return ListRequest(entry_request, entry, take_profit, stop_loss)
    if take_profit is None or stop_loss is None:
        raise UnsupportedOrderError("An Alpaca OCO needs a take-profit limit and a stop-loss")
    base = to_order_request(take_profit)
    fields = base.model_dump(exclude_none=True, exclude={"limit_price"})
    request = LimitOrderRequest(
        **fields, order_class=OrderClass.OCO, limit_price=take_profit.price.as_decimal(), **legs
    )
    return ListRequest(request, take_profit, None, stop_loss)


def position_intent(order: Any) -> alpaca_enums.PositionIntent | None:
    """A stock sell only closes what is held, so no order of the platform ever opens a short."""
    if is_crypto(order.instrument_id):
        return None
    if order.side == OrderSide.BUY:
        return alpaca_enums.PositionIntent.BUY_TO_OPEN
    return alpaca_enums.PositionIntent.SELL_TO_CLOSE


def to_order_request(order: Any) -> AlpacaRequest:
    """Express a Nautilus order as an Alpaca request, or raise UnsupportedOrderError."""
    if order.time_in_force not in ALPACA_TIME_IN_FORCE:
        raise UnsupportedOrderError(f"Unsupported time in force {order.time_in_force}")
    common: dict[str, Any] = {
        "symbol": symbol_of(order.instrument_id),
        "qty": order.quantity.as_decimal(),
        "side": alpaca_enums.OrderSide(order.side.name.lower()),
        "time_in_force": ALPACA_TIME_IN_FORCE[order.time_in_force],
        "client_order_id": str(order.client_order_id),
    }
    intent = position_intent(order)
    if intent is not None:
        common["position_intent"] = intent
    if isinstance(order, MarketOrder):
        return MarketOrderRequest(**common)
    if isinstance(order, LimitOrder):
        return LimitOrderRequest(**common, limit_price=order.price.as_decimal())
    if isinstance(order, StopMarketOrder):
        return StopOrderRequest(**common, stop_price=order.trigger_price.as_decimal())
    if isinstance(order, StopLimitOrder):
        return StopLimitOrderRequest(
            **common,
            stop_price=order.trigger_price.as_decimal(),
            limit_price=order.price.as_decimal(),
        )
    if isinstance(order, TrailingStopMarketOrder):
        return TrailingStopOrderRequest(**common, **trailing(order))
    raise UnsupportedOrderError(f"Unsupported order type {order.order_type}")


def commission(
    instrument: AlpacaInstrument,
    side: OrderSide,
    last_qty: Quantity,
    last_px: Price,
    liquidity_side: LiquiditySide,
    maker_fee: Decimal,
    taker_fee: Decimal,
) -> Money:
    """Stocks are commission-free; crypto fees are charged in the asset received.

    Buy fees round up to the currency precision so Nautilus never holds more than Alpaca.
    """
    if not isinstance(instrument, CurrencyPair):
        return Money(0.0, instrument.quote_currency)
    rate = maker_fee if liquidity_side == LiquiditySide.MAKER else taker_fee
    if side == OrderSide.BUY:
        fee = last_qty.as_decimal() * rate
        step = Decimal(1).scaleb(-instrument.base_currency.precision)
        return Money(float(fee.quantize(step, rounding=ROUND_UP)), instrument.base_currency)
    notional = last_qty.as_decimal() * last_px.as_decimal()
    return Money(float(notional * rate), instrument.quote_currency)


TRIGGERED: Final = (OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP_MARKET)


def parse_order_status_report(
    order: Order, *, account_id: AccountId, instrument: AlpacaInstrument, ts_init: int
) -> OrderStatusReport | None:
    """Return None for orders Nautilus cannot represent (unknown types or TIFs, fractions)."""
    if order.side is None or order.order_type is None or order.qty is None:
        return None
    order_type = ORDER_TYPES.get(order.order_type)
    time_in_force = TIME_IN_FORCE.get(order.time_in_force)
    if order_type is None or time_in_force is None:
        return None
    quantity = exact_qty(instrument, order.qty)
    filled_qty = exact_qty(instrument, order.filled_qty or 0)
    if quantity is None or filled_qty is None:
        return None
    limit_price = order.limit_price
    stop_price = order.stop_price
    filled_avg_price = order.filled_avg_price
    offset: Decimal | None = None
    offset_type: TrailingOffsetType | None = None
    if order_type == OrderType.TRAILING_STOP_MARKET:
        if order.trail_percent is not None:
            offset = Decimal(str(order.trail_percent)) * 100
            offset_type = TrailingOffsetType.BASIS_POINTS
        elif order.trail_price is not None:
            offset, offset_type = Decimal(str(order.trail_price)), TrailingOffsetType.PRICE
    return OrderStatusReport(
        account_id=account_id,
        instrument_id=instrument.id,
        venue_order_id=VenueOrderId(str(order.id)),
        order_side=ORDER_SIDES[order.side],
        order_type=order_type,
        time_in_force=time_in_force,
        order_status=ORDER_STATUSES[order.status],
        quantity=quantity,
        filled_qty=filled_qty,
        ts_accepted=dt_to_unix_nanos(order.submitted_at),
        ts_last=dt_to_unix_nanos(order.updated_at),
        ts_init=ts_init,
        client_order_id=ClientOrderId(order.client_order_id),
        price=instrument.make_price(float(limit_price)) if limit_price else None,
        trigger_price=instrument.make_price(float(stop_price)) if stop_price else None,
        trigger_type=TriggerType.LAST_PRICE if order_type in TRIGGERED else None,
        avg_px=Decimal(str(filled_avg_price)) if filled_avg_price else None,
        trailing_offset=offset,
        trailing_offset_type=offset_type,
    )


def parse_fill_report(
    order: Order,
    *,
    account_id: AccountId,
    instrument: AlpacaInstrument,
    maker_fee: Decimal,
    taker_fee: Decimal,
    ts_init: int,
) -> FillReport | None:
    """One aggregate fill per order; alpaca-py exposes per-fill activity only in its Broker API."""
    filled_qty = float(order.filled_qty or 0)
    last_qty = exact_qty(instrument, filled_qty)
    if order.side is None or filled_qty <= 0 or last_qty is None or not order.filled_avg_price:
        return None
    side = ORDER_SIDES[order.side]
    last_px = instrument.make_price(float(order.filled_avg_price))
    liquidity_side = LiquiditySide.NO_LIQUIDITY_SIDE
    fee = commission(instrument, side, last_qty, last_px, liquidity_side, maker_fee, taker_fee)
    return FillReport(
        account_id=account_id,
        instrument_id=instrument.id,
        venue_order_id=VenueOrderId(str(order.id)),
        trade_id=TradeId(str(order.id)),
        order_side=side,
        last_qty=last_qty,
        last_px=last_px,
        commission=fee,
        liquidity_side=liquidity_side,
        ts_event=dt_to_unix_nanos(order.filled_at or order.updated_at),
        ts_init=ts_init,
        client_order_id=ClientOrderId(order.client_order_id),
    )


def parse_position_report(
    position: Position, *, account_id: AccountId, instrument: AlpacaInstrument, ts_init: int
) -> PositionStatusReport | None:
    """Return None for positions Nautilus cannot hold exactly, such as fractional shares."""
    quantity = exact_qty(instrument, abs(float(position.qty)))
    if quantity is None:
        return None
    is_long = position.side == alpaca_enums.PositionSide.LONG
    return PositionStatusReport(
        account_id=account_id,
        instrument_id=instrument.id,
        position_side=PositionSide.LONG if is_long else PositionSide.SHORT,
        quantity=quantity,
        ts_last=ts_init,
        ts_init=ts_init,
        avg_px_open=Decimal(position.avg_entry_price),
    )


def parse_balances(account: TradeAccount, positions: Sequence[Position]) -> list[AccountBalance]:
    """USD cash (less what open orders reserve) plus each crypto position as a base balance."""
    total = Decimal(account.cash or 0)
    free = max(Decimal(0), min(total, Decimal(account.non_marginable_buying_power or 0)))
    balances = [
        AccountBalance(
            Money(float(total), USD), Money(float(total - free), USD), Money(float(free), USD)
        )
    ]
    for position in positions:
        if position.asset_class != alpaca_enums.AssetClass.CRYPTO:
            continue
        symbol = normalize_symbol(position.symbol, position.asset_class)
        currency = Currency.from_str(symbol.split("/")[0])
        quantity = float(position.qty)
        balances.append(
            AccountBalance(
                Money(quantity, currency), Money(0.0, currency), Money(quantity, currency)
            )
        )
    return balances
