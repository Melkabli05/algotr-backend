import asyncio
from decimal import Decimal
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import QueryOrderStatus, TradeEvent
from alpaca.trading.models import Order, Position, TradeAccount, TradeUpdate
from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
from alpaca.trading.stream import TradingStream
from nautilus_trader import live
from nautilus_trader.common import Logger
from nautilus_trader.core import dt_to_unix_nanos, unix_nanos_to_dt
from nautilus_trader.live.clients import ExecutionClient
from nautilus_trader.model import (
    AccountId,
    AccountType,
    ClientOrderId,
    FillReport,
    LiquiditySide,
    OmsType,
    OrderStatus,
    OrderStatusReport,
    OrderType,
    PositionStatusReport,
    TradeId,
    TraderId,
    VenueOrderId,
)
from requests import RequestException

from trading_node.alpaca_adapter.config import AlpacaExecClientConfig
from trading_node.alpaca_adapter.constants import ALPACA, ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import (
    ALPACA_ORDER_SIDES,
    AlpacaInstrument,
    ListRequest,
    UnsupportedOrderError,
    commission,
    instrument_id_of,
    normalize_symbol,
    parse_balances,
    parse_fill_report,
    parse_order_status_report,
    parse_position_report,
    symbol_of,
    to_list_request,
    to_order_request,
)
from trading_node.alpaca_adapter.providers import AlpacaInstrumentProvider
from trading_node.alpaca_adapter.sdk import (
    call,
    call_list,
    call_soon,
    error_reason,
    is_definitive_rejection,
)
from trading_node.alpaca_adapter.streams import StreamThread

ORDER_LIMIT = 500
UNKNOWN_RETRY_SECS = (1, 2, 4, 8)
POLICY_RESOLUTION = "Policy resolution:"
POSITION_TOLERANCE = Decimal("0.00000001")


class AlpacaExecutionClient(ExecutionClient):
    """Places market and limit orders on Alpaca and reports fills from its trade-update stream."""

    def __init__(
        self,
        *,
        name: str,
        config: AlpacaExecClientConfig,
        cache: live.ClientCache,
        clock: Any,
        trader_id: TraderId,
        instrument_provider: AlpacaInstrumentProvider,
        trading: TradingClient,
    ) -> None:
        mode = "PAPER" if config.paper else "LIVE"
        super().__init__(
            name=name,
            config=config,
            cache=cache,
            clock=clock,
            trader_id=trader_id,
            venue=ALPACA_VENUE,
            account_id=AccountId(f"{ALPACA}-{mode}-001"),
            account_type=AccountType.CASH,
            oms_type=OmsType.NETTING,
            instrument_provider=instrument_provider,
            position_reconciliation_tolerance=POSITION_TOLERANCE,
        )
        self._alpaca = config
        self._provider = instrument_provider
        self._trading = trading
        self._log = Logger(type(self).__name__)
        self._legs: dict[str, ClientOrderId] = {}
        self._updates: StreamThread | None = None

    async def _connect(self) -> None:
        await self._provider.initialize()
        for instrument in self._provider.list_all():
            self._handle_instrument(instrument)
        await self._refresh_account()
        stream = TradingStream(
            self._alpaca.api_key, self._alpaca.api_secret, paper=self._alpaca.paper
        )
        stream.subscribe_trade_updates(self._on_trade_update)
        self._updates = StreamThread(stream, "alpaca-trade-updates")
        self._updates.start()

    async def _disconnect(self) -> None:
        if self._updates is not None:
            await asyncio.to_thread(self._updates.stop)
            self._updates = None

    async def _refresh_account(self) -> None:
        account = await call(self._trading.get_account, TradeAccount)
        positions = await call_list(self._trading.get_all_positions, Position)
        self.generate_account_state(
            balances=parse_balances(account, positions),
            margins=[],
            reported=True,
            ts_event=self.clock.timestamp_ns(),
        )

    async def _instrument(self, symbol: str) -> AlpacaInstrument | None:
        instrument_id = instrument_id_of(symbol)
        await self._provider.load_async(instrument_id)
        return self._provider.find_instrument(instrument_id)

    async def _on_trade_update(self, update: TradeUpdate | dict[Any, Any]) -> None:
        if isinstance(update, TradeUpdate):
            call_soon(self.loop, self._process_update, update)

    def _process_update(self, update: TradeUpdate) -> None:
        """Translate an Alpaca trade update into the matching Nautilus order event."""
        alpaca_order = update.order
        leg = self._legs.get(str(alpaca_order.id))
        order = self.cache.order(leg or ClientOrderId(alpaca_order.client_order_id))
        instrument = self._provider.find_instrument(instrument_id_of(alpaca_order.symbol or ""))
        if order is None or instrument is None:
            return
        try:
            event = TradeEvent(update.event)
        except ValueError:
            return
        venue_order_id = VenueOrderId(str(alpaca_order.id))
        ts_event = dt_to_unix_nanos(update.timestamp)
        match event:
            case TradeEvent.NEW | TradeEvent.ACCEPTED:
                if order.status == OrderStatus.SUBMITTED:
                    self.generate_order_accepted(order, venue_order_id, ts_event)
            case TradeEvent.FILL | TradeEvent.PARTIAL_FILL:
                self._process_fill(update, order, instrument, venue_order_id, ts_event)
            case TradeEvent.CANCELED:
                self.generate_order_canceled(order, venue_order_id, ts_event)
            case TradeEvent.EXPIRED:
                self.generate_order_expired(order, venue_order_id, ts_event)
            case TradeEvent.REJECTED:
                self.generate_order_rejected(order, "Rejected by Alpaca", ts_event, False)
            case TradeEvent.REPLACED:
                self.generate_order_updated(
                    order,
                    venue_order_id,
                    instrument.make_qty(float(alpaca_order.qty or 0)),
                    instrument.make_price(float(alpaca_order.limit_price))
                    if alpaca_order.limit_price
                    else None,
                    None,
                    None,
                    ts_event,
                )

    def _process_fill(
        self,
        update: TradeUpdate,
        order: Any,
        instrument: AlpacaInstrument,
        venue_order_id: VenueOrderId,
        ts_event: int,
    ) -> None:
        last_qty = instrument.make_qty(update.qty or 0)
        last_px = instrument.make_price(update.price or 0)
        liquidity_side = (
            LiquiditySide.TAKER
            if order.order_type == OrderType.MARKET
            else LiquiditySide.NO_LIQUIDITY_SIDE
        )
        fee = commission(
            instrument,
            order.side,
            last_qty,
            last_px,
            liquidity_side,
            self._alpaca.crypto_maker_fee,
            self._alpaca.crypto_taker_fee,
        )
        self.generate_order_filled(
            order,
            venue_order_id,
            None,
            TradeId(str(update.execution_id or f"{venue_order_id}-{ts_event}")),
            last_qty,
            last_px,
            instrument.quote_currency,
            fee,
            liquidity_side,
            ts_event,
        )
        self.create_task(self._refresh_account(), "refresh-account")

    async def _submit_order(self, command: live.SubmitOrder) -> None:
        order = command.order
        try:
            request = to_order_request(order)
        except UnsupportedOrderError as exc:
            self.generate_order_denied(order, str(exc))
            return
        self.generate_order_submitted(order)
        try:
            await asyncio.to_thread(self._trading.submit_order, request)
        except APIError as exc:
            if is_definitive_rejection(exc):
                self.generate_order_rejected(
                    order, error_reason(exc), self.clock.timestamp_ns(), False
                )
            else:
                self._log.error(f"Outcome of {order.client_order_id} unknown: {exc}")
                await self._resolve_unknown(order)
        except RequestException as exc:
            self._log.error(f"Outcome of {order.client_order_id} unknown: {exc}")
            await self._resolve_unknown(order)

    async def _submit_order_list(self, command: live.SubmitOrderList) -> None:
        """A bracket, OTO or OCO as one Alpaca request; its legs are tracked by Alpaca's IDs."""
        orders = [self.cache.order(i) for i in command.order_list.client_order_ids()]
        present = [order for order in orders if order is not None]
        try:
            found = to_list_request(present)
        except UnsupportedOrderError as exc:
            for order in present:
                self.generate_order_denied(order, str(exc))
            return
        for order in present:
            self.generate_order_submitted(order)
        try:
            placed = await call(self._trading.submit_order, Order, found.request)
        except APIError as exc:
            if is_definitive_rejection(exc):
                for order in present:
                    self.generate_order_rejected(
                        order, error_reason(exc), self.clock.timestamp_ns(), False
                    )
            else:
                self._log.error(f"Outcome of {found.parent.client_order_id} unknown: {exc}")
                await self._resolve_unknown(found.parent)
            return
        except RequestException as exc:
            self._log.error(f"Outcome of {found.parent.client_order_id} unknown: {exc}")
            await self._resolve_unknown(found.parent)
            return
        self._remember_legs(placed, found)

    def _remember_legs(self, placed: Order, found: ListRequest) -> None:
        """Alpaca names the legs itself; map each to the Nautilus order it stands for."""
        for leg in placed.legs or []:
            kind = leg.order_type or leg.type
            target = found.take_profit if kind == AlpacaOrderType.LIMIT else found.stop_loss
            if target is not None:
                self._legs[str(leg.id)] = target.client_order_id

    async def _resolve_unknown(self, order: Any) -> None:
        """Ask Alpaca whether a submit with an unknown outcome arrived; never resend it.

        Found: the order's report is applied, as for any other venue answer. Never found after
        the retries: the order is rejected locally with a reason that says this was a policy
        resolution, not a venue answer, so the reconciliation view can ask for a person.
        """
        for delay in UNKNOWN_RETRY_SECS:
            await asyncio.sleep(delay)
            try:
                found = await self._fetch_order(order.client_order_id, None)
            except RequestException as exc:
                self._log.warning(f"Still cannot reach Alpaca for {order.client_order_id}: {exc}")
                continue
            if found is not None:
                report = await self._report_for(found)
                if report is not None:
                    self._handle_report(report)
                    return
        self.generate_order_rejected(
            order,
            f"{POLICY_RESOLUTION} not found at Alpaca after an unknown submit outcome",
            self.clock.timestamp_ns(),
            False,
        )

    def _venue_order_id(
        self, client_order_id: ClientOrderId, venue_order_id: VenueOrderId | None
    ) -> VenueOrderId | None:
        return venue_order_id or self.cache.venue_order_id(client_order_id)

    async def _cancel_order(self, command: live.CancelOrder) -> None:
        order = self.cache.order(command.client_order_id)
        venue_order_id = self._venue_order_id(command.client_order_id, command.venue_order_id)
        try:
            if venue_order_id is None:
                raise ValueError("The order has no venue order ID yet")
            await asyncio.to_thread(self._trading.cancel_order_by_id, str(venue_order_id))
        except APIError as exc:
            self._reject_cancel(order, venue_order_id, error_reason(exc))
        except ValueError as exc:
            self._reject_cancel(order, venue_order_id, str(exc))
        except RequestException as exc:
            self._log.error(f"Outcome of cancelling {command.client_order_id} unknown: {exc}")

    def _reject_cancel(self, order: Any, venue_order_id: VenueOrderId | None, reason: str) -> None:
        if order is not None:
            self.generate_order_cancel_rejected(
                order, venue_order_id, reason, self.clock.timestamp_ns()
            )

    async def _cancel_all_orders(self, command: live.CancelAllOrders) -> None:
        order_side = command.order_side
        side = ALPACA_ORDER_SIDES.get(order_side) if order_side is not None else None
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[symbol_of(command.instrument_id)], side=side
        )
        for open_order in await call_list(self._trading.get_orders, Order, request):
            try:
                await asyncio.to_thread(self._trading.cancel_order_by_id, open_order.id)
            except APIError as exc:
                self._log.warning(f"Could not cancel {open_order.id}: {error_reason(exc)}")

    async def _modify_order(self, command: live.ModifyOrder) -> None:
        order = self.cache.order(command.client_order_id)
        venue_order_id = self._venue_order_id(command.client_order_id, command.venue_order_id)
        if venue_order_id is None:
            self._reject_modify(order, None, "The order has no venue order ID yet")
            return
        request = ReplaceOrderRequest(
            qty=command.quantity.as_decimal() if command.quantity else None,
            limit_price=command.price.as_decimal() if command.price else None,
        )
        try:
            await asyncio.to_thread(self._trading.replace_order_by_id, str(venue_order_id), request)
        except APIError as exc:
            self._reject_modify(order, venue_order_id, error_reason(exc))
        except RequestException as exc:
            self._log.error(f"Outcome of modifying {command.client_order_id} unknown: {exc}")

    def _reject_modify(self, order: Any, venue_order_id: VenueOrderId | None, reason: str) -> None:
        if order is not None:
            self.generate_order_modify_rejected(
                order, venue_order_id, reason, self.clock.timestamp_ns()
            )

    async def _query_account(self, command: live.QueryAccount) -> None:
        await self._refresh_account()

    async def _query_order(self, command: live.QueryOrder) -> None:
        report = await self._order_report(command.client_order_id, command.venue_order_id)
        if report is not None:
            self._handle_report(report)

    async def _fetch_order(
        self, client_order_id: ClientOrderId | None, venue_order_id: VenueOrderId | None
    ) -> Order | None:
        try:
            if venue_order_id is not None:
                return await call(self._trading.get_order_by_id, Order, str(venue_order_id))
            if client_order_id is not None:
                return await call(self._trading.get_order_by_client_id, Order, str(client_order_id))
        except APIError as exc:
            self._log.warning(f"Could not fetch order: {error_reason(exc)}")
        return None

    async def _order_report(
        self, client_order_id: ClientOrderId | None, venue_order_id: VenueOrderId | None
    ) -> OrderStatusReport | None:
        order = await self._fetch_order(client_order_id, venue_order_id)
        return None if order is None else await self._report_for(order)

    async def _report_for(self, order: Order) -> OrderStatusReport | None:
        instrument = await self._instrument(self._order_symbol(order))
        if instrument is None:
            return None
        return parse_order_status_report(
            order,
            account_id=self.account_id,
            instrument=instrument,
            ts_init=self.clock.timestamp_ns(),
        )

    @staticmethod
    def _order_symbol(order: Order) -> str:
        symbol = order.symbol or ""
        return normalize_symbol(symbol, order.asset_class) if order.asset_class else symbol

    async def _orders_between(
        self,
        status: QueryOrderStatus,
        instrument_id: Any,
        start: int | None,
        end: int | None,
    ) -> list[Order]:
        request = GetOrdersRequest(
            status=status,
            limit=ORDER_LIMIT,
            after=unix_nanos_to_dt(start) if start else None,
            until=unix_nanos_to_dt(end) if end else None,
            symbols=[symbol_of(instrument_id)] if instrument_id else None,
        )
        return await call_list(self._trading.get_orders, Order, request)

    async def _generate_order_status_report(
        self, command: live.GenerateOrderStatusReport
    ) -> OrderStatusReport | None:
        return await self._order_report(command.client_order_id, command.venue_order_id)

    async def _generate_order_status_reports(
        self, command: live.GenerateOrderStatusReports
    ) -> list[OrderStatusReport]:
        status = QueryOrderStatus.OPEN if command.open_only else QueryOrderStatus.ALL
        orders = await self._orders_between(
            status, command.instrument_id, command.start, command.end
        )
        reports = [await self._report_for(order) for order in orders]
        return [report for report in reports if report is not None]

    async def _generate_fill_reports(self, command: live.GenerateFillReports) -> list[FillReport]:
        orders = await self._orders_between(
            QueryOrderStatus.CLOSED, command.instrument_id, command.start, command.end
        )
        reports: list[FillReport] = []
        for order in orders:
            instrument = await self._instrument(self._order_symbol(order))
            if instrument is None:
                continue
            report = parse_fill_report(
                order,
                account_id=self.account_id,
                instrument=instrument,
                maker_fee=self._alpaca.crypto_maker_fee,
                taker_fee=self._alpaca.crypto_taker_fee,
                ts_init=self.clock.timestamp_ns(),
            )
            if report is not None and self._wanted(report, command.venue_order_id):
                reports.append(report)
        return reports

    @staticmethod
    def _wanted(report: FillReport, venue_order_id: VenueOrderId | None) -> bool:
        return venue_order_id is None or report.venue_order_id == venue_order_id

    async def _generate_position_status_reports(
        self, command: live.GeneratePositionStatusReports
    ) -> list[PositionStatusReport]:
        reports = []
        for position in await call_list(self._trading.get_all_positions, Position):
            symbol = normalize_symbol(position.symbol, position.asset_class)
            if command.instrument_id and instrument_id_of(symbol) != command.instrument_id:
                continue
            instrument = await self._instrument(symbol)
            if instrument is None:
                continue
            report = parse_position_report(
                position,
                account_id=self.account_id,
                instrument=instrument,
                ts_init=self.clock.timestamp_ns(),
            )
            if report is None:
                self._log.warning(f"Ignoring {symbol}: quantity {position.qty} is not exact")
            else:
                reports.append(report)
        return reports
