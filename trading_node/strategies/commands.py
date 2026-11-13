import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, Self

import sqlalchemy as sa
from nautilus_trader.model import (
    BarType,
    ClientOrderId,
    InstrumentId,
    OrderSide,
    OrderType,
    PriceType,
    TimeInForce,
    TrailingOffsetType,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from app.bridge.commands import unsupported
from app.core import breakers
from trading_node import commands, metrics
from trading_node.alpaca_adapter.constants import ALPACA_VENUE
from trading_node.alpaca_adapter.parsing import USD, instrument_id_of, is_crypto
from trading_node.exec_algorithms.family import STOP_TOPIC, family_outcome, is_algorithm_primary
from trading_node.exec_algorithms.iceberg import ICEBERG, iceberg_params
from trading_node.exec_algorithms.scheduled import SCHEDULED, schedule_params
from trading_node.state_writer import StateWriter
from trading_node.strategies.plan import whole_units
from trading_node.strategies.rebalance import (
    DEADLINE,
    PRICE_WAIT,
    SETTLE_WAIT,
    Leg,
    Rebalance,
    legs_of,
    outcome,
)

POLL = timedelta(milliseconds=500)
SNAPSHOT = timedelta(seconds=5)
WAIT_SECONDS = 15
PRICE_HISTORY = timedelta(days=7)
TIME_IN_FORCE = {
    "day": TimeInForce.DAY,
    "gtc": TimeInForce.GTC,
    "ioc": TimeInForce.IOC,
    "fok": TimeInForce.FOK,
    "opg": TimeInForce.AT_THE_OPEN,
    "cls": TimeInForce.AT_THE_CLOSE,
}
PLACING = ("submit", "rebalance")
Decision = Literal["attach", "requeue", "fail", "stopped"]
UNKNOWN_AFTER_RESTART = "Sent before a restart and not found at Alpaca; check before resending"
EXPIRED = "Expired before the node took it; nothing was sent"
STOPPED_BY_RESTART = (
    "The node restarted during the algorithm; what filled stays filled, the rest was not sent"
)


def side_of(payload: dict[str, Any]) -> OrderSide:
    return OrderSide.BUY if payload["side"] == "buy" else OrderSide.SELL


def time_in_force_of(payload: dict[str, Any], instrument: Any) -> TimeInForce:
    default = "gtc" if is_crypto(instrument.id) else "day"
    return TIME_IN_FORCE[payload.get("time_in_force") or default]


def nanos(moment: str) -> int:
    return int(datetime.fromisoformat(moment).timestamp() * 1e9)


def algorithm_of(payload: dict[str, Any]) -> dict[str, Any]:
    """The execution algorithm that works the order, if any, as Nautilus order arguments."""
    algo = payload.get("algo")
    if not algo:
        return {}
    if algo["name"] == "iceberg":
        return {
            "exec_algorithm_id": ICEBERG,
            "exec_algorithm_params": iceberg_params(float(algo["display_qty"])),
        }
    intervals = [(nanos(start), float(weight)) for start, weight in algo["intervals"]]
    params = schedule_params(intervals, nanos(algo["end"]), float(algo.get("tilt") or 0.0))
    return {
        "exec_algorithm_id": SCHEDULED,
        "exec_algorithm_params": params,
        "time_in_force": TimeInForce.IOC,
    }


def build_order(
    factory: Any, payload: dict[str, Any], instrument: Any, client_id: ClientOrderId
) -> Any:
    """One Nautilus order from a command's payload, under the command's client order ID."""
    kind = payload.get("type", "market")
    side, quantity = side_of(payload), instrument.make_qty(float(payload["qty"]))
    common = {
        "time_in_force": time_in_force_of(payload, instrument),
        "client_order_id": client_id,
        **algorithm_of(payload),
    }

    def price(key: str) -> Any:
        return instrument.make_price(float(payload[key]))

    if kind == "limit":
        return factory.limit(instrument.id, side, quantity, price("limit_price"), **common)
    if kind == "stop":
        return factory.stop_market(instrument.id, side, quantity, price("stop_price"), **common)
    if kind == "stop_limit":
        return factory.stop_limit(
            instrument.id, side, quantity, price("limit_price"), price("stop_price"), **common
        )
    if kind == "trailing_stop":
        if payload.get("trail_percent") is not None:
            offset = Decimal(str(payload["trail_percent"])) * 100
            offset_type = TrailingOffsetType.BASIS_POINTS
        else:
            offset, offset_type = Decimal(str(payload["trail_price"])), TrailingOffsetType.PRICE
        return factory.trailing_stop_market(
            instrument.id,
            side,
            quantity,
            trailing_offset=offset,
            trailing_offset_type=offset_type,
            **common,
        )
    return factory.market(instrument.id, side, quantity, **common)


def sweep_decision(command: dict[str, Any], found_at_alpaca: bool) -> Decision:
    """After a restart: an order Alpaca has is followed; one never sent is sent; else a person."""
    if found_at_alpaca:
        return "attach"
    if command["kind"] not in PLACING or command["status"] == "claimed":
        return "requeue"
    if command["kind"] == "rebalance" or command["payload"].get("algo"):
        return "stopped"
    return "fail"


def epoch(nanos: int) -> datetime:
    return datetime.fromtimestamp(nanos / 1e9, UTC)


def number(value: Any) -> Decimal | None:
    if value is None:
        return None
    found = value.as_decimal() if hasattr(value, "as_decimal") else Decimal(str(value))
    return found


def order_row(order: Any, last_event: str, reason: str | None = None) -> dict[str, Any]:
    client_id = str(order.client_order_id)
    command_id = commands.command_id_of(client_id)
    price = getattr(order, "price", None)
    trigger = getattr(order, "trigger_price", None)
    venue = order.venue_order_id
    algorithm = getattr(order, "exec_algorithm_id", None)
    spawn = getattr(order, "exec_spawn_id", None)
    return {
        "client_order_id": client_id,
        "venue_order_id": None if venue is None else str(venue),
        "command_id": command_id,
        "primary_id": None if spawn is None else str(spawn),
        "exec_algorithm": None if algorithm is None else str(algorithm),
        "strategy_id": str(order.strategy_id),
        "instrument_id": str(order.instrument_id),
        "side": order.side.name.lower(),
        "order_type": order.order_type.name.lower(),
        "time_in_force": order.time_in_force.name.lower(),
        "quantity": number(order.quantity),
        "filled_qty": number(order.filled_qty) or Decimal(0),
        "avg_px": number(order.avg_px) if order.filled_qty else None,
        "price": number(price),
        "trigger_price": number(trigger),
        "status": order.status.name.lower(),
        "is_open": bool(order.is_open),
        "is_closed": bool(order.is_closed),
        "created": epoch(order.ts_init),
        "updated": datetime.now(UTC),
        "last_event": last_event,
        "reason": reason,
    }


def fill_row(event: Any) -> dict[str, Any]:
    commission = getattr(event, "commission", None)
    return {
        "trade_id": str(event.trade_id),
        "client_order_id": str(event.client_order_id),
        "venue_order_id": str(event.venue_order_id),
        "instrument_id": str(event.instrument_id),
        "side": event.order_side.name.lower(),
        "quantity": number(event.last_qty),
        "price": number(event.last_px),
        "commission": None if commission is None else number(commission),
        "commission_currency": None if commission is None else str(commission.currency),
        "liquidity_side": event.liquidity_side.name.lower(),
        "time": epoch(event.ts_event),
    }


def command_outcome(
    order: Any, reason: str | None
) -> tuple[str, dict[str, Any] | None, str | None]:
    """The command's status from its order: sent while working, finished once the order is."""
    status = order.status.name.lower()
    if status == "denied":
        return "denied", None, reason
    if status == "rejected":
        return "failed", None, reason
    if order.is_closed:
        result = {
            "status": status,
            "filled_qty": float(order.filled_qty.as_decimal()),
            "avg_px": float(order.avg_px) if order.filled_qty else None,
        }
        return "done", result, None
    return "sent", None, None


@dataclass(frozen=True)
class TradeLimits:
    """The trade-level breaker: one order's size, price and the book it would hit."""

    order_share: float = 0.25
    price_deviation: float = 0.10
    book_imbalance: float = 0.6
    book_depth: int = 5


def imbalance_against(book: Any, side: str, depth: int) -> float | None:
    """How far the book leans against the trade: -1 all in its favour, 1 all against it."""
    bids = sum(float(level.size()) for level in book.bids(depth))
    asks = sum(float(level.size()) for level in book.asks(depth))
    total = bids + asks
    if total <= 0:
        return None
    against = asks - bids if side == "buy" else bids - asks
    return against / total


def trade_problem(
    payload: dict[str, Any],
    last: float | None,
    equity: float | None,
    book: Any,
    limits: TradeLimits,
) -> str | None:
    """Why the trade-level breaker rejects this one order, or None."""
    for key in ("limit_price", "stop_price"):
        given = payload.get(key)
        if given is not None and last:
            gap = abs(float(given) - last) / last
            if gap > limits.price_deviation:
                return (
                    f"Trade breaker: the {key.replace('_', ' ')} {float(given):g} is {gap:.0%} "
                    f"from the last price {last:g}, more than {limits.price_deviation:.0%}"
                )
    price = payload.get("limit_price") or last
    if price and equity:
        notional = float(payload["qty"]) * float(price)
        if notional > limits.order_share * equity:
            return (
                f"Trade breaker: the order is ${notional:,.0f}, more than "
                f"{limits.order_share:.0%} of the account's ${equity:,.0f}"
            )
    if payload.get("type", "market") == "market" and book is not None:
        against = imbalance_against(book, payload["side"], limits.book_depth)
        if against is not None and against > limits.book_imbalance:
            return (
                f"Trade breaker: the order book leans {against:.0%} against this "
                f"{payload['side']}, more than {limits.book_imbalance:.0%}"
            )
    return None


@dataclass
class Waiting:
    command: dict[str, Any]
    instrument_id: InstrumentId
    since: datetime
    asked: set[str] = field(default_factory=set)


class CommandStrategy(Strategy):
    """Carries out the API's order commands; the only way the platform's orders reach Alpaca."""

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Self:
        return super().__new__(cls)

    def __init__(
        self,
        writer: StateWriter,
        engine: sa.engine.Engine,
        node_name: str,
        poll: timedelta = POLL,
        snapshot_every: timedelta = SNAPSHOT,
        trade_limits: TradeLimits | None = None,
        mode: str = "paper",
    ) -> None:
        super().__init__(StrategyConfig(order_id_tag="CMD"))
        self.mode = mode
        self.poll_every = poll
        self.snapshot_every = snapshot_every
        self.trade_limits = trade_limits or TradeLimits()
        self.writer = writer
        self.engine = engine
        self.node_name = node_name
        self.waiting: list[Waiting] = []
        self.rebalances: list[Rebalance] = []
        self.reasons: dict[str, str] = {}

    def on_start(self) -> None:
        self.sweep()
        self.writer.status(
            "node", {"state": "running", "node": self.node_name, "mode": self.mode}, announce=True
        )
        self.clock.set_timer("command-poll", self.poll_every, callback=lambda _: self.poll())
        self.clock.set_timer(
            "state-snapshot", self.snapshot_every, callback=lambda _: self.snapshot()
        )
        self.snapshot()

    def sweep(self) -> None:
        with self.engine.begin() as connection:
            for command in commands.unfinished(connection):
                client_id = command["client_order_id"]
                found = (
                    client_id is not None and self.cache.order(ClientOrderId(client_id)) is not None
                )
                decision = sweep_decision(command, found)
                if decision == "attach":
                    commands.mark(
                        connection, command["id"], "sent", {"attached": "found at Alpaca"}
                    )
                elif decision == "requeue":
                    commands.requeue(connection, command["id"], "the node restarted")
                elif decision == "stopped":
                    commands.mark(connection, command["id"], "failed", error=STOPPED_BY_RESTART)
                    self.writer.reconciliation(
                        "algorithm_stopped",
                        "warning",
                        STOPPED_BY_RESTART,
                        client_order_id=client_id,
                    )
                else:
                    commands.mark(
                        connection,
                        command["id"],
                        "unknown",
                        error=UNKNOWN_AFTER_RESTART,
                    )
                    self.writer.reconciliation(
                        "unknown_command",
                        "critical",
                        "Command outcome unknown after a restart",
                        client_order_id=client_id,
                    )

    def poll(self) -> None:
        self.retry_waiting()
        self.advance_rebalances()
        with self.engine.begin() as connection:
            taken = commands.claim(connection, self.node_name)
            halted, why = commands.is_halted(connection)
            for command in taken:
                self.handle(connection, command, halted, why)

    def handle(
        self, connection: sa.Connection, command: dict[str, Any], halted: bool, why: str | None
    ) -> None:
        kind, payload = command["kind"], command["payload"]
        try:
            expires = payload.get("expires_at")
            if expires and datetime.fromisoformat(expires) <= self.clock.utc_now():
                commands.mark(connection, command["id"], "denied", error=EXPIRED)
                return
            if kind == "submit":
                self.prepare_submit(connection, command, halted, why)
            elif kind == "cancel":
                self.cancel(ClientOrderId(payload["client_order_id"]))
                commands.mark(connection, command["id"], "done", {"cancel": "sent"})
            elif kind == "modify":
                self.modify(connection, command)
            elif kind == "cancel_all":
                self.cancel_everything(connection, command)
            elif kind == "close_position":
                self.close(connection, command, halted)
            elif kind == "rebalance":
                self.start_rebalance(connection, command, halted, why)
            else:
                commands.mark(
                    connection, command["id"], "failed", error=f"{kind} is not supported yet"
                )
        except (KeyError, ValueError) as exc:
            commands.mark(connection, command["id"], "failed", error=f"Bad command: {exc}")

    def prepare_submit(
        self, connection: sa.Connection, command: dict[str, Any], halted: bool, why: str | None
    ) -> None:
        payload = command["payload"]
        refusal = self.platform_check(connection, payload, halted, why)
        if refusal is not None:
            commands.mark(connection, command["id"], "denied", error=refusal)
            return
        self.waiting.append(
            Waiting(command, instrument_id_of(payload["symbol"]), self.clock.utc_now())
        )
        self.retry_waiting()

    def platform_check(
        self, connection: sa.Connection, payload: dict[str, Any], halted: bool, why: str | None
    ) -> str | None:
        """The platform's own rules, before Nautilus' RiskEngine applies the standard ones."""
        reason = unsupported(payload)
        if reason is not None:
            return reason
        if payload["side"] != "buy":
            return None
        if halted:
            return f"Trading is halted: {why or 'kill switch'}"
        symbol = payload["symbol"]
        if "/" in symbol:
            return None
        found = commands.verdict(connection, symbol)
        if found is None:
            return f"{symbol} has no Shariah verdict on record; the screen fails closed"
        allowed, reason = found
        return None if allowed else f"{symbol} fails the Shariah screen: {reason}"

    def retry_waiting(self) -> None:
        still = []
        for item in self.waiting:
            if not self.try_submit(item):
                still.append(item)
        self.waiting = still

    def try_submit(self, item: Waiting) -> bool:
        payload, instrument_id = item.command["payload"], item.instrument_id
        expired = self.clock.utc_now() - item.since > timedelta(seconds=WAIT_SECONDS)
        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            if expired:
                self.finish(item.command, "failed", error=f"{instrument_id} could not be loaded")
                return True
            if "instrument" not in item.asked:
                item.asked.add("instrument")
                self.request_instrument(instrument_id)
            return False
        market = payload.get("type", "market") == "market"
        priced = self.cache.price(instrument_id, PriceType.LAST) is not None
        if market and not priced and not expired:
            if "price" not in item.asked:
                item.asked.add("price")
                bar_type = BarType.from_str(f"{instrument_id}-1-DAY-LAST-EXTERNAL")
                self.request_bars(bar_type, start=self.clock.utc_now() - PRICE_HISTORY)
            return False
        problem = self.trade_check(payload, instrument_id)
        if problem is not None:
            self.finish(item.command, "denied", error=problem)
            self.trade_tripped(problem)
            return True
        self.submit(item.command, instrument)
        return True

    def equity(self) -> float | None:
        account = self.cache.account_for_venue(ALPACA_VENUE)
        balance = None if account is None else account.balance_total(USD)
        if balance is None:
            return None
        held = 0.0
        for position in self.cache.positions_open():
            last = self.last_price(position.instrument_id)
            held += float(position.signed_qty) * (last or 0.0)
        return float(balance.as_double()) + held

    def last_price(self, instrument_id: InstrumentId) -> float | None:
        for kind in (PriceType.LAST, PriceType.MID):
            found = self.cache.price(instrument_id, kind)
            if found is not None:
                return found.as_double()
        return None

    def trade_check(self, payload: dict[str, Any], instrument_id: InstrumentId) -> str | None:
        return trade_problem(
            payload,
            self.last_price(instrument_id),
            self.equity(),
            self.cache.order_book(instrument_id),
            self.trade_limits,
        )

    def trade_tripped(self, reason: str) -> None:
        """A trade-level trip rejects only that order; it is recorded and the next is re-checked."""
        now = datetime.now(UTC)

        def work(connection: sa.Connection) -> None:
            breakers.record(connection, "trade", "trade", "closed", "closed", "auto", reason, now)

        self.writer.put(work)

    def submit(self, command: dict[str, Any], instrument: Any) -> None:
        payload = command["payload"]
        client_id = ClientOrderId(command["client_order_id"])
        if payload.get("order_class", "simple") == "bracket":
            self.submit_order_list(self.bracket(payload, instrument, client_id))
        else:
            self.submit_order(build_order(self.order_factory, payload, instrument, client_id))
        self.finish(command, "sent", detail={"submitted": str(client_id)})

    def bracket(self, payload: dict[str, Any], instrument: Any, client_id: ClientOrderId) -> Any:
        limit = payload.get("type") == "limit"
        return self.order_factory.bracket(
            instrument.id,
            side_of(payload),
            instrument.make_qty(float(payload["qty"])),
            time_in_force=time_in_force_of(payload, instrument),
            entry_order_type=OrderType.LIMIT if limit else OrderType.MARKET,
            entry_price=instrument.make_price(float(payload["limit_price"])) if limit else None,
            entry_client_order_id=client_id,
            tp_price=instrument.make_price(float(payload["take_profit"])),
            tp_client_order_id=ClientOrderId(f"{client_id}-TP"),
            sl_trigger_price=instrument.make_price(float(payload["stop_loss"])),
            sl_client_order_id=ClientOrderId(f"{client_id}-SL"),
        )

    def start_rebalance(
        self, connection: sa.Connection, command: dict[str, Any], halted: bool, why: str | None
    ) -> None:
        payload = command["payload"]
        legs = legs_of(payload, command["client_order_id"])
        for leg in legs:
            refusal = self.platform_check(connection, leg.payload(leg.qty), halted, why)
            if refusal is not None:
                leg.status = f"denied: {refusal}"
        cancel = [instrument_id_of(symbol) for symbol in payload.get("cancel", [])]
        for instrument_id in cancel:
            self.cancel_all_orders(instrument_id, strategy_only=False)
        item = Rebalance(command, legs, cancel, self.clock.utc_now())
        for instrument_id in item.instrument_ids:
            if self.cache.instrument(instrument_id) is None:
                self.request_instrument(instrument_id)
        self.rebalances.append(item)
        commands.mark(connection, command["id"], "sent", {"rebalance": "started"})
        self.advance_rebalances()

    def advance_rebalances(self) -> None:
        self.rebalances = [item for item in self.rebalances if not self.advance_rebalance(item)]

    def advance_rebalance(self, item: Rebalance) -> bool:
        """Moves one rebalance on a phase when it can; true once it is finished."""
        now = self.clock.utc_now()
        if item.phase == "loading":
            self.ask_prices(item)
            ready = self.rebalance_ready(item)
            if ready or item.waited(now, PRICE_WAIT):
                self.place_legs(item, "sell")
                item.phase, item.phase_started = "selling", now
        sold = item.phase == "selling" and (
            self.legs_closed(item.side("sell")) or item.waited(now, SETTLE_WAIT)
        )
        if sold:
            self.place_legs(item, "buy")
            item.phase, item.phase_started = "buying", now
        if item.phase == "buying":
            closed = self.legs_closed(item.legs)
            if closed or now - item.started >= DEADLINE:
                self.finish_rebalance(item)
                return True
        return False

    def ask_prices(self, item: Rebalance) -> None:
        for instrument_id in item.instrument_ids:
            key = str(instrument_id)
            if key in item.asked or self.cache.instrument(instrument_id) is None:
                continue
            item.asked.add(key)
            bar_type = BarType.from_str(f"{instrument_id}-1-DAY-LAST-EXTERNAL")
            self.request_bars(bar_type, start=self.clock.utc_now() - PRICE_HISTORY)

    def rebalance_ready(self, item: Rebalance) -> bool:
        """Every instrument loaded and priced, and the stale orders it cancels gone."""
        loaded = all(
            self.cache.instrument(i) is not None and self.cache.price(i, PriceType.LAST) is not None
            for i in item.instrument_ids
        )
        cleared = all(self.cache.orders_open_count(instrument_id=i) == 0 for i in item.cancel)
        return loaded and cleared

    def place_legs(self, item: Rebalance, side: str) -> None:
        for leg in item.side(side):
            if leg.status is not None:
                continue
            instrument = self.cache.instrument(leg.instrument_id)
            if instrument is None:
                leg.status = "rejected: instrument is not tradable"
                continue
            quantity = whole_units(instrument, float(leg.qty))
            if quantity <= 0:
                leg.status = "skipped: below the smallest tradable size"
                continue
            problem = self.trade_check(leg.payload(quantity), leg.instrument_id)
            if problem is not None:
                leg.status = f"denied: {problem}"
                self.trade_tripped(problem)
                continue
            order = build_order(
                self.order_factory, leg.payload(quantity), instrument, leg.client_id
            )
            leg.sent = True
            self.submit_order(order)

    def legs_closed(self, legs: list[Leg]) -> bool:
        return all(
            not leg.sent or (order := self.cache.order(leg.client_id)) is None or order.is_closed
            for leg in legs
        )

    def finish_rebalance(self, item: Rebalance) -> None:
        orders = [
            outcome(leg, self.cache.order(leg.client_id), self.reasons.get(str(leg.client_id)))
            for leg in item.legs
        ]
        self.finish(item.command, "done", {"rebalance": "finished"}, {"orders": orders})

    def modify(self, connection: sa.Connection, command: dict[str, Any]) -> None:
        payload = command["payload"]
        client_id = ClientOrderId(payload["client_order_id"])
        order = self.cache.order(client_id)
        if order is None:
            commands.mark(connection, command["id"], "failed", error="No such order on the node")
            return
        instrument = self.cache.instrument(order.instrument_id)
        if instrument is None:
            commands.mark(
                connection, command["id"], "failed", error="The order's instrument is not loaded"
            )
            return
        self.modify_order(
            client_id,
            quantity=None
            if payload.get("qty") is None
            else instrument.make_qty(float(payload["qty"])),
            price=None
            if payload.get("limit_price") is None
            else instrument.make_price(float(payload["limit_price"])),
            trigger_price=None
            if payload.get("stop_price") is None
            else instrument.make_price(float(payload["stop_price"])),
        )
        commands.mark(connection, command["id"], "done", {"modify": "sent"})

    def cancel(self, client_id: ClientOrderId) -> None:
        """An algorithm's primary is stopped through its algorithm; any other order is canceled."""
        order = self.cache.order(client_id)
        if order is not None and is_algorithm_primary(order):
            self.publish_message(STOP_TOPIC, str(client_id))
        else:
            self.cancel_order(client_id)

    def working_algorithms(self) -> list[Any]:
        return [
            order
            for order in self.cache.orders()
            if is_algorithm_primary(order) and not order.is_closed
        ]

    def cancel_everything(self, connection: sa.Connection, command: dict[str, Any]) -> None:
        primaries = self.working_algorithms()
        for primary in primaries:
            self.publish_message(STOP_TOPIC, str(primary.client_order_id))
        instruments = {order.instrument_id for order in self.cache.orders_open()}
        instruments |= {primary.instrument_id for primary in primaries}
        for instrument_id in instruments:
            self.cancel_all_orders(instrument_id, strategy_only=False)
        commands.mark(
            connection,
            command["id"],
            "done",
            result={"instruments": sorted(str(i) for i in instruments)},
        )

    def close(self, connection: sa.Connection, command: dict[str, Any], halted: bool) -> None:
        """A one-leg rebalance: priced first, and it closes a position the node did not open."""
        symbol = command["payload"]["symbol"]
        instrument_id = instrument_id_of(symbol)
        quantity = sum(
            float(p.signed_qty)
            for p in self.cache.positions_open()
            if p.instrument_id == instrument_id
        )
        if quantity == 0:
            commands.mark(connection, command["id"], "failed", error="No open position on the node")
            return
        side = "sell" if quantity > 0 else "buy"
        leg = {"symbol": symbol, "side": side, "qty": abs(quantity)}
        closing = {**command, "payload": {"orders": [leg], "cancel": [symbol]}}
        self.start_rebalance(connection, closing, halted, None)

    def finish(
        self,
        command: dict[str, Any],
        status: str,
        detail: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        command_id: uuid.UUID = command["id"]
        self.writer.put(
            lambda connection: commands.mark(connection, command_id, status, detail, result, error)
        )

    def on_order_event(self, event: Any) -> None:
        client_id = str(event.client_order_id)
        reason = getattr(event, "reason", None)
        if reason:
            self.reasons[client_id] = str(reason)
        order = self.cache.order(event.client_order_id)
        if order is None:
            return
        name = type(event).__name__
        metrics.ORDERS.labels(order.status.name.lower()).inc()
        self.writer.order(order_row(order, name, self.reasons.get(client_id)))
        if name == "OrderFilled":
            self.writer.fill(fill_row(event))
        if getattr(event, "reconciliation", False):
            self.writer.reconciliation(
                "auto_healed",
                "warning",
                f"{name} applied from Alpaca's report",
                client_order_id=client_id,
            )
        if reason and str(reason).startswith("Policy resolution:"):
            self.writer.reconciliation(
                "policy_resolution", "critical", str(reason), client_order_id=client_id
            )
        command_id = commands.command_id_of(client_id)
        if command_id is None:
            return
        if any(item.command["id"] == command_id for item in self.rebalances):
            self.advance_rebalances()
            return
        primary = self.cache.order(order.exec_spawn_id) if order.exec_spawn_id else None
        if primary is None and client_id != f"CMD-{command_id.hex}":
            return
        if primary is not None:
            primary_reason = self.reasons.get(str(primary.client_order_id))
            status, result, error = family_outcome(self.cache, primary, primary_reason)
        else:
            status, result, error = command_outcome(order, self.reasons.get(client_id))
        self.writer.put(
            lambda connection: commands.mark(
                connection, command_id, status, {"event": name}, result, error
            )
        )

    def snapshot(self) -> None:
        rows = [
            {
                "instrument_id": str(position.instrument_id),
                "side": position.side.name.lower(),
                "quantity": Decimal(str(position.signed_qty)),
                "avg_px_open": Decimal(str(position.avg_px_open)),
                "realized_pnl": number(getattr(position, "realized_pnl", None)),
                "updated": datetime.now(UTC),
            }
            for position in self.cache.positions_open()
        ]
        self.writer.positions(rows)
        metrics.HEARTBEAT.set_to_current_time()
        self.writer.status(
            "heartbeat",
            {
                "time": datetime.now(UTC).isoformat(),
                "node": self.node_name,
                "open_orders": len(self.cache.orders_open()),
                "positions": len(rows),
                "waiting": len(self.waiting),
            },
        )
