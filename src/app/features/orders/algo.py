import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from uuid import uuid4
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel
from requests import RequestException

from app.core.book import BookGuard
from app.core.errors import broker_message
from app.features.orders import quotes, risk, schedule, service, shortfall
from app.features.orders.schemas import (
    Algo,
    AlgoPlan,
    AlgoRequest,
    Child,
    Interval,
    OrderRequest,
    PlannedSlice,
)
from app.quant import history, volume
from app.quant.volume import Profile

logger = logging.getLogger(__name__)

PATH = Path("data/algos.json")
JOB_ID = "algo-slices"
EVERY = timedelta(seconds=3)
KEPT_FINISHED = 50
LEFT = 1e-9
MIN_COIN_COST = 10.0
NEW_YORK = ZoneInfo("America/New_York")


class AlgoNotFoundError(LookupError):
    """Raised when an algorithm order id is unknown."""


class State(BaseModel):
    algos: list[Algo] = []


class Session(BaseModel):
    is_open: bool
    next_close: datetime


class Holding(BaseModel):
    qty_available: float


class AlgoDesk:
    """Runs a parent order as immediate-or-cancel child orders on a schedule, and keeps count."""

    def __init__(
        self,
        trading: TradingClient,
        stocks: StockHistoricalDataClient,
        crypto: CryptoHistoricalDataClient,
        feed: DataFeed,
        path: Path = PATH,
        guard: BookGuard | None = None,
    ) -> None:
        self.guard = guard
        self.trading = trading
        self.stocks = stocks
        self.crypto = crypto
        self.feed = feed
        self.path = path
        self.lock = Lock()
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.state = self._load()

    def _load(self) -> State:
        try:
            return State.model_validate_json(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return State()

    def _save(self) -> None:
        working = [item for item in self.state.algos if item.status == "working"]
        finished = [item for item in self.state.algos if item.status != "working"]
        kept = [*working, *finished[-KEPT_FINISHED:]]
        self.state.algos = sorted(kept, key=lambda item: item.created)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.state.model_dump_json(indent=2))

    def algos(self) -> list[Algo]:
        with self.lock:
            return [item.model_copy(deep=True) for item in reversed(self.state.algos)]

    def plan(self, request: AlgoRequest, now: datetime | None = None) -> AlgoPlan:
        """The schedule an order would run on; nothing is placed."""
        start = now or datetime.now(UTC)
        end = start + timedelta(minutes=request.duration)
        self._check_session(request, start, end)
        self._check_holdings(request)
        book = quotes.top_of_book(self.stocks, self.crypto, request.symbol, self.feed)
        arrival = (book.mid if book else None) or risk.last_price(
            self.stocks, self.crypto, request.symbol, self.feed
        )
        intervals, notes = self._intervals(request, start, end)
        guard = self._book_for(request.symbol, request.limit_price)
        cap = guard.max_size(request.symbol, request.side) if guard else None
        if cap is not None:
            notes["book_cap"] = cap
        sizes = schedule.preview(
            request.qty, [item.weight for item in intervals], whole="/" not in request.symbol
        )
        self._check_cost(request, sizes, arrival)
        slices = [
            PlannedSlice(due=item.start, qty=size)
            for item, size in zip(intervals, sizes, strict=True)
            if size > 0
        ]
        return AlgoPlan(
            algo=request.algo,
            start=start,
            end=end,
            arrival_price=arrival,
            intervals=intervals,
            slices=slices,
            notes=notes,
        )

    def place(self, request: AlgoRequest, now: datetime | None = None) -> Algo:
        """Start a parent order; its first slice goes out on the next tick."""
        plan = self.plan(request, now)
        item = Algo(
            id=uuid4().hex[:12],
            created=plan.start,
            algo=request.algo,
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            duration=request.duration,
            limit_price=request.limit_price,
            tilt=request.tilt,
            risk_aversion=request.risk_aversion,
            start=plan.start,
            end=plan.end,
            arrival_price=plan.arrival_price,
            intervals=plan.intervals,
            notes=plan.notes,
        )
        with self.lock:
            self.state.algos = [*self.state.algos, item]
            self._save()
            return item.model_copy(deep=True)

    def cancel(self, algo_id: str) -> Algo:
        """Stop a parent order and cancel the child on show; what already filled stays filled."""
        with self.lock:
            item = self._find(algo_id)
            if item.status == "working":
                live = self._live(item)
                if live is not None:
                    self._cancel(live)
                self._refresh(item)
                item.status, item.detail = "canceled", "Canceled"
                self._save()
            return item.model_copy(deep=True)

    def tick(self, now: datetime | None = None) -> None:
        moment = now or datetime.now(UTC)
        with self.lock:
            changed = False
            for item in self.state.algos:
                if item.status != "working" and self._live(item) is None:
                    continue
                try:
                    changed = self._advance(item, moment) or changed
                except APIError as exc:
                    if exc.status_code in service.TRANSIENT:
                        logger.warning("Algorithm order %s will retry: %s", item.id, exc)
                        continue
                    item.status, item.detail = "failed", broker_message(exc)
                    changed = True
                except RequestException as exc:
                    logger.warning("Algorithm order %s will retry: %s", item.id, exc)
            if changed:
                self._save()

    def _find(self, algo_id: str) -> Algo:
        for item in self.state.algos:
            if item.id == algo_id:
                return item
        raise AlgoNotFoundError(algo_id)

    def _intervals(
        self, request: AlgoRequest, start: datetime, end: datetime
    ) -> tuple[list[Interval], dict[str, float | str]]:
        if request.algo == "twap":
            return schedule.uniform(start, request.duration, request.slices or 0), {}
        try:
            curve = volume.profile(self.stocks, self.crypto, request.symbol, start)
        except history.NotEnoughHistoryError as exc:
            raise service.AlgoRefusedError(str(exc)) from exc
        if request.algo == "vwap":
            return schedule.volume_weighted(start, end, curve), {"sessions": curve.sessions}
        return self._shortfall(request, start, curve)

    def _shortfall(
        self, request: AlgoRequest, start: datetime, curve: Profile
    ) -> tuple[list[Interval], dict[str, float | str]]:
        quoted = volume.spread(self.stocks, request.symbol, start)
        if quoted is None:
            raise service.AlgoRefusedError(f"No recent quotes of {request.symbol} for its spread")
        market = shortfall.market_of(curve, quoted)
        if market.sigma <= 0 or market.volume <= 0:
            raise service.AlgoRefusedError(
                f"{request.symbol} traded too little in its last sessions to size its volatility"
            )
        count = request.slices or 0
        days = request.duration / curve.session.minutes
        risk = request.risk_aversion
        trades = shortfall.weights(request.qty, days, count, risk, market)
        notes: dict[str, float | str] = {
            "sessions": curve.sessions,
            "volatility": market.sigma,
            "volume": market.volume,
            "spread": market.spread,
            "urgency": shortfall.decay(risk, days, count, market) * days,
        }
        return schedule.uniform(start, request.duration, count, trades), notes

    def _check_session(self, request: AlgoRequest, start: datetime, end: datetime) -> None:
        if "/" in request.symbol:
            return
        session = Session.model_validate(self.trading.get_clock())
        if not session.is_open:
            raise service.AlgoRefusedError(
                "The market is closed; an algorithm runs in market hours"
            )
        if end > session.next_close:
            close = session.next_close.astimezone(NEW_YORK)
            minutes = int((session.next_close - start).total_seconds() // 60)
            raise service.AlgoRefusedError(
                f"The market closes at {close:%H:%M} ET; use at most {minutes} minutes"
            )

    def _check_holdings(self, request: AlgoRequest) -> None:
        if request.side != "sell":
            return
        try:
            position = self.trading.get_open_position(request.symbol.replace("/", ""))
        except APIError as exc:
            if exc.status_code != 404:
                raise
            raise service.AlgoRefusedError(f"You hold no {request.symbol} to sell") from exc
        held = Holding.model_validate(position).qty_available
        if request.qty > held + LEFT:
            raise service.AlgoRefusedError(
                f"You can sell {held:g} {request.symbol}; the order is for {request.qty:g}"
            )

    @staticmethod
    def _check_cost(request: AlgoRequest, sizes: list[float], arrival: float | None) -> None:
        price = request.limit_price or arrival
        smallest = min(sizes) if request.algo == "twap" else request.qty
        if "/" not in request.symbol or not price or smallest * price >= MIN_COIN_COST:
            return
        most = int(request.qty * price // MIN_COIN_COST)
        if most < 2 or request.algo != "twap":
            raise service.AlgoRefusedError(
                f"The order costs ${request.qty * price:,.2f}, too little to slice: "
                f"a slice must cost at least ${MIN_COIN_COST:g}"
            )
        raise service.AlgoRefusedError(
            f"A slice must cost at least ${MIN_COIN_COST:g}; use at most {most} slices"
        )

    def _advance(self, item: Algo, now: datetime) -> bool:
        changed = self._refresh(item)
        if item.status != "working":
            return changed
        live = self._live(item)
        if live is not None:
            return self._expire(item, live, now) or changed
        left = round(item.qty - item.filled_qty, schedule.PRECISION)
        if left <= LEFT:
            item.status = "done"
            return True
        if now >= item.end or item.next_interval >= len(item.intervals):
            return self._finish(item, left)
        due = self._due(item, now)
        if due is None or due < item.next_interval:
            return changed
        size = self._size(item, left, due)
        if size > 0:
            self._place(item, size, due)
        item.next_interval = due + 1
        if size <= 0 and due >= len(item.intervals) - 1:
            return self._finish(item, left)
        return True

    def _refresh(self, item: Algo) -> bool:
        changed = False
        for child in item.children:
            if child.status not in service.LIVE:
                continue
            order = service.get_order(self.trading, child.order)
            before = child.model_copy()
            child.status = order.status
            child.filled_qty = order.filled_qty
            child.average_price = order.filled_avg_price
            changed = changed or child != before
        return changed

    @staticmethod
    def _live(item: Algo) -> Child | None:
        return next(
            (child for child in reversed(item.children) if child.status in service.LIVE), None
        )

    @staticmethod
    def _due(item: Algo, now: datetime) -> int | None:
        started = [index for index, span in enumerate(item.intervals) if span.start <= now]
        return started[-1] if started else None

    def _expire(self, item: Algo, live: Child, now: datetime) -> bool:
        """A child still open when its interval is over is canceled, once."""
        due = self._due(item, now)
        overdue = now >= item.end or (due is not None and due > live.interval)
        if not overdue or live.cancel_sent:
            return False
        self._cancel(live)
        return True

    def _cancel(self, child: Child) -> None:
        child.cancel_sent = True
        try:
            self.trading.cancel_order_by_id(child.order)
        except APIError as exc:
            logger.info("Algorithm child %s could not be canceled: %s", child.order, exc)

    @staticmethod
    def _finish(item: Algo, left: float) -> bool:
        item.status = "ended"
        item.detail = f"{left:g} of {item.qty:g} not filled"
        return True

    def _size(self, item: Algo, left: float, index: int) -> float:
        lean = 0.0
        if item.tilt:
            book = quotes.top_of_book(self.stocks, self.crypto, item.symbol, self.feed)
            lean = item.tilt * book.lean(item.side) if book else 0.0
        size = schedule.next_slice(
            left,
            [span.weight for span in item.intervals],
            index,
            whole="/" not in item.symbol,
            lean=lean,
        )
        guard = self._book_for(item.symbol, item.limit_price)
        if guard is not None:
            size = self._fit(guard, item, size)
        price = item.limit_price or item.arrival_price
        if "/" in item.symbol and price and size * price < MIN_COIN_COST:
            return 0.0
        return size

    def _book_for(self, symbol: str, limit_price: float | None) -> BookGuard | None:
        """The book that judges a market slice of a coin; none for a stock or a priced slice."""
        return self.guard if "/" in symbol and not limit_price else None

    @staticmethod
    def _fit(guard: BookGuard, item: Algo, size: float) -> float:
        """A market slice cut to what the order book takes in budget, or none while it says wait."""
        cap = guard.max_size(item.symbol, item.side)
        size = round(size if cap is None else min(size, cap), schedule.PRECISION)
        if size <= 0:
            return 0.0
        reason = guard.refusal(item.symbol, item.side, size)
        if reason is not None:
            item.detail = f"A slice was held back: {reason}"
            return 0.0
        return size

    def _place(self, item: Algo, size: float, index: int) -> None:
        request = OrderRequest.model_construct(
            symbol=item.symbol,
            side=item.side,
            type="limit" if item.limit_price else "market",
            qty=size,
            limit_price=item.limit_price,
            time_in_force="ioc",
        )
        order = service.submit_keyed(self.trading, request, f"algo-{item.id}-{index + 1}")
        item.children.append(
            Child(
                order=order.id,
                interval=index,
                qty=size,
                status=order.status,
                filled_qty=order.filled_qty,
                average_price=order.filled_avg_price,
            )
        )

    async def _tick(self) -> None:
        await asyncio.to_thread(self.tick)

    def start(self) -> None:
        self.scheduler.add_job(
            self._tick,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
