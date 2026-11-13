from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

Method = Literal["fifo", "average", "specific"]
Side = Literal["long", "short"]
ZERO = Decimal(0)
WASH_WINDOW = timedelta(days=30)


@dataclass(frozen=True)
class Execution:
    """One order's fills as a single buy, sell or write-off at the volume-weighted price."""

    order: str
    symbol: str
    kind: Literal["buy", "sell", "drop"]
    time: datetime
    qty: Decimal
    price: Decimal
    paid: Decimal | None = None

    @property
    def cost(self) -> Decimal:
        return self.paid if self.paid is not None else self.qty * self.price


@dataclass
class Lot:
    """What one order left open: the quantity still held (or owed, for a short) and its cost."""

    id: str
    symbol: str
    opened: datetime
    qty: Decimal
    cost: Decimal
    original: Decimal
    side: Side = "long"
    wash: Decimal = ZERO
    replaced: Decimal = ZERO

    @property
    def unit_cost(self) -> Decimal:
        return self.cost / self.qty

    @property
    def signed_qty(self) -> Decimal:
        return self.qty if self.side == "long" else -self.qty


@dataclass(frozen=True)
class Close:
    """Part of a lot closed by one order. `gain` is `proceeds` less `basis` for either side:
    a long is bought for the basis and sold for the proceeds, a short sold for the proceeds
    and bought back for the basis."""

    lot: str | None
    symbol: str
    side: Side
    opened: datetime
    closed: datetime
    order: str
    qty: Decimal
    basis: Decimal
    proceeds: Decimal
    disallowed: Decimal = ZERO

    @property
    def gain(self) -> Decimal:
        return self.proceeds - self.basis

    @property
    def reported(self) -> Decimal:
        """The gain to report: a loss less what the wash-sale rule disallows."""
        return self.gain + self.disallowed


@dataclass
class Book:
    lots: list[Lot]
    closes: list[Close]


@dataclass
class Washes:
    """Loss sales still waiting for a replacement buy in the 30 days after them."""

    exempt: Collection[str]
    waiting: dict[str, list[list[int | Decimal]]] = field(default_factory=dict)

    def applies(self, symbol: str) -> bool:
        return symbol not in self.exempt


Taken = list[tuple[Lot, Decimal, Decimal]]


def replay(
    executions: Iterable[Execution],
    method: Method,
    picks: Mapping[str, Sequence[tuple[str, Decimal]]] | None = None,
    wash_exempt: Collection[str] | None = None,
) -> Book:
    """Apply the executions in time order; an order closes lots by `method`, then opens the rest.

    With `wash_exempt` given (the symbols the rule does not cover, such as coins), the
    wash-sale rule applies to everything else (`wash_back`, `wash_forward`).
    """
    chosen = picks or {}
    washes = None if wash_exempt is None else Washes(wash_exempt)
    pools: dict[str, list[Lot]] = defaultdict(list)
    closes: list[Close] = []
    for item in sorted(executions, key=lambda execution: (execution.time, execution.order)):
        pool = pools[item.symbol]
        closing: Side = "short" if item.kind == "buy" else "long"
        left = item.qty
        if (pool and pool[0].side == closing) or item.kind == "drop":
            wanted = chosen.get(item.order, ()) if method == "specific" else ()
            taken = close_lots(pool, item.qty, method, wanted)
            for lot, qty, basis in taken:
                closes.append(settle(item, lot, qty, basis))
                if washes is not None and item.kind == "sell" and washes.applies(item.symbol):
                    wash_back(closes, len(closes) - 1, pool, washes)
            left -= sum((qty for _, qty, _ in taken), ZERO)
        if left > ZERO and item.kind != "drop":
            side: Side = "long" if item.kind == "buy" else "short"
            cost = item.cost if left == item.qty else item.cost * left / item.qty
            opened = open_lot(pool, item, left, cost, side, method)
            if washes is not None and side == "long" and washes.applies(item.symbol):
                wash_forward(closes, opened, left, item.time, washes)
    lots = [lot for pool in pools.values() for lot in pool]
    return Book(sorted(lots, key=lambda lot: (lot.symbol, lot.opened, lot.id)), closes)


def settle(item: Execution, lot: Lot, qty: Decimal, opened_for: Decimal) -> Close:
    """The close of `qty` of `lot` by `item`, whose share of the lot's cost was `opened_for`."""
    at_close = opened_for if item.kind == "drop" else qty * item.price
    basis, proceeds = (opened_for, at_close) if lot.side == "long" else (at_close, opened_for)
    return Close(
        lot.id, lot.symbol, lot.side, lot.opened, item.time, item.order, qty, basis, proceeds
    )


def open_lot(
    pool: list[Lot], item: Execution, qty: Decimal, cost: Decimal, side: Side, method: Method
) -> Lot:
    if method == "average" and pool:
        pool[0].qty += qty
        pool[0].cost += cost
        pool[0].original += qty
        return pool[0]
    lot_id = f"average:{item.symbol}" if method == "average" else item.order
    lot = Lot(lot_id, item.symbol, item.time, qty, cost, qty, side)
    pool.append(lot)
    return lot


def disallow(closes: list[Close], index: int, qty: Decimal, lot: Lot) -> Decimal:
    """Move `qty` shares' worth of a loss sale's loss into `lot`'s basis; returns the amount."""
    close = closes[index]
    amount = -close.gain * qty / close.qty
    closes[index] = replace(close, disallowed=close.disallowed + amount)
    lot.cost += amount
    lot.wash += amount
    lot.replaced += qty
    return amount


def wash_back(closes: list[Close], index: int, pool: list[Lot], washes: Washes) -> None:
    """A loss sale against shares bought in the 30 days before it and still held.

    IRS Publication 550: a loss on a sale is disallowed when substantially identical shares are
    bought within 30 days before or after it; the disallowed loss is added to their basis.
    Each replacement share replaces one sold share once. What finds no replacement now waits
    for buys in the next 30 days.
    """
    close = closes[index]
    if close.gain >= ZERO or close.side != "long":
        return
    left = close.qty
    for lot in sorted(pool, key=lambda found: (found.opened, found.id)):
        recent = close.closed - WASH_WINDOW <= lot.opened < close.closed
        free = lot.qty - lot.replaced
        if left <= ZERO or not recent or lot.id == close.lot or free <= ZERO:
            continue
        used = min(left, free)
        disallow(closes, index, used, lot)
        left -= used
    if left > ZERO:
        washes.waiting.setdefault(close.symbol, []).append([index, left])


def wash_forward(
    closes: list[Close], lot: Lot, qty: Decimal, bought: datetime, washes: Washes
) -> None:
    """A buy that replaces shares sold at a loss in the 30 days before it."""
    pending = washes.waiting.get(lot.symbol, [])
    left = qty
    for entry in pending:
        index, waiting = int(entry[0]), Decimal(entry[1])
        if left <= ZERO or waiting <= ZERO or bought - closes[index].closed > WASH_WINDOW:
            continue
        used = min(left, waiting)
        disallow(closes, index, used, lot)
        entry[1] = waiting - used
        left -= used
    washes.waiting[lot.symbol] = [
        entry
        for entry in pending
        if Decimal(entry[1]) > ZERO and bought - closes[int(entry[0])].closed <= WASH_WINDOW
    ]


def close_lots(
    pool: list[Lot], qty: Decimal, method: Method, picks: Sequence[tuple[str, Decimal]]
) -> Taken:
    taken: Taken = []
    left = qty
    by_id = {lot.id: lot for lot in pool}
    for lot_id, size in dict(picks).items():
        if lot_id in by_id:
            left -= take(by_id[lot_id], min(size, left), taken)
    for lot in sorted(pool, key=lambda lot: (lot.opened, lot.id)):
        if left <= ZERO:
            break
        left -= take(lot, min(lot.qty, left), taken)
    pool[:] = [lot for lot in pool if lot.qty > ZERO]
    return taken


def take(lot: Lot, qty: Decimal, taken: Taken) -> Decimal:
    qty = min(qty, lot.qty)
    if qty <= ZERO:
        return ZERO
    share = lot.cost if qty == lot.qty else lot.cost * qty / lot.qty
    lot.qty -= qty
    lot.cost -= share
    taken.append((lot, qty, share))
    return qty
