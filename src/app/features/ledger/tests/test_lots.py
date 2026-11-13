from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import numpy as np
import pytest

from app.features.ledger.lots import ZERO, Execution, Method, replay
from app.features.ledger.tests.support import row

METHODS: list[Method] = ["fifo", "average", "specific"]


def at(day: int) -> datetime:
    return datetime(2026, 11, day, 14, 30, tzinfo=UTC)


def buy(order: str, day: int, qty: str, price: str, paid: str | None = None) -> Execution:
    return Execution(
        order, "ACME", "buy", at(day), Decimal(qty), Decimal(price), Decimal(paid) if paid else None
    )


def sell(
    order: str, day: int, qty: str, price: str, kind: Literal["sell", "drop"] = "sell"
) -> Execution:
    return Execution(order, "ACME", kind, at(day), Decimal(qty), Decimal(price))


LADDER = [buy("b1", 2, "10", "10"), buy("b2", 3, "10", "20")]


def test_fifo_closes_the_oldest_lot_first() -> None:
    book = replay([*LADDER, sell("s1", 4, "15", "30")], "fifo")
    first, second = book.closes
    assert row(first.lot, first.qty, first.basis, first.proceeds) == ("b1", 10, 100, 300)
    assert row(second.lot, second.qty, second.basis, second.proceeds) == ("b2", 5, 100, 150)
    assert sum(close.gain for close in book.closes) == 250
    [left] = book.lots
    assert row(left.id, left.qty, left.cost, left.unit_cost) == ("b2", 5, 100, 20)


def test_average_cost_closes_at_the_pooled_price() -> None:
    book = replay([*LADDER, sell("s1", 4, "15", "30")], "average")
    [close] = book.closes
    assert row(close.qty, close.basis, close.proceeds, close.gain) == (15, 225, 450, 225)
    [left] = book.lots
    assert row(left.id, left.qty, left.cost, left.unit_cost) == ("average:ACME", 5, 75, 15)
    assert left.opened == at(2)


def test_specific_lot_closes_the_picked_lot_then_the_oldest() -> None:
    picks = {"s1": [("b2", Decimal(10))]}
    book = replay([*LADDER, sell("s1", 4, "15", "30")], "specific", picks)
    by_lot = {close.lot: close for close in book.closes}
    assert row(by_lot["b2"].qty, by_lot["b2"].basis) == (10, 200)
    assert row(by_lot["b1"].qty, by_lot["b1"].basis) == (5, 50)
    assert sum(close.gain for close in book.closes) == 200
    [left] = book.lots
    assert row(left.id, left.qty, left.cost) == ("b1", 5, 50)


def test_specific_lot_without_a_pick_closes_fifo() -> None:
    fifo = replay([*LADDER, sell("s1", 4, "15", "30")], "fifo")
    specific = replay([*LADDER, sell("s1", 4, "15", "30")], "specific")
    assert fifo.closes == specific.closes and fifo.lots == specific.lots


def test_a_pick_is_capped_by_the_lot_and_the_quantity_sold_and_unknown_lots_are_skipped() -> None:
    picks = {"s1": [("nope", Decimal(3)), ("b2", Decimal(50))]}
    book = replay([*LADDER, sell("s1", 4, "4", "30")], "specific", picks)
    [close] = book.closes
    assert row(close.lot, close.qty, close.basis) == ("b2", 4, 80)
    assert [lot.qty for lot in book.lots] == [10, 6]


def test_every_method_realizes_the_same_total_once_the_position_is_flat() -> None:
    flat = [*LADDER, sell("s1", 4, "12", "30"), sell("s2", 5, "8", "25")]
    totals = {method: sum(c.gain for c in replay(flat, method).closes) for method in METHODS}
    assert set(totals.values()) == {Decimal(12 * 30 + 8 * 25 - 300)}


def test_selling_more_than_is_held_opens_a_short_for_the_rest() -> None:
    book = replay([buy("b1", 2, "3", "10"), sell("s1", 3, "5", "12")], "fifo")
    [close] = book.closes
    assert row(close.side, close.qty, close.basis, close.proceeds) == ("long", 3, 30, 36)
    [short] = book.lots
    assert row(short.side, short.id, short.qty, short.cost, short.signed_qty) == (
        "short",
        "s1",
        2,
        24,
        -2,
    )


def test_a_buy_covers_a_short_before_it_opens_anything() -> None:
    steps = [sell("s1", 2, "4", "100"), buy("b1", 3, "4", "90")]
    for method in METHODS:
        book = replay(steps, method)
        [close] = book.closes
        assert row(close.side, close.qty, close.basis, close.proceeds, close.gain) == (
            "short",
            4,
            360,
            400,
            40,
        )
        assert book.lots == []


def test_a_buy_larger_than_the_short_covers_it_and_opens_a_long() -> None:
    book = replay([sell("s1", 2, "3", "100"), buy("b1", 3, "5", "90")], "fifo")
    [close] = book.closes
    assert row(close.side, close.qty, close.gain) == ("short", 3, 30)
    [lot] = book.lots
    assert row(lot.side, lot.qty, lot.cost, lot.opened) == ("long", 2, 180, at(3))


def test_shorts_are_covered_oldest_first_and_pooled_for_average_cost() -> None:
    steps = [sell("s1", 2, "2", "100"), sell("s2", 3, "2", "110"), buy("b1", 4, "3", "90")]
    fifo = replay(steps, "fifo")
    assert [row(c.lot, c.qty, c.gain) for c in fifo.closes] == [("s1", 2, 20), ("s2", 1, 20)]
    assert [row(lot.id, lot.qty, lot.cost) for lot in fifo.lots] == [("s2", 1, 110)]
    average = replay(steps, "average")
    [pooled] = average.lots
    assert row(pooled.side, pooled.qty, pooled.cost) == ("short", 1, 105)
    assert sum(c.gain for c in average.closes) == 3 * (105 - 90)


def test_a_write_off_clears_a_short_too() -> None:
    book = replay([sell("s1", 2, "4", "100"), sell("w1", 3, "4", "0", kind="drop")], "fifo")
    [close] = book.closes
    assert row(close.side, close.gain, book.lots) == ("short", 0, [])


def test_a_coin_fee_leaves_fewer_coins_for_the_same_money() -> None:
    fee_in_kind = buy("b1", 2, "0.9975", "100", paid="100")
    [lot] = replay([fee_in_kind], "fifo").lots
    assert row(lot.qty, lot.cost) == (Decimal("0.9975"), 100)
    assert lot.unit_cost == Decimal(100) / Decimal("0.9975")


def test_a_write_off_removes_quantity_without_a_gain() -> None:
    book = replay([*LADDER, sell("w1", 4, "4", "0", kind="drop")], "fifo")
    [close] = book.closes
    assert row(close.qty, close.basis, close.proceeds, close.gain) == (4, 40, 40, 0)
    assert sum(lot.qty for lot in book.lots) == 16


def test_the_average_pool_starts_again_after_the_position_goes_flat() -> None:
    book = replay(
        [buy("b1", 2, "10", "10"), sell("s1", 3, "10", "12"), buy("b2", 4, "5", "20")], "average"
    )
    [lot] = book.lots
    assert row(lot.qty, lot.cost, lot.opened) == (5, 100, at(4))


def test_partial_sells_keep_the_remaining_cost_exact() -> None:
    steps = [buy("b1", 2, "3", "10.1"), sell("s1", 3, "1", "11"), sell("s2", 4, "2", "11")]
    for method in METHODS:
        book = replay(steps, method)
        assert book.lots == []
        assert sum(close.basis for close in book.closes) == Decimal("30.3")


def test_symbols_are_kept_apart() -> None:
    other = Execution("c1", "OTHER", "buy", at(2), Decimal(5), Decimal(7))
    book = replay([*LADDER, other, sell("s1", 4, "20", "30")], "fifo")
    assert [row(lot.symbol, lot.qty) for lot in book.lots] == [("OTHER", 5)]


def test_executions_are_applied_in_time_order_whatever_the_input_order() -> None:
    shuffled = [sell("s1", 4, "15", "30"), LADDER[1], LADDER[0]]
    assert (
        replay(shuffled, "fifo").closes
        == replay([*LADDER, sell("s1", 4, "15", "30")], "fifo").closes
    )


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("method", METHODS)
def test_cost_and_quantity_are_conserved_whatever_the_sequence(method: Method, seed: int) -> None:
    rng = np.random.default_rng(seed)
    steps, held, bought, paid, order = [], ZERO, ZERO, ZERO, 0
    for day in range(1, 29):
        order += 1
        if rng.random() < 0.55 or held == 0:
            qty, price = (
                Decimal(int(rng.integers(1, 10))) / 4,
                Decimal(int(rng.integers(50, 150))) / 8,
            )
            steps.append(buy(f"b{order}", day, str(qty), str(price)))
            held, bought, paid = held + qty, bought + qty, paid + qty * price
        else:
            qty = min(held, Decimal(int(rng.integers(1, 10))) / 4)
            steps.append(
                sell(f"s{order}", day, str(qty), str(Decimal(int(rng.integers(50, 150))) / 8))
            )
            held -= qty
    picks = {f"s{n}": [(f"b{n - 1}", Decimal(1))] for n in range(2, order + 1)}
    book = replay(steps, method, picks)
    tolerance = Decimal("1e-20")
    assert {lot.side for lot in book.lots} <= {"long"}
    assert sum((lot.qty for lot in book.lots), ZERO) == held
    assert abs(sum((c.qty for c in book.closes), ZERO) + held - bought) <= tolerance
    kept = sum((lot.cost for lot in book.lots), ZERO) + sum((c.basis for c in book.closes), ZERO)
    assert abs(kept - paid) <= tolerance
