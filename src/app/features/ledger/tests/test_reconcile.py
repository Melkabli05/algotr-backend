from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from app.features.ledger.lots import Book, Execution, replay
from app.features.ledger.reconcile import Held, compare
from app.features.ledger.tests.support import row

WHEN = datetime(2026, 11, 2, tzinfo=UTC)
Step = tuple[str, Literal["buy", "sell"], str, str]


def book(*steps: Step) -> Book:
    executions = [
        Execution(f"o{n}", symbol, kind, WHEN.replace(hour=n), Decimal(qty), Decimal(price))
        for n, (symbol, kind, qty, price) in enumerate(steps, start=1)
    ]
    return replay(executions, "average")


def held(symbol: str, qty: str, average: str | None = None) -> Held:
    return Held(symbol, Decimal(qty), Decimal(average) if average else None)


def test_a_ledger_that_matches_the_broker_has_nothing_to_report() -> None:
    mine = book(
        ("SPUS", "buy", "10", "50"), ("SPUS", "buy", "10", "60"), ("SPUS", "sell", "5", "70")
    )
    assert compare(mine, [held("SPUS", "15", "55")], {}, {}) == []


def test_a_different_quantity_is_a_break_that_says_who_holds_more() -> None:
    mine = book(("SPUS", "buy", "10", "50"))
    [more] = compare(mine, [held("SPUS", "8", "50")], {}, {})
    assert row(more.kind, more.ledger_qty, more.broker_qty) == ("quantity", 10, 8)
    assert more.note == "The ledger holds 2 more than the other"
    [less] = compare(mine, [held("SPUS", "12", "50")], {}, {})
    assert less.note == "Alpaca holds 2 more than the other"


def test_a_position_only_one_side_knows_about_is_a_break() -> None:
    mine = book(("SPUS", "buy", "10", "50"))
    found = compare(mine, [held("AAPL", "3", "340")], {}, {})
    assert [row(d.symbol, d.kind, d.ledger_qty, d.broker_qty) for d in found] == [
        ("AAPL", "quantity", 0, 3),
        ("SPUS", "quantity", 10, 0),
    ]


def test_a_small_shortfall_on_a_coin_is_a_pending_fee_until_it_exceeds_the_allowance() -> None:
    mine = book(("BTCUSD", "buy", "0.001", "80000"))
    owed = {"BTCUSD": Decimal("0.000005")}
    [pending] = compare(mine, [held("BTCUSD", "0.0009975", "80000")], owed, {})
    assert (pending.kind, pending.note) == ("pending", "Coin fees post the next day")
    [broken] = compare(mine, [held("BTCUSD", "0.0009", "80000")], owed, {})
    assert broken.kind == "quantity"
    [more] = compare(mine, [held("BTCUSD", "0.0011", "80000")], owed, {})
    assert more.kind == "quantity"


def test_an_average_cost_that_drifts_is_a_note_not_a_break() -> None:
    mine = book(("SPUS", "buy", "10", "100"))
    [note] = compare(mine, [held("SPUS", "10", "101")], {}, {})
    assert note.kind == "cost"
    assert compare(mine, [held("SPUS", "10", "100.005")], {}, {}) == []


def test_the_cost_of_a_dust_position_is_not_compared() -> None:
    mine = book(("BTCUSD", "buy", "0.000000009", "78098"))
    assert compare(mine, [held("BTCUSD", "0.000000009", "78078")], {}, {}) == []


def test_a_corporate_action_on_record_is_named_as_the_likely_cause() -> None:
    mine = book(("ACME", "buy", "10", "100"))
    [split] = compare(mine, [held("ACME", "40", "25")], {}, {"ACME": "SPLIT"})
    assert "SPLIT" in split.note and split.kind == "quantity"


def test_a_short_is_compared_with_the_brokers_negative_quantity() -> None:
    mine = book(("ACME", "sell", "3", "100"))
    assert compare(mine, [held("ACME", "-3", "100")], {}, {}) == []
    [broken] = compare(mine, [held("ACME", "0")], {}, {})
    assert row(broken.kind, broken.ledger_qty) == ("quantity", -3)
