import logging
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app.features.ledger.store import Store
from app.features.ledger.tests.support import coin_fee, fill, row


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "ledger.sqlite")


def test_activities_are_kept_once_whatever_the_number_of_times_they_are_added(store: Store) -> None:
    rows = [
        fill("2026-10-28T11:00:00", "o1", "buy", "1", "100"),
        coin_fee("2026-10-28T11:00:04", "-1", "1"),
    ]
    assert store.add(rows) == 2
    assert store.add(rows) == 0
    assert store.add([*rows, fill("2026-10-28T12:00:00", "o2", "sell", "1", "101")]) == 1
    assert store.count() == 3
    assert [a.activity_type for a in store.activities()] == ["CFEE", "FILL", "FILL"]


def test_the_last_day_is_the_trade_date_of_a_fill_and_the_date_of_a_fee(store: Store) -> None:
    assert store.last_day() is None
    store.add([fill("2026-10-28T23:59:00", "o1", "buy", "1", "100")])
    assert store.last_day() == date(2026, 10, 28)
    store.add([coin_fee("2026-10-30T01:00:00", "-1", "1")])
    assert store.last_day() == date(2026, 10, 30)


def test_a_row_that_cannot_be_read_is_skipped_and_logged(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    store.add([{"id": "bad", "activity_type": "FILL", "qty": "not a number", "date": "2026-10-28"}])
    store.add([fill("2026-10-28T11:00:00", "o1", "buy", "1", "100")])
    with caplog.at_level(logging.WARNING):
        found = store.activities()
    assert len(found) == 1 and "could not be read" in caplog.text


def test_the_lots_chosen_for_a_sale_replace_the_earlier_choice(store: Store) -> None:
    store.set_picks("s1", [("a", Decimal(2)), ("b", Decimal("1.5"))])
    store.set_picks("s2", [("a", Decimal(1))])
    assert store.picks() == {
        "s1": [("a", Decimal(2)), ("b", Decimal("1.5"))],
        "s2": [("a", Decimal(1))],
    }
    store.set_picks("s1", [("c", Decimal(3))])
    store.set_picks("s2", [])
    assert store.picks() == {"s1": [("c", Decimal(3))]}


def test_adjustments_are_kept_in_the_order_they_were_made(store: Store) -> None:
    store.adjust("ACME", "open", Decimal(5), Decimal("12.5"), "first")
    store.adjust("ACME", "drop", Decimal(2), Decimal(0), "second")
    first, second = store.adjustments()
    assert row(first.id, first.kind, first.qty, first.price, first.reason) == (
        1,
        "open",
        5,
        Decimal("12.5"),
        "first",
    )
    assert row(second.id, second.kind, second.qty) == (2, "drop", 2)
