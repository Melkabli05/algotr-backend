from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.features.ledger import activities
from app.features.ledger.activities import Activity
from app.features.ledger.lots import replay
from app.features.ledger.tests.support import Broker, coin_fee, fill, row, usd_fee

BTC = "BTC/USD"


def parse(rows: list[dict[str, Any]]) -> list[Activity]:
    return [Activity.model_validate(row) for row in rows]


def test_an_activity_is_read_with_exact_numbers_and_unknown_keys_ignored() -> None:
    row = fill("2026-10-28T11:31:41.473979", "o1", "buy", "0.00024605", "81284.36", BTC)
    [found] = parse([{**row, "swap_rate": "1", "cusip": None}])
    assert (found.qty, found.price, found.key) == (
        Decimal("0.00024605"),
        Decimal("81284.36"),
        "BTCUSD",
    )
    assert found.moment == datetime(2026, 10, 28, 11, 31, 41, 473979, tzinfo=UTC)


def test_the_moment_of_a_row_without_a_time_is_the_start_of_its_date() -> None:
    [found] = parse([{"id": "x", "activity_type": "FEE", "date": "2026-11-02", "net_amount": "-1"}])
    assert found.moment == datetime(2026, 11, 2, tzinfo=UTC)
    assert found.day == date(2026, 11, 2)


def test_the_fills_of_an_order_become_one_execution_at_the_volume_weighted_price() -> None:
    rows = [
        fill("2026-10-28T11:31:41.1", "o1", "buy", "1", "100", partial=True),
        fill("2026-10-28T11:31:42.1", "o1", "buy", "3", "104", partial=True),
    ]
    [item] = activities.executions(parse(rows))
    assert row(item.order, item.kind, item.qty, item.price) == ("o1", "buy", 4, Decimal(103))
    assert item.time == datetime(2026, 10, 28, 11, 31, 41, 100000, tzinfo=UTC)
    assert item.paid == 412


def test_a_buy_is_net_of_its_coin_fee_but_costs_what_was_paid() -> None:
    rows = [
        fill("2026-10-28T11:31:41", "o1", "buy", "0.001", "80000", BTC),
        coin_fee("2026-10-28T11:31:45", "-0.0000025", "80000"),
        usd_fee("2026-10-28T11:31:46", "-0.05"),
    ]
    [item] = activities.executions(parse(rows))
    assert (item.qty, item.paid) == (Decimal("0.0009975"), Decimal(80))


def test_a_fee_belongs_to_the_latest_buy_before_it() -> None:
    rows = [
        fill("2026-10-28T11:00:00", "early", "buy", "1", "100", BTC),
        fill("2026-10-28T11:30:00", "late", "buy", "1", "101", BTC),
        coin_fee("2026-10-28T11:30:04", "-0.01", "101"),
        fill("2026-10-28T11:40:00", "sale", "sell", "0.5", "102", BTC),
    ]
    found = {item.order: item for item in activities.executions(parse(rows))}
    assert row(found["early"].qty, found["late"].qty, found["sale"].qty) == (
        1,
        Decimal("0.99"),
        Decimal("0.5"),
    )


def test_a_fee_before_any_buy_or_on_another_symbol_is_left_out() -> None:
    rows = [
        coin_fee("2026-10-28T10:00:00", "-0.01", "100"),
        fill("2026-10-28T11:00:00", "o1", "buy", "1", "100", BTC),
        coin_fee("2026-10-28T11:00:03", "-0.01", "100", symbol="ETHUSD"),
    ]
    [item] = activities.executions(parse(rows))
    assert item.qty == 1


def test_rows_that_are_not_fills_with_an_order_are_skipped() -> None:
    odd = {**fill("2026-10-28T11:00:00", "o1", "buy", "1", "100"), "order_id": None}
    rows = [odd, {"id": "j", "activity_type": "JNLC", "date": "2026-10-28", "net_amount": "100000"}]
    assert activities.executions(parse(rows)) == []


def test_replaying_the_fills_and_fees_reproduces_the_brokers_coin_quantity() -> None:
    rows = [
        fill("2026-10-28T11:31:41", "b1", "buy", "0.00024605", "81284.36", BTC, partial=True),
        fill("2026-10-28T11:31:41.5", "b1", "buy", "0.00075395", "81307.1", BTC),
        coin_fee("2026-10-28T11:31:45", "-0.0000006151", "81284.36"),
        coin_fee("2026-10-28T11:31:45.5", "-0.0000018849", "81307.1"),
        fill("2026-10-28T11:32:10", "s1", "sell", "0.000997498", "81260", BTC),
    ]
    book = replay(activities.executions(parse(rows)), "average")
    [lot] = book.lots
    assert lot.qty == Decimal("0.001") - Decimal("0.0000025") - Decimal("0.000997498")


def test_recent_coin_buys_may_still_owe_a_fee_and_stocks_never_do() -> None:
    now = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
    rows = [
        fill("2026-11-03T09:00:00", "b1", "buy", "2", "100", BTC),
        fill("2026-10-25T09:00:00", "old", "buy", "9", "100", BTC),
        fill("2026-11-03T09:00:00", "stock", "buy", "9", "100", "SPUS"),
    ]
    assert activities.pending_fees(parse(rows), now) == {"BTCUSD": Decimal("0.010")}


def test_the_latest_corporate_action_per_symbol_is_reported() -> None:
    rows = [
        {"id": "a", "activity_type": "SPLIT", "date": "2026-11-01", "symbol": "ACME"},
        {"id": "b", "activity_type": "NC", "date": "2026-11-02", "symbol": "ACME"},
        {"id": "c", "activity_type": "DIV", "date": "2026-11-02", "symbol": "OTHER"},
    ]
    assert activities.corporate_actions(parse(rows)) == {"ACME": "NC"}


def test_all_activity_is_fetched_page_by_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(activities, "PAGE", 2)
    rows = [fill(f"2026-10-28T11:00:0{n}", f"o{n}", "buy", "1", "100") for n in range(5)]
    broker = Broker(rows, [])
    assert [row["order_id"] for row in activities.fetch_all(broker.client)] == [
        f"o{n}" for n in range(5)
    ]
    assert [call.get("page_token") is None for call in broker.calls] == [True, False, False]


def test_one_day_is_fetched_by_date_and_fees_are_listed_a_day_late() -> None:
    rows = [
        fill("2026-11-02T10:00:00", "o1", "buy", "1", "100", BTC),
        coin_fee("2026-11-02T10:00:04", "-0.01", "100"),
    ]
    broker = Broker(rows, [])
    assert [r["activity_type"] for r in activities.fetch_day(broker.client, date(2026, 11, 2))] == [
        "FILL"
    ]
    assert [r["activity_type"] for r in activities.fetch_day(broker.client, date(2026, 11, 3))] == [
        "CFEE"
    ]
    assert broker.calls[0] == {"date": "2026-11-02", "direction": "asc"}


def test_a_day_without_activity_is_empty() -> None:
    assert activities.fetch_day(Broker([], []).client, date.today() - timedelta(days=400)) == []


def test_each_coin_fee_is_attached_to_the_latest_buy_fill_before_it() -> None:
    rows = [
        fill("2026-10-28T11:00:00", "early", "buy", "1", "100", BTC),
        fill("2026-10-28T11:30:00", "late", "buy", "1", "101", BTC, partial=True),
        fill("2026-10-28T11:30:01", "late", "buy", "2", "101", BTC),
        coin_fee("2026-10-28T11:30:04", "-0.01", "101"),
        coin_fee("2026-10-28T11:30:05", "-0.02", "101"),
        fill("2026-10-28T11:40:00", "sale", "sell", "0.5", "102", BTC),
        coin_fee("2026-10-28T11:40:03", "-0.03", "102"),
        coin_fee("2026-10-28T09:00:00", "-0.5", "100"),
    ]
    parsed = parse(rows)
    ids = {(a.order_id, a.qty): a.id for a in parsed if a.activity_type == "FILL"}
    assert activities.coin_fees(parsed) == {ids[("late", Decimal(2))]: Decimal("0.06")}
