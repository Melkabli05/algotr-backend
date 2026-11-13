import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.core.audit import AuditLog
from app.core.tests.support import api_error, until
from app.features.ledger import service as module
from app.features.ledger.schemas import Pick, Picks, Trigger
from app.features.ledger.service import Ledger, NothingToAdoptError
from app.features.ledger.tests.support import Broker, coin_fee, fill, position, row

STOCKS = [
    fill("2026-10-28T14:00:00", "o1", "buy", "10", "50"),
    fill("2026-10-29T14:00:00", "o2", "buy", "10", "60"),
    fill("2026-10-30T14:00:00", "o3", "sell", "5", "70"),
]
HELD = [position("SPUS", "15", "55", price="65")]


def ledger_for(
    tmp_path: Path,
    rows: list[dict[str, object]],
    positions: list[dict[str, object]],
    log: AuditLog | None = None,
) -> tuple[Ledger, Broker]:
    broker = Broker(rows, positions)
    desk = Ledger(broker.client, tmp_path / "ledger.sqlite", log)
    return desk, broker


def test_the_first_sync_reads_all_history_and_later_ones_a_window_of_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "utc_today", lambda: date(2026, 10, 31))
    desk, broker = ledger_for(tmp_path, [*STOCKS], HELD)
    assert desk.sync() == 3
    assert all("date" not in call for call in broker.calls)
    broker.calls.clear()
    broker.rows.append(fill("2026-10-31T14:00:00", "o4", "buy", "1", "66"))
    assert desk.sync() == 1
    assert [call["date"] for call in broker.calls] == [
        "2026-10-28",
        "2026-10-29",
        "2026-10-30",
        "2026-10-31",
        "2026-11-01",
    ]
    assert desk.sync() == 0


def test_holdings_follow_the_method_chosen(tmp_path: Path) -> None:
    desk, _ = ledger_for(tmp_path, STOCKS, HELD)
    desk.sync()
    [fifo] = desk.holdings("fifo")
    assert row(fifo.symbol, fifo.qty, fifo.basis, fifo.price, fifo.value) == (
        "SPUS",
        15,
        850,
        65,
        975,
    )
    assert row(fifo.unrealized, fifo.realized, fifo.lots) == (125, 100, 2)
    [average] = desk.holdings("average")
    assert row(
        average.basis, average.average_cost, average.unrealized, average.realized, average.lots
    ) == (
        825,
        55,
        150,
        75,
        1,
    )


def test_lots_are_marked_at_the_brokers_price(tmp_path: Path) -> None:
    desk, _ = ledger_for(tmp_path, STOCKS, HELD)
    desk.sync()
    first, second = desk.lots("fifo")
    assert row(first.id, first.qty, first.original, first.unit_cost, first.unrealized) == (
        "o1",
        5,
        10,
        50,
        75,
    )
    assert row(second.id, second.qty, second.value, second.unrealized_pct) == (
        "o2",
        10,
        650,
        50 / 600,
    )
    assert first.days >= 0 and first.side == "long"
    assert desk.lots("fifo", "AAPL") == []
    assert [lot.id for lot in desk.lots("fifo", "spus")] == ["o1", "o2"]


def test_realized_rows_come_newest_first_with_totals_and_can_be_limited_by_days(
    tmp_path: Path,
) -> None:
    recent = datetime.now(UTC) - timedelta(hours=2)
    rows = [
        fill("2026-10-28T14:00:00", "o1", "buy", "10", "50"),
        fill("2026-10-29T14:00:00", "o2", "sell", "4", "70"),
        fill(f"{recent:%Y-%m-%dT%H:%M:%S}", "o3", "sell", "6", "55"),
    ]
    desk, _ = ledger_for(tmp_path, rows, [])
    desk.sync()
    report = desk.realized("fifo")
    assert [close.order for close in report.rows] == ["o3", "o2"]
    assert row(
        report.summary.count, report.summary.basis, report.summary.proceeds, report.summary.gain
    ) == (
        2,
        500,
        610,
        110,
    )
    assert row(report.rows[1].gain, report.rows[1].gain_pct) == (80, 0.4)
    assert [close.order for close in desk.realized("fifo", days=1).rows] == ["o3"]
    assert desk.realized("fifo", symbol="AAPL").rows == []


def test_a_ledger_that_matches_the_broker_reconciles(tmp_path: Path) -> None:
    desk, _ = ledger_for(tmp_path, STOCKS, HELD)
    result = desk.refresh()
    assert (result.ok, result.differences, result.activities) == (True, [], 3)
    assert desk.status().reconciliation == result and desk.status().error is None


def test_a_quantity_the_fills_do_not_explain_is_a_break(tmp_path: Path) -> None:
    desk, _ = ledger_for(tmp_path, STOCKS, [position("SPUS", "20", "55")])
    result = desk.refresh()
    [difference] = result.differences
    assert (result.ok, difference.kind, difference.ledger_qty, difference.broker_qty) == (
        False,
        "quantity",
        15,
        20,
    )


def test_a_coin_fee_that_has_not_posted_yet_is_pending_and_then_settles(tmp_path: Path) -> None:
    moment = datetime.now(UTC) - timedelta(hours=1)
    stamp = f"{moment:%Y-%m-%dT%H:%M:%S}"
    rows = [fill(stamp, "b1", "buy", "0.001", "80000", "BTC/USD")]
    desk, broker = ledger_for(tmp_path, rows, [position("BTCUSD", "0.0009975", "80000")])
    first = desk.refresh()
    assert first.ok and [d.kind for d in first.differences] == ["pending"]
    broker.rows.append(
        coin_fee(f"{moment + timedelta(seconds=4):%Y-%m-%dT%H:%M:%S}", "-0.0000025", "80000")
    )
    broker.rows[-1]["date"] = (moment - timedelta(days=1)).date().isoformat()
    second = desk.refresh()
    assert second.ok and second.differences == []


def test_adopting_the_brokers_quantity_adds_a_lot_and_leaves_an_audit_entry(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    desk, _ = ledger_for(tmp_path, STOCKS, [position("SPUS", "20", "57")], log)
    desk.refresh()
    result = desk.adopt("spus")
    assert result.ok and result.differences == []
    [added] = [lot for lot in desk.lots("fifo") if lot.id.startswith("adjustment")]
    assert row(added.qty, added.unit_cost) == (5, 63)
    [entry] = log.entries()
    assert entry.kind == "ledger-adopt" and entry.payload["symbol"] == "SPUS"


def test_adopting_a_smaller_brokers_quantity_writes_the_difference_off_at_no_gain(
    tmp_path: Path,
) -> None:
    desk, _ = ledger_for(tmp_path, STOCKS, [position("SPUS", "10", "55")])
    desk.refresh()
    assert desk.adopt("SPUS").ok
    assert desk.holdings("fifo")[0].qty == 10
    assert desk.realized("fifo").summary.gain == 100


def test_there_is_nothing_to_adopt_when_the_quantities_agree_or_the_symbol_is_unknown(
    tmp_path: Path,
) -> None:
    desk, _ = ledger_for(tmp_path, STOCKS, HELD)
    desk.refresh()
    for symbol in ("SPUS", "NOPE"):
        with pytest.raises(NothingToAdoptError):
            desk.adopt(symbol)


def test_picks_choose_the_lots_a_sale_closes_under_the_specific_method(tmp_path: Path) -> None:
    desk, _ = ledger_for(tmp_path, STOCKS, HELD)
    desk.sync()
    assert desk.realized("specific").summary.gain == 100
    saved = desk.set_picks("o3", Picks(lots=[Pick(lot="o2", qty=3), Pick(lot="o2", qty=2)]))
    assert [(pick.lot, pick.qty) for pick in saved.lots] == [("o2", 5)]
    assert desk.realized("specific").summary.gain == 5 * (70 - 60)
    assert desk.realized("fifo").summary.gain == 100
    desk.set_picks("o3", Picks(lots=[]))
    assert desk.realized("specific").summary.gain == 100


def test_a_short_that_was_covered_leaves_no_position_and_a_realized_gain(tmp_path: Path) -> None:
    rows = [
        fill("2026-10-28T14:00:00", "s1", "sell", "4", "100", "SPY"),
        fill("2026-10-28T14:05:00", "b1", "buy", "4", "90", "SPY"),
    ]
    desk, _ = ledger_for(tmp_path, rows, [])
    result = desk.refresh()
    assert result.ok and desk.holdings("fifo") == []
    [close] = desk.realized("fifo").rows
    assert (close.side, close.gain) == ("short", 40)


def test_an_open_short_is_held_as_a_negative_quantity_with_its_own_gain(tmp_path: Path) -> None:
    rows = [fill("2026-10-28T14:00:00", "s1", "sell", "4", "100", "SPY")]
    desk, _ = ledger_for(tmp_path, rows, [position("SPY", "-4", "100", price="90")])
    assert desk.refresh().ok
    [holding] = desk.holdings("fifo")
    assert row(holding.qty, holding.value, holding.unrealized) == (-4, -360, 40)
    [lot] = desk.lots("fifo")
    assert (lot.side, lot.qty) == ("short", -4)


def test_a_failed_sync_is_kept_for_the_status_and_raised(tmp_path: Path) -> None:
    desk, broker = ledger_for(tmp_path, STOCKS, HELD)
    desk.refresh()
    broker.client.get.side_effect = api_error("unavailable", 503)
    with pytest.raises(type(api_error("x"))):
        desk.refresh()
    status = desk.status()
    assert status.error is not None and "unavailable" in status.error
    assert status.reconciliation is not None and status.reconciliation.ok


def test_the_scheduled_sync_survives_a_failure_and_the_next_one_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "FIRST_RUN", timedelta(milliseconds=30))
    monkeypatch.setattr(module, "EVERY", timedelta(milliseconds=60))

    async def scenario() -> bool:
        desk, broker = ledger_for(tmp_path, STOCKS, HELD)
        real = broker.client.get.side_effect
        broker.client.get.side_effect = api_error("unavailable", 503)
        desk.start()
        await until(lambda: desk.error is not None)
        broker.client.get.side_effect = real
        await until(lambda: desk.last is not None)
        desk.stop()
        return desk.error is None

    assert asyncio.run(scenario())


def stamp(moment: datetime) -> str:
    return f"{moment:%Y-%m-%dT%H:%M:%S}"


def test_fills_come_newest_first_with_their_value_and_whether_they_were_partial(
    tmp_path: Path,
) -> None:
    rows = [
        fill("2026-10-28T14:00:00", "o1", "buy", "10", "50"),
        fill("2026-10-29T14:00:00", "o2", "buy", "4", "60", partial=True),
        fill("2026-10-29T14:00:01", "o2", "buy", "6", "61"),
        fill("2026-10-30T14:00:00", "o3", "sell", "5", "70", "AAPL"),
    ]
    desk, _ = ledger_for(tmp_path, rows, [])
    desk.sync()
    found = desk.fills()
    assert [item.order for item in found] == ["o3", "o2", "o2", "o1"]
    assert [item.partial for item in found] == [False, False, True, False]
    sale = found[0]
    assert row(sale.symbol, sale.side, sale.qty, sale.price, sale.value) == (
        "AAPL",
        "sell",
        5,
        70,
        350,
    )
    assert all(item.fee_qty is None for item in found)


def test_fills_can_be_limited_by_symbol_days_and_count(tmp_path: Path) -> None:
    recent = datetime.now(UTC) - timedelta(hours=2)
    rows = [
        fill("2026-10-28T14:00:00", "o1", "buy", "10", "50"),
        fill(stamp(recent), "o2", "buy", "1", "51"),
        fill(stamp(recent), "o3", "buy", "1", "9", "BTC/USD"),
    ]
    desk, _ = ledger_for(tmp_path, rows, [])
    desk.sync()
    assert [item.order for item in desk.fills(symbol="spus")] == ["o2", "o1"]
    assert [item.order for item in desk.fills(symbol="BTC/USD")] == ["o3"]
    assert [item.order for item in desk.fills(symbol="BTCUSD")] == ["o3"]
    assert {item.order for item in desk.fills(days=1)} == {"o2", "o3"}
    assert len(desk.fills(limit=1)) == 1


def test_a_coin_buy_lists_the_fee_that_came_with_it(tmp_path: Path) -> None:
    moment = datetime.now(UTC) - timedelta(days=2)
    rows = [
        fill(stamp(moment), "b1", "buy", "0.001", "80000", "BTC/USD"),
        coin_fee(stamp(moment + timedelta(seconds=4)), "-0.0000025", "80000"),
    ]
    desk, _ = ledger_for(tmp_path, rows, [])
    desk.sync()
    [item] = desk.fills()
    assert (item.symbol, item.fee_qty) == ("BTC/USD", 0.0000025)


def test_a_burst_of_fills_makes_one_sync_a_few_seconds_after_the_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "FILL_DELAY", timedelta(milliseconds=80))

    async def scenario() -> tuple[int, int]:
        desk, _ = ledger_for(tmp_path, STOCKS, HELD)
        syncs: list[int] = []
        real = desk.refresh

        def counting(trigger: Trigger = "manual") -> Any:
            syncs.append(1)
            assert trigger == "fill"
            return real(trigger)

        monkeypatch.setattr(desk, "refresh", counting)
        desk.scheduler.start()
        for _ in range(5):
            desk.nudge()
        pending = len(desk.scheduler.get_jobs())
        await until(lambda: desk.last is not None)
        await asyncio.sleep(0.3)
        desk.stop()
        return pending, len(syncs)

    assert asyncio.run(scenario()) == (1, 1)


def test_a_stream_of_fills_cannot_postpone_the_sync_for_ever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "FILL_DELAY", timedelta(seconds=60))

    async def scenario() -> float:
        desk, _ = ledger_for(tmp_path, STOCKS, HELD)
        desk.scheduler.start()
        desk.nudge()
        patient = desk.scheduler.get_job(module.FILL_JOB_ID).next_run_time
        desk._waiting = datetime.now(UTC) - timedelta(minutes=5)
        desk.nudge()
        impatient = desk.scheduler.get_job(module.FILL_JOB_ID).next_run_time
        desk.stop()
        return float((patient - impatient).total_seconds())

    assert asyncio.run(scenario()) > 50


def test_a_fill_while_the_broker_is_down_is_retried_by_the_next_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "FILL_DELAY", timedelta(milliseconds=30))

    async def scenario() -> bool:
        desk, broker = ledger_for(tmp_path, STOCKS, HELD)
        real = broker.client.get.side_effect
        broker.client.get.side_effect = api_error("unavailable", 503)
        desk.scheduler.start()
        desk.nudge()
        await until(lambda: desk.error is not None)
        broker.client.get.side_effect = real
        desk.nudge()
        await until(lambda: desk.last is not None)
        desk.stop()
        return desk.error is None

    assert asyncio.run(scenario())


def test_the_comparison_lists_every_symbol_with_orphans_and_zombies() -> None:
    from decimal import Decimal

    from app.features.ledger.schemas import Difference
    from app.features.ledger.service import state_of

    quantity = Difference(
        symbol="X",
        kind="quantity",
        ledger_qty=1,
        broker_qty=0,
        ledger_cost=None,
        broker_cost=None,
        note="",
    )
    assert state_of(None, Decimal(1), Decimal(1)) == "match"
    assert state_of(quantity, Decimal(1), Decimal(0)) == "orphan"
    assert state_of(quantity, Decimal(0), Decimal(2)) == "zombie"
    assert state_of(quantity, Decimal(1), Decimal(2)) == "quantity"
