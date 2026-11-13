from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from alpaca.trading.client import TradingClient

from app.core.tests.support import api_error
from app.features.systematic import execution
from app.features.systematic.schemas import ExecutedOrder, PlannedOrder, RunRecord

NOW = datetime(2026, 11, 5, 1, 0, tzinfo=UTC)
FILLED_AT = "2026-11-04T14:30:02Z"
CLIENT_ID = "O-20261104-125651-001-002-1"
ALPACA_ID = "0b7c1c5e-8d55-4c1b-9f51-2f6c2a7f3c11"


def alpaca_order(**fields: Any) -> dict[str, Any]:
    base = {
        "id": ALPACA_ID,
        "symbol": "SPUS",
        "side": "buy",
        "type": "limit",
        "status": "filled",
        "qty": "10",
        "filled_qty": "10",
        "filled_avg_price": "50.10",
        "submitted_at": "2026-11-04T00:05:10Z",
        "filled_at": FILLED_AT,
    }
    return {**base, **fields}


def sent(**fields: Any) -> ExecutedOrder:
    base: dict[str, Any] = {
        "symbol": "SPUS",
        "side": "buy",
        "qty": 10,
        "order_id": CLIENT_ID,
        "status": "accepted",
        "price": 49.90,
        "limit_price": 50.89,
    }
    return ExecutedOrder(**{**base, **fields})


def run_of(*orders: ExecutedOrder, time: datetime = NOW - timedelta(hours=25)) -> RunRecord:
    return RunRecord(
        time=time,
        strategies=["halal-etf-trend"],
        mode="paper",
        status="ok",
        executed=list(orders),
    )


def opens(price: float | None = 50.0) -> execution.OpenPrice:
    calls: list[tuple[str, date]] = []

    def open_on(symbol: str, day: date) -> float | None:
        calls.append((symbol, day))
        return price

    open_on.calls = calls  # type: ignore[attr-defined]
    return open_on


def trading_with(order: dict[str, Any]) -> MagicMock:
    trading: MagicMock = create_autospec(TradingClient, instance=True)
    trading.get_order_by_client_id.return_value = order
    trading.get_order_by_id.return_value = order
    return trading


def test_a_filled_stock_buy_is_costed_against_the_open_of_its_session() -> None:
    open_on = opens(50.0)
    run = run_of(sent())
    assert execution.settle(trading_with(alpaca_order()), [run], open_on, NOW)
    [item] = run.executed
    assert item.settled and item.status == "filled"
    assert item.filled_price == 50.10
    assert item.model_price == 50.0
    assert item.cost_bps == pytest.approx(20.0)
    assert item.shortfall_bps == pytest.approx((50.10 - 49.90) / 49.90 * 1e4)
    assert open_on.calls == [("SPUS", date(2026, 11, 4))]  # type: ignore[attr-defined]


def test_a_sell_below_the_open_is_a_cost() -> None:
    run = run_of(sent(side="sell", price=51.0, limit_price=None))
    order = alpaca_order(side="sell", filled_avg_price="49.95")
    execution.settle(trading_with(order), [run], opens(50.0), NOW)
    assert run.executed[0].cost_bps == pytest.approx(10.0)


def test_a_coin_is_costed_against_its_decision_price_plus_the_coin_fee() -> None:
    coin = sent(symbol="BTC/USD", price=80_000.0, limit_price=None, qty=0.1)
    order = alpaca_order(symbol="BTCUSD", filled_avg_price="80040", qty="0.1", filled_qty="0.1")
    open_on = opens()
    run = run_of(coin)
    execution.settle(trading_with(order), [run], open_on, NOW)
    item = run.executed[0]
    assert item.model_price == 80_000.0
    assert item.cost_bps == pytest.approx(5.0 + execution.CRYPTO_TAKER_BPS)
    assert open_on.calls == []  # type: ignore[attr-defined]


def test_an_order_still_working_is_left_for_later() -> None:
    run = run_of(sent())
    assert not execution.settle(
        trading_with(alpaca_order(status="new", filled_qty="0", filled_avg_price=None)),
        [run],
        opens(),
        NOW,
    )
    assert not run.executed[0].settled


def test_a_stock_fill_waits_until_its_open_is_known() -> None:
    run = run_of(sent())
    assert not execution.settle(trading_with(alpaca_order()), [run], opens(None), NOW)
    assert not run.executed[0].settled


def test_a_limit_the_open_gapped_past_settles_unfilled() -> None:
    run = run_of(sent())
    order = alpaca_order(status="expired", filled_qty="0", filled_avg_price=None, filled_at=None)
    execution.settle(trading_with(order), [run], opens(), NOW)
    item = run.executed[0]
    assert item.settled and item.status == "expired"
    assert item.filled_price is None and item.cost_bps is None


def test_orders_are_read_by_alpacas_id_or_by_nautilus_client_id() -> None:
    trading = trading_with(alpaca_order())
    execution.settle(trading, [run_of(sent(order_id=ALPACA_ID))], opens(), NOW)
    trading.get_order_by_id.assert_called_once_with(ALPACA_ID)
    execution.settle(trading, [run_of(sent())], opens(), NOW)
    trading.get_order_by_client_id.assert_called_once_with(CLIENT_ID)


def test_orders_never_sent_or_already_settled_or_old_are_not_read() -> None:
    trading = trading_with(alpaca_order())
    runs = [
        run_of(sent(status="denied: no cash"), sent(order_id=None, status="rejected: x")),
        run_of(sent(settled=True)),
        run_of(sent(), time=NOW - timedelta(days=execution.SETTLE_DAYS + 1)),
    ]
    assert not execution.settle(trading, runs, opens(), NOW)
    trading.get_order_by_client_id.assert_not_called()


def test_an_order_that_cannot_be_read_is_tried_again_later(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trading = trading_with(alpaca_order())
    trading.get_order_by_client_id.side_effect = api_error("not found", 404)
    run = run_of(sent())
    assert not execution.settle(trading, [run], opens(), NOW)
    assert not run.executed[0].settled
    assert "Could not read order" in caplog.text


def test_costs_are_weighted_by_notional_per_asset_class() -> None:
    small = sent(settled=True, filled_qty=1, filled_price=100, cost_bps=50, shortfall_bps=60)
    large = sent(settled=True, filled_qty=9, filled_price=100, cost_bps=0, shortfall_bps=10)
    coin = sent(symbol="BTC/USD", settled=True, filled_qty=0.1, filled_price=80_000, cost_bps=30)
    [stock, crypto] = execution.summaries([run_of(small, large, coin)])
    assert (stock.asset, stock.orders, stock.notional) == ("stock", 2, 1_000)
    assert stock.cost_bps == pytest.approx(5.0)
    assert stock.shortfall_bps == pytest.approx(15.0)
    assert stock.assumed_bps == 5
    assert (crypto.asset, crypto.cost_bps, crypto.shortfall_bps) == ("crypto", 30, None)
    assert crypto.assumed_bps == 25


def test_nothing_filled_reports_no_summary() -> None:
    assert execution.summaries([run_of(sent())]) == []


def test_the_sent_orders_take_the_decision_price_and_limit_of_their_plan() -> None:
    planned = [
        PlannedOrder(
            symbol="SPUS", side="buy", qty=10, notional=499, price=49.9, limit_price=50.89
        ),
    ]
    done = [ExecutedOrder(symbol="SPUS", side="buy", qty=10, status="accepted")]
    [item] = execution.with_prices(done, planned)
    assert (item.price, item.limit_price) == (49.9, 50.89)
    [other] = execution.with_prices([sent(symbol="HLAL", price=None)], planned)
    assert other.price is None


def test_the_open_is_read_from_the_daily_bar_of_that_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.quant import history

    seen: list[tuple[str, datetime, datetime]] = []

    def bars(_s: object, _c: object, symbol: str, start: datetime, end: datetime) -> object:
        seen.append((symbol, start, end))
        return [{"t": "2026-11-04T05:00:00Z", "o": 50.25, "c": 50.5}]

    monkeypatch.setattr(history, "bars", bars)
    open_on = execution.bar_open(MagicMock(), MagicMock())
    assert open_on("SPUS", date(2026, 11, 4)) == 50.25
    assert open_on("SPUS", date(2026, 11, 5)) is None
    assert seen[0][1] == datetime(2026, 11, 4, tzinfo=UTC)
