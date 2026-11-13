import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pandas as pd
import pytest
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.core.settings import Settings
from app.core.tests.support import until
from app.features.systematic import automation as module
from app.features.systematic import research, targets
from app.features.systematic.automation import (
    Automation,
    AutomationDisabledError,
    LiveAccountError,
)
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.nav import Book
from app.features.systematic.schemas import (
    AutomationConfig,
    RunRecord,
    StrategyName,
)
from app.features.systematic.tests.conftest import FakeLink
from app.features.systematic.tests.support import (
    LIVE_SETTINGS,
    SETTINGS,
    fake_history,
    position,
    serve,
    trading_with,
)
from app.quant import history
from app.quant.history import NotEnoughHistoryError
from app.quant.performance import Check

BTC = ["btc-trend-200"]


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    return fake_history(monkeypatch)


def build(
    tmp_path: Path,
    trading: MagicMock | None = None,
    settings: Settings = SETTINGS,
    warm_cache: bool = False,
) -> Automation:
    return Automation(
        settings,
        trading or trading_with(),
        create_autospec(StockHistoricalDataClient, instance=True),
        create_autospec(CryptoHistoricalDataClient, instance=True),
        state_path=tmp_path / "automation.json",
        warm_cache=warm_cache,
    )


def test_automation_starts_off_with_the_halal_etf_strategy_selected(tmp_path: Path) -> None:
    status = build(tmp_path).status()
    assert status.config == AutomationConfig(
        mode="off",
        allocation=0.1,
        automatic=True,
        crypto=False,
        strategies=["halal-etf-trend"],
        max_drawdown=0.15,
    )
    assert (status.runs, status.next_run, status.halted) == ([], None, None)
    assert (status.book, status.books) == (Book(), {})


def test_running_while_off_is_refused(tmp_path: Path, market: dict[str, str]) -> None:
    with pytest.raises(AutomationDisabledError):
        build(tmp_path).run()


def test_a_dry_run_plans_and_records_but_never_submits(
    tmp_path: Path, market: dict[str, str], link: FakeLink
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="dry-run", allocation=0.2, strategies=BTC))
    record = automation.run()
    assert (record.status, record.mode, record.strategies) == ("ok", "dry-run", BTC)
    assert record.plan is not None
    assert [o.side for o in record.plan.orders] == ["buy"]
    assert record.executed == []
    assert link.written == []


def test_a_paper_run_leaves_the_orders_to_the_node_and_remembers_what_it_plans(
    tmp_path: Path, market: dict[str, str], link: FakeLink
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="paper", strategies=BTC))
    record = automation.run()
    assert record.status == "ok"
    assert [e.status for e in record.executed] == ["accepted"]
    assert link.written == [[]]
    assert link.configs[-1].mode == "paper"
    trading.submit_order.assert_not_called()
    assert automation.state.managed == ["BTC/USD"]


def test_the_agents_ensemble_is_written_as_a_target_for_the_arbiter() -> None:
    from app.features.systematic.schemas import Plan, StrategyBook, Target

    plan = Plan(
        strategies=[],
        allocation=0.5,
        equity=100_000,
        signals={},
        targets=[Target(symbol="SPUS", weight=0.1, target_value=10_000, current_value=0)],
        orders=[],
        books={"ai-ensemble": StrategyBook(capital=50_000, targets={"SPUS": 20_000}, prices={})},
    )

    [row] = targets.api_targets(plan, "ai-ensemble")

    assert (row["strategy"], row["source"], row["weights"]) == ("ai-ensemble", "api", {"SPUS": 0.2})


def test_a_dry_run_does_not_claim_any_symbols(tmp_path: Path, market: dict[str, str]) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    automation.run()
    assert automation.state.managed == []


def test_a_second_run_after_filling_places_nothing(tmp_path: Path, market: dict[str, str]) -> None:
    price = float(history.cached_closes(MagicMock(), MagicMock(), ["BTC/USD"])["BTC/USD"].iloc[-1])
    trading = trading_with([position("BTCUSD", 10_000 / price, price)])
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="paper", strategies=BTC))
    record = automation.run()
    assert record.plan is not None and record.plan.orders == []


def test_deselecting_every_strategy_sells_what_the_automation_bought(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with([position("BTCUSD", 0.5, 80_000)])
    automation = build(tmp_path, trading)
    automation.state.managed = ["BTC/USD"]
    automation.configure(AutomationConfig(mode="paper", engine="alpaca", strategies=[]))
    record = automation.run()
    assert record.status == "ok"
    assert record.plan is not None and record.plan.orders[-1].side == "sell"


def test_a_live_account_is_traded_only_once_the_settings_open_it(tmp_path: Path) -> None:
    automation = build(tmp_path, settings=LIVE_SETTINGS)
    with pytest.raises(LiveAccountError, match="ALPACA_TRADING_MODE=live"):
        automation.configure(AutomationConfig(mode="paper"))
    assert automation.configure(AutomationConfig(mode="dry-run")).config.mode == "dry-run"
    opened = build(
        tmp_path / "opened",
        settings=LIVE_SETTINGS.model_copy(update={"alpaca_trading_mode": "live"}),
    )
    assert opened.configure(AutomationConfig(mode="paper")).config.mode == "paper"


def test_missing_history_is_a_skip_and_unexpected_failures_are_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="paper", engine="alpaca", strategies=BTC))

    def no_history(*_: object) -> pd.DataFrame:
        raise NotEnoughHistoryError("Alpaca has no daily bars for BTC/USD")

    serve(monkeypatch, no_history)
    assert automation.run().status == "skipped"

    def broken(*_: object) -> pd.DataFrame:
        raise RuntimeError("boom")

    serve(monkeypatch, broken)
    record = automation.run()
    assert (record.status, record.detail) == ("error", "RuntimeError: boom")


def held_after_first_run(automation: Automation, trading: MagicMock, drop: float) -> float:
    """Run once, then move the price of the BTC the book holds by `drop`; returns the new price."""
    assert automation.run().status == "ok"
    held = automation.state.books["btc-trend-200"].holdings["BTC/USD"]
    price = held.price * (1 - drop)
    trading.get_all_positions.return_value = [position("BTCUSD", held.qty, price)]
    return price


def test_a_drawdown_past_the_limit_closes_positions_and_halts_until_resumed(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(
        AutomationConfig(mode="paper", engine="alpaca", strategies=BTC, max_drawdown=0.1)
    )
    held_after_first_run(automation, trading, 0.15)
    record = automation.run()
    assert record.status == "halted"
    assert record.plan is not None and record.plan.flat
    assert record.plan.orders[-1].side == "sell"
    assert "15.0%" in (automation.status().halted or "")
    assert automation.state.book.drawdown == pytest.approx(0.15)
    assert automation.state.books["btc-trend-200"].holdings == {}

    skipped = automation.run()
    assert skipped.status == "skipped"
    assert (skipped.detail or "").startswith("Halted")

    automation.resume()
    assert automation.status().halted is None
    assert automation.state.book.drawdown == 0
    assert automation.run().status == "ok"


def test_half_the_limit_halves_the_strategy_and_the_rest_keeps_trading(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC, max_drawdown=0.1))
    held_after_first_run(automation, trading, 0.06)
    record = automation.run()
    assert record.status == "ok"
    assert record.plan is not None and record.plan.scales == {"btc-trend-200": 0.5}


def test_the_reduction_can_be_switched_off(tmp_path: Path, market: dict[str, str]) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(
        AutomationConfig(mode="dry-run", strategies=BTC, max_drawdown=0.1, reduce_at=1.0)
    )
    held_after_first_run(automation, trading, 0.06)
    record = automation.run()
    assert record.plan is not None and record.plan.scales == {}


def test_a_deposit_or_a_trade_by_hand_never_moves_the_drawdown(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC, max_drawdown=0.1))
    held_after_first_run(automation, trading, 0.0)
    trading.get_account.return_value = {**trading.get_account.return_value, "equity": "60000"}
    trading.get_all_positions.return_value = [
        *trading.get_all_positions.return_value,
        position("AAPL", 100, 1.0),
    ]
    assert automation.run().status == "ok"
    assert automation.state.book.drawdown == pytest.approx(0)


def test_a_small_drawdown_keeps_trading(tmp_path: Path, market: dict[str, str]) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC, max_drawdown=0.15))
    held_after_first_run(automation, trading, 0.03)
    assert automation.run().status == "ok"


def test_switching_automation_on_starts_the_books_afresh(
    tmp_path: Path, market: dict[str, str]
) -> None:
    automation = build(tmp_path)
    automation.state.book = Book(nav=0.8, peak=1.0)
    automation.state.books = {"btc-trend-200": Book(nav=0.8, peak=1.0)}
    automation.configure(AutomationConfig(mode="dry-run"))
    assert (automation.state.book, automation.state.books) == (Book(), {})


def test_a_strategy_no_longer_selected_holds_nothing_in_its_book(
    tmp_path: Path, market: dict[str, str]
) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    automation.run()
    assert automation.state.books["btc-trend-200"].holdings
    automation.configure(AutomationConfig(mode="dry-run", strategies=["halal-etf-trend"]))
    automation.run()
    assert automation.state.books["btc-trend-200"].holdings == {}
    assert automation.state.books["btc-trend-200"].capital == 0
    assert set(automation.state.books["halal-etf-trend"].holdings) == {"SPUS", "HLAL"}


def test_the_check_between_runs_closes_positions_once_live_prices_breach_the_limit(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(
        AutomationConfig(mode="paper", engine="alpaca", strategies=BTC, max_drawdown=0.1)
    )
    held_after_first_run(automation, trading, 0.05)
    asyncio.run(automation._caught_up())
    assert [run.status for run in automation.state.runs] == ["ok"]
    held_after_first_run_price = automation.state.books["btc-trend-200"].holdings["BTC/USD"]
    trading.get_all_positions.return_value = [
        position("BTCUSD", held_after_first_run_price.qty, held_after_first_run_price.price * 0.8)
    ]
    asyncio.run(automation._caught_up())
    assert [run.status for run in automation.state.runs] == ["ok", "halted"]


def test_state_survives_a_restart_and_keeps_the_newest_runs_first(
    tmp_path: Path, market: dict[str, str]
) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", allocation=0.3, strategies=BTC))
    automation.run()
    automation.run()
    again = build(tmp_path)
    assert again.state.config.allocation == 0.3
    assert again.state.config.strategies == BTC
    assert len(again.status().runs) == 2
    assert again.status().runs[0].time >= again.status().runs[1].time


def test_only_the_latest_runs_are_kept(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_RUNS", 3)
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    for _ in range(5):
        automation.run()
    assert len(automation.state.runs) == 3


def test_a_corrupt_or_old_state_file_falls_back_to_defaults(tmp_path: Path) -> None:
    (tmp_path / "automation.json").write_text("{not json")
    assert build(tmp_path).state.config.mode == "off"
    (tmp_path / "automation.json").write_text('{"config": {"strategy": "btc-trend-200"}}')
    assert build(tmp_path).state.config.strategies == ["halal-etf-trend"]


def test_overview_has_every_strategy_with_research_the_plan_and_the_status(
    tmp_path: Path, market: dict[str, str]
) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC, allocation=0.2))
    overview = automation.overview()
    assert [s.spec.name for s in overview.strategies] == [
        "btc-trend-200",
        "crypto-donchian",
        "halal-etf-trend",
        "halal-multi-trend",
        "halal-etf-rotation",
    ]
    assert [s.active for s in overview.strategies] == [True, False, False, False, False]
    assert all(s.research and len(s.research.checks) == 5 for s in overview.strategies)
    assert overview.plan is not None and overview.plan.allocation == 0.2
    assert overview.equity == 100_000
    assert overview.drawdown == 0.0
    assert overview.status.config.mode == "dry-run"


def test_overview_reports_the_drawdown_of_the_books_at_live_prices(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    held_after_first_run(automation, trading, 0.1)
    assert automation.overview().drawdown == pytest.approx(0.1)
    assert automation.state.book.drawdown == 0


def test_overview_keeps_going_when_one_strategy_has_no_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = fake_history(monkeypatch)
    fetch = history.daily_prices

    def sometimes_missing(
        stocks: object, crypto: object, symbols: list[str], years: float
    ) -> object:
        if "UMMA" in symbols:
            raise NotEnoughHistoryError("Alpaca has no daily bars for UMMA")
        return fetch(stocks, crypto, symbols, years).closes  # type: ignore[arg-type]

    serve(monkeypatch, sometimes_missing)
    overview = build(tmp_path).overview()
    rotation = next(s for s in overview.strategies if s.spec.name == "halal-etf-rotation")
    assert rotation.research is None
    assert "UMMA" in (rotation.error or "")
    assert real["default"] == "up"
    assert overview.plan is not None


def test_overview_explains_why_there_is_no_plan_when_trading_is_blocked(
    tmp_path: Path, market: dict[str, str]
) -> None:
    overview = build(tmp_path, trading_with(trading_blocked=True)).overview()
    assert overview.plan is None
    assert "blocked" in (overview.plan_error or "")
    assert overview.equity is None
    assert len(overview.strategies) == 5


def test_a_run_is_missed_when_none_happened_today_after_the_run_time(tmp_path: Path) -> None:
    automation = build(tmp_path)
    assert automation._missed_today() is (datetime.now(UTC).time() >= module.RUN_AT_TIME)


def test_a_run_today_is_not_missed(tmp_path: Path, market: dict[str, str]) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    automation.run()
    assert automation._missed_today() is False
    automation.state.runs[-1].time = datetime.now(UTC) - timedelta(days=1, hours=1)
    assert automation._missed_today() is (datetime.now(UTC).time() >= module.RUN_AT_TIME)


def test_a_failed_run_stays_missed_until_the_day_runs_out_of_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    automation = build(tmp_path)
    failure = RunRecord(
        time=datetime.now(UTC), strategies=BTC, mode="paper", status="error", detail="offline"
    )
    automation.state.runs = [failure] * (module.MAX_FAILURES - 1)
    assert automation._missed_today() is True
    automation.state.runs = [failure] * module.MAX_FAILURES
    assert automation._missed_today() is False
    automation.state.runs = [failure, failure.model_copy(update={"status": "skipped"})]
    assert automation._missed_today() is False


def test_the_catch_up_check_repeats_a_run_that_failed_offline_and_then_stops(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    trading.get_account.side_effect = ConnectionError("offline")
    asyncio.run(automation._caught_up())
    trading.get_account.side_effect = None
    asyncio.run(automation._caught_up())
    asyncio.run(automation._caught_up())
    assert [run.status for run in automation.state.runs] == ["error", "ok"]


def test_the_catch_up_check_leaves_a_run_in_progress_alone(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    with automation.lock:
        asyncio.run(automation._caught_up())
    assert automation.state.runs == []


def test_the_catch_up_check_is_scheduled_every_ten_minutes(tmp_path: Path) -> None:
    async def scenario() -> object:
        automation = build(tmp_path)
        automation.start()
        job = automation.scheduler.get_job(module.CATCH_UP_JOB_ID)
        automation.stop()
        return job.trigger.interval

    assert asyncio.run(scenario()) == module.CATCH_UP_EVERY


def test_the_scheduler_runs_daily_at_five_past_midnight_utc(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, object, object]:
        automation = build(tmp_path)
        automation.start()
        job = automation.scheduler.get_job(module.JOB_ID)
        off = automation.status().next_run
        automation.configure(AutomationConfig(mode="dry-run"))
        on = automation.status().next_run
        automation.stop()
        return str(job.trigger), off, on

    trigger, off, on = asyncio.run(scenario())
    assert "hour='0'" in trigger and "minute='5'" in trigger
    assert off is None
    assert isinstance(on, datetime) and (on.hour, on.minute) == (0, 5)


def test_changing_the_settings_deploys_straight_away_without_a_button(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "CATCH_UP_DELAY", timedelta(milliseconds=50))

    async def scenario() -> tuple[int, int]:
        automation = build(tmp_path)
        automation.start()
        automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
        await until(lambda: len(automation.state.runs) == 1)
        after_switching_on = len(automation.state.runs)
        automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
        await asyncio.sleep(0.3)
        unchanged = len(automation.state.runs)
        automation.configure(AutomationConfig(mode="dry-run", strategies=BTC, allocation=0.5))
        await until(lambda: len(automation.state.runs) == unchanged + 1)
        automation.stop()
        return after_switching_on, len(automation.state.runs) - unchanged

    first, after_change = asyncio.run(scenario())
    assert first == 1
    assert after_change == 1


def test_a_missed_run_is_caught_up_on_start(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "RUN_AT_TIME", datetime.min.time())
    monkeypatch.setattr(module, "CATCH_UP_DELAY", timedelta(milliseconds=50))

    async def scenario() -> int:
        automation = build(tmp_path)
        automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
        automation.start()
        await until(lambda: len(automation.state.runs) == 1)
        automation.stop()
        return len(automation.state.runs)

    assert asyncio.run(scenario()) == 1


def test_resuming_while_automation_is_on_redeploys(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "CATCH_UP_DELAY", timedelta(milliseconds=50))

    async def scenario() -> int:
        automation = build(tmp_path)
        automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
        automation.state.halted = "drawdown"
        automation.start()
        automation.resume()
        await until(lambda: len(automation.state.runs) >= 1)
        automation.stop()
        return len(automation.state.runs)

    assert asyncio.run(scenario()) >= 1


def test_the_caches_are_warmed_in_the_background_when_asked(
    tmp_path: Path, market: dict[str, str]
) -> None:
    async def scenario() -> int:
        automation = build(tmp_path, warm_cache=True)
        automation.start()
        await until(lambda: len(research.RESULTS) == 4)
        automation.stop()
        return len(research.RESULTS)

    assert asyncio.run(scenario()) == 4


def test_warming_skips_strategies_without_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def none(*_: object) -> pd.DataFrame:
        raise NotEnoughHistoryError("no bars")

    serve(monkeypatch, none)
    build(tmp_path)._warm()
    assert len(research.RESULTS) == 0


def test_the_scheduled_job_does_nothing_while_off(tmp_path: Path) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    asyncio.run(automation._scheduled())
    trading.get_account.assert_not_called()


def passing_checks(monkeypatch: pytest.MonkeyPatch, passing: set[str]) -> None:
    """Make exactly the named strategies pass all four research checks."""
    real = research.research

    def fake(stocks: object, crypto: object, spec: object, *args: object, **_: object) -> object:
        result = real(stocks, crypto, spec, *args)  # type: ignore[arg-type]
        checks = [Check(name=c.name, passed=result.strategy.name in passing) for c in result.checks]
        return result.model_copy(update={"checks": checks})

    monkeypatch.setattr(research, "research", fake)


def test_automatic_mode_runs_every_non_crypto_strategy_that_passes_its_checks(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"btc-trend-200", "halal-etf-trend"})
    automation = build(tmp_path)
    assert automation.selected(AutomationConfig()) == ["halal-etf-trend"]


class Stages:
    """A lifecycle where the given strategies trade at the given shares of their capital."""

    def __init__(self, capital: dict[str, float]) -> None:
        self.shares = capital

    def trading(self) -> list[str]:
        return list(self.shares)

    def capital(self, name: str) -> float:
        return self.shares.get(name, 0.0)


def test_with_a_lifecycle_automatic_mode_trades_its_canary_and_live_strategies(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-rotation"})
    automation = build(tmp_path)
    automation.lifecycle = Stages({"halal-etf-trend": 0.1, "btc-trend-200": 1.0})

    assert automation.selected(AutomationConfig()) == ["halal-etf-trend"]
    assert automation.selected(AutomationConfig(crypto=True)) == [
        "btc-trend-200",
        "halal-etf-trend",
    ]


def test_a_canary_is_planned_at_its_share_of_the_capital(
    tmp_path: Path, market: dict[str, str]
) -> None:
    automation = build(tmp_path)
    automation.lifecycle = Stages({"btc-trend-200": 0.1})
    automation.configure(AutomationConfig(mode="dry-run", crypto=True))

    plan = automation.plan(["btc-trend-200"], 0.5)

    assert plan.scales == {"btc-trend-200": 0.1}
    automation.configure(AutomationConfig(mode="dry-run", automatic=False, strategies=BTC))
    assert automation.plan(["btc-trend-200"], 0.5).scales == {}


def test_automatic_mode_includes_crypto_only_when_allowed(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"btc-trend-200", "halal-etf-trend"})
    automation = build(tmp_path)
    assert automation.selected(AutomationConfig(crypto=True)) == [
        "btc-trend-200",
        "halal-etf-trend",
    ]


def test_automatic_mode_selects_nothing_when_no_strategy_passes(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, set())
    assert build(tmp_path).selected(AutomationConfig(crypto=True)) == []


def test_automatic_mode_skips_a_strategy_without_history(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend", "halal-etf-rotation"})
    real = research.research

    def broken(stocks: object, crypto: object, spec: object, *args: object, **kw: object) -> object:
        if spec.name == "halal-etf-rotation":  # type: ignore[attr-defined]
            raise NotEnoughHistoryError("too short")
        return real(stocks, crypto, spec, *args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(research, "research", broken)
    assert build(tmp_path).selected(AutomationConfig()) == ["halal-etf-trend"]


def test_manual_mode_keeps_exactly_the_listed_strategies(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, set())
    config = AutomationConfig(strategies=["crypto-donchian"])
    assert build(tmp_path).selected(config) == ["crypto-donchian"]


def test_listing_strategies_without_saying_automatic_means_manual() -> None:
    assert AutomationConfig(strategies=BTC).automatic is False
    assert AutomationConfig(strategies=BTC, automatic=True).automatic is True
    assert AutomationConfig().automatic is True


def test_a_run_records_and_trades_the_automatically_chosen_strategies(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend"})
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="paper", engine="alpaca"))
    record = automation.run()
    assert record.strategies == ["halal-etf-trend"]
    assert {e.symbol for e in record.executed} == {"SPUS", "HLAL"}


def test_overview_marks_the_automatically_chosen_strategies_active(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend", "halal-etf-rotation"})
    overview = build(tmp_path).overview()
    assert [(s.spec.name, s.active) for s in overview.strategies] == [
        ("btc-trend-200", False),
        ("crypto-donchian", False),
        ("halal-etf-trend", True),
        ("halal-multi-trend", False),
        ("halal-etf-rotation", True),
    ]
    assert overview.plan is not None
    assert overview.plan.strategies == ["halal-etf-trend", "halal-etf-rotation"]


def test_every_run_is_written_to_the_audit_log_with_what_it_saw_and_did(
    tmp_path: Path, market: dict[str, str]
) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))

    record = automation.run()

    entries = automation.audit.entries()
    assert [e.kind for e in entries] == ["settings-change", "run"]
    assert entries[0].payload["area"] == "autopilot"
    payload = entries[1].payload
    assert payload["run"]["status"] == record.status == "ok"
    assert payload["manifest"]["config"]["strategies"] == BTC
    assert payload["manifest"]["data"]["btc-trend-200"]["rows"] > 250
    assert payload["compliance"] == {
        "short_positions": [],
        "negative_cash": False,
        "compliant": True,
    }
    assert automation.audit.verify().valid


def test_the_audit_flags_a_short_position_or_borrowed_cash(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with([position("AAPL", -5, 200.0)], cash="-50")
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))

    automation.run()

    compliance = automation.audit.entries()[-1].payload["compliance"]
    assert compliance == {
        "short_positions": ["AAPL"],
        "negative_cash": True,
        "compliant": False,
    }


def test_a_fault_in_the_audit_log_never_stops_a_run(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run", strategies=BTC))
    monkeypatch.setattr(automation.audit, "record", MagicMock(side_effect=OSError("disk full")))

    assert automation.run().status == "ok"


class Cut:
    def __init__(self, **scales: float) -> None:
        self.scales = scales
        self.asked: list[str] = []

    def scale(self, name: str) -> float:
        self.asked.append(name)
        return self.scales.get(name, 1.0)


def test_the_plan_asks_the_sizing_guard_about_each_selected_strategy(
    tmp_path: Path, market: dict[str, str]
) -> None:
    names: list[StrategyName] = ["btc-trend-200"]
    automation = build(tmp_path)
    plain = automation.plan(names, 0.2)
    guard = Cut(**{"btc-trend-200": 0.5})
    automation.sizing = guard
    cut = automation.plan(names, 0.2)
    assert guard.asked == names
    assert cut.targets[0].target_value == pytest.approx(plain.targets[0].target_value / 2)
    assert cut.scales == {"btc-trend-200": 0.5} and plain.scales == {}


def test_without_a_sizing_guard_nothing_is_cut(tmp_path: Path, market: dict[str, str]) -> None:
    assert build(tmp_path).plan(["btc-trend-200"], 0.2).scales == {}


def filled_coin(price: str = "100.0") -> dict[str, object]:
    return {
        "id": "order-BTC/USD",
        "symbol": "BTCUSD",
        "side": "buy",
        "type": "market",
        "status": "filled",
        "qty": "1",
        "filled_qty": "1",
        "filled_avg_price": price,
        "submitted_at": "2026-11-04T00:05:00Z",
        "filled_at": "2026-11-04T00:05:01Z",
    }


def test_each_order_sent_keeps_the_decision_price_of_its_plan(
    tmp_path: Path, market: dict[str, str]
) -> None:
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="paper", engine="alpaca", strategies=BTC))
    record = automation.run()
    assert record.plan is not None
    assert [e.price for e in record.executed] == [o.price for o in record.plan.orders]
    assert record.executed[0].price is not None


def test_the_next_run_records_how_the_last_orders_filled(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="paper", engine="alpaca", strategies=BTC))
    first = automation.run()
    decided = first.executed[0].price
    assert decided is not None
    trading.get_order_by_client_id.return_value = filled_coin(str(decided * 1.001))
    automation.run()
    item = automation.state.runs[0].executed[0]
    assert item.settled and item.filled_price == pytest.approx(decided * 1.001)
    assert item.cost_bps == pytest.approx(10 + 25)
    [summary] = automation.overview().execution
    assert (summary.asset, summary.orders, summary.assumed_bps) == ("crypto", 1, 25)


def test_the_catch_up_check_settles_finished_orders_between_runs(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="paper", engine="alpaca", strategies=BTC))
    automation.run()
    trading.get_order_by_client_id.return_value = filled_coin()
    asyncio.run(automation._caught_up())
    assert automation.state.runs[0].executed[0].settled
    assert build(tmp_path).state.runs[0].executed[0].settled


def test_the_execution_report_lists_every_settled_fill(
    tmp_path: Path, market: dict[str, str]
) -> None:
    from app.features.systematic import execution

    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="paper", engine="alpaca", strategies=BTC))
    automation.run()
    trading.get_order_by_client_id.return_value = filled_coin()
    automation.settle_fills()
    report = execution.report(automation.state.runs)
    assert [fill.symbol for fill in report.fills] == ["BTC/USD"]


def test_the_first_judgement_admits_what_passes_today(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend"})
    automation = build(tmp_path)
    automation.judge(automation.search(), date(2026, 11, 5))
    assert automation.state.active == ["halal-etf-trend"]
    assert automation.selected(AutomationConfig()) == ["halal-etf-trend"]


def test_the_first_judgement_keeps_what_the_last_run_traded_even_if_it_now_fails(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing: set[str] = {"halal-etf-trend"}
    passing_checks(monkeypatch, passing)
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run"))
    automation.run()
    automation.state.active, automation.state.judged = None, None
    passing.clear()
    automation.judge(automation.search(), date(2026, 11, 5))
    assert automation.state.active == ["halal-etf-trend"]


def test_a_strategy_leaves_only_after_failing_ten_days_in_a_row(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing: set[str] = {"halal-etf-trend"}
    passing_checks(monkeypatch, passing)
    automation = build(tmp_path)
    search = automation.search()
    start = date(2026, 11, 5)
    automation.judge(search, start)
    passing.clear()
    for day in range(1, module.EXIT_DAYS):
        automation.judge(search, start + timedelta(days=day))
        assert automation.selected(AutomationConfig()) == ["halal-etf-trend"]
    automation.judge(search, start + timedelta(days=module.EXIT_DAYS))
    assert automation.selected(AutomationConfig()) == []
    assert automation.state.streaks["halal-etf-trend"] == -module.EXIT_DAYS


def test_a_pass_in_between_resets_the_count(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing: set[str] = {"halal-etf-trend"}
    passing_checks(monkeypatch, passing)
    automation = build(tmp_path)
    search = automation.search()
    start = date(2026, 11, 5)
    automation.judge(search, start)
    passing.clear()
    for day in range(1, 6):
        automation.judge(search, start + timedelta(days=day))
    passing.add("halal-etf-trend")
    automation.judge(search, start + timedelta(days=6))
    passing.clear()
    for day in range(7, 7 + module.EXIT_DAYS - 1):
        automation.judge(search, start + timedelta(days=day))
    assert automation.selected(AutomationConfig()) == ["halal-etf-trend"]


def test_a_strategy_joins_only_after_passing_ten_days_in_a_row(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing: set[str] = set()
    passing_checks(monkeypatch, passing)
    automation = build(tmp_path)
    search = automation.search()
    start = date(2026, 11, 5)
    automation.judge(search, start)
    passing.add("halal-etf-rotation")
    for day in range(1, module.ENTER_DAYS):
        automation.judge(search, start + timedelta(days=day))
    assert automation.selected(AutomationConfig()) == []
    automation.judge(search, start + timedelta(days=module.ENTER_DAYS))
    assert automation.selected(AutomationConfig()) == ["halal-etf-rotation"]


def test_judging_twice_in_a_day_counts_once(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing: set[str] = {"halal-etf-trend"}
    passing_checks(monkeypatch, passing)
    automation = build(tmp_path)
    search = automation.search()
    start = date(2026, 11, 5)
    automation.judge(search, start)
    passing.clear()
    for _ in range(module.EXIT_DAYS + 2):
        automation.judge(search, start + timedelta(days=1))
    assert automation.state.streaks["halal-etf-trend"] == -1
    assert automation.selected(AutomationConfig()) == ["halal-etf-trend"]


def test_a_run_judges_and_the_judgement_survives_a_restart(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend"})
    automation = build(tmp_path)
    automation.configure(AutomationConfig(mode="dry-run"))
    assert automation.run().strategies == ["halal-etf-trend"]
    again = build(tmp_path).state
    assert again.active == ["halal-etf-trend"]
    assert again.judged == datetime.now(UTC).date()


def test_the_gate_counts_every_rule_and_variant_including_new_ones_the_lab_tried(
    tmp_path: Path, market: dict[str, str]
) -> None:
    from app.features.systematic import lab

    automation = build(tmp_path)
    defined = len(STRATEGIES) + sum(
        len(lab.variants(spec)) - 1 for spec in STRATEGIES.values() if lab.variants(spec)
    )
    assert automation.search().trials == defined == 24
    labels = ["200-day average", "300-day average"]
    automation.audit.record(
        "lab",
        {
            "result": {
                "families": [
                    {
                        "family": "halal-etf-trend",
                        "candidates": [{"label": label, "sharpe": 1.0} for label in labels],
                    }
                ]
            }
        },
    )
    assert automation.search().trials == defined + 1
    trend = next(s for s in automation.overview().strategies if s.spec.name == "halal-etf-trend")
    assert trend.research is not None
    assert f"({defined + 1} trials)" in trend.research.checks[-1].name


def test_new_market_history_makes_research_run_again(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.quant import proxy

    automation = build(tmp_path)
    research.RESULTS[("x", 1, 1.0, 4, None)] = None  # type: ignore[assignment]
    monkeypatch.setattr(proxy, "refresh", lambda: False)
    automation._refresh_proxy()
    assert len(research.RESULTS) == 1
    monkeypatch.setattr(proxy, "refresh", lambda: True)
    automation._refresh_proxy()
    assert len(research.RESULTS) == 0


def test_the_market_history_is_refreshed_daily(tmp_path: Path) -> None:
    async def scenario() -> object:
        automation = build(tmp_path)
        automation.start()
        job = automation.scheduler.get_job(module.PROXY_JOB_ID)
        automation.stop()
        return job

    job = asyncio.run(scenario())
    assert job is not None
    assert job.trigger.interval == timedelta(hours=24)  # type: ignore[attr-defined]


def test_in_defensive_the_autopilot_only_sells_and_the_status_says_so(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(
        AutomationConfig(mode="dry-run", strategies=["halal-etf-trend"], max_drawdown=0.1)
    )
    automation.run()
    book = automation.state.books["halal-etf-trend"]
    spus = book.holdings["SPUS"]
    automation.state.books["halal-etf-trend"] = book.model_copy(update={"holdings": {"SPUS": spus}})
    trading.get_all_positions.return_value = [position("SPUS", spus.qty, spus.price * 0.84)]
    record = automation.run()
    assert automation.kill_state() == "defensive"
    assert record.plan is not None
    assert all(order.side == "sell" for order in record.plan.orders)
    assert "HLAL" not in automation.state.books["halal-etf-trend"].holdings
    assert automation.status().kill_state == "defensive"


def test_the_kill_switch_reads_normal_while_off_and_flatten_while_halted(tmp_path: Path) -> None:
    automation = build(tmp_path)
    automation.state.halted = "test"
    assert automation.kill_state() == "normal"
    automation.state.config = AutomationConfig(mode="paper")
    assert automation.kill_state() == "flatten"


class Agents:
    """Stands in for the agent desk: an ensemble with a target and a record."""

    def __init__(self, eligible: bool) -> None:
        self.eligible = eligible

    def ensemble_eligible(self) -> bool:
        return self.eligible

    def ensemble_target(self) -> dict[str, float] | None:
        return {"SPUS": 0.6, "HLAL": 0.3}

    def ensemble_returns(self) -> pd.Series | None:
        index = pd.bdate_range("2025-01-01", periods=300)
        return pd.Series(np.random.default_rng(2).normal(0.0004, 0.008, 300), index=index)


def test_the_ensemble_trades_only_after_passing_the_gates_ten_days_in_a_row(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend"})
    automation = build(tmp_path)
    automation.agents = Agents(eligible=True)
    search = automation.search()
    start = date(2026, 11, 5)
    automation.state.active = ["halal-etf-trend"]
    automation.state.judged = start
    for day in range(1, module.ENTER_DAYS):
        automation.judge(search, start + timedelta(days=day))
    config = AutomationConfig(mode="dry-run", allocation=0.5)
    automation.state.config = config
    assert not automation.ensemble_admitted(config)
    automation.judge(search, start + timedelta(days=module.ENTER_DAYS))
    assert automation.ensemble_admitted(config)
    plan = automation.plan(["halal-etf-trend"], 0.5)
    assert set(plan.books) == {"halal-etf-trend", "ai-ensemble"}
    assert plan.signals["ai-ensemble"] == {"HLAL": 0.3, "SPUS": 0.6}
    assert sum(book.capital for book in plan.books.values()) == pytest.approx(50_000)
    assert not automation.ensemble_admitted(config.model_copy(update={"automatic": False}))


def test_an_ensemble_that_fails_the_gates_never_trades(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend"})
    automation = build(tmp_path)
    automation.agents = Agents(eligible=False)
    automation.configure(AutomationConfig(mode="dry-run"))
    record = automation.run()
    assert record.plan is not None and "ai-ensemble" not in record.plan.books
    assert automation.overview().ensemble_active is False


def test_the_manual_kill_switch_sells_what_the_autopilot_holds_and_halts(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = build(tmp_path, trading)
    automation.configure(AutomationConfig(mode="paper", engine="alpaca", strategies=BTC))
    automation.run()
    held = automation.state.books["btc-trend-200"].holdings["BTC/USD"]
    trading.get_all_positions.return_value = [position("BTCUSD", held.qty, held.price)]
    record = automation.halt("testing the switch")
    assert record.status == "halted"
    assert [o.side for o in record.executed] == ["sell"]
    assert automation.kill_state() == "flatten"
    assert (automation.status().halted or "").endswith("testing the switch")
    assert automation.state.books["btc-trend-200"].holdings == {}
    assert automation.run().status == "skipped"


def test_a_paused_strategy_is_left_out_in_either_mode(
    tmp_path: Path, market: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    passing_checks(monkeypatch, {"halal-etf-trend", "halal-etf-rotation"})
    automation = build(tmp_path)
    paused = AutomationConfig(paused=["halal-etf-trend"])
    assert automation.selected(paused) == ["halal-etf-rotation"]
    manual = AutomationConfig(
        strategies=["halal-etf-trend", "btc-trend-200"], paused=["halal-etf-trend"]
    )
    assert automation.selected(manual) == ["btc-trend-200"]
