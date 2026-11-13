from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pandas as pd
import pytest
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

from app.core.sleeve import Criterion, Holding, Sleeve
from app.features.systematic import rebalance
from app.features.systematic.automation import Automation
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.rebalance import SleeveRequest
from app.features.systematic.schemas import (
    AutomationConfig,
    Plan,
    SleeveConfig,
    StrategyName,
)
from app.features.systematic.tests.conftest import FakeLink
from app.features.systematic.tests.support import (
    SETTINGS,
    fake_history,
    position,
    trading_with,
    uptrend,
)

TREND: StrategyName = "halal-etf-trend"
CAPITAL = 10_000.0
SHARE = 0.2
PASSED = [Criterion(name="Sharpe above buy and hold", passed=True)]
FAILED = [Criterion(name="Sharpe above buy and hold", passed=False)]


def closes() -> dict[StrategyName, pd.DataFrame]:
    return {TREND: uptrend(STRATEGIES[TREND].universe)}


def holding(symbol: str, weight: float, blocked: str | None = None) -> Holding:
    return Holding(symbol=symbol, weight=weight, price=100.0, name=f"{symbol} Inc", blocked=blocked)


def offer(
    *holdings: Holding, criteria: list[Criterion] = PASSED, problems: list[str] | None = None
) -> Sleeve:
    return Sleeve(
        variant="rank_avg top 20",
        session=date(2026, 10, 30),
        holdings=list(holdings),
        criteria=criteria,
        problems=problems or [],
    )


def request(
    offered: Sleeve | None,
    mode: str = "trade",
    held: tuple[str, ...] = (),
    override: bool = False,
    share: float = SHARE,
) -> SleeveRequest:
    config = SleeveConfig.model_validate(
        {"mode": mode, "share": share, "override": override, "variant": "rank_avg top 20"}
    )
    return SleeveRequest(config, offered, held)


def plan_with(scanner: SleeveRequest | None, positions: list[dict[str, Any]] | None = None) -> Plan:
    return rebalance.build_plan(trading_with(positions), closes(), 0.1, True, scanner=scanner)


def strategy_total(plan: Plan) -> float:
    return sum(item.target_value for item in plan.targets if item.source == "strategy")


def test_with_the_sleeve_off_the_plan_is_exactly_what_it_was() -> None:
    base = plan_with(None)
    off = plan_with(request(None, mode="off", share=0.0))
    assert base.sleeve is None
    assert off.model_dump(exclude={"sleeve"}) == base.model_dump(exclude={"sleeve"})
    assert off.sleeve is not None and not off.sleeve.targets and not off.sleeve.orders
    assert all(item.source == "strategy" for item in base.targets)
    assert all(item.source == "strategy" for item in base.orders)


def test_a_trading_sleeve_takes_its_share_of_the_allocation_before_the_strategies() -> None:
    plan = plan_with(request(offer(holding("AAA", 0.6), holding("BBB", 0.4))))
    values = {item.symbol: item.target_value for item in plan.targets if item.source == "sleeve"}
    assert strategy_total(plan) == pytest.approx(CAPITAL * (1 - SHARE))
    assert values == pytest.approx({"AAA": CAPITAL * SHARE * 0.6, "BBB": CAPITAL * SHARE * 0.4})
    orders = {item.symbol: item for item in plan.orders if item.source == "sleeve"}
    assert orders["AAA"].side == "buy" and orders["AAA"].qty == pytest.approx(12.0)
    assert plan.sleeve is not None and not plan.sleeve.preview and plan.sleeve.passed


def test_the_sleeves_orders_are_fractional_quantities() -> None:
    plan = plan_with(request(offer(Holding(symbol="AAA", weight=1.0, price=333.0))))
    [order] = [item for item in plan.orders if item.source == "sleeve"]
    assert order.qty == pytest.approx(2000 / 333.0, abs=1e-6) and order.qty != int(order.qty)


def test_a_record_that_has_not_passed_keeps_the_sleeve_closed_and_the_strategies_whole() -> None:
    base = plan_with(None)
    plan = plan_with(request(offer(holding("AAA", 1.0), criteria=FAILED)))
    assert plan.model_dump(exclude={"sleeve"}) == base.model_dump(exclude={"sleeve"})
    assert plan.sleeve is not None and not plan.sleeve.passed and not plan.sleeve.orders
    assert plan.sleeve.problems == ["The promotion checks have not passed"]
    assert plan.sleeve.criteria == FAILED


def test_an_override_trades_a_failed_record_and_says_so() -> None:
    plan = plan_with(request(offer(holding("AAA", 1.0), criteria=FAILED), override=True))
    assert [item.symbol for item in plan.orders if item.source == "sleeve"] == ["AAA"]
    assert plan.sleeve is not None and plan.sleeve.override and not plan.sleeve.passed


def test_in_shadow_the_sleeve_shows_its_orders_and_places_none_of_them() -> None:
    base = plan_with(None)
    plan = plan_with(request(offer(holding("AAA", 0.6), holding("BBB", 0.4)), mode="shadow"))
    assert plan.model_dump(exclude={"sleeve"}) == base.model_dump(exclude={"sleeve"})
    assert plan.sleeve is not None and plan.sleeve.preview
    assert [(item.symbol, item.side) for item in plan.sleeve.orders] == [
        ("AAA", "buy"),
        ("BBB", "buy"),
    ]


def test_only_three_new_names_are_started_in_one_run() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    plan = plan_with(request(offer(*[holding(name, 0.2) for name in names])))
    assert [item.symbol for item in plan.orders if item.source == "sleeve"] == names[:3]


def test_names_already_held_do_not_count_as_new() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    held = [position("AAA", 2000 / 6 / 100, 100.0), position("BBB", 2000 / 6 / 100, 100.0)]
    plan = plan_with(request(offer(*[holding(name, 1 / 6) for name in names])), held)
    assert [item.symbol for item in plan.orders if item.source == "sleeve"] == ["CCC", "DDD", "EEE"]


def test_a_name_with_a_setback_gets_no_new_buy_and_no_addition() -> None:
    offered = offer(holding("AAA", 0.5, "Guidance cut"), holding("BBB", 0.5, "Guidance cut"))
    plan = plan_with(request(offered), [position("BBB", 2.0, 100.0)])
    assert [item.symbol for item in plan.orders if item.source == "sleeve"] == []
    rows = {item.symbol: item.target_value for item in plan.targets if item.source == "sleeve"}
    assert rows == pytest.approx({"BBB": 200.0})


def test_names_the_sleeve_is_not_trading_are_not_in_its_targets() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    plan = plan_with(request(offer(*[holding(name, 0.2) for name in names])))
    assert plan.sleeve is not None
    assert [item.symbol for item in plan.sleeve.targets] == names[:3]


def test_a_name_with_a_setback_can_still_be_trimmed_to_its_target() -> None:
    offered = offer(holding("BBB", 0.5, "Guidance cut"))
    plan = plan_with(request(offered), [position("BBB", 50.0, 100.0)])
    [order] = [item for item in plan.orders if item.source == "sleeve"]
    assert (order.symbol, order.side) == ("BBB", "sell") and order.notional == pytest.approx(4000)


def test_a_stale_scan_freezes_the_sleeve_in_place_and_keeps_its_capital_reserved() -> None:
    base = plan_with(None)
    stale = offer(
        holding("AAA", 1.0), problems=["The data was last refreshed more than 36 hours ago"]
    )
    plan = plan_with(request(stale, held=("CCC",)), [position("CCC", 5.0, 100.0)])
    assert [item for item in plan.orders if item.source == "sleeve"] == []
    assert strategy_total(plan) == pytest.approx(strategy_total(base) * (1 - SHARE))
    assert plan.sleeve is not None and plan.sleeve.problems == stale.problems


def test_without_a_scan_the_sleeve_does_nothing_and_the_strategies_are_untouched() -> None:
    held = [position("CCC", 5.0, 100.0)]
    base, plan = plan_with(None, held), plan_with(request(None), held)
    assert plan.model_dump(exclude={"sleeve"}) == base.model_dump(exclude={"sleeve"})
    assert plan.sleeve is not None
    assert plan.sleeve.problems == ["There is no scan to take holdings from"]


def test_turning_the_sleeve_off_sells_what_it_bought_and_nothing_else() -> None:
    held = [position("AAA", 5.0, 100.0), position("ZZZ", 3.0, 100.0)]
    plan = plan_with(request(None, mode="off", held=("AAA",)), held)
    sold = [item for item in plan.orders if item.source == "sleeve"]
    assert [(item.symbol, item.side, item.qty) for item in sold] == [("AAA", "sell", 5.0)]


def test_a_sleeve_in_shadow_leaves_what_it_bought_alone() -> None:
    plan = plan_with(
        request(offer(holding("AAA", 1.0)), mode="shadow", held=("AAA",)),
        [position("AAA", 20.0, 100.0)],
    )
    assert [item for item in plan.orders if item.source == "sleeve"] == []


def test_a_drawdown_halt_sells_the_sleeve_too_whatever_its_mode() -> None:
    held = [position("AAA", 5.0, 100.0)]
    flat = rebalance.build_plan(
        trading_with(held),
        closes(),
        0.1,
        True,
        flat=True,
        scanner=request(offer(holding("AAA", 1.0)), mode="shadow", held=("AAA",)),
    )
    [order] = [item for item in flat.orders if item.source == "sleeve"]
    assert (order.symbol, order.side) == ("AAA", "sell")
    assert flat.sleeve is not None and not flat.sleeve.preview


def test_a_name_a_strategy_already_targets_is_left_to_the_strategy() -> None:
    plan = plan_with(request(offer(holding("SPUS", 0.5), holding("AAA", 0.5))))
    assert {item.symbol for item in plan.targets if item.source == "sleeve"} == {"AAA"}


def test_part_keeps_only_the_targets_and_orders_of_one_source() -> None:
    plan = plan_with(request(offer(holding("AAA", 1.0))))
    sleeve, strategy = plan.part("sleeve"), plan.part("strategy")
    assert {item.source for item in sleeve.targets} == {"sleeve"}
    assert {item.source for item in sleeve.orders} == {"sleeve"}
    assert {item.source for item in strategy.targets} == {"strategy"}
    assert {item.source for item in strategy.orders} == {"strategy"}
    assert len(sleeve.orders) + len(strategy.orders) == len(plan.orders)


class Provider:
    def __init__(self, offered: Sleeve | None, fault: bool = False) -> None:
        self.offered = offered
        self.fault = fault
        self.asked: list[str] = []

    def sleeve(self, variant: str) -> Sleeve | None:
        self.asked.append(variant)
        if self.fault:
            raise RuntimeError("scanner broke")
        return self.offered


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    return fake_history(monkeypatch)


def automation_with(
    tmp_path: Path, provider: Provider | None, trading: MagicMock | None = None
) -> Automation:
    found = Automation(
        SETTINGS,
        trading or trading_with(),
        create_autospec(StockHistoricalDataClient, instance=True),
        create_autospec(CryptoHistoricalDataClient, instance=True),
        state_path=tmp_path / "automation.json",
    )
    found.sleeve = provider
    return found


def configured(share: float = SHARE, mode: str = "trade", **sleeve: Any) -> AutomationConfig:
    return AutomationConfig.model_validate(
        {
            "mode": "paper",
            "engine": "nautilus",
            "strategies": ["btc-trend-200"],
            "sleeve": {"mode": mode, "share": share, **sleeve},
        }
    )


def test_the_sleeve_is_written_as_a_target_for_the_nodes_arbiter(
    tmp_path: Path, market: dict[str, str], link: FakeLink
) -> None:
    trading = trading_with()
    automation = automation_with(tmp_path, Provider(offer(holding("AAA", 1.0))), trading)
    automation.configure(configured())
    record = automation.run()
    assert record.status == "ok"
    [rows] = link.written
    [sleeve] = [row for row in rows if row["strategy"] == "scanner-sleeve"]
    assert set(sleeve["weights"]) == {"AAA"} and sleeve["source"] == "api"
    assert automation.state.sleeve_managed == ["AAA"] and automation.state.managed == ["BTC/USD"]


def test_the_plan_cancels_the_queued_orders_of_the_names_the_sleeve_trades_and_no_others(
    tmp_path: Path, market: dict[str, str]
) -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    trading = trading_with()
    provider = Provider(offer(*[holding(name, 0.2) for name in names]))
    automation = automation_with(tmp_path, provider, trading)
    automation.configure(configured())
    record = automation.run()
    assert record.plan is not None
    targets = {target.symbol for target in record.plan.targets}
    assert "AAA" in targets and "EEE" not in targets


def test_a_shadow_run_places_nothing_for_the_sleeve_and_remembers_nothing(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = automation_with(tmp_path, Provider(offer(holding("AAA", 1.0))), trading)
    automation.configure(configured(mode="shadow").model_copy(update={"engine": "alpaca"}))
    record = automation.run()
    assert record.plan is not None
    symbols = [order.symbol for order in record.plan.orders]
    assert "AAA" not in symbols and automation.state.sleeve_managed == []
    assert record.plan is not None and record.plan.sleeve is not None and record.plan.sleeve.preview


def test_turning_the_sleeve_off_sells_what_it_bought_on_the_next_run(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with([position("AAA", 12.0, 100.0)])
    automation = automation_with(tmp_path, Provider(offer(holding("AAA", 1.0))), trading)
    automation.state.sleeve_managed = ["AAA"]
    automation.configure(configured(mode="off", share=0.0).model_copy(update={"engine": "alpaca"}))
    record = automation.run()
    assert record.plan is not None
    sells = [order for order in record.plan.orders if order.symbol == "AAA"]
    assert [(order.side, order.qty) for order in sells] == [("sell", 12.0)]


def test_a_fault_in_the_scanner_never_stops_the_strategies(
    tmp_path: Path, market: dict[str, str]
) -> None:
    trading = trading_with()
    automation = automation_with(tmp_path, Provider(None, fault=True), trading)
    automation.configure(configured().model_copy(update={"engine": "alpaca"}))
    record = automation.run()
    assert record.status == "ok" and [e.symbol for e in record.executed] == ["BTC/USD"]
    assert record.plan is not None and record.plan.sleeve is not None
    assert record.plan.sleeve.problems == ["There is no scan to take holdings from"]


def test_the_sleeve_is_not_asked_for_while_it_is_off_and_holds_nothing(
    tmp_path: Path, market: dict[str, str]
) -> None:
    provider = Provider(offer(holding("AAA", 1.0)))
    automation = automation_with(tmp_path, provider)
    automation.configure(configured(mode="off", share=0.0))
    automation.overview()
    assert provider.asked == [] and automation.state.sleeve_managed == []


def test_the_variant_in_the_settings_is_the_one_asked_for(
    tmp_path: Path, market: dict[str, str]
) -> None:
    provider = Provider(offer(holding("AAA", 1.0)))
    automation = automation_with(tmp_path, provider)
    automation.configure(configured(mode="shadow", variant="mom_12_1 top 10"))
    automation.overview()
    assert provider.asked and set(provider.asked) == {"mom_12_1 top 10"}


def test_the_sleeve_settings_survive_a_restart_and_an_old_state_file_gets_the_defaults(
    tmp_path: Path, market: dict[str, str]
) -> None:
    first = automation_with(tmp_path, None)
    first.configure(configured(share=0.15, override=True))
    saved = automation_with(tmp_path, None).state.config.sleeve
    assert (saved.mode, saved.share, saved.override) == ("trade", 0.15, True)
    (tmp_path / "automation.json").write_text('{"config": {"mode": "off"}, "managed": ["SPUS"]}')
    older = automation_with(tmp_path, None).state
    assert older.config.sleeve == SleeveConfig() and older.sleeve_managed == []
    assert older.managed == ["SPUS"]


def test_the_share_of_the_sleeve_cannot_pass_a_quarter_of_the_allocation() -> None:
    with pytest.raises(ValueError, match="less than or equal to 0.25"):
        SleeveConfig(share=0.3)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        SleeveConfig(share=-0.1)
