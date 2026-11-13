from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest

from app.features.systematic import research
from app.features.systematic.automation import Automation
from app.features.systematic.schemas import AutomationConfig, ExecutedOrder, Plan
from app.features.systematic.targets import PlatformLink


@pytest.fixture(autouse=True)
def fresh_caches() -> Iterator[None]:
    research.clear()
    yield
    research.clear()


class FakeLink(PlatformLink):
    """Stands in for Postgres: keeps the settings published and the targets written."""

    def __init__(self) -> None:
        self.configs: list[AutomationConfig] = []
        self.written: list[list[dict[str, Any]]] = []
        self.last_plan: Plan | None = None
        self.sent: list[Plan] = []

    def executed(self, after: datetime) -> list[ExecutedOrder]:
        """The arbiter's part, played straight: the plan's own orders, accepted, once each."""
        plan = self.last_plan
        if plan is None or plan in self.sent:
            return []
        self.sent.append(plan)
        return [
            ExecutedOrder(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                order_id=f"order-{order.symbol}",
                status="accepted",
            )
            for order in plan.orders
        ]

    def publish(self, config: AutomationConfig, managed: list[str]) -> None:
        self.configs.append(config)
        self.managed = managed

    def write(self, rows: list[dict[str, Any]]) -> None:
        self.written.append(rows)


@pytest.fixture(autouse=True)
def link(monkeypatch: pytest.MonkeyPatch) -> FakeLink:
    """Every autopilot built in a test talks to this link instead of the database."""
    fake = FakeLink()
    original = Automation.__init__

    def init(self: Automation, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self.link = fake

    planned = Automation.plan

    def plan(self: Automation, *args: Any, **kwargs: Any) -> Plan:
        found = planned(self, *args, **kwargs)
        fake.last_plan = found
        return found

    monkeypatch.setattr(Automation, "__init__", init)
    monkeypatch.setattr(Automation, "plan", plan)
    return fake
