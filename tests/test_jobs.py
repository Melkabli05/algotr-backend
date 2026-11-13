import re
import tomllib
from pathlib import Path

import pytest

from app import jobs

ROOT = Path(__file__).resolve().parent.parent
CALL = re.compile(r"app\.jobs\s+([a-z][a-z-]*)")


def commands_named_in_the_repository() -> set[str]:
    found = set(CALL.findall((ROOT / "pyproject.toml").read_text()))
    for chart in (ROOT / "deploy" / "helm" / "algotr" / "templates").glob("*.yaml"):
        found |= set(re.findall(r'"app\.jobs"\s+"([a-z][a-z-]*)"', chart.read_text()))
    return found


def test_the_tasks_in_the_project_file_call_commands_that_exist() -> None:
    tasks = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["poe"]["tasks"]
    assert "ai-evaluate" in tasks and "news-backfill" in tasks
    assert {"ai-evaluate", "news-backfill"} <= commands_named_in_the_repository()


@pytest.mark.parametrize("command", sorted(commands_named_in_the_repository()))
def test_every_command_the_repository_names_is_accepted(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []

    def evaluate() -> dict[str, bool]:
        ran.append("evaluate")
        return {}

    def news_backfill(since: object) -> dict[str, int]:
        ran.append("news")
        return {}

    def lifecycle() -> int:
        ran.append("lifecycle")
        return 0

    def backtest(name: str, results: Path) -> Path:
        ran.append("backtest")
        return results

    monkeypatch.setattr(jobs, "evaluate", evaluate)
    monkeypatch.setattr(jobs, "news_backfill", news_backfill)
    monkeypatch.setattr(jobs, "lifecycle", lifecycle)
    monkeypatch.setattr(jobs, "backtest", backtest)
    arguments = ["halal-etf-trend"] if command == "backtest" else []
    jobs.main([command, *arguments])
    assert len(ran) == 1
