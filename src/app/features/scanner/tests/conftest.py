from collections.abc import Iterator
from pathlib import Path

import pytest
import time_machine

from app.features.scanner import filings, lists
from app.features.scanner.tests.support import NOW


@pytest.fixture(autouse=True)
def on_the_day_of_the_fixtures() -> Iterator[None]:
    """The fixtures' Shariah lists are dated 2 November 2026 and the scanner reads the clock, so
    the tests run on that afternoon rather than whenever they are run."""
    with time_machine.travel(NOW, tick=True):
        yield


@pytest.fixture(autouse=True)
def no_real_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scanner test must hand the service its own HTTP client, never reach the internet."""

    def refuse(*_: object) -> None:
        raise AssertionError("A test tried to open a real HTTP client")

    monkeypatch.setattr(lists, "client", refuse)
    monkeypatch.setattr(filings, "client", refuse)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return tmp_path / "lists"
