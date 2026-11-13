from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.audit import AuditLog
from app.core.limits import LimitStore, defaults
from app.core.settings import get_settings
from app.main import app


@pytest.fixture
def state(tmp_path: Path) -> Iterator[SimpleNamespace]:
    names = ("limits", "audit", "book")
    saved = {name: getattr(app.state, name, None) for name in names}
    app.state.limits = LimitStore(get_settings(), tmp_path / "limits.json")
    app.state.audit = AuditLog(tmp_path / "audit.jsonl")
    app.state.book = SimpleNamespace(max_imbalance=0.0)
    yield SimpleNamespace(limits=app.state.limits, audit=app.state.audit, book=app.state.book)
    for name, value in saved.items():
        setattr(app.state, name, value)


def test_limits_start_at_the_environment_defaults(
    client: TestClient, state: SimpleNamespace
) -> None:
    assert client.get("/settings/limits").json() == defaults(get_settings()).model_dump()


def test_a_change_is_saved_audited_and_handed_to_the_book(
    client: TestClient, state: SimpleNamespace
) -> None:
    before = client.get("/settings/limits").json()
    after = {**before, "position_share": 0.05, "book_imbalance": 0.7}
    assert client.put("/settings/limits", json=after).json() == after
    assert state.limits.get().position_share == 0.05
    assert state.book.max_imbalance == 0.7
    [entry] = state.audit.entries()
    assert entry.kind == "settings-change"
    assert entry.payload == {"area": "limits", "before": before, "after": after}


def test_limits_out_of_range_are_refused(client: TestClient, state: SimpleNamespace) -> None:
    before = client.get("/settings/limits").json()
    response = client.put("/settings/limits", json={**before, "daily_loss": 0.9})
    assert response.status_code == 422
    assert state.audit.entries() == []


def test_a_reset_goes_back_to_the_environment(client: TestClient, state: SimpleNamespace) -> None:
    before = client.get("/settings/limits").json()
    client.put("/settings/limits", json={**before, "spread_bps": 5})
    assert client.delete("/settings/limits").json() == before
    assert [entry.kind for entry in state.audit.entries()] == ["settings-change"] * 2
