import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

from app.core.compliance import Screening
from app.core.tables import strategies, strategy_versions
from app.features.strategies.router import get_screen, get_selected
from app.features.systematic.catalog import STRATEGIES, TEMPLATES, seed
from app.features.systematic.tests.support import uptrend
from app.main import app

REFUSED = {"XOM"}


def screen(symbol: str) -> Screening:
    allowed = symbol not in REFUSED
    return Screening(
        symbol=symbol, allowed=allowed, reason="passes" if allowed else "not held by SPUS"
    )


@pytest.fixture
def api(client: TestClient, database: Engine) -> Iterator[TestClient]:
    with database.begin() as connection:
        seed(connection)
    STRATEGIES.refresh()
    app.dependency_overrides[get_screen] = lambda: screen
    app.dependency_overrides[get_selected] = lambda: {"halal-etf-trend"}
    yield client
    app.dependency_overrides.pop(get_screen, None)
    app.dependency_overrides.pop(get_selected, None)


def draft(**changes: Any) -> dict[str, Any]:
    base = {
        "name": "my-trend",
        "title": "My SPUS trend",
        "description": "SPUS above its 150-day average",
        "universe": ["SPUS"],
        "signal": "sma",
        "window": 150,
        "fee_bps": 5,
    }
    return base | changes


def test_the_templates_are_listed_as_strategies_and_offered_as_starting_points(
    api: TestClient,
) -> None:
    listed = api.get("/strategies").json()
    assert [item["name"] for item in listed] == list(TEMPLATES)
    assert {item["source"] for item in listed} == {"template"}
    templates = api.get("/strategies/templates").json()
    assert templates[2]["name"] == "halal-etf-trend"
    assert "periods_per_year" not in templates[2]


def test_a_new_strategy_is_saved_as_version_one_and_reaches_the_registry(
    api: TestClient,
) -> None:
    made = api.post("/strategies", json={"draft": draft(), "note": "first try"})
    assert made.status_code == 201
    body = made.json()
    assert (body["version"], body["source"], body["stage"]) == (1, "user", None)
    assert body["spec"]["periods_per_year"] == 252
    assert STRATEGIES["my-trend"].window == 150
    assert STRATEGIES.trials() == len(TEMPLATES) + 1
    again = api.post("/strategies", json={"draft": draft()})
    assert again.status_code == 409


def test_an_edit_is_a_new_version_and_an_unchanged_save_makes_none(api: TestClient) -> None:
    api.post("/strategies", json={"draft": draft()})
    edited = api.put("/strategies/my-trend", json={"draft": draft(window=100), "note": "faster"})
    assert edited.status_code == 200
    body = edited.json()
    assert body["version"] == 2
    assert [item["version"] for item in body["history"]] == [2, 1]
    assert STRATEGIES["my-trend"].window == 100
    first = api.get("/strategies/my-trend/versions/1").json()
    assert first["window"] == 150
    unchanged = api.put("/strategies/my-trend", json={"draft": draft(window=100)})
    assert unchanged.status_code == 409
    renamed = api.put("/strategies/my-trend", json={"draft": draft(name="other")})
    assert renamed.status_code == 422


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"universe": ["SPUS", "XOM"]}, "Shariah"),
        ({"universe": ["SPUS", "BTC/USD"]}, "coins or stocks"),
        ({"signal": "donchian"}, "lookback"),
        ({"signal": "momentum", "top_n": 3}, "more assets"),
        ({"fee_bps": 0.5}, "cost of a trade"),
        ({"universe": ["BTC/USD"], "fee_bps": 25, "proxy": True}, "century"),
        ({"name": "Not A Slug"}, "pattern"),
    ],
)
def test_a_strategy_the_rules_refuse_is_not_saved(
    api: TestClient, changes: dict[str, Any], reason: str
) -> None:
    refused = api.post("/strategies", json={"draft": draft(**changes)})
    assert refused.status_code == 422
    assert reason in str(refused.json()["detail"])
    assert "my-trend" not in STRATEGIES


def test_archiving_is_refused_while_a_strategy_trades_or_is_selected(
    api: TestClient, database: Engine
) -> None:
    version = uuid.uuid4()
    with database.begin() as connection:
        connection.execute(
            sa.insert(strategy_versions).values(
                id=version, strategy="halal-multi-trend", code_hash="x", params={}, engine="t"
            )
        )
        connection.execute(
            sa.insert(strategies).values(
                name="halal-multi-trend",
                stage="live",
                version_id=version,
                stage_since=datetime.now(UTC),
            )
        )
    live = api.post("/strategies/halal-multi-trend/archive")
    assert live.status_code == 422 and "at live" in live.json()["detail"]
    selected = api.post("/strategies/halal-etf-trend/archive")
    assert selected.status_code == 422 and "autopilot" in selected.json()["detail"]


def test_an_archived_strategy_leaves_the_registry_until_it_is_restored(api: TestClient) -> None:
    archived = api.post("/strategies/btc-trend-200/archive")
    assert archived.status_code == 200 and archived.json()["archived"]
    assert "btc-trend-200" not in STRATEGIES
    assert "btc-trend-200" not in [item["name"] for item in api.get("/strategies").json()]
    shown = api.get("/strategies", params={"archived": True}).json()
    assert "btc-trend-200" in [item["name"] for item in shown]
    edit = api.put("/strategies/btc-trend-200", json={"draft": draft(name="btc-trend-200")})
    assert edit.status_code == 422
    api.post("/strategies/btc-trend-200/restore")
    assert "btc-trend-200" in STRATEGIES


def test_an_export_imports_back_as_a_copy_and_never_overwrites(api: TestClient) -> None:
    api.post("/strategies", json={"draft": draft()})
    file = api.get("/strategies/my-trend/export").json()
    assert file["format"] == "algotr.strategies" and file["format_version"] == 1
    assert file["strategies"][0]["draft"]["window"] == 150
    clash = api.post("/strategies/import", json=file)
    assert clash.status_code == 409
    copied = api.post("/strategies/import", params={"copy": True}, json=file)
    assert copied.status_code == 201
    assert copied.json() == {"names": ["my-trend-2"], "renamed": {"my-trend": "my-trend-2"}}
    made = api.get("/strategies/my-trend-2").json()
    assert (made["source"], made["version"], made["stage"]) == ("import", 1, None)
    assert STRATEGIES["my-trend-2"].window == 150


def test_an_import_is_all_or_nothing(api: TestClient) -> None:
    file = {
        "exported": datetime.now(UTC).isoformat(),
        "strategies": [
            {"draft": draft(name="good-one")},
            {"draft": draft(name="bad-one", universe=["XOM"])},
        ],
    }
    refused = api.post("/strategies/import", json=file)
    assert refused.status_code == 422
    assert "good-one" not in [item["name"] for item in api.get("/strategies").json()]


def test_the_whole_library_exports_and_the_schema_is_published(api: TestClient) -> None:
    every = api.get("/strategies/export").json()
    assert [item["draft"]["name"] for item in every["strategies"]] == list(TEMPLATES)
    schema = api.get("/strategies/schema").json()
    assert "strategies" in schema["properties"]
    assert api.get("/strategies/nothing").status_code == 404


@pytest.fixture
def prices(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Days of history per symbol; backtests run on a steady rise of that length."""
    from app.features.systematic import backtest
    from app.features.systematic import research as research_module
    from app.quant import history

    days = {"SPUS": 900, "HLAL": 900}

    def closes(_stocks: object, _crypto: object, symbols: list[str]) -> object:
        if symbols[0] not in days:
            raise ValueError("no bars")
        return uptrend(symbols, days[symbols[0]])

    def research(_s: object, _c: object, spec: Any, trials: int) -> object:
        return backtest.run(uptrend(spec.universe, 900), spec, trials=trials)

    monkeypatch.setattr(history, "cached_closes", closes)
    monkeypatch.setattr(research_module, "research", research)
    return days


def test_a_draft_is_backtested_without_being_saved_and_a_new_rule_counts_once(
    api: TestClient, prices: dict[str, int]
) -> None:
    first = api.post("/strategies/preview", json={"draft": draft()})
    assert first.status_code == 200
    body = first.json()
    assert (body["new_trial"], body["trials"]) == (True, len(TEMPLATES) + 1)
    assert body["result"]["metrics"]["sharpe"] > 0
    assert {check["name"] for check in body["result"]["checks"]}
    again = api.post("/strategies/preview", json={"draft": draft(title="Renamed")}).json()
    assert (again["new_trial"], again["trials"]) == (False, len(TEMPLATES) + 1)
    assert api.get("/strategies/my-trend").status_code == 404


def test_a_copy_of_a_rule_with_new_words_is_no_new_trial(
    api: TestClient, prices: dict[str, int]
) -> None:
    copy = TEMPLATES["halal-etf-trend"].model_dump(exclude={"periods_per_year"})
    copy |= {"name": "etf-trend-copy", "title": "Another name", "note": None}
    preview = api.post("/strategies/preview", json={"draft": copy}).json()
    assert preview["new_trial"] is False
    api.post("/strategies", json={"draft": copy})
    assert STRATEGIES.trials() == len(TEMPLATES)


def test_a_draft_the_screen_refuses_is_not_backtested_nor_counted(
    api: TestClient, prices: dict[str, int]
) -> None:
    refused = api.post("/strategies/preview", json={"draft": draft(universe=["XOM"])})
    assert refused.status_code == 422
    assert STRATEGIES.trials() == len(TEMPLATES)


def test_a_check_reports_each_symbol_its_twins_and_short_histories(
    api: TestClient, prices: dict[str, int]
) -> None:
    prices["XOM"] = 100
    copy = TEMPLATES["halal-etf-trend"].model_dump(exclude={"periods_per_year"})
    copy |= {"name": "etf-trend-copy"}
    body = {"universe": ["spus", "XOM", "NEWCO"], "draft": copy}
    found = api.post("/strategies/check", json=body).json()
    by_symbol = {item["symbol"]: item for item in found["symbols"]}
    assert by_symbol["SPUS"]["allowed"] and by_symbol["SPUS"]["days"] == 900
    assert not by_symbol["XOM"]["allowed"]
    assert by_symbol["NEWCO"]["days"] is None
    assert found["needs_days"] == 200 + 20 + 60
    assert found["same_rule_as"] == ["halal-etf-trend"]
    assert any("XOM has 100 days" in warning for warning in found["warnings"])
    assert any("NEWCO: no price history" in warning for warning in found["warnings"])
    assert STRATEGIES.trials() == len(TEMPLATES)


def test_each_version_says_what_changed_from_the_one_before(api: TestClient) -> None:
    api.post("/strategies", json={"draft": draft()})
    api.put("/strategies/my-trend", json={"draft": draft(window=100, fee_bps=8)})
    history = api.get("/strategies/my-trend").json()["history"]
    assert history[0]["changes"] == [
        {"field": "window", "before": 150, "after": 100},
        {"field": "fee_bps", "before": 5.0, "after": 8.0},
    ]
    assert history[1]["changes"] == []


def test_a_half_written_draft_is_checked_and_told_what_is_missing(
    api: TestClient, prices: dict[str, int]
) -> None:
    typed = draft(name="Bad Name", signal="donchian")
    found = api.post("/strategies/check", json={"universe": ["SPUS"], "draft": typed})
    assert found.status_code == 200
    problems = found.json()["problems"]
    assert "name: use lower case letters, digits and dashes, 2 to 48 of them" in problems
    assert found.json()["symbols"][0]["allowed"]
