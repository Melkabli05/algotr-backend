import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from app.core.tables import strategy_definition_versions, strategy_definitions
from app.features.systematic.catalog import (
    STRATEGIES,
    TEMPLATES,
    StrategyRegistry,
    backfill_trials,
    open_registry,
    seed,
    spec_hash,
)


def test_until_it_is_read_the_registry_is_the_built_in_templates() -> None:
    assert list(STRATEGIES) == list(TEMPLATES)
    assert STRATEGIES.trials() == len(TEMPLATES)


def test_the_templates_are_seeded_once_and_read_back_in_their_order(database: Engine) -> None:
    assert open_registry()
    with database.begin() as connection:
        assert seed(connection) == []
        versions = connection.execute(sa.select(strategy_definition_versions)).all()
    assert list(STRATEGIES) == list(TEMPLATES)
    assert STRATEGIES["halal-etf-trend"] == TEMPLATES["halal-etf-trend"]
    assert {row.version for row in versions} == {1}
    assert STRATEGIES.trials() == len(TEMPLATES)


def test_archived_strategies_are_left_out_but_their_versions_still_count(
    database: Engine,
) -> None:
    edited = TEMPLATES["halal-etf-trend"].model_copy(update={"window": 150})
    with database.begin() as connection:
        seed(connection)
        connection.execute(
            sa.insert(strategy_definition_versions).values(
                name="halal-etf-trend",
                version=2,
                spec=edited.model_dump(mode="json"),
                spec_hash=spec_hash(edited),
            )
        )
        connection.execute(
            sa.update(strategy_definitions)
            .where(strategy_definitions.c.name == "halal-etf-trend")
            .values(current_version=2)
        )
        connection.execute(
            sa.update(strategy_definitions)
            .where(strategy_definitions.c.name == "btc-trend-200")
            .values(archived=True)
        )
        assert backfill_trials(connection) == 1
    assert STRATEGIES.refresh()
    assert "btc-trend-200" not in STRATEGIES
    assert STRATEGIES["halal-etf-trend"].window == 150
    assert STRATEGIES.trials() == len(TEMPLATES) + 1


def test_a_failed_read_keeps_the_strategies_already_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = StrategyRegistry(TEMPLATES)
    assert not registry.refresh()
    assert list(registry) == list(TEMPLATES)
