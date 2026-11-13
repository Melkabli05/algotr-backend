from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from app.core.tables import strategy_health

if TYPE_CHECKING:
    from app.features.models.drift import DriftFinding

FailureType = Literal["none", "technical", "statistical"]
STATISTICAL_FACTOR = 0.5
STALE_AFTER = timedelta(days=2)


class StrategyHealth(BaseModel):
    """Technical (same inputs, different outputs: blocked) or statistical (signal decayed:
    halved and retrained), after Jansen's failure taxonomy."""

    subject: str
    failure_type: FailureType
    factor: float
    reason: str
    since: datetime
    updated: datetime


def classify(findings: Iterable["DriftFinding"], now: datetime) -> dict[str, StrategyHealth]:
    """Each measured subject's failure type from tonight's findings; technical outranks."""
    found: dict[str, StrategyHealth] = {}
    for finding in findings:
        if finding.drift_severity == "low":
            kind: FailureType = "none"
        elif finding.kind == "technical":
            kind = "technical" if finding.drift_severity == "high" else "none"
        else:
            kind = "statistical"
        current = found.get(finding.subject)
        rank = {"none": 0, "statistical": 1, "technical": 2}
        if current is not None and rank[current.failure_type] >= rank[kind]:
            continue
        found[finding.subject] = StrategyHealth(
            subject=finding.subject,
            failure_type=kind,
            factor={"none": 1.0, "statistical": STATISTICAL_FACTOR, "technical": 0.0}[kind],
            reason=finding.detail if kind != "none" else "Every measure within its bounds",
            since=now,
            updated=now,
        )
    return found


def write(connection: Connection, found: dict[str, StrategyHealth]) -> None:
    """Keep each subject's state; `since` stays put while the failure type does not change."""
    for health in found.values():
        values = health.model_dump(exclude={"subject"})
        statement = insert(strategy_health).values(subject=health.subject, **values)
        keep_since = sa.case(
            (
                strategy_health.c.failure_type == statement.excluded.failure_type,
                strategy_health.c.since,
            ),
            else_=statement.excluded.since,
        )
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=[strategy_health.c.subject],
                set_={**values, "since": keep_since},
            )
        )


def read(connection: Connection, now: datetime) -> dict[str, StrategyHealth]:
    """Every subject's state, as the risk actor reads it before an order; a state the nightly
    job has not confirmed for two days no longer counts."""
    rows = connection.execute(
        sa.select(strategy_health).where(strategy_health.c.updated >= now - STALE_AFTER)
    )
    return {row.subject: StrategyHealth.model_validate(row._mapping) for row in rows}


def blocked(found: dict[str, StrategyHealth]) -> dict[str, str]:
    return {name: item.reason for name, item in found.items() if item.failure_type == "technical"}


def factors(found: dict[str, StrategyHealth]) -> dict[str, float]:
    return {name: item.factor for name, item in found.items() if item.failure_type == "statistical"}
