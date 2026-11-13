from datetime import UTC, date, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from app.core.database import transaction
from app.core.tables import arbiter_runs, autopilot, order_commands, strategy_targets
from app.features.systematic.schemas import AutomationConfig, ExecutedOrder, Plan

SLEEVE = "scanner-sleeve"
API_SOURCES_STAGE = "live"


def api_targets(plan: Plan, ensemble: str) -> list[dict[str, Any]]:
    """The API's own sources as targets for the arbiter: the agents' ensemble and the sleeve.

    Each is a share of the account's value, so its allocation is one; the rule strategies send
    theirs from the trading node.
    """
    equity = plan.equity or 0.0
    if equity <= 0:
        return []
    found = []
    book = plan.books.get(ensemble)
    if book is not None:
        found.append((ensemble, {s: v / equity for s, v in book.targets.items() if v}))
    sleeve = {t.symbol: t.target_value / equity for t in plan.targets if t.source == "sleeve"}
    if plan.sleeve is not None and not plan.sleeve.preview:
        found.append((SLEEVE, {s: w for s, w in sleeve.items() if w}))
    session = date.today()
    return [
        {
            "strategy": name,
            "source": "api",
            "stage": API_SOURCES_STAGE,
            "allocation": 1.0,
            "weights": weights,
            "version_id": None,
            "session": session,
            "updated": datetime.now(UTC),
        }
        for name, weights in found
    ]


class PlatformLink:
    """How the autopilot reaches the trading node: its settings and its sources' targets."""

    def publish(self, config: AutomationConfig, managed: list[str]) -> None:
        """The node's arbiter trades only these symbols and the ones its targets name, so a
        position placed by hand is never sold; symbols the arbiter has bought stay in the list."""
        with transaction() as connection:
            known: list[str] = (
                connection.execute(sa.select(autopilot.c.managed)).scalar_one_or_none() or []
            )
            values = {
                "mode": config.mode,
                "allocation": config.allocation,
                "crypto": config.crypto,
                "managed": sorted({*known, *managed}),
                "updated": datetime.now(UTC),
            }
            statement = insert(autopilot).values(id=1, **values)
            connection.execute(
                statement.on_conflict_do_update(index_elements=[autopilot.c.id], set_=values)
            )

    def write(self, rows: list[dict[str, Any]]) -> None:
        """Replace every API-sourced target with `rows`; a source left out wants nothing."""
        with transaction() as connection:
            connection.execute(
                sa.delete(strategy_targets).where(strategy_targets.c.source == "api")
            )
            if rows:
                connection.execute(sa.insert(strategy_targets), rows)

    def executed(self, after: datetime) -> list[ExecutedOrder]:
        """The orders the node's arbiter sent since `after`, once their command has finished."""
        with transaction() as connection:
            rows = connection.execute(
                sa.select(order_commands.c.result)
                .join(arbiter_runs, arbiter_runs.c.command_id == order_commands.c.id)
                .where(arbiter_runs.c.time >= after, order_commands.c.status == "done")
                .order_by(arbiter_runs.c.id)
            )
            return [
                ExecutedOrder.model_validate(item)
                for row in rows
                for item in (row.result or {}).get("orders", [])
            ]
