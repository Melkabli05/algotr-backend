import json
import logging
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from celery import Celery
from celery.schedules import crontab

from app import watch
from app.core import audit
from app.core.broker import get_crypto_data_client, get_stock_data_client
from app.core.database import transaction
from app.core.settings import get_settings
from app.core.tables import strategies, strategy_versions
from app.features.drift.monitor import PATH as DRIFT_PATH
from app.features.lifecycle import gates
from app.features.lifecycle.service import Lifecycle, by_day
from app.features.models import drift
from app.features.systematic import agents, explanations, signals
from app.features.systematic.agents import ROOT as AGENTS_ROOT
from app.features.systematic.agents import State as AgentState
from app.features.systematic.catalog import STRATEGIES
from app.quant import history

logger = logging.getLogger(__name__)

app = Celery("algotr", broker=get_settings().celery_broker_url)
app.conf.update(
    timezone="UTC",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "nightly-drift": {
            "task": "app.worker.nightly_drift",
            "schedule": crontab(hour=1, minute=30),
        }
    },
)
DEPLOYED = ("shadow", "canary", "live")
AGENT_PERIODS = 252


def deployed() -> list[str]:
    with transaction() as connection:
        rows = connection.execute(
            sa.select(strategies.c.name).where(strategies.c.stage.in_(DEPLOYED))
        )
        return [row.name for row in rows]


def replay_gap(name: str) -> float | None:
    """The live decisions of the strategy's current version against the rule replayed now."""
    spec = STRATEGIES.get(name)
    if spec is None:
        return None
    with transaction() as connection:
        row = connection.execute(sa.select(strategies).where(strategies.c.name == name)).one()
        recorded = Lifecycle.decisions(connection, name, row.version_id, date(1970, 1, 1))
    if not recorded:
        return None
    stocks, crypto = get_stock_data_client(), get_crypto_data_client()
    closes = history.cached_prices(stocks, crypto, spec.universe).closes
    return gates.divergence(recorded, by_day(signals.weights(closes, spec), recorded))


def strategy_returns(name: str) -> tuple[pd.Series, int] | None:
    return watch.strategy_returns(name, datetime(1970, 1, 1, tzinfo=UTC))


def inputs(name: str) -> pd.Series | None:
    """The daily returns of the market a strategy or agent decides on, averaged across it."""
    spec = STRATEGIES.get(name)
    if spec is not None:
        symbols = list(spec.universe)
    else:
        shadow = agent_state().shadow.get(name)
        symbols = [asset for asset in shadow.assets if asset != "cash"] if shadow else []
    if not symbols:
        return None
    closes = history.cached_closes(get_stock_data_client(), get_crypto_data_client(), symbols)
    return closes.pct_change().mean(axis=1).dropna()


def agent_state(root: Path = AGENTS_ROOT) -> AgentState:
    try:
        return AgentState.model_validate_json((root / "state.json").read_text())
    except (FileNotFoundError, ValueError):
        return AgentState()


def expected_sharpe(name: str) -> float | None:
    """The Sharpe the lifecycle measured for the version a strategy holds."""
    with transaction() as connection:
        found = connection.execute(
            sa.select(strategy_versions.c.metrics)
            .join(strategies, strategies.c.version_id == strategy_versions.c.id)
            .where(strategies.c.name == name)
        ).scalar_one_or_none()
    metrics: dict[str, Any] = found or {}
    value = metrics.get("sharpe")
    return float(value) if value is not None else None


def calibrations() -> dict[str, tuple[list[float], list[bool]]]:
    """Each agent's stated confidence against what its moves then earned."""
    stocks, crypto = get_stock_data_client(), get_crypto_data_client()
    found = {}
    for name, shadow in agent_state().shadow.items():
        if not shadow.confidence:
            continue
        closes = history.cached_closes(stocks, crypto, list(shadow.assets))
        found[name] = agents.calibration_points(shadow, closes)
    return found


def model_returns() -> dict[str, tuple[pd.Series, int]]:
    """Each agent's paper returns, as its model chose them."""
    return {
        name: (pd.Series(shadow.returns, index=pd.to_datetime(shadow.dates)), AGENT_PERIODS)
        for name, shadow in agent_state().shadow.items()
        if shadow.returns
    }


def model_decisions() -> dict[str, list[Mapping[str, Any]]]:
    """Every explained decision of each agent, oldest first, from the audit log."""
    found: dict[str, list[Mapping[str, Any]]] = {}
    for entry in audit.AuditLog(audit.AUDIT_PATH).entries():
        if entry.kind != explanations.KIND:
            continue
        for name, record in entry.payload.get("agents", {}).items():
            found.setdefault(name, []).extend(record.get("decisions", []))
    return found


def adwin_changes(path: Path = DRIFT_PATH, since: timedelta = timedelta(days=1)) -> dict[str, int]:
    """Changes the drift monitor's ADWIN detectors found per stream over the last day."""
    try:
        recorded = json.loads(path.read_text()).get("recorded", [])
    except (FileNotFoundError, ValueError):
        return {}
    cutoff = (datetime.now(UTC) - since).date()
    counts: dict[str, int] = {}
    for item in recorded:
        if date.fromisoformat(str(item["time"])[:10]) >= cutoff:
            counts[item["stream"]] = counts.get(item["stream"], 0) + 1
    return counts


@app.task(name="app.worker.nightly_drift")  # type: ignore[untyped-decorator]
def nightly_drift() -> int:
    """Measure drift of every deployed strategy and model and write it to model_drift."""
    STRATEGIES.refresh()
    sources = drift.DriftSources(
        deployed=deployed,
        replay_gap=replay_gap,
        returns=strategy_returns,
        inputs=inputs,
        model_returns=model_returns,
        model_decisions=model_decisions,
        adwin_changes=adwin_changes,
        expected_sharpe=expected_sharpe,
        calibrations=calibrations,
    )
    with transaction() as connection:
        found = drift.nightly(
            connection, sources, get_settings().strategy_min_sharpe, datetime.now(UTC)
        )
    logger.info("Drift: %d measurements", len(found))
    return len(found)
