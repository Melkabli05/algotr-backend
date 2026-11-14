import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.engine import Connection

from app.core.tables import models

ModelType = Literal["rl_agent", "classifier", "llm"]
Stage = Literal["staging", "production", "archived", "rolled_back"]


class ImmutableError(ValueError):
    """A model id already names other weights: a version identifies one artifact for good."""


class GateError(ValueError):
    """A model that has not passed every gate recorded for it cannot be promoted."""


class RegisteredModel(BaseModel):
    model_id: str
    name: str
    model_type: ModelType
    strategy: str | None
    training_data_version: str
    hyperparameters: dict[str, Any]
    artifact_path: str
    artifact_hash: str | None
    feature_version: dict[str, Any]
    metrics: dict[str, Any]
    deployment_stage: Stage
    promoted_at: datetime | None
    promoted_by: str | None
    promotion_reason: str | None
    created: datetime

    @property
    def gates(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = self.metrics.get("gates", [])
        return found

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(gate.get("passed") for gate in self.gates)


def hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def artifact_hash(path: str) -> str | None:
    """SHA-256 of a model's weights: of the file, or of every file of a folder in order.

    Hidden folders are left out: Hugging Face writes download timestamps into `.cache`, which would
    give the same weights a new hash on every fresh download.
    """
    found = Path(path)
    if found.is_file():
        return hashlib.sha256(found.read_bytes()).hexdigest()
    if found.is_dir():
        digest = hashlib.sha256()
        files = (p for p in found.rglob("*") if p.is_file() and not hidden(p.relative_to(found)))
        for item in sorted(files):
            digest.update(str(item.relative_to(found)).encode())
            digest.update(hashlib.sha256(item.read_bytes()).digest())
        return digest.hexdigest()
    return None


def register(
    connection: Connection,
    model_id: str,
    name: str,
    model_type: ModelType,
    training_data_version: str,
    hyperparameters: dict[str, Any],
    artifact_path: str,
    metrics: dict[str, Any],
    strategy: str | None = None,
    feature_version: dict[str, Any] | None = None,
) -> RegisteredModel:
    """Record a trained model in staging, once: the same id and weights again changes nothing,
    the same id with other weights is refused."""
    found_hash = artifact_hash(artifact_path)
    existing = find(connection, model_id)
    if existing is not None:
        if existing.artifact_hash is None and found_hash is not None:
            connection.execute(
                sa.update(models)
                .where(models.c.model_id == model_id)
                .values(artifact_hash=found_hash)
            )
            return get(connection, model_id)
        if found_hash is not None and existing.artifact_hash != found_hash:
            raise ImmutableError(f"{model_id} already names other weights; train a new version")
        return existing
    connection.execute(
        sa.insert(models).values(
            model_id=model_id,
            name=name,
            model_type=model_type,
            strategy=strategy,
            training_data_version=training_data_version,
            hyperparameters=hyperparameters,
            artifact_path=artifact_path,
            artifact_hash=found_hash,
            feature_version=feature_version or {},
            metrics=metrics,
        )
    )
    return get(connection, model_id)


def find(connection: Connection, model_id: str) -> RegisteredModel | None:
    row = connection.execute(sa.select(models).where(models.c.model_id == model_id)).one_or_none()
    return None if row is None else RegisteredModel.model_validate(row._mapping)


def get(connection: Connection, model_id: str) -> RegisteredModel:
    found = find(connection, model_id)
    if found is None:
        raise LookupError(model_id)
    return found


def set_stage(
    connection: Connection,
    model_id: str,
    stage: Stage,
    by: str | None = None,
    reason: str | None = None,
    now: datetime | None = None,
) -> None:
    values: dict[str, Any] = {"deployment_stage": stage}
    if stage == "production":
        values |= {"promoted_at": now or datetime.now(UTC), "promoted_by": by}
        values |= {"promotion_reason": reason}
    connection.execute(sa.update(models).where(models.c.model_id == model_id).values(**values))


def note(connection: Connection, model_id: str, reason: str) -> None:
    """Why a model left staging without being promoted."""
    connection.execute(
        sa.update(models).where(models.c.model_id == model_id).values(promotion_reason=reason)
    )


def promote(
    connection: Connection,
    model_id: str,
    by: str,
    reason: str,
    now: datetime | None = None,
    require_gates: bool = True,
) -> RegisteredModel:
    """Make one model the production model of its name; the one it replaces is archived.

    A candidate must have passed every gate recorded for it, unless it was production before.
    """
    found = get(connection, model_id)
    if require_gates and not found.passed and found.promoted_at is None:
        failed = [gate["name"] for gate in found.gates if not gate.get("passed")]
        raise GateError(f"{model_id} has not passed {', '.join(failed) if failed else 'any gate'}")
    connection.execute(
        sa.update(models)
        .where(models.c.name == found.name, models.c.deployment_stage == "production")
        .values(deployment_stage="archived")
    )
    set_stage(connection, model_id, "production", by, reason, now)
    return get(connection, model_id)


def rollback(
    connection: Connection, name: str, by: str, reason: str, now: datetime | None = None
) -> RegisteredModel:
    """Put the model that was production before back, and mark the current one rolled back:
    two database writes, picked up by the next restart or reload."""
    current = production(connection, name)
    if current is None:
        raise LookupError(f"{name} has no production model")
    earlier = connection.execute(
        sa.select(models.c.model_id)
        .where(
            models.c.name == name,
            models.c.model_id != current.model_id,
            models.c.promoted_at.is_not(None),
            models.c.deployment_stage == "archived",
        )
        .order_by(models.c.promoted_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if earlier is None:
        raise LookupError(f"{name} has no earlier production model to roll back to")
    set_stage(connection, current.model_id, "rolled_back")
    set_stage(connection, earlier, "production", by, reason, now)
    return get(connection, earlier)


def production(connection: Connection, name: str) -> RegisteredModel | None:
    row = connection.execute(
        sa.select(models).where(models.c.name == name, models.c.deployment_stage == "production")
    ).one_or_none()
    return None if row is None else RegisteredModel.model_validate(row._mapping)


def listed(connection: Connection) -> list[RegisteredModel]:
    rows = connection.execute(sa.select(models).order_by(models.c.name, models.c.created.desc()))
    return [RegisteredModel.model_validate(row._mapping) for row in rows]


def productions(connection: Connection) -> dict[str, str]:
    """Each name's production model id: what a node restart loads."""
    rows = connection.execute(
        sa.select(models.c.name, models.c.model_id).where(models.c.deployment_stage == "production")
    )
    return {row.name: row.model_id for row in rows}
