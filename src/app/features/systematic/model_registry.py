from pathlib import Path
from typing import Any

from app.core import registry
from app.core.agents import ENSEMBLE
from app.core.database import transaction
from app.features.systematic.schemas_models import AgentModel

HYPERPARAMETER_KEYS = ("seed", "timesteps")


def model_id(algorithm: str, version: int) -> str:
    return f"{algorithm}-v{version}"


class AgentRegistry:
    """The agents' models in the platform's model registry: each version trained is registered
    in staging with its admission gates, promoted only by the desk's judgement, and the desk
    always runs the production version."""

    def publish(
        self,
        model: AgentModel,
        artifact: str,
        extra: dict[str, object],
        gates: list[dict[str, Any]],
        features: dict[str, str],
    ) -> None:
        with transaction() as connection:
            registry.register(
                connection,
                model_id(model.algorithm, model.version),
                model.algorithm,
                "rl_agent",
                f"{model.first}..{model.last} ({model.days} sessions)",
                {**{key: getattr(model, key) for key in HYPERPARAMETER_KEYS}, **extra},
                artifact,
                {
                    "in_sample_sharpe": model.in_sample_sharpe,
                    "model": model.model_dump(mode="json"),
                    "gates": gates,
                },
                strategy=ENSEMBLE,
                feature_version=features,
            )

    def promote(self, model: AgentModel, by: str, reason: str) -> None:
        with transaction() as connection:
            registry.promote(connection, model_id(model.algorithm, model.version), by, reason)

    def reject(self, model: AgentModel, reason: str) -> None:
        with transaction() as connection:
            registry.set_stage(connection, model_id(model.algorithm, model.version), "archived")
            registry.note(connection, model_id(model.algorithm, model.version), reason)

    def feature_version(self, model: AgentModel) -> dict[str, Any] | None:
        with transaction() as connection:
            found = registry.find(connection, model_id(model.algorithm, model.version))
        return None if found is None or not found.feature_version else found.feature_version

    def production(self, algorithm: str) -> tuple[AgentModel, str] | None:
        with transaction() as connection:
            found = registry.production(connection, algorithm)
        if found is None:
            return None
        return AgentModel.model_validate(found.metrics["model"]), found.artifact_path


def artifacts(artifact: str) -> dict[str, str]:
    """The model file and its scaling, from the registry's artifact path."""
    path = Path(artifact)
    return {"model_path": str(path), "scaler_path": str(path.with_suffix(".json"))}
