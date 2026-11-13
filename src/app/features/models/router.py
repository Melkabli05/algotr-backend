from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core import audit, registry
from app.core.database import transaction
from app.core.registry import RegisteredModel
from app.features.models import health
from app.features.orders import trail
from app.features.scanner import sentiment

router = APIRouter(prefix="/models", tags=["models"])
Trail = Annotated[audit.AuditLog, Depends(trail.log_of)]


class Promotion(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


def register_finbert() -> RegisteredModel:
    """The headline classifier the scanner scores with: pinned weights, so always production."""
    with transaction() as connection:
        registry.register(
            connection,
            sentiment.MODEL,
            "finbert",
            "classifier",
            "Financial PhraseBank (as published with the weights)",
            {
                "repo": sentiment.REPO,
                "revision": sentiment.REVISION,
                "fallback_below": sentiment.FALLBACK_BELOW,
            },
            str(sentiment.MODEL_DIR.resolve()),
            {"gates": [{"name": "Pinned weights at a fixed revision", "passed": True}]},
            strategy="scanner-sleeve",
            feature_version={"input": "headline text"},
        )
        found = registry.production(connection, "finbert")
        if found is None or found.model_id != sentiment.MODEL:
            reason = f"Published weights pinned at revision {sentiment.REVISION}"
            return registry.promote(
                connection, sentiment.MODEL, "pinned", reason, require_gates=False
            )
        return found


@router.get("/health")
def read_health() -> list[health.StrategyHealth]:
    """Each strategy's and model's failure type from the last nightly check: technical failures
    are blocked, statistical ones trade at their degradation factor."""
    with transaction() as connection:
        found = health.read(connection, datetime.now(UTC))
    return sorted(found.values(), key=lambda item: item.subject)


@router.get("")
def read_models() -> list[RegisteredModel]:
    """Every registered model with its stage; one production model per name."""
    with transaction() as connection:
        return registry.listed(connection)


@router.post("/{model_id}/promote")
def promote_model(log: Trail, model_id: str, body: Promotion) -> RegisteredModel:
    """Make a version the production model of its name, if it passed every gate on record or
    was production before."""
    with transaction() as connection:
        try:
            found = registry.promote(connection, model_id, "manual", body.reason)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="No such model") from exc
        except registry.GateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    log.record("model", {"model": model_id, "stage": "production", "reason": body.reason})
    return found


@router.post("/{name}/rollback")
def rollback_model(log: Trail, name: str, body: Promotion) -> RegisteredModel:
    """Put back the model that was production before; the current one is marked rolled back."""
    with transaction() as connection:
        try:
            found = registry.rollback(connection, name, "manual", body.reason)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    log.record("model", {"model": found.model_id, "rollback": name, "reason": body.reason})
    return found
