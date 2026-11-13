from datetime import date
from itertools import islice
from typing import Any

from pydantic import BaseModel

from app.core import audit

KIND = "xai"
PLACES = 6
Named = dict[str, dict[str, float]]


class Decision(BaseModel):
    """Why an agent chose its weights on one day: each indicator's push on each weight."""

    decided: date
    version: int
    trained: date
    weights: dict[str, float]
    base: dict[str, float]
    values: Named
    inputs: Named
    residual: float


class Overall(BaseModel):
    """How much each indicator moves each weight on average over a sample of training days."""

    days: int
    values: Named


class AgentExplanations(BaseModel):
    algorithm: str
    version: int | None
    assets: list[str]
    groups: list[str]
    decisions: list[Decision]
    overall: Overall | None
    across: Named
    unexplained: int


def by_asset(assets: list[str], values: list[float]) -> dict[str, float]:
    return {asset: round(value, PLACES) for asset, value in zip(assets, values, strict=True)}


def named(names: list[str], assets: list[str], rows: list[list[float]]) -> Named:
    return {name: by_asset(assets, row) for name, row in zip(names, rows, strict=True)}


def inputs_of(columns: list[str], observation: list[float]) -> Named:
    """What the agent saw, standardised, for each indicator and asset."""
    seen: Named = {}
    for label, value in zip(columns, observation, strict=True):
        name, _, asset = label.partition(":")
        seen.setdefault(name, {})[asset] = round(value, PLACES)
    return seen


def build(
    found: dict[str, Any],
    queued: list[tuple[date, list[float]]],
    model: dict[str, Any],
    sample: int,
) -> dict[str, Any]:
    """One agent's part of an audit entry, from the sidecar's arrays and the decisions asked."""
    assets, groups = found["assets"], found["groups"]
    decisions = [
        {
            "decided": day.isoformat(),
            "weights": by_asset(assets, item["weights"]),
            "base": by_asset(assets, item["base"]),
            "values": named(groups, assets, item["values"]),
            "inputs": inputs_of(found["columns"], observation),
            "residual": float(f"{item['residual']:.3g}"),
        }
        for (day, observation), item in zip(queued, found["decisions"], strict=True)
    ]
    overall = (
        {"days": found["sample"], "values": named(groups, assets, found["overall"])}
        if sample
        else None
    )
    return {
        "model": model,
        "shap": found["shap"],
        "assets": assets,
        "groups": groups,
        "background": found["background"],
        "decisions": decisions,
        "overall": overall,
    }


def mean_abs(decisions: list[Decision]) -> Named:
    """The average size of each indicator's push on each weight, over the decisions that have it."""
    names = dict.fromkeys(name for item in decisions for name in item.values)
    averages: Named = {}
    for name in names:
        having = [item.values[name] for item in decisions if name in item.values]
        averages[name] = {
            asset: sum(abs(pushes[asset]) for pushes in having) / len(having) for asset in having[0]
        }
    return averages


def read(entries: list[audit.Entry], algorithm: str, limit: int) -> AgentExplanations:
    """The newest decisions of an agent from the audit log, and the overall of its latest model."""
    decisions: list[Decision] = []
    overall: Overall | None = None
    version: int | None = None
    assets: list[str] = []
    groups: list[str] = []
    unexplained = 0
    for entry in reversed(entries):
        if entry.kind != KIND:
            continue
        unexplained += sum(item["agent"] == algorithm for item in entry.payload["unexplained"])
        record = entry.payload["agents"].get(algorithm)
        if record is None:
            continue
        assets, groups = assets or record["assets"], groups or record["groups"]
        version = record["model"]["version"] if version is None else version
        if overall is None and record["overall"] is not None:
            overall = Overall.model_validate(record["overall"])
        room = max(limit - len(decisions), 0)
        made_by = {"version": record["model"]["version"], "trained": record["model"]["trained"]}
        decisions += [
            Decision.model_validate({**item, **made_by})
            for item in islice(reversed(record["decisions"]), room)
        ]
    return AgentExplanations(
        algorithm=algorithm,
        version=version,
        assets=assets,
        groups=groups,
        decisions=decisions,
        overall=overall,
        across=mean_abs(decisions),
        unexplained=unexplained,
    )
