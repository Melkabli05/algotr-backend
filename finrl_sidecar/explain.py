"""SHAP values of a trained agent's target weights, over the groups of what it observes.

Each indicator is one player, with its value for both assets, and the weights held are one player:
apart, the held weights of the day and of a background day would mix into something that is not a
portfolio. Thirteen players are few enough to enumerate every coalition, so the values are exact
and the same inputs always give the same numbers.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import shap
from weights_env import SCALE, softmax

BACKGROUND = 64
SEED = 0
HELD = "held"
Weights = Callable[[np.ndarray], np.ndarray]


def layout(columns: list[str], assets: int) -> tuple[list[str], list[list[int]]]:
    """The indicators in order with the observation columns of each, then the weights held."""
    names = list(dict.fromkeys(column.split(":")[0] for column in columns))
    groups = [
        [index for index, column in enumerate(columns) if column.split(":")[0] == name]
        for name in names
    ]
    return [*names, HELD], [*groups, list(range(len(columns), len(columns) + assets + 1))]


def weights_of(agent: Any) -> Weights:
    """The target weights the agent's policy gives for each row of observations."""

    def weights(rows: np.ndarray) -> np.ndarray:
        actions, _ = agent.predict(np.asarray(rows, dtype=np.float32), deterministic=True)
        return softmax(SCALE * actions.astype(float))

    return weights


def background_of(pool: np.ndarray, rows: int = BACKGROUND, seed: int = SEED) -> np.ndarray:
    """A reproducible sample of the observations the agent met in training."""
    return pool[np.random.default_rng(seed).choice(len(pool), min(rows, len(pool)), replace=False)]


def evenly(pool: np.ndarray, rows: int) -> np.ndarray:
    """`rows` observations spread over the pool, first to last."""
    return pool[np.linspace(0, len(pool) - 1, min(rows, len(pool))).astype(int)]


def grouped_shap(
    weights: Weights, instance: np.ndarray, background: np.ndarray, groups: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Each group's Shapley value for each weight, and the weights' mean over the background."""

    def mask_rows(present: np.ndarray, _: np.ndarray) -> np.ndarray:
        kept = np.zeros(instance.size, dtype=bool)
        for members, here in zip(groups, present, strict=True):
            kept[members] = here
        return np.where(kept, instance, background)

    result = shap.ExactExplainer(weights, mask_rows)(np.ones((1, len(groups))))
    return result.values[0], result.base_values[0]


def explain(
    weights: Weights,
    observations: np.ndarray,
    background: np.ndarray,
    groups: list[list[int]],
) -> list[dict[str, Any]]:
    """For each observation: the weights chosen, the base, the values and how far they miss."""
    found = []
    for row in observations:
        values, base = grouped_shap(weights, row, background, groups)
        chosen = weights(row[None])[0]
        miss = np.abs(values.sum(axis=0) + base - chosen).max()
        found.append(
            {
                "weights": chosen.tolist(),
                "base": base.tolist(),
                "values": values.tolist(),
                "residual": float(miss),
            }
        )
    return found
