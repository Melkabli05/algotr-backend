import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

FLOOR = 0.1
TEMPERATURE = 0.1
FIRST_SPREAD = 0.03
MIN_ROUNDS = 3


def reward(value: float, spread: float) -> float:
    """A round's return as a reward in (0, 1), the range UCB1 needs; `spread` is a typical round."""
    return 0.5 * (1 + math.tanh(value / spread))


def scores(means: Sequence[float], counts: Sequence[int]) -> list[float]:
    """UCB1 (Auer, Cesa-Bianchi and Fischer, 2002): the mean reward plus sqrt(2 ln n / n_j)."""
    total = sum(counts)
    return [
        mean + math.sqrt(2 * math.log(total) / count)
        for mean, count in zip(means, counts, strict=True)
    ]


def allocate(
    rewards: Sequence[Sequence[float]], floor: float = FLOOR, temperature: float = TEMPERATURE
) -> list[float]:
    """Each arm's weight from the rewards it has earned: softmax of its UCB1 score over a floor.

    Until every arm has a round they share equally; after that every arm keeps at least `floor`.
    """
    arms = len(rewards)
    if not all(rewards):
        return [1 / arms] * arms
    ranked = scores([float(np.mean(each)) for each in rewards], [len(each) for each in rewards])
    shifted = np.exp((np.array(ranked) - max(ranked)) / temperature)
    mix: list[float] = (floor + (1 - arms * floor) * shifted / shifted.sum()).tolist()
    return mix


class Replay:
    """The ensemble's daily returns and monthly weights, and the weights it would use next."""

    def __init__(self, returns: pd.Series, weights: pd.DataFrame, latest: dict[str, float]) -> None:
        self.returns = returns
        self.weights = weights
        self.latest = latest


def replay(agents: pd.DataFrame, floor: float = FLOOR, temperature: float = TEMPERATURE) -> Replay:
    """Run UCB1 over the agents' daily returns, a round a month, using only the months before.

    The weights of a month come from the rewards of the months before it; at its end each agent's
    return for the month becomes one more reward. Daily returns blend the agents' by those weights.
    """
    rewards: dict[str, list[float]] = {name: [] for name in agents}
    monthly: list[float] = []
    chosen: dict[pd.Period, list[float]] = {}
    parts: list[pd.Series] = []
    for month, block in agents.groupby(agents.index.to_period("M")):
        mix = allocate([rewards[name] for name in agents], floor, temperature)
        chosen[month] = mix
        parts.append(block @ np.array(mix))
        earned = (1 + block).prod() - 1
        monthly += earned.tolist()
        spread = float(np.std(monthly)) if len(monthly) >= MIN_ROUNDS else FIRST_SPREAD
        for name in agents:
            rewards[name].append(reward(float(earned[name]), max(spread, 1e-4)))
    latest = allocate([rewards[name] for name in agents], floor, temperature)
    return Replay(
        pd.concat(parts) if parts else pd.Series(dtype=float),
        pd.DataFrame(chosen, index=list(agents)).T,
        dict(zip(agents, latest, strict=True)),
    )
