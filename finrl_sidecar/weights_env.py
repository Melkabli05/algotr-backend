"""The weight-vector environment of the PPO, SAC and A2C agents, and the features it shows them.

FinRL's portfolio environments do not fit: `StockPortfolioEnv` rewards the portfolio value
instead of a return, never applies its transaction cost and shows no portfolio state, and
`PortfolioOptimizationEnv` speaks the old gym API for FinRL's own agent. So this small Gymnasium
environment holds what the agents need: the weights over the assets and cash, a cost on turnover,
and the weights the portfolio holds now in the observation.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

SCALE = 4.0
REWARD_SCALE = 100.0
CLIP = 5.0
EPISODE_DAYS = 252
LOOKBACKS = (1, 5, 20, 60)


def softmax(values: np.ndarray) -> np.ndarray:
    """Along the last axis, so one vector or a row of them."""
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def agreement(
    sampled: np.ndarray, chosen: np.ndarray, held: np.ndarray, move: float = 0.02
) -> np.ndarray:
    """For each weight, the share of sampled weights that make the chosen move from `held`.

    A move of at least `move` counts by its direction; a smaller one is a hold, matched by
    samples that also stay within `move` of what is held.
    """
    change = chosen - held
    drift = sampled - held
    moving = np.abs(change) >= move
    same = np.where(moving, np.sign(drift) == np.sign(change), np.abs(drift) < move)
    return same.mean(axis=0)


def policy_confidence(
    agent: object, observation: np.ndarray, held: np.ndarray, samples: int = 256, seed: int = 0
) -> np.ndarray:
    """How often the agent's own stochastic policy makes the move its deterministic action makes."""
    import torch

    torch.manual_seed(seed)
    chosen, _ = agent.predict(observation, deterministic=True)
    batch = np.repeat(np.asarray(observation)[None, :], samples, axis=0)
    drawn, _ = agent.predict(batch, deterministic=False)
    return agreement(
        softmax(SCALE * np.asarray(drawn, dtype=float)),
        softmax(SCALE * np.asarray(chosen, dtype=float)),
        np.asarray(held, dtype=float),
    )


def wide(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    table = frame.pivot(index="date", columns="tic", values=column)
    table.index = pd.to_datetime(table.index)
    return table


def asset_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and daily returns, a row for each day, from FinRL's long layout with indicators.

    When the frame carries the market's headline `sentiment` (FinBERT, -1 to +1) and `news_risk`
    (1 to 5), each is one more feature, scaled to about -1 to +1; a day without headlines reads
    as neutral (0).
    """
    close = wide(frame, "close")
    returns = close.pct_change()
    blocks = {f"ret_{days}": close.pct_change(days) for days in LOOKBACKS}
    blocks["vol_20"] = returns.rolling(20).std()
    blocks["macd"] = wide(frame, "macd") / close
    bands = wide(frame, "boll_ub") - wide(frame, "boll_lb")
    blocks["boll"] = (close - wide(frame, "boll_lb")) / bands - 0.5
    blocks["rsi"] = wide(frame, "rsi_30") / 100 - 0.5
    blocks["cci"] = wide(frame, "cci_30") / 200
    blocks["dx"] = wide(frame, "dx_30") / 100
    blocks["sma_30"] = close / wide(frame, "close_30_sma") - 1
    blocks["sma_60"] = close / wide(frame, "close_60_sma") - 1
    features = pd.concat(blocks, axis=1)
    features.columns = [f"{name}:{tic}" for name, tic in features.columns]
    if "sentiment" in frame.columns:
        features["sentiment"] = wide(frame, "sentiment").mean(axis=1).fillna(0.0)
        features["news_risk"] = ((wide(frame, "news_risk").mean(axis=1) - 1) / 4).fillna(0.0)
    days = features.dropna().index.intersection(returns.dropna().index)
    return features.loc[days], returns.loc[days]


class WeightsEnv(gym.Env):
    """Each day the agent splits the portfolio over the assets and cash: net log return."""

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        cost: float,
        *,
        episode_days: int | None = None,
        initial: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.features, self.returns, self.cost = features, returns, cost
        self.assets = returns.shape[1]
        self.episode_days = episode_days
        self.initial = np.eye(self.assets + 1)[-1] if initial is None else np.asarray(initial)
        self.action_space = spaces.Box(-1.0, 1.0, (self.assets + 1,), np.float32)
        width = features.shape[1] + self.assets + 1
        self.observation_space = spaces.Box(-CLIP, CLIP, (width,), np.float32)
        self.day = self.end = 0
        self.held = self.initial.copy()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        length = len(self.features)
        if self.episode_days and length > self.episode_days + 1:
            self.day = int(self.np_random.integers(0, length - self.episode_days))
            self.end = self.day + self.episode_days
        else:
            self.day, self.end = 0, length - 1
        self.held = self.initial.copy()
        return self.observation(), {}

    def observation(self) -> np.ndarray:
        shown = np.concatenate([self.features[self.day], self.held])
        return np.clip(shown, -CLIP, CLIP).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        target = softmax(SCALE * np.asarray(action, dtype=float))
        turnover = float(np.abs(target - self.held).sum())
        moves = self.returns[self.day + 1]
        net = float(target[:-1] @ moves) - self.cost * turnover
        grown = target * np.append(1 + moves, 1.0)
        self.held = grown / grown.sum()
        self.day += 1
        reward = REWARD_SCALE * float(np.log1p(max(net, -0.99)))
        return (
            self.observation(),
            reward,
            self.day >= self.end,
            False,
            {"net": net, "weights": target},
        )


def roll_out(
    model: Any,
    features: np.ndarray,
    returns: np.ndarray,
    cost: float,
    initial: np.ndarray,
    seen: list[np.ndarray] | None = None,
) -> tuple[list[float], list[list[float]], np.ndarray]:
    """Run a trained policy through the days given: net returns, weights, weights held after.

    When `seen` is given, each observation the policy acted on is appended to it.
    """
    env = WeightsEnv(features, returns, cost, initial=initial)
    observation, _ = env.reset()
    nets: list[float] = []
    weights: list[list[float]] = []
    done = False
    while not done:
        if seen is not None:
            seen.append(observation)
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _, info = env.step(action)
        nets.append(info["net"])
        weights.append(info["weights"].tolist())
    return nets, weights, env.held
