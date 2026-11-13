import numpy as np
import pandas as pd

from app.features.systematic.schemas import StrategySpec

VOLATILITY_WINDOW = 20


def sma_trend(close: pd.Series, window: int) -> pd.Series:
    """1 while the close is above its trailing average, else 0."""
    return (close > close.rolling(window).mean()).astype(float)


def donchian_state(close: pd.Series, lookback: int) -> pd.Series:
    """Long on a close above the prior `lookback`-day high, out below a trailing midpoint stop."""
    high = close.rolling(lookback).max()
    midpoint = (high + close.rolling(lookback).min()) / 2
    breakout = high.shift(1).to_numpy()
    prices, midpoints = close.to_numpy(), midpoint.to_numpy()
    state = np.zeros(len(close))
    holding, stop = False, -np.inf
    for index, price in enumerate(prices):
        if holding:
            stop = max(stop, midpoints[index])
            holding = price >= stop
        elif price > breakout[index]:
            holding, stop = True, midpoints[index]
        state[index] = holding
    return pd.Series(state, index=close.index)


def donchian_ensemble(close: pd.Series, lookbacks: list[int]) -> pd.Series:
    return pd.concat([donchian_state(close, n) for n in lookbacks], axis=1).mean(axis=1)


def trend_votes(close: pd.Series, lookbacks: list[int]) -> pd.Series:
    """The share of the trend windows in which the close is above its average, from 0 to 1."""
    return pd.concat([sma_trend(close, n) for n in lookbacks], axis=1).mean(axis=1)


def volatility_scale(
    closes: pd.DataFrame,
    target_vol: float,
    periods_per_year: int,
    window: int = VOLATILITY_WINDOW,
) -> pd.Series:
    """Exposure that targets `target_vol` for an equal-weight basket, capped at 1: no leverage."""
    basket = closes.pct_change().mean(axis=1)
    realized = basket.rolling(window).std() * np.sqrt(periods_per_year)
    return (target_vol / realized).clip(upper=1.0).fillna(0.0)


def momentum_selection(closes: pd.DataFrame, window: int, top_n: int) -> pd.DataFrame:
    """Dual momentum: the `top_n` best performers over `window` days, only if they gained."""
    gain = closes.pct_change(window)
    rank = gain.rank(axis=1, ascending=False, method="first")
    return ((rank <= top_n) & (gain > 0)).astype(float) / top_n


def no_trade_band(aim: pd.DataFrame, width: float) -> pd.DataFrame:
    """Hold each weight until the aim has moved more than `width` away from it, then move to it."""
    targets = aim.to_numpy()
    held = np.zeros(targets.shape)
    current = np.zeros(targets.shape[1])
    for index, row in enumerate(targets):
        current = np.where(np.abs(row - current) > width, row, current)
        held[index] = current
    return pd.DataFrame(held, index=aim.index, columns=aim.columns)


def partial_adjustment(aim: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Move each weight `rate` of the way to its aim every day (Garleanu and Pedersen, 2013)."""
    return aim.ewm(alpha=rate, adjust=False).mean()


def raw_weights(closes: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    if spec.signal == "momentum":
        return momentum_selection(closes, spec.window, spec.top_n)
    count = len(closes.columns)
    if spec.signal == "sma":
        return pd.DataFrame({s: sma_trend(closes[s], spec.window) for s in closes}) / count
    if spec.signal == "multitrend":
        return pd.DataFrame({s: trend_votes(closes[s], spec.lookbacks) for s in closes}) / count
    return pd.DataFrame({s: donchian_ensemble(closes[s], spec.lookbacks) for s in closes}) / count


def weights(closes: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Share of the strategy's capital each asset should hold at every close, summing to at most 1.

    With a `target_vol` the exposure is scaled to that annual volatility and never above 1x, so
    risk stays steady and nothing is borrowed. Each row uses only prices up to that close.
    """
    raw = raw_weights(closes, spec)
    if spec.target_vol is not None:
        scale = volatility_scale(closes, spec.target_vol, spec.periods_per_year, spec.vol_window)
        raw = raw.mul(scale, axis=0)
    if spec.trade_rate < 1.0:
        raw = partial_adjustment(raw, spec.trade_rate)
    return no_trade_band(raw, spec.band) if spec.band else raw


def warmup_days(spec: StrategySpec) -> int:
    """Bars a rule needs before its first valid weight."""
    longest = max(spec.lookbacks) if spec.signal in ("donchian", "multitrend") else spec.window
    return longest + max(VOLATILITY_WINDOW, spec.vol_window)
