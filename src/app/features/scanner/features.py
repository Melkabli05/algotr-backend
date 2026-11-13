from dataclasses import dataclass

import numpy as np
import pandas as pd

TREND_WINDOW = 200
LONG = 252
MEDIUM = 126
SKIP = 21
CLENOW_WINDOW = 90
GAP_WINDOW = 90
ADV_WINDOW = 20
VOL_WINDOW = 60
BETA_WINDOW = 252
YEAR = 252


@dataclass(frozen=True)
class Features:
    """Per date and symbol, everything the ranking reads; each value uses data up to that date."""

    close: pd.DataFrame
    open: pd.DataFrame
    trend: pd.DataFrame
    mom_12_1: pd.DataFrame
    mom_6_1: pd.DataFrame
    clenow_90: pd.DataFrame
    resmom_12_1: pd.DataFrame
    adv20: pd.DataFrame
    gap90: pd.DataFrame
    vol60: pd.DataFrame
    history: pd.DataFrame


def wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """One column per symbol, one row per session."""
    return panel.pivot(index="date", columns="symbol", values=field).sort_index()


def momentum(close: pd.DataFrame, window: int, skip: int = SKIP) -> pd.DataFrame:
    """Return from `window` sessions ago to `skip` sessions ago, leaving out the latest month."""
    return close.shift(skip) / close.shift(window) - 1


def clenow(close: pd.DataFrame, window: int = CLENOW_WINDOW) -> pd.DataFrame:
    """Annualized exponential-regression slope times R-squared (Clenow's momentum score).

    The slope of log price on the session number comes from rolling sums, which is the least
    squares formula with the window's own positions as the regressor.
    """
    logs = np.log(close)
    position = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    first = position.shift(window - 1)
    centre = (window - 1) / 2
    spread = window * (window**2 - 1) / 12
    total = logs.rolling(window).sum()
    weighted = logs.mul(position, axis=0).rolling(window).sum()
    slope = (weighted.sub(total.mul(first + centre, axis=0))) / spread
    r_squared = slope**2 * spread / (logs.rolling(window).var(ddof=0) * window)
    score = (np.exp(slope * YEAR) - 1) * r_squared
    return score.where(np.isfinite(score))


def residual_momentum(
    close: pd.DataFrame,
    window: int = LONG,
    skip: int = SKIP,
    beta_window: int = BETA_WINDOW,
) -> pd.DataFrame:
    """Momentum of what is left of each stock's return once the market's part is taken out.

    The market is the equal-weight return of the names given. Each day's leftover is the stock's
    return less yesterday's rolling beta times the market's, so a score never uses a later day.
    The score is the sum of the leftovers from `window` to `skip` sessions ago over their
    standard deviation, which ranks stocks with a steady edge above one with a lucky month
    (Blitz, Huij and Martens, 2011, with the market as the only factor).
    """
    daily = close.pct_change()
    market = daily.mean(axis=1)
    needed = beta_window // 2
    beta = (
        daily.rolling(beta_window, min_periods=needed)
        .cov(market)
        .div(market.rolling(beta_window, min_periods=needed).var(), axis=0)
    )
    leftover = (daily - beta.shift(1).mul(market, axis=0)).shift(skip)
    span = window - skip
    score = leftover.rolling(span).sum() / leftover.rolling(span).std()
    return score.where(np.isfinite(score))


def compute(panel: pd.DataFrame) -> Features:
    close, volume = wide(panel, "close"), wide(panel, "volume")
    daily = close.pct_change()
    return Features(
        close=close,
        open=wide(panel, "open"),
        trend=close > close.rolling(TREND_WINDOW).mean(),
        mom_12_1=momentum(close, LONG),
        mom_6_1=momentum(close, MEDIUM),
        clenow_90=clenow(close),
        resmom_12_1=residual_momentum(close),
        adv20=(close * volume).rolling(ADV_WINDOW).mean(),
        gap90=daily.abs().rolling(GAP_WINDOW).max(),
        vol60=daily.rolling(VOL_WINDOW).std(),
        history=close.notna().cumsum(),
    )
