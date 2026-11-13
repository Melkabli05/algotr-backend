import empyrical
import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = (1, 5)
CONFIDENCES = (0.95, 0.99)


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily returns of holding today's weights through the history given."""
    aligned = returns[weights.index].dropna()
    return aligned @ weights


def compounded(returns: pd.Series, horizon: int) -> pd.Series:
    """Overlapping `horizon`-day returns."""
    return (1 + returns).rolling(horizon).apply(np.prod, raw=True).dropna() - 1


def value_at_risk(returns: pd.Series, confidence: float, horizon: int) -> dict[str, float]:
    """Loss not exceeded with `confidence` over `horizon` days: historical, normal, and the mean
    loss beyond the historical one (expected shortfall), each as a positive fraction."""
    spans = compounded(returns, horizon)
    cutoff = 1 - confidence
    mean, spread = float(returns.mean()), float(returns.std(ddof=1))
    normal = -(mean * horizon + float(stats.norm.ppf(cutoff)) * spread * np.sqrt(horizon))
    return {
        "historical": max(-float(empyrical.value_at_risk(spans, cutoff=cutoff)), 0.0),
        "parametric": max(normal, 0.0),
        "shortfall": max(-float(empyrical.conditional_value_at_risk(spans, cutoff=cutoff)), 0.0),
    }
