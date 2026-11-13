import numpy as np
from scipy.optimize import minimize


def equal_risk(covariance: np.ndarray) -> np.ndarray:
    """Long-only weights summing to 1 whose contributions to portfolio risk are all equal.

    Spinu's convex form (2013): minimise half the portfolio variance minus the mean log weight;
    the minimiser, scaled to sum to 1, is the equal risk contribution portfolio. Correlation
    counts, so two strategies that move together share one risk budget.
    """
    count = len(covariance)
    if count == 1:
        return np.ones(1)
    scale = float(np.mean(np.diag(covariance)))
    sigma = covariance / scale

    def objective(w: np.ndarray) -> float:
        return float(0.5 * w @ sigma @ w - np.log(w).sum() / count)

    def gradient(w: np.ndarray) -> np.ndarray:
        return np.asarray(sigma @ w - 1 / (count * w))

    start = 1 / np.sqrt(np.diag(sigma))
    found = minimize(
        objective,
        start / start.sum(),
        jac=gradient,
        bounds=[(1e-9, None)] * count,
        method="L-BFGS-B",
    )
    weights = np.asarray(found.x)
    return np.asarray(weights / weights.sum())


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Each weight's share of the portfolio variance."""
    marginal = covariance @ weights
    return np.asarray(weights * marginal / (weights @ marginal))
