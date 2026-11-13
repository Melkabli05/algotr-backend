import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

STATES = 3
SEARCHES = 10
SEED = 7
LABELS = ("trending", "ranging", "volatile")


@dataclass(frozen=True)
class State:
    label: str
    mean: float
    volatility: float
    stay: float

    @property
    def expected_days(self) -> float:
        return 1 / (1 - self.stay) if self.stay < 1 else float("inf")


@dataclass(frozen=True)
class Regimes:
    """A three-state hidden Markov model of daily returns and each day's filtered probabilities.

    Each state has its own mean and variance and a Markov chain moves between them. The
    probabilities on a day use only returns up to that day; the parameters are fitted on the
    whole history.
    """

    states: list[State]
    probabilities: pd.DataFrame

    def labels(self) -> pd.Series:
        return self.probabilities.idxmax(axis=1)


def label_states(means: np.ndarray, variances: np.ndarray) -> list[str]:
    """The highest variance is volatile; of the rest, the higher mean is trending."""
    order = list(np.argsort(variances))
    volatile = order[-1]
    calm = sorted(order[:-1], key=lambda index: -means[index])
    found = [""] * len(means)
    found[volatile] = "volatile"
    found[calm[0]] = "trending"
    for index in calm[1:]:
        found[index] = "ranging"
    return found


def fit(closes: pd.Series) -> Regimes:
    returns = closes.dropna().pct_change().dropna() * 100
    model = MarkovRegression(returns.to_numpy(), k_regimes=STATES, switching_variance=True)
    np.random.seed(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(search_reps=SEARCHES, disp=False)
    names = list(model.param_names)
    params = np.asarray(result.params)
    means = np.array([params[names.index(f"const[{state}]")] for state in range(STATES)])
    variances = np.array([params[names.index(f"sigma2[{state}]")] for state in range(STATES)])
    stays = np.diag(np.asarray(result.regime_transition)[:, :, 0])
    labels = label_states(means, variances)
    filtered = np.asarray(result.filtered_marginal_probabilities)
    probabilities = pd.DataFrame(filtered, index=returns.index, columns=labels)
    states = [
        State(
            label=labels[index],
            mean=float(means[index]) / 100,
            volatility=float(np.sqrt(variances[index])) / 100,
            stay=float(stays[index]),
        )
        for index in range(STATES)
    ]
    return Regimes(states=states, probabilities=probabilities)
