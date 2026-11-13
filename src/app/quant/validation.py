import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from scipy import stats

EULER_GAMMA = 0.5772156649015329
CONFIDENCE = 0.95


def probabilistic_sharpe(returns: pd.Series, benchmark: float = 0.0) -> float:
    """Chance the true per-period Sharpe ratio beats `benchmark`, allowing for skew and fat tails.

    Bailey and López de Prado, "The Sharpe ratio efficient frontier" (2012).
    """
    if returns.std(ddof=1) == 0:
        return 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1))
    skew = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns, fisher=False))
    spread = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    return float(stats.norm.cdf((sharpe - benchmark) * np.sqrt(len(returns) - 1) / np.sqrt(spread)))


def expected_max_sharpe(trials: int, observations: int, spread: float | None = None) -> float:
    """The best per-period Sharpe ratio that `trials` strategies with no skill would show by luck.

    The trials' Sharpe ratios scatter by `spread`, the measured standard deviation of their
    per-period Sharpe ratios, but never by less than one standard error of a Sharpe estimate:
    a narrow measured spread from near-identical variants must not lower the bar.
    """
    if trials < 2:
        return 0.0
    spread = max(1 / np.sqrt(observations), spread or 0.0)
    return float(
        spread
        * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / trials)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (trials * np.e))
        )
    )


def deflated_sharpe(returns: pd.Series, trials: int, spread: float | None = None) -> float:
    """Chance of real skill after discounting the luck of picking the best of `trials` strategies.

    Bailey and López de Prado, "The Deflated Sharpe Ratio" (2014).
    """
    return probabilistic_sharpe(returns, expected_max_sharpe(trials, len(returns), spread))


HURDLE_T = 3.0
PBO_PARTITIONS = 8


def sharpe_t_statistic(returns: pd.Series) -> float:
    """The Sharpe ratio expressed as a t-statistic: per-period Sharpe times the square root of T.

    Harvey, Liu and Zhu (2016) argue a new finding needs a t-statistic above 3, not 2.
    """
    if returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns)))


def probability_of_overfitting(returns: pd.DataFrame, partitions: int = PBO_PARTITIONS) -> float:
    """How often the best candidate in one half of the history ranks below the median in the other.

    Combinatorially symmetric cross-validation (Bailey, Borwein, López de Prado and Zhu, 2017):
    every way of choosing half of `partitions` equal time slices as the training set is tried.
    A value near 0 means the winner stays a winner; near 0.5 or more means picking it was luck.
    """
    if returns.shape[1] < 2:
        raise ValueError("Overfitting needs at least two candidates to compare")
    slices = np.array_split(np.arange(len(returns)), partitions)
    values = returns.to_numpy()
    logits = []
    for train_ids in itertools.combinations(range(partitions), partitions // 2):
        train = np.concatenate([slices[i] for i in train_ids])
        test = np.concatenate([slices[i] for i in range(partitions) if i not in train_ids])
        best = int(np.argmax(sharpe_of(values[train])))
        rank = stats.rankdata(sharpe_of(values[test]))[best] / (values.shape[1] + 1)
        logits.append(np.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))


def sharpe_of(values: np.ndarray) -> np.ndarray:
    spread = values.std(axis=0, ddof=1)
    return np.divide(values.mean(axis=0), spread, out=np.zeros(values.shape[1]), where=spread > 0)


def consistency(returns: pd.Series, partitions: int = PBO_PARTITIONS) -> float:
    """Share of equal time slices in which the strategy made money.

    A high Sharpe earned in one lucky stretch scores low here; a steady edge scores high.
    """
    slices = np.array_split(returns.to_numpy(), partitions)
    return float(np.mean([piece.mean() > 0 for piece in slices]))


def reliability(sharpes: Sequence[float], floor: float = 0.0) -> dict[str, float]:
    """How stable a result is across independent runs of the same recipe (seeds).

    The spread matters as much as the median: a recipe that is good on average but wildly
    different from seed to seed cannot be trusted with one training run.
    """
    values = np.asarray(sharpes, dtype=float)
    low, high = np.percentile(values, [25, 75])
    return {
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "spread": float(high - low),
        "share_above": float(np.mean(values > floor)),
    }


SHARPE_TEST_LEVEL = 0.10
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260101


def sharpe_gap(values: np.ndarray) -> float:
    spread = values.std(axis=0, ddof=1)
    if np.any(spread == 0):
        return 0.0
    return float(values[:, 0].mean() / spread[0] - values[:, 1].mean() / spread[1])


def sharpe_difference_p(
    strategy: pd.Series, benchmark: pd.Series, reps: int = BOOTSTRAP_REPS
) -> float:
    """Chance of a Sharpe lead this large if the strategy were no better than the benchmark.

    A one-sided test of the difference of two Sharpe ratios on the same dates by the stationary
    bootstrap (Ledoit and Wolf, 2008), which keeps the serial dependence of daily returns; the
    mean block length is Politis and White's optimum for the difference of the two series.
    """
    paired = pd.concat([strategy, benchmark], axis=1, join="inner").dropna().to_numpy()
    observed = sharpe_gap(paired)
    if len(paired) < 3 * 2:
        return 1.0
    gaps = paired[:, 0] - paired[:, 1]
    if np.all(gaps == gaps[0]):
        return 0.0 if observed > 0 else 1.0
    block = float(optimal_block_length(gaps)["stationary"].iloc[0])
    bootstrap = StationaryBootstrap(max(round(block), 1), paired, seed=BOOTSTRAP_SEED)
    draws = bootstrap.apply(sharpe_gap, reps).ravel()
    return float(np.mean(draws - observed >= observed))
