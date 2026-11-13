import numpy as np
import pandas as pd

from app.quant.regimes import fit, label_states


def test_labels_follow_variance_then_mean() -> None:
    assert label_states(np.array([0.1, -0.3, 0.0]), np.array([0.6, 13.0, 2.4])) == [
        "trending",
        "volatile",
        "ranging",
    ]


def test_a_calm_rise_and_a_crash_land_in_different_states() -> None:
    rng = np.random.default_rng(3)
    calm = rng.normal(0.001, 0.005, 400)
    crash = rng.normal(-0.004, 0.04, 60)
    days = pd.bdate_range("2023-01-02", periods=461)
    closes = pd.Series(100 * np.cumprod(np.concatenate([[1.0], 1 + calm, 1 + crash])), index=days)
    found = fit(closes)
    labels = found.labels()
    assert labels.iloc[100:390].value_counts().idxmax() != "volatile"
    assert labels.iloc[-30:].value_counts().idxmax() == "volatile"
    assert np.allclose(found.probabilities.sum(axis=1), 1)
