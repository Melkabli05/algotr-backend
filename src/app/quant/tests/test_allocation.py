import numpy as np
import pytest

from app.quant import allocation


def test_uncorrelated_assets_get_weights_inverse_to_their_volatility() -> None:
    covariance = np.diag([0.01**2, 0.03**2])
    weights = allocation.equal_risk(covariance)
    assert weights == pytest.approx([0.75, 0.25], abs=1e-4)


def test_every_asset_contributes_the_same_risk() -> None:
    vols = np.array([0.01, 0.02, 0.04])
    correlation = np.array([[1, 0.8, 0.1], [0.8, 1, 0.2], [0.1, 0.2, 1]])
    covariance = correlation * np.outer(vols, vols)
    weights = allocation.equal_risk(covariance)
    assert weights.sum() == pytest.approx(1)
    assert allocation.risk_contributions(weights, covariance) == pytest.approx(
        [1 / 3] * 3, abs=1e-4
    )


def test_two_assets_that_move_together_share_one_budget() -> None:
    vols = np.array([0.01, 0.01, 0.01])
    correlation = np.array([[1, 0.95, 0], [0.95, 1, 0], [0, 0, 1]])
    weights = allocation.equal_risk(correlation * np.outer(vols, vols))
    assert weights[2] > weights[0] + weights[1] - 0.2
    assert weights[0] == pytest.approx(weights[1], abs=1e-4)


def test_a_single_asset_takes_everything() -> None:
    assert allocation.equal_risk(np.array([[0.04]])).tolist() == [1.0]
