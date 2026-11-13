import numpy as np
import pandas as pd
import pytest

from app.quant import risk


def test_the_var_of_a_normal_series_matches_both_methods() -> None:
    returns = pd.Series(np.random.default_rng(4).normal(0.0, 0.01, 20_000))
    found = risk.value_at_risk(returns, 0.95, 1)
    assert found["historical"] == pytest.approx(0.01645, rel=0.05)
    assert found["parametric"] == pytest.approx(0.01645, rel=0.05)
    assert found["shortfall"] > found["historical"]


def test_a_longer_horizon_or_a_higher_confidence_risks_more() -> None:
    returns = pd.Series(np.random.default_rng(5).normal(0.0, 0.01, 5_000))
    one = risk.value_at_risk(returns, 0.95, 1)["historical"]
    assert risk.value_at_risk(returns, 0.95, 5)["historical"] > one
    assert risk.value_at_risk(returns, 0.99, 1)["historical"] > one


def test_the_portfolio_holds_todays_weights_through_the_history() -> None:
    returns = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.0]})
    found = risk.portfolio_returns(returns, pd.Series({"A": 0.5, "B": 0.25}))
    assert found.tolist() == pytest.approx([0.0125, -0.01])
