import empyrical
import pandas as pd
from pydantic import BaseModel

from app.quant import validation

SENSITIVITY_FEES = (10.0, 25.0, 50.0)
IN_SAMPLE_SHARE = 0.7


class Metrics(BaseModel):
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    exposure: float
    turnover: float


class Sensitivity(BaseModel):
    fee_bps: float
    cagr: float
    sharpe: float


class Check(BaseModel):
    name: str
    passed: bool


def strategy_returns(
    closes: pd.DataFrame,
    held: pd.DataFrame,
    fee_bps: float,
    opens: pd.DataFrame | None = None,
) -> pd.Series:
    """Daily returns of holding `held` (row t: weights held through day t), after costs.

    Without `opens` a day's new weights earn from the previous close, as if filled there. With
    them the trade fills at the day's open: the night still earns the weights held before.
    """
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    cost = turnover * fee_bps / 10_000
    if opens is None:
        return (held * closes.pct_change()).sum(axis=1) - cost
    before = held.shift(1).fillna(0.0)
    night = (before * (opens / closes.shift(1) - 1)).sum(axis=1)
    day = (held * (closes / opens - 1)).sum(axis=1)
    return (1 + night) * (1 + day) - 1 - cost


def metrics(returns: pd.Series, held: pd.DataFrame, periods_per_year: int) -> Metrics:
    return Metrics(
        total_return=float(empyrical.cum_returns_final(returns)),
        cagr=float(empyrical.annual_return(returns, annualization=periods_per_year)),
        volatility=float(empyrical.annual_volatility(returns, annualization=periods_per_year)),
        sharpe=float(empyrical.sharpe_ratio(returns, annualization=periods_per_year)),
        max_drawdown=float(empyrical.max_drawdown(returns)),
        exposure=float(held.sum(axis=1).mean()),
        turnover=float(held.diff().abs().sum(axis=1).mean() * periods_per_year),
    )


def sensitivity_at(
    fee_bps: float,
    closes: pd.DataFrame,
    held: pd.DataFrame,
    periods_per_year: int,
    opens: pd.DataFrame | None = None,
) -> Sensitivity:
    returns = strategy_returns(closes, held, fee_bps, opens)
    return Sensitivity(
        fee_bps=fee_bps,
        cagr=float(empyrical.annual_return(returns, annualization=periods_per_year)),
        sharpe=float(empyrical.sharpe_ratio(returns, annualization=periods_per_year)),
    )


def checks(
    strategy: Metrics,
    benchmark: Metrics,
    out_of_sample: Metrics,
    sensitivity: list[Sensitivity],
    deflated: float,
    trials: int,
    sharpe_p: float | None = None,
    evidence: str | None = None,
) -> list[Check]:
    """The five gates. With `sharpe_p` the Sharpe lead must also be unlikely to be luck.

    `evidence` names the longer history the two significance checks were run on, when they were.
    """
    on = f", on {evidence}" if evidence else ""
    lead = strategy.sharpe > benchmark.sharpe
    level = validation.SHARPE_TEST_LEVEL
    return [
        Check(
            name="Sharpe above buy and hold"
            if sharpe_p is None
            else f"Sharpe above buy and hold, p under {level:.0%}{on}",
            passed=lead if sharpe_p is None else sharpe_p < level,
        ),
        Check(
            name="Smaller max drawdown than buy and hold",
            passed=strategy.max_drawdown > benchmark.max_drawdown,
        ),
        Check(name="Profitable in the last 30%", passed=out_of_sample.cagr > 0),
        Check(name="Profitable at 50 bps per trade", passed=sensitivity[-1].cagr > 0),
        Check(
            name=f"Deflated Sharpe above {validation.CONFIDENCE:.0%} ({trials} trials{on})",
            passed=deflated >= validation.CONFIDENCE,
        ),
    ]
