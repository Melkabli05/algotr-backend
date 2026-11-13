import empyrical
import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.features.systematic import lab, signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import StrategyName
from app.quant import performance, validation

FAMILY: StrategyName = "halal-etf-trend"
STABLE_SHARE = 0.8
GENERALIZATION = 0.5
CONSISTENT_SHARE = 0.75
PERIODS = 252


def sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe; a series that never moves (an agent holding only cash) scores 0."""
    value = float(empyrical.sharpe_ratio(returns, annualization=PERIODS))
    return value if np.isfinite(value) else 0.0


class Gate(BaseModel):
    name: str
    passed: bool
    detail: str


class Row(BaseModel):
    label: str
    in_sample_sharpe: float | None
    sharpe: float
    cagr: float
    max_drawdown: float
    beats_trend: bool


class Judgement(BaseModel):
    """How a set of candidate return series fares against the rule they would replace."""

    start: str
    end: str
    days: int
    rows: list[Row]
    trend_sharpe: float
    hold_sharpe: float
    reliability: dict[str, float]
    reality_check_p: float
    probability_of_overfitting: float
    deflated_sharpe: float
    trials: int
    gates: list[Gate]
    eligible: bool


def judge(
    closes: pd.DataFrame,
    candidates: pd.DataFrame,
    in_sample: dict[str, float | None],
    trials: int,
) -> Judgement:
    """Apply the gates to candidates (one column each) over the dates they all cover.

    The same five gates apply whatever produced the returns: our own agents, FinRL's agents, its
    ensemble or its portfolio environments. A candidate without training figures cannot show that
    it holds up on unseen data, so that gate fails rather than passes by default.
    """
    spec = STRATEGIES[FAMILY]
    candidates = candidates.dropna()
    window = candidates.index
    held = signals.weights(closes, spec).shift(1).fillna(0.0)
    trend = performance.strategy_returns(closes, held, spec.fee_bps).loc[window]
    hold = closes.pct_change().mean(axis=1).loc[window]
    sharpes = {label: sharpe(candidates[label]) for label in candidates}
    trend_sharpe, hold_sharpe = sharpe(trend), sharpe(hold)
    rows = [
        Row(
            label=str(label),
            in_sample_sharpe=in_sample.get(str(label)),
            sharpe=sharpes[label],
            cagr=float(empyrical.annual_return(candidates[label], annualization=PERIODS)),
            max_drawdown=float(empyrical.max_drawdown(candidates[label])),
            beats_trend=sharpes[label] > trend_sharpe,
        )
        for label in candidates
    ]
    median_label = sorted(sharpes, key=lambda label: sharpes[label])[len(sharpes) // 2]
    reliable = validation.reliability(list(sharpes.values()), floor=trend_sharpe)
    superior = lab.superior_models(-trend, -candidates)
    p_value = lab.reality_check_p(-trend, -candidates)
    overfit = validation.probability_of_overfitting(candidates) if candidates.shape[1] > 1 else 0.0
    steady = validation.consistency(candidates[median_label])
    deflated = validation.deflated_sharpe(candidates[median_label], trials)
    known = [value for value in in_sample.values() if value is not None]
    median_in = float(np.median(known)) if known else None
    ratio = reliable["median"] / median_in if median_in and median_in > 0 else None
    count = len(sharpes)
    gates = [
        Gate(
            name="Stable across seeds",
            passed=reliable["share_above"] >= STABLE_SHARE,
            detail=f"{round(reliable['share_above'] * count)} of {count} seeds beat "
            f"the trend rule (needs {STABLE_SHARE:.0%})",
        ),
        Gate(
            name="Beats the trend rule after the search",
            passed=str(median_label) in superior,
            detail=f"chance the best agent is luck against trend: {p_value:.1%}",
        ),
        Gate(
            name="Real skill after counting every trial",
            passed=deflated >= validation.CONFIDENCE,
            detail=f"{deflated:.1%} sure after {trials} trials (needs {validation.CONFIDENCE:.0%})",
        ),
        Gate(
            name="Holds up on data it never saw",
            passed=ratio is not None and ratio >= GENERALIZATION,
            detail="training performance was not reported, so this cannot be shown"
            if ratio is None
            else f"out-of-sample Sharpe is {ratio:.0%} of in-sample (needs {GENERALIZATION:.0%})",
        ),
        Gate(
            name="Consistent through time",
            passed=steady >= CONSISTENT_SHARE,
            detail=f"profitable in {steady:.0%} of the time slices (needs {CONSISTENT_SHARE:.0%})",
        ),
    ]
    return Judgement(
        start=window[0].isoformat(),
        end=window[-1].isoformat(),
        days=len(window),
        rows=rows,
        trend_sharpe=trend_sharpe,
        hold_sharpe=hold_sharpe,
        reliability=reliable,
        reality_check_p=p_value,
        probability_of_overfitting=overfit,
        deflated_sharpe=deflated,
        trials=trials,
        gates=gates,
        eligible=all(gate.passed for gate in gates),
    )
