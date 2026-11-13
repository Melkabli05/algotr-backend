from collections.abc import Collection
from typing import NamedTuple

import numpy as np
import pandas as pd
from pydantic import ValidationError

from app.core import audit
from app.features.scanner import features, rank, universe
from app.features.scanner.features import Features
from app.features.scanner.schemas import Evaluation, VariantResult
from app.quant import performance, validation
from app.quant.history import NotEnoughHistoryError

FEE_BPS = 10.0
MIN_DAYS = 500
TRIALS = len(rank.VARIANTS)
YEAR = features.YEAR
NOTES = [
    "The Shariah lists are today's: a stock dropped from them in the past is missing, and one "
    "that joined after a rise is present throughout.",
    "Delisted stocks are missing from Alpaca's data. Both gaps flatter every result, the "
    "benchmark included.",
    "Judge a variant against the benchmark, which holds the same names without picking.",
]


class Run(NamedTuple):
    returns: pd.Series
    held: pd.DataFrame


def run_variant(
    found: Features, ok: pd.DataFrame, scored: dict[str, pd.DataFrame], v: rank.Variant
) -> Run:
    chosen = rank.select(scored[v.score], ok, v.top_n)
    target = rank.targets(rank.weights(chosen, found.vol60), ok)
    held = target.shift(1).fillna(0.0)
    return Run(performance.strategy_returns(found.close, held, FEE_BPS, found.open), held)


def run_benchmark(found: Features) -> Run:
    """Every liquid name in the set, equally weighted every day, without costs."""
    liquid = (found.close >= universe.MIN_PRICE) & (found.adv20 >= universe.MIN_DOLLAR_VOLUME)
    weights = liquid.astype(float).div(liquid.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    return Run(performance.strategy_returns(found.close, held, 0.0), held)


def first_holding(held: pd.DataFrame) -> pd.Timestamp:
    invested = held.sum(axis=1) > 0
    if not invested.any():
        raise NotEnoughHistoryError("No variant ever holds anything on this data")
    return pd.Timestamp(invested.idxmax())


def evaluate(panel: pd.DataFrame, members: Collection[str], lists: list[str]) -> Evaluation:
    """Run the eight pre-registered variants on the Shariah-listed names and judge them together."""
    named = panel[panel["symbol"].isin(members)]
    if named.empty:
        raise NotEnoughHistoryError("None of the listed names has price history yet")
    found = features.compute(named)
    ok = rank.eligible(found)
    scored = rank.scores(found, ok)
    runs = {v.label: run_variant(found, ok, scored, v) for v in rank.VARIANTS}
    start = max(first_holding(run.held) for run in runs.values())
    window = found.close.index[found.close.index >= start]
    if len(window) < MIN_DAYS:
        raise NotEnoughHistoryError(f"Need {MIN_DAYS} sessions of holdings; have {len(window)}")
    close, opens = found.close.loc[window], found.open.loc[window]
    bench = run_benchmark(found)
    bench_held = bench.held.loc[window]
    bench_metrics = performance.metrics(bench.returns.loc[window], bench_held, YEAR)
    split = int(len(window) * performance.IN_SAMPLE_SHARE)
    results = []
    for v in rank.VARIANTS:
        returns, held = runs[v.label].returns.loc[window], runs[v.label].held.loc[window]
        overall = performance.metrics(returns, held, YEAR)
        out_of_sample = performance.metrics(returns.iloc[split:], held.iloc[split:], YEAR)
        sensitivity = [
            performance.sensitivity_at(bps, close, held, YEAR, opens)
            for bps in performance.SENSITIVITY_FEES
        ]
        deflated = validation.deflated_sharpe(returns, TRIALS)
        results.append(
            VariantResult(
                label=v.label,
                score=v.score,
                top_n=v.top_n,
                metrics=overall,
                out_of_sample=out_of_sample,
                sensitivity=sensitivity,
                t_statistic=validation.sharpe_t_statistic(returns),
                deflated_sharpe=deflated,
                checks=performance.checks(
                    overall,
                    bench_metrics,
                    out_of_sample,
                    sensitivity,
                    deflated,
                    TRIALS,
                    validation.sharpe_difference_p(returns, bench.returns.loc[window]),
                ),
            )
        )
    candidates = pd.DataFrame({label: run.returns.loc[window] for label, run in runs.items()})
    return Evaluation(
        start=window[0].date(),
        end=window[-1].date(),
        days=len(window),
        symbols=int(found.close.shape[1]),
        lists=lists,
        trials=TRIALS,
        luck_bar=float(validation.expected_max_sharpe(TRIALS, len(window)) * np.sqrt(YEAR)),
        probability_of_overfitting=validation.probability_of_overfitting(candidates),
        benchmark=bench_metrics,
        variants=results,
        notes=NOTES,
    )


def latest(log: audit.AuditLog) -> Evaluation | None:
    """The newest stored evaluation; one written by an incompatible older version reads as none."""
    for entry in reversed(log.entries()):
        if entry.kind == "scan-evaluation":
            try:
                return Evaluation.model_validate(entry.payload["result"])
            except ValidationError:
                return None
    return None
