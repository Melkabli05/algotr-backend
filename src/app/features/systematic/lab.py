import time
from collections.abc import Collection
from typing import Any

import empyrical
import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from arch.bootstrap import SPA
from pydantic import BaseModel, ValidationError

from app.core import audit
from app.features.systematic import backtest, parallel, signals
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import StrategyName, StrategySpec
from app.quant import history, performance, validation

SMA_WINDOWS = (100, 150, 200, 250)
MOMENTUM_WINDOWS = (63, 126, 189, 252)
MOMENTUM_TOP = (1, 2)
TREND_VOLATILITY = (0.10, 0.15)
TREND_SPEEDS = {
    "faster speeds": [25, 50, 100, 150, 200],
    "slower speeds": [100, 150, 200, 250, 300],
}
SIZE = 0.05
BOOTSTRAP_REPS = 1000
BLOCK_SIZE = 20
SEED = 20260101
MIN_DAYS = 250
SEARCHES = ("lab", "sweep")


class Candidate(BaseModel):
    """One variant of a published rule set, identified by what makes it different."""

    label: str
    spec: StrategySpec

    @property
    def config_hash(self) -> str:
        return audit.digest(self.spec.model_dump(mode="json"))


class CandidateResult(BaseModel):
    label: str
    sharpe: float
    cagr: float
    max_drawdown: float
    t_statistic: float
    deflated_sharpe: float
    consistency: float | None = None
    clears_hurdle: bool
    beats_cash: bool
    survives: bool


class FamilyResult(BaseModel):
    family: StrategyName
    title: str
    days: int
    start: str
    end: str
    trials: int
    cumulative_trials: int
    reality_check_p: float
    probability_of_overfitting: float
    candidates: list[CandidateResult]
    candidate_hashes: list[str]


class LabResult(BaseModel):
    families: list[FamilyResult]
    errors: dict[str, str]
    seconds: float
    workers: int


def variants(base: StrategySpec) -> list[Candidate]:
    """The base rule and its neighbours, so the search is as wide as the people who tuned it."""
    if base.signal == "sma":
        return [
            Candidate(
                label=f"{window}-day average", spec=base.model_copy(update={"window": window})
            )
            for window in SMA_WINDOWS
        ]
    if base.signal == "multitrend":
        assert base.target_vol is not None
        steps: dict[str, dict[str, Any]] = {f"{base.target_vol:.0%} volatility": {}}
        steps |= {f"{vol:.0%} volatility": {"target_vol": vol} for vol in TREND_VOLATILITY}
        steps |= {label: {"lookbacks": speeds} for label, speeds in TREND_SPEEDS.items()}
        steps["partial 50% a day"] = {"trade_rate": 0.5}
        steps["5% band"] = {"trade_rate": 1.0, "band": 0.05}
        return [
            Candidate(label=label, spec=base.model_copy(update=change))
            for label, change in steps.items()
        ]
    if base.signal == "momentum":
        return [
            Candidate(
                label=f"{window}-day, top {top}",
                spec=base.model_copy(update={"window": window, "top_n": top}),
            )
            for window in MOMENTUM_WINDOWS
            for top in MOMENTUM_TOP
            if top < len(base.universe)
        ]
    return []


def family_returns(
    closes: pd.DataFrame, candidates: list[Candidate], opens: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Daily returns of every candidate over the dates all of them can trade."""
    held = {c.label: signals.weights(closes, c.spec).shift(1).fillna(0.0) for c in candidates}
    start = max(signals.warmup_days(c.spec) for c in candidates)
    window = closes.iloc[start:]
    returns = {}
    for c in candidates:
        fills = backtest.fill_prices(c.spec, opens)
        returns[c.label] = performance.strategy_returns(
            window,
            held[c.label].iloc[start:],
            c.spec.fee_bps,
            None if fills is None else fills.iloc[start:],
        )
    return pd.DataFrame(returns)


def reality_check_p(benchmark: pd.Series, losses: pd.DataFrame, seed: int = SEED) -> float:
    """Chance that the best of these models only looks better than the benchmark through luck."""
    test = SPA(benchmark, losses, block_size=BLOCK_SIZE, reps=BOOTSTRAP_REPS, seed=seed)
    test.compute()
    value = float(test.pvalues["consistent"])
    return value if np.isfinite(value) else 1.0


def superior_models(
    benchmark: pd.Series, losses: pd.DataFrame, seed: int = SEED, size: float = SIZE
) -> list[str]:
    """Romano-Wolf step-down over arch's SPA, since arch 8.0.0's StepM crashes on later steps."""
    remaining: list[str] = [str(label) for label in losses.columns]
    superior: list[str] = []
    while remaining:
        test = SPA(
            benchmark, losses[remaining], block_size=BLOCK_SIZE, reps=BOOTSTRAP_REPS, seed=seed
        )
        test.compute()
        found = [remaining[int(position)] for position in test.better_models(size)]
        if not found:
            break
        superior.extend(found)
        remaining = [label for label in remaining if label not in found]
    return superior


def evaluate_family(
    family: StrategyName,
    closes: pd.DataFrame,
    candidates: list[Candidate],
    prior_trials: int,
    opens: pd.DataFrame | None = None,
    seed: int = SEED,
) -> FamilyResult:
    """Judge every variant together, so the luck of picking the best one is accounted for."""
    returns = family_returns(closes, candidates, opens)
    if len(returns) < MIN_DAYS:
        raise ValueError(f"Only {len(returns)} common trading days; need {MIN_DAYS}")
    periods = candidates[0].spec.periods_per_year
    cumulative = max(len(candidates), prior_trials)
    cash = pd.Series(0.0, index=returns.index)
    beating_cash = set(superior_models(-cash, -returns, seed))
    spread = float(np.std(validation.sharpe_of(returns.to_numpy()), ddof=1))
    rows = []
    for candidate in candidates:
        series = returns[candidate.label]
        deflated = validation.deflated_sharpe(series, cumulative, spread)
        t_statistic = validation.sharpe_t_statistic(series)
        beats = candidate.label in beating_cash
        rows.append(
            CandidateResult(
                label=candidate.label,
                sharpe=float(empyrical.sharpe_ratio(series, annualization=periods)),
                cagr=float(empyrical.annual_return(series, annualization=periods)),
                max_drawdown=float(empyrical.max_drawdown(series)),
                t_statistic=t_statistic,
                deflated_sharpe=deflated,
                consistency=validation.consistency(series),
                clears_hurdle=t_statistic >= validation.HURDLE_T,
                beats_cash=beats,
                survives=beats and deflated >= validation.CONFIDENCE,
            )
        )
    return FamilyResult(
        family=family,
        title=STRATEGIES[family].title,
        days=len(returns),
        start=returns.index[0].isoformat(),
        end=returns.index[-1].isoformat(),
        trials=len(candidates),
        cumulative_trials=cumulative,
        reality_check_p=reality_check_p(-cash, -returns, seed),
        probability_of_overfitting=validation.probability_of_overfitting(returns),
        candidates=rows,
        candidate_hashes=[c.config_hash for c in candidates],
    )


def known_trials(log: audit.AuditLog, family: StrategyName, now: Collection[str]) -> int:
    """Distinct variants ever tried for this family, so repeats are not double counted.

    A variant is known by its label ("200-day average"), which stays the same when fields that
    do not change the rule (a cost, a note, how fills are modelled) are added to the spec.
    """
    seen = set(now)
    for entry in log.entries():
        if entry.kind in SEARCHES:
            for item in entry.payload["result"]["families"]:
                if item["family"] == family:
                    seen.update(row["label"] for row in item.get("candidates", []))
    return len(seen)


class Search(BaseModel):
    """How wide the search for a tradable rule has been, for the gate that decides what trades."""

    trials: int
    spreads: dict[str, float]


def search(log: audit.AuditLog) -> Search:
    """Every rule and every lab variant ever tried, and the spread of each family's Sharpe ratios.

    The catalog's rules count once each; a family's lab variants add the ones that are not the
    rule itself. The spread is the standard deviation of the per-period Sharpe ratios of the
    family's variants in the latest lab run.
    """
    trials = STRATEGIES.trials()
    for name, spec in STRATEGIES.items():
        labels = [c.label for c in variants(spec)]
        if labels:
            trials += known_trials(log, name, labels) - 1
    spreads: dict[str, float] = {}
    found = latest(log)
    for family in found.families if found else []:
        periods = STRATEGIES[family.family].periods_per_year
        sharpes = [c.sharpe / np.sqrt(periods) for c in family.candidates]
        if len(sharpes) > 1:
            spreads[family.family] = float(np.std(sharpes, ddof=1))
    return Search(trials=trials, spreads=spreads)


def run_lab(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    log: audit.AuditLog,
    workers: int | None = None,
) -> LabResult:
    """Test every family of variants at once, in parallel, and write the run to the audit log."""
    started = time.monotonic()
    families = [(name, spec) for name, spec in STRATEGIES.items() if variants(spec)]
    prices = {name: history.cached_prices(stocks, crypto, spec.universe) for name, spec in families}
    data: dict[str, pd.DataFrame] = {name: found.closes for name, found in prices.items()}
    jobs = []
    for name, spec in families:
        candidates = variants(spec)
        prior = known_trials(log, name, [c.label for c in candidates])
        jobs.append((name, data[name], candidates, prior, prices[name].opens))
    used = workers or parallel.default_workers(len(jobs))
    outcomes = parallel.run_parallel(evaluate_family, jobs, used)
    result = LabResult(
        families=[o.result for o in outcomes if o.result is not None],
        errors={jobs[o.index][0]: o.error for o in outcomes if o.error},
        seconds=round(time.monotonic() - started, 2),
        workers=used,
    )
    log.record(
        "lab",
        {
            "manifest": audit.manifest(
                {
                    "candidates": {
                        name: [c.label for c in variants(spec)] for name, spec in families
                    }
                },
                data,
                seed=SEED,
            ),
            "result": result.model_dump(mode="json"),
        },
    )
    return result


def latest(log: audit.AuditLog) -> LabResult | None:
    """The newest stored lab result; one written by an incompatible older version reads as none."""
    for entry in reversed(log.entries()):
        if entry.kind == "lab":
            try:
                return LabResult.model_validate(entry.payload["result"])
            except ValidationError:
                return None
    return None


def checks_pass(result: FamilyResult) -> list[str]:
    """The variants that survive every test, best Sharpe first."""
    winners = [c for c in result.candidates if c.survives]
    return [c.label for c in sorted(winners, key=lambda c: -c.sharpe)]
