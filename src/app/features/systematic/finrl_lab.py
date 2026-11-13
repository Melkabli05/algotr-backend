import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from pydantic import BaseModel, ValidationError

from app.core import audit, sidecar
from app.features.systematic import judging, parallel
from app.features.systematic.catalog import STRATEGIES
from app.quant import history

FAMILY = judging.FAMILY
ALGORITHMS = ("a2c", "ddpg", "ppo", "sac", "td3")
PORTFOLIO_ALGORITHMS = ("a2c", "ppo", "ddpg")
FOLDS = 3
TRAIN_FRACTION = 0.55
SEEDS = (1, 2, 3)
TIMESTEPS = 10_000
EPISODES = 20
TUNING_TRIALS = 6
WINDOW_DAYS = 126
VALIDATION_DAYS = 63
HMAX = 100
INITIAL_AMOUNT = 100_000
MARKET = ("SPY", "QQQ", "TLT", "GLD", "IWM", "EFA")
MIN_TEST_DAYS = 90

Role = Literal["used", "excluded", "unavailable"]


class FeatureSpec(BaseModel):
    key: str
    title: str
    module: str
    role: Role
    note: str


class FeatureStatus(FeatureSpec):
    """A FinRL feature with whether it imports here and what the last run made of it."""

    imports: bool | None = None
    detail: str = ""
    ran: str = "not run"


CATALOG: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        key="features",
        title="Feature engineering: technical indicators and the turbulence index",
        module="finrl.meta.preprocessor.preprocessors",
        role="used",
        note="FeatureEngineer builds the indicators; turbulence comes from six market ETFs.",
    ),
    FeatureSpec(
        key="stock_env",
        title="Stock trading environment (long-only, cash-constrained)",
        module="finrl.meta.env_stock_trading.env_stocktrading",
        role="used",
        note="The environment behind the five agents, ensemble and tuning below.",
    ),
    FeatureSpec(
        key="sb3_agents",
        title="Stable-Baselines3 agents: A2C, DDPG, PPO, SAC, TD3",
        module="finrl.agents.stablebaselines3.models",
        role="used",
        note="Three seeds each, over anchored walk-forward folds, judged by our gates.",
    ),
    FeatureSpec(
        key="ensemble",
        title="Ensemble agent (rolling retraining, best validation Sharpe trades)",
        module="finrl.agents.stablebaselines3.models",
        role="used",
        note="Retrained every 126 days; the validation Sharpe picks the model for each window.",
    ),
    FeatureSpec(
        key="portfolio",
        title="Portfolio-allocation environment (weights over the assets)",
        module="finrl.meta.env_portfolio_allocation.env_portfolio",
        role="used",
        note="Fully invested weights from a covariance-aware state; no cash slot.",
    ),
    FeatureSpec(
        key="eiie",
        title="Portfolio-optimisation environment and EIIE policy-gradient agent",
        module="finrl.agents.portfolio_optimization.models",
        role="used",
        note="FinRL's EIIE agent on its own environment.",
    ),
    FeatureSpec(
        key="elegantrl",
        title="ElegantRL backend on the vectorised environment",
        module="finrl.agents.elegantrl.models",
        role="unavailable",
        note="ElegantRL's own training loop fails on current PyTorch (\"Can't detach views "
        'in-place", elegantrl 0.3.10) and its crashed worker leaves the parent waiting.',
    ),
    FeatureSpec(
        key="tune",
        title="Optuna hyperparameter tuning",
        module="finrl.agents.stablebaselines3.tune_sb3",
        role="used",
        note="Scored on a validation window before the test period, never on the test data.",
    ),
    FeatureSpec(
        key="stats",
        title="Backtest statistics (pyfolio)",
        module="finrl.plot",
        role="used",
        note="FinRL's own performance table for each candidate's median run.",
    ),
    FeatureSpec(
        key="rllib",
        title="RLlib backend",
        module="finrl.agents.rllib.models",
        role="unavailable",
        note="Upstream imports an A2C algorithm that current Ray no longer ships.",
    ),
    FeatureSpec(
        key="paper_trading",
        title="Alpaca paper and live trading module",
        module="finrl.meta.paper_trading.alpaca",
        role="excluded",
        note="Would place orders around our safeguards (drawdown halt, no shorts, audit, Nautilus "
        "risk checks) and uses the retired alpaca-trade-api. Orders go through autopilot only.",
    ),
    FeatureSpec(
        key="data_sources",
        title="Data processors for Yahoo, WRDS, CCXT, JoinQuant, Sinopac and others",
        module="finrl.meta.data_processor",
        role="excluded",
        note="Need accounts and keys outside our universe. Prices come from Alpaca through the "
        "audited loader and are handed to FinRL as files.",
    ),
    FeatureSpec(
        key="crypto_envs",
        title="Cryptocurrency environments",
        module="finrl.meta.env_cryptocurrency_trading.env_multiple_crypto",
        role="excluded",
        note="Crypto stays off by default here; scholarly opinion is divided.",
    ),
)


class GroupResult(BaseModel):
    name: str
    task: str
    verdict: judging.Judgement
    note: str = ""
    stats: dict[str, float | None] = {}


class FinrlConfig(BaseModel):
    algorithms: tuple[str, ...] = ALGORITHMS
    portfolio_algorithms: tuple[str, ...] = PORTFOLIO_ALGORITHMS
    seeds: tuple[int, ...] = SEEDS
    folds: int = FOLDS
    train_fraction: float = TRAIN_FRACTION
    timesteps: int = TIMESTEPS
    episodes: int = EPISODES
    tuning_trials: int = TUNING_TRIALS
    cost: float = 0.0005

    def trial_hash(self, task: str, name: str, seed: int = 0) -> str:
        return audit.digest(
            {
                **self.model_dump(exclude={"seeds", "algorithms", "portfolio_algorithms"}),
                "task": task,
                "name": name,
                "seed": seed,
            }
        )


class FinrlResult(BaseModel):
    commit: str | None
    packages: dict[str, str]
    features: list[FeatureStatus]
    groups: list[GroupResult]
    tuning: dict[str, Any] | None
    turbulence: dict[str, Any] | None
    total_trials: int
    trial_hashes: list[str]
    errors: dict[str, str]
    seconds: float
    workers: int
    config: FinrlConfig


class FinrlState(BaseModel):
    available: bool
    catalog: list[FeatureSpec]
    result: FinrlResult | None


def available() -> bool:
    return sidecar.available()


def fold_dates(index: pd.DatetimeIndex, config: FinrlConfig) -> list[dict[str, str]]:
    """Anchored walk-forward folds as the date strings FinRL's data_split takes."""
    first = int(len(index) * config.train_fraction)
    edges = np.linspace(first, len(index) - 1, config.folds + 1).astype(int)
    folds = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        validation_start = max(start - VALIDATION_DAYS * 4, 1)
        folds.append(
            {
                "train_start": index[0].strftime("%Y-%m-%d"),
                "train_end": index[start].strftime("%Y-%m-%d"),
                "validation_start": index[validation_start].strftime("%Y-%m-%d"),
                "test_start": index[start].strftime("%Y-%m-%d"),
                "test_end": (index[end] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            }
        )
    return folds


def series_of(run: dict[str, Any]) -> pd.Series:
    return pd.Series(run["returns"], index=pd.to_datetime(run["dates"]), dtype=float)


def stitched(parts: Sequence[pd.Series]) -> pd.Series:
    merged = pd.concat(list(parts)).sort_index()
    return merged[~merged.index.duplicated(keep="last")]


def group_runs(
    runs: list[dict[str, Any]], by: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """name -> seed label -> that seed's runs (one per fold)."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run in runs:
        groups.setdefault(by(run), {}).setdefault(f"seed {run['seed']}", []).append(run)
    return groups


def judged(
    closes: pd.DataFrame,
    name: str,
    task: str,
    seeds: dict[str, list[dict[str, Any]]],
    trials: int,
    note: str = "",
) -> tuple[GroupResult, dict[str, list[Any]]]:
    frame = pd.DataFrame(
        {label: stitched([series_of(run) for run in parts]) for label, parts in seeds.items()}
    )
    in_sample: dict[str, float | None] = {}
    for label, parts in seeds.items():
        known = [run["in_sample"] for run in parts if run.get("in_sample") is not None]
        in_sample[label] = float(np.mean(known)) if known else None
    verdict = judging.judge(closes, frame, in_sample, trials)
    middle = sorted(verdict.rows, key=lambda row: row.sharpe)[len(verdict.rows) // 2].label
    typical = frame[middle].dropna()
    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in typical.index],
        "returns": typical.tolist(),
    }
    return GroupResult(name=name, task=task, verdict=verdict, note=note), series


def known_trials(log: audit.AuditLog, hashes: Sequence[str]) -> int:
    seen = set(hashes)
    for entry in log.entries():
        if entry.kind in ("finrl", "ai-evaluation"):
            seen.update(entry.payload["result"]["trial_hashes"])
    return len(seen)


def agent_jobs(
    config: FinrlConfig, folds: list[dict[str, str]]
) -> list[tuple[str, dict[str, Any]]]:
    base = {
        "cost": config.cost,
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "folds": folds,
        "timesteps": config.timesteps,
    }
    jobs: list[tuple[str, dict[str, Any]]] = []
    for algorithm in config.algorithms:
        for seed in config.seeds:
            jobs.append(("agents", {**base, "algorithms": [algorithm], "seeds": [seed]}))
    for algorithm in config.portfolio_algorithms:
        for seed in config.seeds:
            jobs.append(("portfolio", {**base, "algorithms": [algorithm], "seeds": [seed]}))
    for seed in config.seeds:
        jobs.append(("eiie", {**base, "seeds": [seed], "episodes": config.episodes}))
    return jobs


def run_finrl_lab(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    log: audit.AuditLog,
    config: FinrlConfig | None = None,
    workers: int | None = None,
    run: Callable[..., sidecar.JobResult] = sidecar.run_job,
) -> FinrlResult:
    """Run every FinRL feature we use, judge each by the lab's gates, and log the run."""
    if not available():
        raise sidecar.SidecarUnavailableError("Set FinRL up first: uv run poe finrl-setup")
    config = config or FinrlConfig()
    started = time.monotonic()
    spec = STRATEGIES[FAMILY]
    data = history.ohlcv(stocks, crypto, spec.universe)
    market = history.ohlcv(stocks, crypto, list(MARKET))
    closes = data.pivot(index="date", columns="tic", values="close")[spec.universe]
    closes.index = pd.to_datetime(closes.index)
    if len(closes) * (1 - config.train_fraction) / config.folds < MIN_TEST_DAYS:
        raise ValueError("Not enough history for walk-forward testing with this many folds")
    folds = fold_dates(closes.index, config)
    frames = {"data": data, "market": market}
    base_params = {
        "cost": config.cost,
        "hmax": HMAX,
        "initial_amount": INITIAL_AMOUNT,
        "folds": folds,
        "timesteps": config.timesteps,
    }
    ensemble = {
        **base_params,
        "train_period": [folds[0]["train_start"], folds[0]["train_end"]],
        "val_test_period": [folds[0]["train_end"], folds[-1]["test_end"]],
        "rebalance_window": WINDOW_DAYS,
        "validation_window": VALIDATION_DAYS,
    }
    tune = {
        **base_params,
        "algorithms": ["ppo"],
        "seeds": [1],
        "trials": config.tuning_trials,
    }
    jobs = [
        *agent_jobs(config, folds),
        ("ensemble", ensemble),
        ("tune", tune),
        ("features", {}),
        ("inventory", {"modules": {item.key: item.module for item in CATALOG}}),
    ]
    used = workers or parallel.default_workers(len(jobs))
    with ThreadPoolExecutor(used) as pool:
        outcomes = list(pool.map(lambda job: run(job[0], job[1], frames), jobs))
    result, typical = assemble(closes, config, jobs, outcomes, started, used, log)
    result = attach_stats(result, typical, run)
    result = result.model_copy(update={"seconds": round(time.monotonic() - started, 2)})
    record(log, result, frames)
    return result


def assemble(
    closes: pd.DataFrame,
    config: FinrlConfig,
    jobs: list[tuple[str, dict[str, Any]]],
    outcomes: list[sidecar.JobResult],
    started: float,
    workers: int,
    log: audit.AuditLog,
) -> tuple[FinrlResult, dict[str, dict[str, list[Any]]]]:
    errors: dict[str, str] = {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    extras: dict[str, dict[str, Any]] = {}
    for (task, params), outcome in zip(jobs, outcomes, strict=True):
        label = f"{task} {params.get('algorithms', [''])[0]} {params.get('seeds', [''])[0]}".strip()
        if outcome.status != "ok" or outcome.data is None:
            errors[label] = outcome.error or "failed"
            continue
        if "runs" in outcome.data:
            by_task.setdefault(task, []).extend(outcome.data["runs"])
        else:
            extras[task] = outcome.data
    hashes = [config.trial_hash("agents", a, s) for a in config.algorithms for s in config.seeds]
    hashes += [
        config.trial_hash("portfolio", a, s)
        for a in config.portfolio_algorithms
        for s in config.seeds
    ]
    hashes += [config.trial_hash("eiie", "eiie", s) for s in config.seeds]
    hashes += [config.trial_hash("ensemble", "ensemble")]
    hashes += [config.trial_hash("tune", "ppo", n) for n in range(config.tuning_trials)]
    trials = known_trials(log, hashes)
    groups: list[GroupResult] = []
    typical: dict[str, dict[str, list[Any]]] = {}

    def add(name: str, task: str, seeds: dict[str, list[dict[str, Any]]], note: str = "") -> None:
        group, series = judged(closes, name, task, seeds, trials, note)
        groups.append(group)
        typical[name] = series

    for task, label in (("agents", "FinRL"), ("portfolio", "FinRL portfolio")):
        for algorithm, seeds in group_runs(by_task.get(task, []), lambda r: r["algorithm"]).items():
            add(f"{label} {algorithm.upper()}", task, seeds)
    for _, seeds in group_runs(by_task.get("eiie", []), lambda r: r["algorithm"]).items():
        add("FinRL EIIE", "eiie", seeds)
    ensemble = extras.get("ensemble")
    if ensemble:
        one = {
            "seed 0": [
                {
                    "dates": ensemble["dates"],
                    "returns": ensemble["returns"],
                    "in_sample": ensemble.get("in_sample"),
                    "seed": 0,
                }
            ]
        }
        used = ", ".join(sorted(set(ensemble["models_used"])))
        add("FinRL ensemble", "ensemble", one, f"Models chosen: {used}")
    result = finish(closes, config, groups, extras, errors, trials, hashes, started, workers, log)
    return result, typical


def attach_stats(
    result: FinrlResult,
    series: dict[str, dict[str, list[Any]]],
    run: Callable[..., sidecar.JobResult],
) -> FinrlResult:
    """FinRL's own pyfolio statistics for each group's typical run."""
    outcome = run("stats", {"series": series}, {})
    ok = outcome.status == "ok" and outcome.data is not None
    stats = outcome.data["stats"] if ok and outcome.data else {}
    # pyfolio cannot define some measures (skew, ratios) for a series that never moves: store
    # those as "not defined" rather than as a NaN that JSON cannot hold.
    stats = {
        name: {k: (v if v is not None and math.isfinite(v) else None) for k, v in table.items()}
        for name, table in stats.items()
    }
    errors = dict(result.errors)
    if not ok:
        errors["stats"] = outcome.error or "failed"
    groups = [g.model_copy(update={"stats": stats.get(g.name, {})}) for g in result.groups]
    features = [
        f.model_copy(update={"ran": "ran" if ok else "failed"}) if f.key == "stats" else f
        for f in result.features
    ]
    return result.model_copy(update={"groups": groups, "features": features, "errors": errors})


def finish(
    closes: pd.DataFrame,
    config: FinrlConfig,
    groups: list[GroupResult],
    extras: dict[str, dict[str, Any]],
    errors: dict[str, str],
    trials: int,
    hashes: list[str],
    started: float,
    workers: int,
    log: audit.AuditLog,
) -> FinrlResult:
    inventory = extras.get("inventory", {})
    ran = {group.task for group in groups} | set(extras)
    statuses = []
    for spec in CATALOG:
        found = inventory.get("features", {}).get(spec.key)
        works = spec.key in ran or {
            "sb3_agents": "agents" in ran,
            "stock_env": "agents" in ran,
        }.get(spec.key, False)
        statuses.append(
            FeatureStatus(
                **spec.model_dump(),
                imports=None if found is None else bool(found["ok"]),
                detail="" if found is None else str(found["detail"]),
                ran="ran" if works else ("not used" if spec.role != "used" else "failed"),
            )
        )
    tuning = extras.get("tune")
    return FinrlResult(
        commit=inventory.get("commit"),
        packages=inventory.get("packages", {}),
        features=statuses,
        groups=groups,
        tuning=tuning,
        turbulence=extras.get("features"),
        total_trials=trials,
        trial_hashes=hashes,
        errors=errors,
        seconds=round(time.monotonic() - started, 2),
        workers=workers,
        config=config,
    )


def record(log: audit.AuditLog, result: FinrlResult, data: dict[str, pd.DataFrame]) -> None:
    log.record(
        "finrl",
        {
            "manifest": audit.manifest(result.config.model_dump(), data),
            "finrl": {"commit": result.commit, "packages": result.packages},
            "result": result.model_dump(mode="json"),
        },
    )


def latest(log: audit.AuditLog) -> FinrlResult | None:
    for entry in reversed(log.entries()):
        if entry.kind == "finrl":
            try:
                return FinrlResult.model_validate(entry.payload["result"])
            except ValidationError:
                return None
    return None
