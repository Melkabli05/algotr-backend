import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from pydantic import BaseModel

from app.core import audit, sidecar
from app.core.agents import MarketMood
from app.features.systematic import agents, ensemble, finrl_lab, parallel, regimes
from app.quant import history

KIND = "ai-evaluation"
SEEDS = (1, 2, 3)
SINCE = "2023-05-01"
KRONOS = "kronos"
KRONOS_DIR = Path("data/models/kronos")
KRONOS_SINCE = "2025-09-10"
KRONOS_NAME = "Kronos small (zero-shot)"


class AiConfig(BaseModel):
    algorithms: tuple[str, ...] = agents.ALGORITHMS
    seeds: tuple[int, ...] = SEEDS
    lookback: int = agents.LOOKBACK
    purge: int = agents.PURGE_DAYS
    embargo: int = agents.EMBARGO_DAYS
    timesteps: int = agents.TIMESTEPS
    sac_train_freq: int = agents.SAC_TRAIN_FREQ
    cost: float = agents.COST
    since: str = SINCE
    floor: float = ensemble.FLOOR
    temperature: float = ensemble.TEMPERATURE
    kronos: bool = False
    kronos_since: str = KRONOS_SINCE
    kronos_context: int = 400
    kronos_samples: int = 3
    kronos_seen_days: int = 252

    def trial_hash(self, name: str, seed: int) -> str:
        if name == KRONOS:
            return audit.digest(
                {
                    **self.model_dump(
                        include={k for k in type(self).model_fields if k.startswith(KRONOS)}
                    ),
                    "cost": self.cost,
                    "task": "ai",
                    "name": name,
                    "seed": seed,
                }
            )
        return audit.digest(
            {
                **self.model_dump(
                    exclude={"algorithms", "seeds"}
                    | {k for k in type(self).model_fields if k.startswith(KRONOS)}
                ),
                "task": "ai",
                "name": name,
                "seed": seed,
            }
        )


class AiEvaluation(BaseModel):
    time: datetime
    config: AiConfig
    data: dict[str, Any]
    retrains: int
    groups: list[finrl_lab.GroupResult]
    weights: dict[str, float]
    trial_hashes: list[str]
    total_trials: int
    errors: dict[str, str]
    seconds: float


def retrain_dates(dates: list[str], since: str) -> list[str]:
    """The first trading day of each month from `since` on."""
    return [day for day in agents.month_starts(dates, 1_000) if day >= since]


def job_list(config: AiConfig, dates: list[str]) -> list[tuple[str, int]]:
    """Every agent and seed, the slow ones first so the workers finish together."""
    order = sorted(config.algorithms, key=lambda name: name != "sac")
    names = [*order, KRONOS] if config.kronos else order
    return [(name, seed) for name in names for seed in config.seeds]


def evaluate(
    stocks: StockHistoricalDataClient,
    crypto: CryptoHistoricalDataClient,
    log: audit.AuditLog,
    config: AiConfig | None = None,
    workers: int | None = None,
    run: agents.RunJob = sidecar.run_job,
    mood: MarketMood | None = None,
) -> AiEvaluation:
    """Retrain every agent each month through the history and judge what it did unseen."""
    if not sidecar.available():
        raise sidecar.SidecarUnavailableError("Set FinRL up first: uv run poe finrl-setup")
    config = config or AiConfig()
    if config.kronos and not (KRONOS_DIR / "revision.json").exists():
        raise sidecar.SidecarUnavailableError("Fetch Kronos first: uv run poe kronos-setup")
    started = time.monotonic()
    data = agents.with_mood(history.ohlcv(stocks, crypto, agents.UNIVERSE), mood)
    dates = retrain_dates(data["date"].tolist(), config.since)
    jobs = job_list(config, dates)

    def work(job: tuple[str, int]) -> sidecar.JobResult:
        algorithm, seed = job
        if algorithm == KRONOS:
            return run(
                "kronos_walk_forward",
                kronos_params(config, seed),
                {"data": data},
                timeout=agents.TIMEOUT,
            )
        params = {
            "algorithm": algorithm,
            "seed": seed,
            "cost": config.cost,
            "timesteps": config.timesteps,
            "train_freq": config.sac_train_freq,
            "lookback": config.lookback,
            "purge": config.purge,
            "embargo": config.embargo,
            "retrain_dates": dates,
        }
        return run("ai_walk_forward", params, {"data": data}, timeout=agents.TIMEOUT)

    with ThreadPoolExecutor(workers or parallel.default_workers(len(jobs))) as pool:
        outcomes = list(pool.map(work, jobs))
    return assemble(data, config, jobs, outcomes, len(dates), started, log)


def kronos_params(config: AiConfig, seed: int) -> dict[str, Any]:
    return {
        "model_dir": str(KRONOS_DIR.resolve()),
        "since": config.kronos_since,
        "seen_days": config.kronos_seen_days,
        "context": config.kronos_context,
        "samples": config.kronos_samples,
        "seed": seed,
        "cost": config.cost,
    }


def assemble(
    data: pd.DataFrame,
    config: AiConfig,
    jobs: list[tuple[str, int]],
    outcomes: list[sidecar.JobResult],
    retrains: int,
    started: float,
    log: audit.AuditLog,
) -> AiEvaluation:
    """Judge the runs by the five gates, with the ensemble built from each seed's agents."""
    closes = data.pivot(index="date", columns="tic", values="close")[agents.UNIVERSE]
    closes.index = pd.to_datetime(closes.index)
    errors: dict[str, str] = {}
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for (algorithm, seed), outcome in zip(jobs, outcomes, strict=True):
        if outcome.status == "ok" and outcome.data is not None:
            runs[algorithm, seed] = {**outcome.data, "seed": seed}
        else:
            errors[f"{algorithm} {seed}"] = outcome.error or "failed"
    hashes = [
        config.trial_hash(name, seed)
        for name in (
            *config.algorithms,
            "ensemble",
            "regime-ensemble",
            *([KRONOS] if config.kronos else []),
        )
        for seed in config.seeds
    ]
    trials = finrl_lab.known_trials(log, hashes)
    groups: list[finrl_lab.GroupResult] = []
    for algorithm in config.algorithms:
        seeds = {
            f"seed {seed}": [runs[algorithm, seed]]
            for seed in config.seeds
            if (algorithm, seed) in runs
        }
        if seeds:
            groups.append(
                finrl_lab.judged(closes, f"{algorithm.upper()} agent", "ai", seeds, trials)[0]
            )
    zero_shot = {
        f"seed {seed}": [runs[KRONOS, seed]] for seed in config.seeds if (KRONOS, seed) in runs
    }
    if zero_shot:
        note = f"Scored from {config.kronos_since}, after the model's published training"
        groups.append(finrl_lab.judged(closes, KRONOS_NAME, "ai", zero_shot, trials, note)[0])
    market = closes.pct_change().mean(axis=1).dropna()
    blends: dict[str, list[dict[str, Any]]] = {}
    regime_blends: dict[str, list[dict[str, Any]]] = {}
    for seed in config.seeds:
        found = blend(runs, config, seed)
        if found is not None:
            blends[f"seed {seed}"] = [found]
        chosen = blend(runs, config, seed, market)
        if chosen is not None:
            regime_blends[f"seed {seed}"] = [chosen]
    weights: dict[str, float] = {}
    if blends:
        note = f"Rounds are months; floor {config.floor:.0%}, temperature {config.temperature}"
        groups.append(finrl_lab.judged(closes, "UCB1 ensemble", "ai", blends, trials, note)[0])
    if regime_blends:
        note = (
            "A four-state HMM on the market's drift and volatility each month; the agents "
            f"with the best Sharpe in the current regime (top {regimes.TOP_K}), equal and "
            f"smaller below {regimes.CONFIDENT:.0%} confidence"
        )
        groups.append(
            finrl_lab.judged(closes, agents.ENSEMBLE_GROUP, "ai", regime_blends, trials, note)[0]
        )
        weights = latest_weights(runs, config, market)
    result = AiEvaluation(
        time=datetime.now(UTC),
        config=config,
        data={
            "symbols": agents.UNIVERSE,
            "rows": len(data),
            "first": data["date"].min(),
            "last": data["date"].max(),
        },
        retrains=retrains,
        groups=groups,
        weights=weights,
        trial_hashes=hashes,
        total_trials=trials,
        errors=errors,
        seconds=round(time.monotonic() - started, 2),
    )
    log.record(
        KIND,
        {
            "manifest": audit.manifest(config.model_dump(), {"data": data}),
            "result": result.model_dump(mode="json"),
        },
    )
    return result


def frame_of(
    runs: dict[tuple[str, int], dict[str, Any]], config: AiConfig, seed: int
) -> pd.DataFrame | None:
    """One seed's agents side by side, on the days they all traded."""
    found = {
        name: pd.Series(
            runs[name, seed]["returns"], index=pd.to_datetime(runs[name, seed]["dates"])
        )
        for name in config.algorithms
        if (name, seed) in runs
    }
    return pd.DataFrame(found).dropna() if len(found) == len(config.algorithms) else None


def blend(
    runs: dict[tuple[str, int], dict[str, Any]],
    config: AiConfig,
    seed: int,
    market: pd.Series | None = None,
) -> dict[str, Any] | None:
    """One seed's agents blended, by UCB1 or, given the market, by regime, as a run the judge
    can read."""
    frame = frame_of(runs, config, seed)
    if frame is None:
        return None
    if market is None:
        result = ensemble.replay(frame, config.floor, config.temperature)
    else:
        result = regimes.replay(frame, market)
    known = [runs[name, seed].get("in_sample") for name in config.algorithms]
    inside = [value for value in known if value is not None]
    return {
        "dates": [day.strftime("%Y-%m-%d") for day in result.returns.index],
        "returns": result.returns.tolist(),
        "in_sample": sum(inside) / len(inside) if inside else None,
        "seed": seed,
    }


def latest_weights(
    runs: dict[tuple[str, int], dict[str, Any]], config: AiConfig, market: pd.Series
) -> dict[str, float]:
    """The weights the first seed's regime ensemble would carry into the next month."""
    frame = frame_of(runs, config, config.seeds[0])
    return {} if frame is None else regimes.replay(frame, market).latest


def latest(log: audit.AuditLog) -> AiEvaluation | None:
    for entry in reversed(log.entries()):
        if entry.kind == KIND:
            try:
                return AiEvaluation.model_validate(entry.payload["result"])
            except ValueError:
                return None
    return None
