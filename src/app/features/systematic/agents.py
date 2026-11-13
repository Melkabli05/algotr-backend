import asyncio
import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel

from app.core import audit, sidecar
from app.core.agents import MarketMood
from app.core.drift import ReturnStream
from app.features.systematic import ensemble, explanations, judging, model_registry, regimes
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas_models import AgentModel as Model
from app.quant import history

logger = logging.getLogger(__name__)

ROOT = Path("data/ai")
ALGORITHMS = ("ppo", "sac", "a2c")
UNIVERSE = ["SPUS", "HLAL"]
LOOKBACK = 756
PURGE_DAYS = 1
EMBARGO_DAYS = 5
TIMESTEPS = 30_000
SAC_TRAIN_FREQ = 4
SEED = 1
BACKFILL_MONTHS = 36
COST = STRATEGIES["halal-etf-trend"].fee_bps / 1e4
JOB_ID = "ai-agents"
EVERY = timedelta(minutes=30)
FIRST_RUN = timedelta(seconds=5)
TIMEOUT = 7200
OVERALL_SAMPLE = 60
EXPLAINER = "shap.ExactExplainer, one player for each indicator and one for the weights held"
PREFIX = "ai-"
ENSEMBLE_GROUP = "Regime ensemble"
RunJob = Callable[..., sidecar.JobResult]


class AgentError(RuntimeError):
    """Raised when a sidecar job for an agent does not come back."""


class Queued(BaseModel):
    """A decision of an agent's model that is waiting for its explanation."""

    date: date
    version: int
    observation: list[float]


class Explained(NamedTuple):
    record: dict[str, Any] | None
    taken: list[Queued]
    dropped: list[Queued]


class Shadow(BaseModel):
    """One agent's pretend portfolio: what it earned each day and what it holds for the next."""

    assets: list[str]
    dates: list[date] = []
    returns: list[float] = []
    applied: list[list[float]] = []
    since: date | None = None
    target: list[float] | None = None
    held: list[float] | None = None
    backfilled: bool = False
    pending: list[Queued] = []
    explained: date | None = None
    confidence: dict[str, list[float]] = {}


MOVE = 0.02
CHALLENGE_DAYS = 30
FEATURE_CODE = sidecar.SIDECAR / "weights_env.py"


def feature_version() -> dict[str, str]:
    """The features an agent decides on: the sidecar code that builds them, and the headline
    sentiment version it reads."""
    code = hashlib.sha256(FEATURE_CODE.read_bytes()).hexdigest()[:16]
    return {"agent_features": code, "sentiment_24h": "v2"}


def admission(model: Model, artifact: str) -> list[dict[str, Any]]:
    """What a freshly trained version must show before it may even challenge the champion."""
    return [
        {
            "name": "Learned from its training window",
            "passed": model.in_sample_sharpe > 0,
            "detail": f"in-sample Sharpe {model.in_sample_sharpe:.2f}",
        },
        {
            "name": "Its weights are on disk",
            "passed": Path(artifact).exists(),
            "detail": artifact,
        },
    ]


def head_to_head(challenger: Shadow, champion: Shadow) -> tuple[float, float, int]:
    """Both books' Sharpe over the sessions both followed."""
    mine, theirs = series_of(challenger), series_of(champion)
    days = mine.index.intersection(theirs.index)
    return judging.sharpe(mine[days]), judging.sharpe(theirs[days]), len(days)


def calibration_points(shadow: Shadow, closes: pd.DataFrame) -> tuple[list[float], list[bool]]:
    """Each weight an agent moved on a day it stated its confidence, and whether the asset then
    went the way it moved: the decision of a day earns the next session's return."""
    stated: list[float] = []
    correct: list[bool] = []
    returns = closes.pct_change()
    index = {day: position for position, day in enumerate(shadow.dates)}
    sessions = list(returns.index.date)
    for decided, confidence in shadow.confidence.items():
        day = date.fromisoformat(decided)
        later = [session for session in sessions if session > day]
        if not later or later[0] not in index:
            continue
        position = index[later[0]]
        if position == 0:
            continue
        chosen, before = shadow.applied[position], shadow.applied[position - 1]
        for column, asset in enumerate(shadow.assets):
            move = chosen[column] - before[column]
            earned = returns.loc[returns.index.date == later[0], asset]
            if abs(move) < MOVE or earned.empty or not np.isfinite(earned.iloc[0]):
                continue
            stated.append(float(confidence[column]))
            correct.append(bool(np.sign(move) == np.sign(earned.iloc[0])))
    return stated, correct


class Challenger(BaseModel):
    """A newly trained version with its own paper book beside the champion's, until judged."""

    model: Model
    artifact: str
    shadow: Shadow


class ChallengerStatus(BaseModel):
    version: int
    days: int
    needed: int
    sharpe: float | None
    champion_sharpe: float | None


class State(BaseModel):
    models: dict[str, Model] = {}
    shadow: dict[str, Shadow] = {}
    retrain: list[str] = []
    challengers: dict[str, Challenger] = {}
    trained: dict[str, int] = {}


class AgentStatus(BaseModel):
    algorithm: str
    model: Model | None
    days: int
    first: date | None
    last: date | None
    sharpe: float | None
    target: dict[str, float]
    unexplained: int = 0
    challenger: ChallengerStatus | None = None


class AgentsStatus(BaseModel):
    enabled: bool
    busy: str | None
    agents: list[AgentStatus]
    ensemble: dict[str, float]
    ensemble_sharpe: float | None


def month_starts(dates: list[str], months: int) -> list[str]:
    """The first trading day of each of the last `months` months in `dates`."""
    days = pd.to_datetime(sorted(set(dates)))
    firsts = pd.Series(days).groupby([days.year, days.month]).first()
    return [day.strftime("%Y-%m-%d") for day in firsts.iloc[-months:]]


def with_mood(data: pd.DataFrame, mood: MarketMood | None) -> pd.DataFrame:
    """The bars with the day's headline sentiment and risk beside them, when there is a source.

    The same value goes to every asset of a day: it is the market's mood, not one fund's.
    """
    if mood is None or data.empty:
        return data
    sessions = sorted({date.fromisoformat(day) for day in data["date"]})
    found = mood.mood(sessions)[["sentiment", "news_risk"]]
    found.index = [day.strftime("%Y-%m-%d") for day in found.index]
    return data.merge(found, left_on="date", right_index=True, how="left")


def series_of(shadow: Shadow) -> pd.Series:
    return pd.Series(shadow.returns, index=pd.to_datetime(shadow.dates), dtype=float)


class AgentDesk:
    """Trains the PPO, SAC and A2C agents each month and follows what they would hold, unseen."""

    def __init__(
        self,
        stocks: StockHistoricalDataClient,
        crypto: CryptoHistoricalDataClient,
        log: audit.AuditLog | None = None,
        root: Path = ROOT,
        run: RunJob = sidecar.run_job,
        today: Callable[[], date] = lambda: datetime.now(UTC).date(),
    ) -> None:
        self.stocks = stocks
        self.crypto = crypto
        self.log = log
        self.registry: model_registry.AgentRegistry | None = None
        self.root = root
        self.job = run
        self.today = today
        self.enabled = False
        self.mood: MarketMood | None = None
        self.busy: str | None = None
        self.blended: tuple[tuple[object, ...], ensemble.Replay] | None = None
        self.lock = Lock()
        self.state = self._load()
        self.scheduler = AsyncIOScheduler(timezone=UTC)

    def _load(self) -> State:
        try:
            return State.model_validate_json((self.root / "state.json").read_text())
        except (FileNotFoundError, ValueError):
            return State()

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "state.json").write_text(self.state.model_dump_json(indent=2))

    def _legacy(self, algorithm: str) -> dict[str, str]:
        folder = (self.root / "models").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        return {
            "model_path": str(folder / f"{algorithm}.zip"),
            "scaler_path": str(folder / f"{algorithm}.json"),
        }

    def _versioned(self, algorithm: str, version: int) -> dict[str, str]:
        """Each trained version keeps its own files, so the registry can roll back to it."""
        folder = (self.root / "models" / algorithm).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        return model_registry.artifacts(str(folder / f"v{version}.zip"))

    def _paths(self, algorithm: str) -> dict[str, str]:
        """Where the agent's production model and scaling are, as absolute paths: the sidecar
        runs elsewhere. A rollback in the registry changes which version this is."""
        if self.registry is not None:
            try:
                found = self.registry.production(algorithm)
            except Exception:
                logger.exception("The model registry could not be read; using the local model")
                found = None
            if found is not None:
                model, artifact = found
                current = self.state.models.get(algorithm)
                if current is None or current.version != model.version:
                    self.state.models[algorithm] = model
                return model_registry.artifacts(artifact)
            current = self.state.models.get(algorithm)
            legacy = self._legacy(algorithm)
            if current is not None and Path(legacy["model_path"]).exists():
                self._publish(current, legacy["model_path"])
            return legacy
        current = self.state.models.get(algorithm)
        if current is not None:
            versioned = self._versioned(algorithm, current.version)
            if Path(versioned["model_path"]).exists():
                return versioned
        return self._legacy(algorithm)

    def _publish(self, model: Model, artifact: str) -> None:
        """A version trained before the registry kept models: registered and made champion."""
        self._register(model, artifact, admission(model, artifact))
        self._promote(model, "migration", "The version running before the registry")

    def _register(self, model: Model, artifact: str, gates: list[dict[str, Any]]) -> None:
        if self.registry is None:
            return
        try:
            extra: dict[str, object] = {"cost": COST, "lookback": LOOKBACK}
            self.registry.publish(model, artifact, extra, gates, feature_version())
        except Exception:
            logger.exception("Could not register %s in the model registry", model.algorithm)

    def _promote(self, model: Model, by: str, reason: str) -> None:
        if self.registry is None:
            return
        try:
            self.registry.promote(model, by, reason)
        except Exception:
            logger.exception("Could not promote %s v%d", model.algorithm, model.version)

    def _reject(self, model: Model, reason: str) -> None:
        if self.log is not None:
            self.log.record(
                "ai",
                {"event": "rejected", "model": model.model_dump(mode="json"), "reason": reason},
            )
        if self.registry is None:
            return
        try:
            self.registry.reject(model, reason)
        except Exception:
            logger.exception("Could not archive %s v%d", model.algorithm, model.version)

    def advance_challenger(self, algorithm: str, data: pd.DataFrame) -> None:
        """Follow the days since the challenger last did, with its own model."""
        challenger = self.state.challengers.get(algorithm)
        if challenger is None:
            return
        shadow = challenger.shadow
        params = {
            "algorithm": algorithm,
            "cost": COST,
            "since": shadow.since.isoformat() if shadow.since else None,
            "target": shadow.target,
            "held": shadow.held,
            **model_registry.artifacts(challenger.artifact),
        }
        found = self._ask("ai_shadow", params, data)
        shadow.dates += [date.fromisoformat(day) for day in found["dates"]]
        shadow.returns += found["returns"]
        shadow.applied += found["applied"]
        shadow.since = date.fromisoformat(found["last"])
        shadow.target, shadow.held = found["target"], found["held"]
        self._save()

    def judge(self, algorithm: str) -> bool | None:
        """After 30 sessions the challenger replaces the champion if it did at least as well over
        the same days, on the features the desk computes now; otherwise it is archived."""
        challenger = self.state.challengers.get(algorithm)
        champion = self.state.shadow.get(algorithm)
        if challenger is None or champion is None:
            return None
        mine, theirs, days = head_to_head(challenger.shadow, champion)
        if days < CHALLENGE_DAYS:
            return None
        verdict = f"Sharpe {mine:.2f} against the champion's {theirs:.2f} over {days} sessions"
        same_features = self._features_match(challenger.model)
        won = mine >= theirs and same_features
        if won:
            self.state.models[algorithm] = challenger.model
            self._promote(challenger.model, "gate: 30-session shadow", verdict)
        else:
            why = verdict if same_features else "its features changed since it was trained"
            self._reject(challenger.model, f"Lost to the champion: {why}")
        if self.log is not None:
            self.log.record(
                "ai",
                {
                    "event": "challenge",
                    "algorithm": algorithm,
                    "version": challenger.model.version,
                    "promoted": won,
                    "verdict": verdict,
                },
            )
        del self.state.challengers[algorithm]
        self._save()
        return won

    def _features_match(self, model: Model) -> bool:
        if self.registry is None:
            return True
        try:
            expected = self.registry.feature_version(model)
        except Exception:
            logger.exception("Could not read what %s v%d expects", model.algorithm, model.version)
            return False
        return expected in (None, feature_version())

    def _ask(self, task: str, params: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
        outcome = self.job(task, params, {"data": data}, timeout=TIMEOUT)
        if outcome.status != "ok" or outcome.data is None:
            raise AgentError(f"{task} {params['algorithm']}: {outcome.error or 'no result'}")
        return outcome.data

    def due(self, algorithm: str) -> bool:
        """Whether an agent needs training: none yet, its files gone, asked for, or last month's;
        never while a challenger it trained is still being judged."""
        if algorithm in self.state.challengers:
            return False
        model = self.state.models.get(algorithm)
        if model is None or algorithm in self.state.retrain:
            return True
        if not all(Path(kept).exists() for kept in self._paths(algorithm).values()):
            return True
        today = self.today()
        return (model.trained.year, model.trained.month) < (today.year, today.month)

    def train(self, algorithm: str, data: pd.DataFrame) -> Model:
        """Train one agent on the last three years. The first version is the champion at once;
        a later one becomes a challenger, judged against the champion on paper."""
        previous = self.state.models.get(algorithm)
        broken = previous is not None and not all(
            Path(kept).exists() for kept in self._paths(algorithm).values()
        )
        known = max(previous.version if previous else 0, self.state.trained.get(algorithm, 0))
        version = known + 1
        self.state.trained[algorithm] = version
        paths = self._versioned(algorithm, version)
        params = {
            "algorithm": algorithm,
            "seed": SEED,
            "cost": COST,
            "timesteps": TIMESTEPS,
            "train_freq": SAC_TRAIN_FREQ,
            "lookback": LOOKBACK,
            **paths,
        }
        found = self._ask("ai_train", params, data)
        model = Model(
            algorithm=algorithm,
            version=version,
            trained=self.today(),
            first=found["first"],
            last=found["last"],
            days=found["days"],
            seed=SEED,
            timesteps=TIMESTEPS,
            in_sample_sharpe=found["in_sample_sharpe"],
        )
        artifact = paths["model_path"]
        gates = admission(model, artifact)
        self._register(model, artifact, gates)
        self.state.retrain = [name for name in self.state.retrain if name != algorithm]
        champion = self.state.shadow.get(algorithm)
        if not all(gate["passed"] for gate in gates):
            failed = ", ".join(gate["name"] for gate in gates if not gate["passed"])
            self._reject(model, f"Failed admission: {failed}")
        elif broken or previous is None or champion is None or not champion.dates:
            self.state.models[algorithm] = model
            reason = "The champion's weights are gone" if broken else "No champion to challenge"
            self._promote(
                model, "gate: first version" if not broken else "gate: replacement", reason
            )
            self.state.shadow.setdefault(algorithm, Shadow(assets=found["assets"]))
        else:
            self.state.challengers[algorithm] = Challenger(
                model=model,
                artifact=artifact,
                shadow=Shadow(assets=found["assets"], since=champion.since, backfilled=True),
            )
        self._save()
        if self.log is not None:
            manifest = audit.manifest({"algorithm": algorithm, "seed": SEED}, {"data": data})
            self.log.record(
                "ai",
                {"event": "trained", "model": model.model_dump(mode="json"), "manifest": manifest},
            )
        return model

    def backfill(self, algorithm: str, data: pd.DataFrame) -> None:
        """Give an agent a record: a monthly retrain for three years, each month unseen."""
        params = {
            "algorithm": algorithm,
            "seed": SEED,
            "cost": COST,
            "timesteps": TIMESTEPS,
            "train_freq": SAC_TRAIN_FREQ,
            "lookback": LOOKBACK,
            "purge": PURGE_DAYS,
            "embargo": EMBARGO_DAYS,
            "retrain_dates": month_starts(data["date"].tolist(), BACKFILL_MONTHS),
        }
        found = self._ask("ai_walk_forward", params, data)
        shadow = Shadow(
            assets=found["assets"],
            dates=found["dates"],
            returns=found["returns"],
            applied=found["weights"],
            since=found["dates"][-1] if found["dates"] else None,
            held=found["held"],
            backfilled=True,
        )
        self.state.shadow[algorithm] = shadow
        self._save()

    def advance(self, algorithm: str, data: pd.DataFrame) -> None:
        """Follow the days since the last one handled, and choose the weights for the next."""
        shadow = self.state.shadow[algorithm]
        params = {
            "algorithm": algorithm,
            "cost": COST,
            "since": shadow.since.isoformat() if shadow.since else None,
            "target": shadow.target,
            "held": shadow.held,
            **self._paths(algorithm),
        }
        found = self._ask("ai_shadow", params, data)
        shadow.dates += [date.fromisoformat(day) for day in found["dates"]]
        shadow.returns += found["returns"]
        shadow.applied += found["applied"]
        shadow.since = date.fromisoformat(found["last"])
        shadow.target, shadow.held = found["target"], found["held"]
        shadow.confidence.update(found.get("confidence", {}))
        version = self.state.models[algorithm].version
        shadow.pending += [
            Queued(date=item["date"], version=version, observation=item["observation"])
            for item in found.get("decisions", [])
        ]
        self._save()

    def owes(self, algorithm: str) -> bool:
        """Whether an agent still owes an explanation: of a decision, or of its model."""
        if self.log is None:
            return False
        model, shadow = self.state.models[algorithm], self.state.shadow[algorithm]
        return bool(shadow.pending) or not model.explained or shadow.explained != shadow.since

    def unqueued(self, algorithm: str) -> bool:
        """Whether an agent's latest decision is neither explained nor waiting to be."""
        shadow = self.state.shadow[algorithm]
        return self.log is not None and not shadow.pending and shadow.explained != shadow.since

    def current(self, algorithm: str, latest: date) -> bool:
        """Whether an agent has its record, is trained, has followed `latest` and is explained."""
        shadow = self.state.shadow.get(algorithm)
        challenger = self.state.challengers.get(algorithm)
        return bool(
            shadow
            and shadow.backfilled
            and shadow.since == latest
            and (challenger is None or challenger.shadow.since == latest)
            and not self.due(algorithm)
            and not self.owes(algorithm)
        )

    def explain(self, algorithm: str, data: pd.DataFrame) -> Explained | None:
        """Explain what the agent's model chose, and once what it relies on, in the sidecar."""
        if self.log is None:
            return None
        model, shadow = self.state.models[algorithm], self.state.shadow[algorithm]
        mine = [item for item in shadow.pending if item.version == model.version]
        dropped = [item for item in shadow.pending if item.version != model.version]
        sample = 0 if model.explained else OVERALL_SAMPLE
        if not mine and not sample:
            return Explained(None, [], dropped) if dropped else None
        params = {
            "algorithm": algorithm,
            "cost": COST,
            "first": model.first.isoformat(),
            "last": model.last.isoformat(),
            "observations": [item.observation for item in mine],
            "sample": sample,
            **self._paths(algorithm),
        }
        found = self._ask("ai_explain", params, data)
        queued = [(item.date, item.observation) for item in mine]
        record = explanations.build(found, queued, model.model_dump(mode="json"), sample)
        return Explained(record, mine, dropped)

    def record_explanations(self, results: dict[str, Explained], data: pd.DataFrame) -> None:
        """Write what was explained to the audit log, then forget the decisions it covers."""
        if not results or self.log is None:
            return
        self.log.record(
            explanations.KIND,
            {
                "agents": {
                    name: item.record for name, item in results.items() if item.record is not None
                },
                "unexplained": [
                    {
                        "agent": name,
                        "decided": lost.date.isoformat(),
                        "reason": "the model that made it was replaced before it was explained",
                    }
                    for name, item in results.items()
                    for lost in item.dropped
                ],
                "manifest": audit.manifest(
                    {"explainer": EXPLAINER, "sample": OVERALL_SAMPLE}, {"data": data}
                ),
            },
        )
        for name, item in results.items():
            shadow, gone = self.state.shadow[name], [*item.taken, *item.dropped]
            shadow.pending = [queued for queued in shadow.pending if queued not in gone]
            if item.taken:
                shadow.explained = max(queued.date for queued in item.taken)
            if item.record is not None and item.record["overall"] is not None:
                self.state.models[name].explained = True
        self._save()

    def explanations_of(self, algorithm: str, limit: int) -> explanations.AgentExplanations:
        """An agent's newest explained decisions and what its model relies on, from the log."""
        entries = self.log.entries() if self.log is not None else []
        return explanations.read(entries, algorithm, limit)

    def run(self) -> None:
        """Bring every agent up to date; the first run also gives each its record."""
        if not self.enabled or not self.lock.acquire(blocking=False):
            return
        results: dict[str, Explained] = {}
        try:
            closes = history.cached_closes(self.stocks, self.crypto, UNIVERSE)
            latest = closes.index[-1].date()
            pending = [name for name in ALGORITHMS if not self.current(name, latest)]
            if not pending:
                return
            data = with_mood(history.ohlcv(self.stocks, self.crypto, UNIVERSE), self.mood)
            for algorithm in pending:
                self.busy = algorithm
                try:
                    shadow = self.state.shadow.get(algorithm)
                    if shadow is None or not shadow.backfilled:
                        self.backfill(algorithm, data)
                    retrained = self.due(algorithm)
                    if retrained:
                        self.train(algorithm, data)
                    behind = self.state.shadow[algorithm].since != latest
                    if retrained or behind or self.unqueued(algorithm):
                        self.advance(algorithm, data)
                    challenger = self.state.challengers.get(algorithm)
                    if challenger is not None and challenger.shadow.since != latest:
                        self.advance_challenger(algorithm, data)
                    self.judge(algorithm)
                    explained = self.explain(algorithm, data)
                    if explained is not None:
                        results[algorithm] = explained
                except AgentError as exc:
                    logger.warning("Agent %s could not be brought up to date: %s", algorithm, exc)
            self.busy = None
            self.record_explanations(results, data)
        finally:
            self.busy = None
            self.lock.release()

    def request_retrain(self, stream: str) -> str:
        """Ask for an agent (every agent, for the ensemble) to be retrained at the next check."""
        name = stream.removeprefix(PREFIX)
        wanted = list(ALGORITHMS) if name == "ensemble" else [name]
        wanted = [algorithm for algorithm in wanted if algorithm in ALGORITHMS]
        if not wanted:
            return "not applicable: not an agent"
        self.state.retrain = sorted({*self.state.retrain, *wanted})
        self._save()
        return f"scheduled for the next check, within {int(EVERY.total_seconds() // 60)} minutes"

    def _blend(self) -> ensemble.Replay | None:
        """The regime-aware ensemble of the agents, once all of them have a record; worked out
        again only when a record grows."""
        found = {name: series_of(item) for name, item in self.state.shadow.items() if item.dates}
        if len(found) < len(ALGORITHMS):
            return None
        frame = pd.DataFrame(found).dropna()
        key = (len(frame), frame.index[-1] if len(frame) else None)
        if self.blended is not None and self.blended[0] == key:
            return self.blended[1]
        try:
            closes = history.cached_closes(self.stocks, self.crypto, UNIVERSE)
            market = closes.pct_change().mean(axis=1).dropna()
            market.index = pd.to_datetime(market.index).tz_localize(None)
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            result = regimes.replay(frame, market)
        except Exception:
            logger.exception("The regime ensemble failed; using UCB1 until it works")
            result = ensemble.replay(frame)
        self.blended = (key, result)
        return result

    def ensemble_eligible(self) -> bool:
        """Whether the latest AI evaluation found the UCB1 ensemble passing every gate."""
        from app.features.systematic import agent_lab

        found = None if self.log is None else agent_lab.latest(self.log)
        if found is None:
            return False
        return any(g.name == ENSEMBLE_GROUP and g.verdict.eligible for g in found.groups)

    def ensemble_target(self) -> dict[str, float] | None:
        """Each agent's target for tomorrow, mixed by the ensemble's weights; cash is the rest."""
        blend = self._blend()
        if blend is None:
            return None
        mixed: dict[str, float] = {}
        for name, share in blend.latest.items():
            shadow = self.state.shadow[name]
            if not shadow.target:
                return None
            for asset, weight in zip(shadow.assets, shadow.target, strict=False):
                mixed[asset] = mixed.get(asset, 0.0) + share * weight
        return mixed

    def ensemble_returns(self) -> pd.Series | None:
        blend = self._blend()
        return None if blend is None else blend.returns

    def streams(self) -> list[ReturnStream]:
        found = {name: series_of(item) for name, item in self.state.shadow.items() if item.dates}
        streams = [
            ReturnStream(f"{PREFIX}{name}", "agent", series) for name, series in found.items()
        ]
        blend = self._blend()
        if blend is not None:
            streams.append(ReturnStream(f"{PREFIX}ensemble", "ensemble", blend.returns))
        return streams

    def status(self) -> AgentsStatus:
        agents = []
        for algorithm in ALGORITHMS:
            shadow = self.state.shadow.get(algorithm)
            series = series_of(shadow) if shadow and shadow.dates else None
            names = [*shadow.assets, "cash"] if shadow else []
            agents.append(
                AgentStatus(
                    algorithm=algorithm,
                    model=self.state.models.get(algorithm),
                    days=0 if series is None else len(series),
                    first=None if series is None else series.index[0].date(),
                    last=None if series is None else series.index[-1].date(),
                    sharpe=None if series is None else judging.sharpe(series),
                    target=dict(zip(names, shadow.target, strict=True))
                    if shadow and shadow.target
                    else {},
                    unexplained=len(shadow.pending) if shadow else 0,
                    challenger=self.challenger_status(algorithm),
                )
            )
        blend = self._blend()
        return AgentsStatus(
            enabled=self.enabled,
            busy=self.busy,
            agents=agents,
            ensemble={} if blend is None else blend.latest,
            ensemble_sharpe=None if blend is None else judging.sharpe(blend.returns),
        )

    def challenger_status(self, algorithm: str) -> ChallengerStatus | None:
        challenger = self.state.challengers.get(algorithm)
        champion = self.state.shadow.get(algorithm)
        if challenger is None or champion is None:
            return None
        days = len(challenger.shadow.returns)
        mine, theirs, _ = head_to_head(challenger.shadow, champion) if days else (None, None, 0)
        return ChallengerStatus(
            version=challenger.model.version,
            days=days,
            needed=CHALLENGE_DAYS,
            sharpe=mine,
            champion_sharpe=theirs,
        )

    def register_current(self) -> None:
        """Give the registry the weights' hash of each version running since before it kept one."""
        for algorithm, model in list(self.state.models.items()):
            artifact = self._paths(algorithm)["model_path"]
            if Path(artifact).exists():
                self._register(model, artifact, admission(model, artifact))

    async def _tick(self) -> None:
        await asyncio.to_thread(self.run)

    def start(self) -> None:
        self.register_current()
        self.scheduler.add_job(
            self._tick,
            IntervalTrigger(seconds=EVERY.total_seconds()),
            id=JOB_ID,
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(UTC) + FIRST_RUN,
        )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
