from datetime import date
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core import audit, sidecar
from app.core.breakers import GLOBAL
from app.core.broker import CryptoData, StockData
from app.features.breakers import router as switches
from app.features.scanner.store import BarStore
from app.features.systematic import (
    agent_lab,
    agents,
    backtest,
    events,
    execution,
    explanations,
    finrl_lab,
    lab,
    navlog,
    outcomes,
    research,
    runs,
    stats,
    trace,
)
from app.features.systematic.automation import (
    Automation,
    AutomationDisabledError,
    LiveAccountError,
)
from app.features.systematic.catalog import STRATEGIES
from app.features.systematic.schemas import (
    MAX_YEARS,
    AutomationConfig,
    AutomationStatus,
    ExecutionReport,
    Overview,
    RunRecord,
    StrategyName,
    StrategySpec,
)
from app.quant import history
from app.quant.history import NotEnoughHistoryError


class AuditTrail(BaseModel):
    verification: audit.Verification
    entries: list[audit.Entry]


class AiState(BaseModel):
    available: bool
    status: agents.AgentsStatus
    evaluation: agent_lab.AiEvaluation | None


router = APIRouter(prefix="/systematic", tags=["systematic"])


def get_automation(request: Request) -> Automation:
    automation: Automation = request.app.state.automation
    return automation


def get_agents(request: Request) -> agents.AgentDesk:
    desk: agents.AgentDesk = request.app.state.agents
    return desk


Scheduler = Annotated[Automation, Depends(get_automation)]
Agents = Annotated[agents.AgentDesk, Depends(get_agents)]


@router.get("/overview")
def read_overview(automation: Scheduler) -> Overview:
    """Strategies with their research, today's plan and the automation status in one response."""
    return automation.overview()


@router.get("/automation")
def read_automation(automation: Scheduler) -> AutomationStatus:
    return automation.status()


def spec_named(name: str) -> StrategySpec:
    found = STRATEGIES.get(name)
    if found is None:
        raise HTTPException(status_code=404, detail=f"No strategy named {name}")
    return found


@router.put("/automation")
def update_automation(config: AutomationConfig, automation: Scheduler) -> AutomationStatus:
    """Save the settings; turning automation on triggers a first run straight away."""
    unknown = sorted({*config.strategies, *config.paused} - set(STRATEGIES))
    if unknown:
        raise HTTPException(status_code=422, detail=f"No strategy named {', '.join(unknown)}")
    try:
        return automation.configure(config)
    except LiveAccountError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def get_backtests(request: Request) -> runs.BacktestDesk:
    desk: runs.BacktestDesk = request.app.state.backtests
    return desk


Backtests = Annotated[runs.BacktestDesk, Depends(get_backtests)]


@router.post("/backtests", status_code=202)
def submit_backtest(body: runs.BacktestRequest, desk: Backtests) -> runs.BacktestRun:
    """Queue a sweep of one rule family; its variants count as trials from now on."""
    return desk.submit(body)


@router.get("/backtests")
def read_backtests(desk: Backtests) -> list[runs.BacktestRun]:
    """The last 50 backtests, newest first."""
    return desk.runs()


@router.get("/backtests/{run_id}")
def read_backtest(run_id: str, desk: Backtests) -> runs.BacktestRun:
    found = desk.get(run_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No such backtest")
    return found


@router.post("/validate/{name}")
def validate_strategy(
    name: StrategyName, stocks: StockData, crypto: CryptoData, request: Request
) -> events.EventCheck:
    """Run a trend rule through NautilusTrader's event-driven backtest beside the vectorised one."""
    spec = spec_named(name)
    prices = history.cached_prices(stocks, crypto, spec.universe)
    closes = research.recent(prices.closes, spec, MAX_YEARS)
    try:
        found = events.check(spec, history.Prices(closes, prices.opens.loc[closes.index]))
    except events.EventCheckError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    log: audit.AuditLog = request.app.state.automation.audit
    log.record("event-check", {"result": found.model_dump(mode="json")})
    return found


@router.get("/execution")
def read_execution(automation: Scheduler) -> ExecutionReport:
    """How the autopilot's recent orders filled against the prices the backtest assumes."""
    return execution.report(automation.state.runs)


class Halt(BaseModel):
    reason: str = Field(min_length=3, max_length=200)


@router.post("/automation/halt")
def halt_automation(body: Halt, automation: Scheduler) -> RunRecord:
    """The manual kill switch: the global switch halts and flattens the node, and the autopilot
    stops until Resume."""
    switches.engage(GLOBAL, True, body.reason, "manual")
    return automation.halt(body.reason)


@router.post("/automation/resume")
def resume_automation(automation: Scheduler) -> AutomationStatus:
    """Clear a drawdown halt, release the global kill switch and restart tracking."""
    switches.engage(GLOBAL, False, "Resumed", "manual")
    return automation.resume()


@router.post("/automation/run")
def run_automation(automation: Scheduler) -> RunRecord:
    """Run the configured rebalance now, in the configured mode."""
    try:
        return automation.run()
    except AutomationDisabledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/lab")
def run_lab(automation: Scheduler, stocks: StockData, crypto: CryptoData) -> lab.LabResult:
    """Test every variant of every rule together, in parallel, with multiple-testing corrections."""
    try:
        return lab.run_lab(stocks, crypto, automation.audit)
    except NotEnoughHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/lab")
def read_lab(automation: Scheduler) -> lab.LabResult | None:
    """The latest lab result, or null before the first one."""
    return lab.latest(automation.audit)


@router.get("/audit")
def read_audit(automation: Scheduler, limit: int = 20) -> AuditTrail:
    """The latest audit entries and whether the log's hash chain is intact."""
    return AuditTrail(
        verification=automation.audit.verify(),
        entries=list(reversed(automation.audit.entries(min(max(limit, 1), 2000)))),
    )


@router.get("/lab/finrl")
def read_finrl_lab(automation: Scheduler) -> finrl_lab.FinrlState:
    """FinRL's feature catalog, whether the sidecar is set up, and the latest run if any."""
    return finrl_lab.FinrlState(
        available=finrl_lab.available(),
        catalog=list(finrl_lab.CATALOG),
        result=finrl_lab.latest(automation.audit),
    )


@router.post("/lab/finrl")
def run_finrl_lab(
    automation: Scheduler, stocks: StockData, crypto: CryptoData
) -> finrl_lab.FinrlResult:
    """Run every FinRL feature we use in its isolated environment and judge it by the lab's gates.

    Takes several minutes. Nothing here changes what autopilot trades.
    """
    try:
        return finrl_lab.run_finrl_lab(stocks, crypto, automation.audit)
    except sidecar.SidecarUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, NotEnoughHistoryError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/agents")
def read_agents(desk: Agents, automation: Scheduler) -> AiState:
    """The agents' record and holdings, the UCB1 weights and the latest judgement by the gates."""
    return AiState(
        available=sidecar.available(),
        status=desk.status(),
        evaluation=agent_lab.latest(automation.audit),
    )


@router.get("/outcomes")
def read_outcomes(
    desk: Agents, automation: Scheduler, stocks: StockData, crypto: CryptoData, request: Request
) -> outcomes.Outcomes:
    """Whether agent signals and FinBERT headlines were right the next day, by window and regime."""

    def make() -> outcomes.Outcomes:
        found = [desk.explanations_of(name, 365) for name in agents.ALGORITHMS]
        records = {
            name: (shadow.assets, shadow.dates, shadow.applied, shadow.confidence)
            for name, shadow in desk.state.shadow.items()
        }
        symbols = sorted({asset for assets, *_ in records.values() for asset in assets})
        closes = {
            symbol: history.cached_closes(stocks, crypto, [symbol])[symbol] for symbol in symbols
        }
        market = history.cached_closes(stocks, crypto, ["SPUS"])["SPUS"]
        panel = BarStore()
        return outcomes.compute(
            found, records, closes, market, lambda wanted: panel.read(wanted), date.today()
        )

    return outcomes.cached("outcomes", make)


@router.get("/agents/{algorithm}/confidence")
def read_confidence(algorithm: str, desk: Agents) -> dict[str, dict[str, float]]:
    """For each decision day, the share of the agent's sampled policy that made the same move."""
    shadow = desk.state.shadow.get(algorithm)
    if shadow is None:
        raise HTTPException(status_code=404, detail=f"No record for {algorithm}")
    names = [*shadow.assets, "cash"]
    return {
        day: dict(zip(names, shares, strict=False)) for day, shares in shadow.confidence.items()
    }


@router.get("/strategies/{name}/stats")
def read_strategy_stats(
    name: StrategyName, stocks: StockData, crypto: CryptoData, automation: Scheduler
) -> stats.StrategyStats:
    """A strategy's trade record, its capital in the latest plan and its recent signals."""
    spec = spec_named(name)
    prices = history.cached_prices(stocks, crypto, spec.universe)
    closes = research.recent(prices.closes, spec, MAX_YEARS)
    found = history.Prices(closes, prices.opens.loc[closes.index])
    return stats.strategy_stats(spec, found, automation.state.runs)


@router.get("/strategies/{name}/tracking")
def read_tracking(
    request: Request, name: StrategyName, stocks: StockData, crypto: CryptoData
) -> navlog.Tracking:
    """The strategy's paper NAV against its backtest over the days it has run on paper."""
    log: navlog.NavLog = getattr(request.app.state, "navlog", None) or navlog.NavLog()
    paper = log.readings().get(name, {})
    if not paper:
        return navlog.tracking(name, paper, pd.Series(dtype=float))
    spec = spec_named(name)
    prices = history.cached_prices(stocks, crypto, spec.universe)
    closes = research.recent(prices.closes, spec, MAX_YEARS)
    model = backtest.daily_returns(closes, spec, opens=prices.opens.loc[closes.index])
    return navlog.tracking(name, paper, model)


@router.get("/strategies")
def read_strategies() -> list[StrategySpec]:
    """The published rules and what each trades, without running anything."""
    return list(STRATEGIES.values())


@router.get("/strategies/{name}/trace")
def read_trace(
    name: StrategyName,
    symbol: str,
    stocks: StockData,
    crypto: CryptoData,
    years: Annotated[int, Query(ge=1, le=MAX_YEARS)] = 3,
) -> trace.Trace:
    """What a rule saw and did for one symbol each day: candles, readings, weights and trades."""
    spec = spec_named(name)
    if symbol not in spec.universe:
        raise HTTPException(status_code=404, detail=f"{name} does not trade {symbol}")
    try:
        prices = history.cached_prices(stocks, crypto, spec.universe)
        closes = research.recent(prices.closes, spec, years)
        found = history.Prices(closes, prices.opens.loc[closes.index])
        bars = trace.bars_of(stocks, crypto, symbol).reindex(closes.index)
    except NotEnoughHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return trace.trace(spec, symbol, found, bars)


class Feedback(BaseModel):
    algorithm: str
    decided: str
    verdict: Literal["agree", "flag"]
    note: str = Field(default="", max_length=500)


@router.post("/signals/feedback")
def record_feedback(body: Feedback, automation: Scheduler) -> Feedback:
    """An operator's verdict on one agent decision, kept in the audit log for model review."""
    automation.audit.record("signal-feedback", body.model_dump())
    return body


@router.get("/signals/feedback")
def read_feedback(automation: Scheduler) -> list[Feedback]:
    """Every verdict given, newest first."""
    return [
        Feedback.model_validate(entry.payload)
        for entry in reversed(automation.audit.entries())
        if entry.kind == "signal-feedback"
    ]


@router.get("/agents/{algorithm}/explanations")
def read_explanations(
    algorithm: str, desk: Agents, limit: int = 30
) -> explanations.AgentExplanations:
    """Why an agent chose its weights on its newest decisions, and what its model relies on."""
    if algorithm not in agents.ALGORITHMS:
        raise HTTPException(status_code=404, detail=f"No agent named {algorithm}")
    return desk.explanations_of(algorithm, min(max(limit, 1), 365))
