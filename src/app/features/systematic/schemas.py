from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, StringConstraints, model_validator

from app.core.compliance import KillState
from app.core.sleeve import Criterion
from app.features.systematic.nav import Book
from app.quant.performance import Check, Metrics, Sensitivity

NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{1,47}$"
StrategyName = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
Mode = Literal["off", "dry-run", "paper"]
Engine = Literal["alpaca", "nautilus"]
SleeveMode = Literal["off", "shadow", "trade"]
Source = Literal["strategy", "sleeve"]
MAX_YEARS = 8
MAX_SLEEVE_SHARE = 0.25


def unique(names: list[StrategyName]) -> list[StrategyName]:
    return list(dict.fromkeys(names))


class StrategySpec(BaseModel):
    """A fixed, published rule set: nothing is fitted, so there is nothing to overfit."""

    name: StrategyName
    title: str
    description: str
    universe: list[str]
    signal: Literal["sma", "donchian", "momentum", "multitrend"]
    window: int = 200
    lookbacks: list[int] = []
    target_vol: float | None = None
    vol_window: int = 20
    band: float = 0.0
    trade_rate: float = Field(default=1.0, gt=0.0, le=1.0)
    top_n: int = 1
    fee_bps: float
    periods_per_year: int
    fills: Literal["next_open", "close"] = "next_open"
    proxy: bool = False
    note: str | None = None


class GrowthPoint(BaseModel):
    time: datetime
    strategy: float
    benchmark: float


class Evidence(BaseModel):
    """The rule on a long history standing in for its universe, for the significance tests."""

    name: str
    start: datetime
    end: datetime
    days: int
    metrics: Metrics
    benchmark: Metrics
    sharpe_p: float
    deflated_sharpe: float


class ResearchResult(BaseModel):
    strategy: StrategySpec
    start: datetime
    end: datetime
    days: int
    fee_bps: float
    metrics: Metrics
    benchmark: Metrics
    in_sample: Metrics
    out_of_sample: Metrics
    sensitivity: list[Sensitivity]
    deflated_sharpe: float
    sharpe_p: float | None = None
    evidence: Evidence | None = None
    checks: list[Check]
    growth: list[GrowthPoint]
    signals: dict[str, float]


class Target(BaseModel):
    symbol: str
    weight: float
    target_value: float
    current_value: float
    source: Source = "strategy"


class PlannedOrder(BaseModel):
    """An order of the plan; `price` is the decision price, `limit_price` the most a buy pays."""

    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    notional: float
    source: Source = "strategy"
    price: float | None = None
    limit_price: float | None = None


class StrategyBook(BaseModel):
    """One strategy's part of the plan: its capital and the dollars it means to hold."""

    capital: float
    targets: dict[str, float]
    prices: dict[str, float]


class SleevePlan(BaseModel):
    """What the scanner's sleeve holds and would trade; a preview when `preview` is set."""

    mode: SleeveMode
    variant: str
    share: float
    session: date | None = None
    preview: bool = False
    override: bool = False
    passed: bool = False
    criteria: list[Criterion] = []
    problems: list[str] = []
    targets: list[Target] = []
    orders: list[PlannedOrder] = []
    book: StrategyBook | None = None


class Plan(BaseModel):
    strategies: list[StrategyName]
    allocation: float
    equity: float
    flat: bool = False
    signals: dict[str, dict[str, float]]
    targets: list[Target]
    orders: list[PlannedOrder]
    sleeve: SleevePlan | None = None
    scales: dict[str, float] = {}
    books: dict[str, StrategyBook] = {}

    def part(self, source: Source) -> "Plan":
        """The targets and orders that come from one source."""
        return self.model_copy(
            update={
                "targets": [item for item in self.targets if item.source == source],
                "orders": [item for item in self.orders if item.source == source],
            }
        )


class ExecutedOrder(BaseModel):
    """An order the autopilot sent, and once it is done, how it filled against its prices.

    `price` is the decision price. `model_price` is where the backtest fills the trade (the
    open for a stock, the decision price for a coin). `cost_bps` is the fill against it plus
    the coin fee, which the backtest's `fee_bps` stands for; `shortfall_bps` is the fill
    against the decision price, the night included.
    """

    symbol: str
    side: str
    qty: float
    order_id: str | None = None
    status: str
    price: float | None = None
    limit_price: float | None = None
    settled: bool = False
    filled_qty: float | None = None
    filled_price: float | None = None
    filled_at: datetime | None = None
    model_price: float | None = None
    cost_bps: float | None = None
    shortfall_bps: float | None = None


class ExecutionSummary(BaseModel):
    """Fills of one asset class against the cost the backtest charges for them."""

    asset: Literal["stock", "crypto"]
    orders: int
    notional: float
    cost_bps: float | None
    shortfall_bps: float | None
    assumed_bps: float


class ExecutionReport(BaseModel):
    fills: list[ExecutedOrder]
    summaries: list[ExecutionSummary]


class SleeveConfig(BaseModel):
    """The scanner's picks as a small extra holding: off, a preview, or placing orders."""

    mode: SleeveMode = "off"
    share: float = Field(default=0.0, ge=0, le=MAX_SLEEVE_SHARE)
    variant: str = "rank_avg top 20"
    override: bool = False


class AutomationConfig(BaseModel):
    mode: Mode = "off"
    allocation: float = Field(default=0.1, gt=0, le=1)
    automatic: bool = True
    crypto: bool = False
    engine: Engine = "nautilus"
    strategies: Annotated[list[StrategyName], AfterValidator(unique)] = ["halal-etf-trend"]
    max_drawdown: float = Field(default=0.15, ge=0.02, le=0.5)
    reduce_at: float = Field(default=0.5, ge=0.1, le=1.0)
    paused: Annotated[list[StrategyName], AfterValidator(unique)] = []
    sleeve: SleeveConfig = SleeveConfig()

    @model_validator(mode="before")
    @classmethod
    def named_strategies_are_manual(cls, data: Any) -> Any:
        """A config that lists strategies without saying `automatic` keeps choosing them by hand."""
        if isinstance(data, dict) and "strategies" in data and "automatic" not in data:
            return {**data, "automatic": False}
        return data


class RunRecord(BaseModel):
    time: datetime
    strategies: list[StrategyName]
    mode: Mode
    status: Literal["ok", "skipped", "error", "halted"]
    detail: str | None = None
    plan: Plan | None = None
    executed: list[ExecutedOrder] = []


class AutomationStatus(BaseModel):
    config: AutomationConfig
    next_run: datetime | None
    halted: str | None
    book: Book
    books: dict[str, Book]
    kill_state: KillState = "normal"
    runs: list[RunRecord]


class StrategyOverview(BaseModel):
    """A strategy, its research, and `streak`: days in a row it passed (+) or failed (-)."""

    spec: StrategySpec
    active: bool
    streak: int = 0
    research: ResearchResult | None = None
    error: str | None = None


class Overview(BaseModel):
    status: AutomationStatus
    strategies: list[StrategyOverview]
    plan: Plan | None = None
    plan_error: str | None = None
    equity: float | None = None
    drawdown: float | None = None
    execution: list[ExecutionSummary] = []
    ensemble_active: bool = False
    ensemble_streak: int = 0
