from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.features.systematic.schemas import ResearchResult, StrategyName, StrategySpec

Signal = Literal["sma", "donchian", "momentum", "multitrend"]
Source = Literal["template", "user", "import"]
CRYPTO_PERIODS, STOCK_PERIODS = 365, 252
MIN_CRYPTO_FEE_BPS = 10.0
MIN_STOCK_FEE_BPS = 1.0
Lookback = Annotated[int, Field(ge=2, le=400)]


class StrategyDraft(BaseModel):
    """A strategy as a person writes it; the platform works out the rest when it is saved."""

    model_config = ConfigDict(extra="forbid")

    name: StrategyName = Field(description="Lower case letters, digits and dashes")
    title: str = Field(min_length=3, max_length=80)
    description: str = Field(min_length=3, max_length=600)
    universe: list[str] = Field(min_length=1, max_length=30, description="Symbols it may hold")
    signal: Signal = Field(description="The rule that turns prices into weights")
    window: int = Field(200, ge=5, le=400, description="Days of the average or momentum")
    lookbacks: list[Lookback] = Field(
        default=[], max_length=10, description="Windows of the Donchian or multi-speed vote"
    )
    target_vol: float | None = Field(
        None, ge=0.02, le=1.0, description="Annual volatility to size to, never above 1x"
    )
    vol_window: int = Field(20, ge=5, le=250, description="Days the volatility is measured on")
    band: float = Field(0.0, ge=0.0, le=0.5, description="Weight change ignored as too small")
    trade_rate: float = Field(
        1.0, gt=0.0, le=1.0, description="Share of the way to the target traded each day"
    )
    top_n: int = Field(1, ge=1, le=30, description="How many assets momentum holds")
    fee_bps: float = Field(ge=0.0, le=500.0, description="Cost of each trade, in basis points")
    fills: Literal["next_open", "close"] = "next_open"
    proxy: bool = Field(
        False, description="Judge significance on the US market's century: US equity funds only"
    )
    note: str | None = Field(None, max_length=300)

    @field_validator("universe")
    @classmethod
    def distinct_symbols(cls, universe: list[str]) -> list[str]:
        found = [symbol.strip().upper() for symbol in universe]
        if len(set(found)) != len(found):
            raise ValueError("A symbol appears twice")
        return found

    @model_validator(mode="after")
    def fits_its_signal(self) -> Self:
        coins = [symbol for symbol in self.universe if "/" in symbol]
        if coins and len(coins) != len(self.universe):
            raise ValueError("A strategy holds coins or stocks, not both")
        if self.signal in ("donchian", "multitrend") and not self.lookbacks:
            raise ValueError(f"A {self.signal} strategy needs at least one lookback")
        if self.signal == "momentum" and self.top_n > len(self.universe):
            raise ValueError("Momentum cannot hold more assets than the universe has")
        floor = MIN_CRYPTO_FEE_BPS if coins else MIN_STOCK_FEE_BPS
        if self.fee_bps < floor:
            raise ValueError(f"The cost of a trade cannot be below {floor:g} bps here")
        if coins and self.proxy:
            raise ValueError("The market's century stands in for US equity funds only")
        return self

    def to_spec(self) -> StrategySpec:
        crypto = any("/" in symbol for symbol in self.universe)
        values = self.model_dump()
        values["lookbacks"] = sorted(set(self.lookbacks))
        return StrategySpec(**values, periods_per_year=CRYPTO_PERIODS if crypto else STOCK_PERIODS)

    @classmethod
    def of(cls, spec: StrategySpec) -> "StrategyDraft":
        return cls.model_validate(spec.model_dump(exclude={"periods_per_year"}))


class SavedStrategy(BaseModel):
    draft: StrategyDraft
    note: str | None = Field(None, max_length=300, description="Why this version was made")


class FieldChange(BaseModel):
    field: str
    before: Any
    after: Any


class DefinitionVersion(BaseModel):
    version: int
    created: datetime
    note: str | None
    spec_hash: str
    changes: list[FieldChange] = Field(description="What differs from the version before it")


class StrategySummary(BaseModel):
    name: str
    title: str
    signal: Signal
    universe: list[str]
    version: int
    versions: int
    archived: bool
    source: Source
    stage: str | None = Field(description="Its lifecycle stage, once the lifecycle has seen it")
    created: datetime
    updated: datetime


class StrategyDefinition(StrategySummary):
    draft: StrategyDraft
    spec: StrategySpec
    history: list[DefinitionVersion]


class ExportedStrategy(BaseModel):
    draft: StrategyDraft
    version: int | None = None
    evidence: dict[str, Any] | None = Field(
        None, description="Results it had where it came from; shown, never trusted"
    )


class StrategyFile(BaseModel):
    """What an export writes and an import reads."""

    format: Literal["algotr.strategies"] = "algotr.strategies"
    format_version: Literal[1] = 1
    exported: datetime
    strategies: list[ExportedStrategy] = Field(min_length=1, max_length=100)


class ImportResult(BaseModel):
    names: list[str]
    renamed: dict[str, str]


class StrategyCheckRequest(BaseModel):
    universe: list[str] = Field(max_length=30)
    draft: dict[str, Any] | None = Field(None, description="The draft as typed so far")


class SymbolCheck(BaseModel):
    symbol: str
    allowed: bool
    reason: str
    days: int | None = Field(description="Daily bars of price history, None when unknown")


class MeasuredCost(BaseModel):
    asset: Literal["stock", "crypto"]
    cost_bps: float
    orders: int


class DraftCheck(BaseModel):
    """What the platform can tell about a draft before it is saved or tested."""

    symbols: list[SymbolCheck]
    needs_days: int = Field(description="Bars the backtest needs: warm-up plus two months")
    same_rule_as: list[str] = Field(description="Strategies with the very same rule")
    measured_cost: MeasuredCost | None
    problems: list[str] = Field(description="Why the draft cannot be saved yet")
    warnings: list[str]


class StrategyPreviewRequest(BaseModel):
    draft: StrategyDraft


class StrategyPreview(BaseModel):
    result: ResearchResult
    trials: int = Field(description="Distinct rules tried so far, this one included")
    new_trial: bool = Field(description="Whether this rule had never been tried before")
