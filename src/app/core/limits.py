from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.settings import Settings

PATH = Path("data/limits.json")


class RiskLimits(BaseModel):
    """The limits every buy is checked against; defaults come from the environment."""

    position_share: float = Field(ge=0.01, le=1)
    daily_loss: float = Field(gt=0, le=0.5)
    adv_share: float = Field(gt=0, le=1)
    spread_bps: float = Field(gt=0, le=1000)
    sector_share: float = Field(gt=0, le=1)
    correlation: float = Field(gt=0, le=1)
    book_imbalance: float = Field(ge=0, le=1)


def defaults(settings: Settings) -> RiskLimits:
    return RiskLimits(
        position_share=settings.max_position_share,
        daily_loss=settings.daily_loss_limit,
        adv_share=settings.max_adv_share,
        spread_bps=settings.max_stock_spread_bps,
        sector_share=settings.max_sector_share,
        correlation=settings.max_correlation,
        book_imbalance=settings.max_book_imbalance,
    )


class LimitStore:
    """The limits in force: what was saved from the terminal, else the environment's."""

    def __init__(self, settings: Settings, path: Path = PATH) -> None:
        self.settings = settings
        self.path = path

    def get(self) -> RiskLimits:
        try:
            return RiskLimits.model_validate_json(self.path.read_text())
        except (FileNotFoundError, ValidationError):
            return defaults(self.settings)

    def set(self, limits: RiskLimits) -> RiskLimits:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_suffix(".json.tmp")
        scratch.write_text(limits.model_dump_json(indent=2))
        scratch.replace(self.path)
        return limits

    def reset(self) -> RiskLimits:
        self.path.unlink(missing_ok=True)
        return self.get()
