from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

from app.features.scanner.crypto import CryptoStatus
from app.features.scanner.filings import FilingsBadge
from app.features.scanner.news import NewsBadge
from app.features.scanner.review import Review
from app.features.scanner.sentiment import SentimentStatus
from app.quant.performance import Check, Metrics, Sensitivity


class ListHealth(BaseModel):
    name: str
    as_of: date | None = None
    age_days: int | None = None
    stale: bool = True
    symbols: int = 0
    unmatched: list[str] = []
    error: str | None = None


class Funnel(BaseModel):
    assets: int
    tradable: int
    liquid: int
    listed: int
    listed_liquid: int
    panel: int


class Progress(BaseModel):
    stage: str
    done: int
    total: int


class Refresh(BaseModel):
    started: datetime
    seconds: float
    status: Literal["ok", "error"]
    detail: str | None = None
    session: date | None = None
    scanned: date | None = None
    funnel: Funnel | None = None
    lists: list[ListHealth] = []
    rows: int = 0
    rebuilt: list[str] = []
    rejected: list[str] = []


class NewsStatus(BaseModel):
    enabled: bool = False
    polled: datetime | None = None
    articles: int = 0
    following: int = 0
    error: str | None = None
    sec: bool = False
    filings_at: datetime | None = None
    filings_error: str | None = None
    sentiment: SentimentStatus = SentimentStatus()


class ScannerStatus(BaseModel):
    running: Progress | None = None
    last: Refresh | None = None
    next_run: datetime | None = None
    news: NewsStatus = NewsStatus()
    review: Review | None = None
    crypto: CryptoStatus = CryptoStatus()


class VariantResult(BaseModel):
    label: str
    score: str
    top_n: int
    metrics: Metrics
    out_of_sample: Metrics
    sensitivity: list[Sensitivity]
    t_statistic: float
    deflated_sharpe: float
    checks: list[Check]


class Evaluation(BaseModel):
    start: date
    end: date
    days: int
    symbols: int
    lists: list[str]
    trials: int
    luck_bar: float
    probability_of_overfitting: float
    benchmark: Metrics
    variants: list[VariantResult]
    notes: list[str]


class Candidate(BaseModel):
    rank: int
    member_rank: int | None = None
    symbol: str
    name: str | None = None
    score: float
    mom_12_1: float | None
    mom_6_1: float | None
    clenow_90: float | None
    price: float
    dollar_volume: float
    lists: list[str]
    picked: bool
    weight: float | None = None
    news: NewsBadge | None = None
    results: FilingsBadge | None = None
    blocked: str | None = None


class ScanFunnel(BaseModel):
    liquid: int
    listed: int
    trending: int
    eligible: int
    listed_eligible: int
    picked: int


class Scan(BaseModel):
    session: date
    variant: str
    funnel: ScanFunnel
    candidates: list[Candidate]
    picks: dict[str, dict[str, float]]
    benchmark: list[str]
    stale_lists: list[str]


class Reason(BaseModel):
    check: str
    passed: bool
    detail: str


class Explanation(BaseModel):
    symbol: str
    name: str | None = None
    session: date | None = None
    verdict: Literal["picked", "eligible", "scan only", "excluded"]
    reasons: list[Reason]


class Shadow(BaseModel):
    session: date
    variant: str
    picks: int
    returns: dict[int, float | None]
    excess_equal_weight: dict[int, float | None]
    excess_spus: dict[int, float | None]


class ShadowSummary(BaseModel):
    variant: str
    scans: int
    completed: dict[int, int]
    mean_return: dict[int, float | None]
    mean_excess_equal_weight: dict[int, float | None]
    mean_excess_spus: dict[int, float | None]


class ShadowReport(BaseModel):
    horizons: list[int]
    summary: list[ShadowSummary]
    rows: list[Shadow]
