from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.sleeve import Sleeve
from app.core.types import Symbol
from app.features.scanner import evaluate, vetoes
from app.features.scanner.crypto import CryptoScan
from app.features.scanner.news import NewsReport
from app.features.scanner.review import Review
from app.features.scanner.schemas import Evaluation, Explanation, Scan, ScannerStatus, ShadowReport
from app.features.scanner.service import AlreadyRunningError, Scanner
from app.quant.history import NotEnoughHistoryError

router = APIRouter(prefix="/scanner", tags=["scanner"])


def get_scanner(request: Request) -> Scanner:
    scanner: Scanner = request.app.state.scanner
    return scanner


Scanning = Annotated[Scanner, Depends(get_scanner)]


@router.get("/status")
def read_status(scanner: Scanning) -> ScannerStatus:
    """The universe funnel, the state of each Shariah list and the latest data refresh."""
    return scanner.status()


@router.post("/run", status_code=202)
def run_refresh(scanner: Scanning) -> ScannerStatus:
    """Refresh lists and bars in the background; poll the status for progress."""
    try:
        scanner.request_run()
    except AlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return scanner.status()


@router.post("/evaluate")
def run_evaluation(scanner: Scanning) -> Evaluation:
    """Run the eight variants on the stored panel and judge them together, with the caveats."""
    try:
        return scanner.evaluate()
    except NotEnoughHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/evaluation")
def read_evaluation(scanner: Scanning) -> Evaluation | None:
    """The latest stored evaluation, or null before the first one."""
    return evaluate.latest(scanner.audit)


@router.post("/vetoes")
def run_veto_study(scanner: Scanning) -> vetoes.VetoStudy:
    """Test whether the names the news veto blocks did worse than the ones it let through."""
    try:
        return scanner.veto_study()
    except NotEnoughHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/vetoes")
def read_veto_study(scanner: Scanning) -> vetoes.VetoStudy | None:
    """The latest stored veto study, or null before the first one."""
    return vetoes.latest(scanner.audit)


@router.get("/latest")
def read_latest(scanner: Scanning) -> Scan | None:
    """Today's ranking across the whole market, the picks of every variant and the funnel."""
    return scanner.latest()


@router.get("/symbol/{symbol}")
def read_symbol(symbol: Symbol, scanner: Scanning) -> Explanation:
    """Every test a stock must pass, and which ones it failed."""
    explanation = scanner.explain(symbol)
    if explanation is None:
        raise HTTPException(status_code=404, detail="No market data yet; run a refresh first")
    return explanation


@router.get("/shadow")
def read_shadow(scanner: Scanning) -> ShadowReport:
    """How each logged day's picks did afterwards, against SPUS and the listed universe."""
    return scanner.shadow()


@router.get("/news/{symbol}")
def read_news(symbol: Symbol, scanner: Scanning) -> NewsReport:
    """Headlines for one stock with their event types, and what the last days of news mean."""
    return scanner.news(symbol)


@router.get("/review")
def read_review(scanner: Scanning) -> Review | None:
    """The latest pre-open review: which queued buys were checked and which were cancelled."""
    return scanner.watch.state.review


@router.get("/crypto")
def read_crypto(scanner: Scanning) -> CryptoScan | None:
    """The listed coins by 4 week return with their 200 day trend; display only."""
    return scanner.crypto_scan()


@router.get("/sleeve")
def read_sleeve(scanner: Scanning, variant: str = "rank_avg top 20") -> Sleeve | None:
    """What a variant holds on its schedule, the names the news blocks, and its promotion record."""
    return scanner.sleeve(variant)
