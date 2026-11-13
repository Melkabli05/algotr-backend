from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError
from sqlalchemy.engine import Connection

from app.core import audit
from app.core.broker import CryptoData, StockData
from app.core.database import transaction
from app.features.orders import trail
from app.features.strategies.schemas import (
    DraftCheck,
    ImportResult,
    SavedStrategy,
    StrategyCheckRequest,
    StrategyDefinition,
    StrategyDraft,
    StrategyFile,
    StrategyPreview,
    StrategyPreviewRequest,
    StrategySummary,
)
from app.features.strategies.service import (
    ConflictError,
    NotFoundError,
    RefusedError,
    Screen,
    StrategyStore,
)
from app.features.systematic import execution, research
from app.features.systematic.catalog import STRATEGIES, TEMPLATES
from app.features.systematic.schemas import ExecutionSummary
from app.quant import history

router = APIRouter(prefix="/strategies", tags=["strategies"])


def get_screen(request: Request) -> Screen:
    screen: Screen = request.app.state.scanner.screen
    return screen


def get_selected(request: Request) -> set[str]:
    """The strategies the autopilot has chosen or been told to use."""
    automation = getattr(request.app.state, "automation", None)
    if automation is None:
        return set()
    return {*automation.state.config.strategies, *(automation.state.active or [])}


def get_summaries(request: Request) -> list[ExecutionSummary]:
    """What the autopilot's own fills cost, per asset class."""
    automation = getattr(request.app.state, "automation", None)
    return [] if automation is None else execution.summaries(automation.state.runs)


def get_store(screen: Annotated[Screen, Depends(get_screen)]) -> StrategyStore:
    return StrategyStore(screen)


Store = Annotated[StrategyStore, Depends(get_store)]
Selected = Annotated[set[str], Depends(get_selected)]
Trail = Annotated[audit.AuditLog, Depends(trail.log_of)]
Summaries = Annotated[list[ExecutionSummary], Depends(get_summaries)]


@contextmanager
def writing() -> Iterator[Connection]:
    """One transaction for the change; the registry is read again once it is committed."""
    try:
        with transaction() as connection:
            yield connection
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No strategy {exc}") from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RefusedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    STRATEGIES.refresh()


def reading[T](work: Callable[[Connection], T]) -> T:
    try:
        with transaction() as connection:
            return work(connection)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No strategy {exc}") from exc


@router.get("")
def read_strategies(store: Store, archived: bool = False) -> list[StrategySummary]:
    """Every strategy with its current version and lifecycle stage; archived ones on request."""
    return reading(lambda connection: store.summaries(connection, archived))


@router.get("/templates")
def read_templates() -> list[StrategyDraft]:
    """The built-in strategies, to start a new one from."""
    return [StrategyDraft.of(spec) for spec in TEMPLATES.values()]


@router.get("/schema")
def read_schema() -> dict[str, Any]:
    """The JSON Schema an import file is checked against."""
    return StrategyFile.model_json_schema()


PLAIN = {
    "string_pattern_mismatch": "use lower case letters, digits and dashes, 2 to 48 of them",
    "missing": "required",
    "string_too_short": "too short",
    "too_short": "add at least one",
}


def plain(error: Any) -> str:
    """A validation error as a person reads it: the field, then what to do."""
    field = ".".join(str(part) for part in error["loc"]) or "draft"
    message = PLAIN.get(error["type"]) or str(error["msg"]).removeprefix("Value error, ")
    return f"{field}: {message}"


def parsed(typed: dict[str, Any] | None) -> tuple[StrategyDraft | None, list[str]]:
    """The draft when it is valid, otherwise what is wrong with it, field by field."""
    if typed is None:
        return None, []
    try:
        return StrategyDraft.model_validate(typed), []
    except ValidationError as exc:
        return None, [plain(error) for error in exc.errors()]


@router.post("/check")
def check_draft(
    store: Store,
    stocks: StockData,
    crypto: CryptoData,
    summaries: Summaries,
    body: StrategyCheckRequest,
) -> DraftCheck:
    """Each symbol's Shariah verdict and price history, strategies with the same rule and what
    this account's fills cost: nothing is saved and nothing counts as a trial."""

    def days_of(symbol: str) -> int | None:
        try:
            return len(history.cached_closes(stocks, crypto, [symbol]))
        except Exception:
            return None

    draft, problems = parsed(body.draft)
    found = reading(
        lambda connection: store.inspect(connection, body.universe, draft, days_of, summaries)
    )
    return found.model_copy(update={"problems": problems})


@router.post("/preview")
def preview_draft(
    store: Store, log: Trail, stocks: StockData, crypto: CryptoData, body: StrategyPreviewRequest
) -> StrategyPreview:
    """Backtest a draft without saving it. A rule never tried before counts as a trial from
    now on, as a saved one does, so testing many drafts is paid for in the Deflated Sharpe."""
    with writing() as connection:
        new = store.try_out(connection, body.draft)
    if new:
        log.record("strategy-trial", {"name": body.draft.name, "draft": body.draft})
    trials = STRATEGIES.trials()
    try:
        result = research.research(stocks, crypto, body.draft.to_spec(), trials=trials)
    except history.NotEnoughHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StrategyPreview(result=result, trials=trials, new_trial=new)


@router.get("/export")
def export_all(store: Store, names: Annotated[list[str] | None, Query()] = None) -> StrategyFile:
    """The named strategies, or every one not archived, as one file."""

    def work(connection: Connection) -> StrategyFile:
        chosen = names or [item.name for item in store.summaries(connection)]
        return store.export(connection, chosen)

    return reading(work)


@router.post("/import", status_code=201)
def import_file(store: Store, log: Trail, file: StrategyFile, copy: bool = False) -> ImportResult:
    """Add a file's strategies, all or none; each starts in research, its evidence unused.

    With `copy`, a name already taken gets a number added instead of refusing the file.
    """
    with writing() as connection:
        found = store.import_file(connection, file, copy)
    log.record("strategy-import", found.model_dump())
    return found


@router.post("", status_code=201)
def create_strategy(store: Store, log: Trail, saved: SavedStrategy) -> StrategyDefinition:
    with writing() as connection:
        found = store.create(connection, saved.draft, saved.note)
    log.record("strategy-create", {"name": found.name, "draft": saved.draft, "note": saved.note})
    return found


@router.get("/{name}")
def read_strategy(store: Store, name: str) -> StrategyDefinition:
    return reading(lambda connection: store.detail(connection, name))


@router.get("/{name}/versions/{version}")
def read_version(store: Store, name: str, version: int) -> StrategyDraft:
    return reading(lambda connection: StrategyDraft.of(store.version(connection, name, version)))


@router.put("/{name}")
def update_strategy(
    store: Store, log: Trail, name: str, saved: SavedStrategy
) -> StrategyDefinition:
    """Save a new version; the lifecycle starts it again at research."""
    with writing() as connection:
        found = store.update(connection, name, saved.draft, saved.note)
    log.record(
        "strategy-update",
        {"name": name, "version": found.version, "draft": saved.draft, "note": saved.note},
    )
    return found


@router.post("/{name}/archive")
def archive_strategy(store: Store, log: Trail, selected: Selected, name: str) -> StrategyDefinition:
    with writing() as connection:
        found = store.archive(connection, name, selected)
    log.record("strategy-archive", {"name": name})
    return found


@router.post("/{name}/restore")
def restore_strategy(store: Store, log: Trail, name: str) -> StrategyDefinition:
    with writing() as connection:
        found = store.restore(connection, name)
    log.record("strategy-restore", {"name": name})
    return found


@router.get("/{name}/export")
def export_one(store: Store, name: str) -> StrategyFile:
    return reading(lambda connection: store.export(connection, [name]))
