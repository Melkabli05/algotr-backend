from collections.abc import Callable, Collection, Iterable
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from app.core.compliance import Screening
from app.core.tables import (
    strategies,
    strategy_definition_versions,
    strategy_definitions,
    strategy_versions,
)
from app.features.strategies.schemas import (
    DefinitionVersion,
    DraftCheck,
    ExportedStrategy,
    FieldChange,
    ImportResult,
    MeasuredCost,
    Source,
    StrategyDefinition,
    StrategyDraft,
    StrategyFile,
    StrategySummary,
    SymbolCheck,
)
from app.features.systematic import signals
from app.features.systematic.backtest import MIN_DAYS
from app.features.systematic.catalog import listed_order, record_trial, rule_hash, spec_hash
from app.features.systematic.schemas import ExecutionSummary, StrategySpec

RUNNING = ("shadow", "canary", "live")
COPY_LIMIT = 50

Screen = Callable[[str], Screening]
DaysOf = Callable[[str], int | None]


class NotFoundError(LookupError):
    pass


class ConflictError(ValueError):
    pass


class RefusedError(ValueError):
    pass


def changes(before: StrategySpec | None, after: StrategySpec) -> list[FieldChange]:
    """The fields a person can edit that differ between two versions."""
    if before is None:
        return []
    old, new = StrategyDraft.of(before).model_dump(), StrategyDraft.of(after).model_dump()
    return [
        FieldChange(field=field, before=old[field], after=new[field])
        for field in new
        if old[field] != new[field]
    ]


def measured_for(universe: list[str], summaries: list[ExecutionSummary]) -> MeasuredCost | None:
    """What this account's own fills have cost for the asset class the universe trades."""
    asset = "crypto" if any("/" in symbol for symbol in universe) else "stock"
    for summary in summaries:
        if summary.asset == asset and summary.cost_bps is not None and summary.orders:
            return MeasuredCost(
                asset=asset, cost_bps=round(summary.cost_bps, 1), orders=summary.orders
            )
    return None


def refused_symbols(draft: StrategyDraft, screen: Screen) -> list[str]:
    """Each symbol the Shariah screen refuses, with its reason."""
    found = (screen(symbol) for symbol in draft.universe)
    return [f"{verdict.symbol}: {verdict.reason}" for verdict in found if not verdict.allowed]


class StrategyStore:
    """Strategies as data: every save is a new version, nothing is deleted, only archived."""

    def __init__(self, screen: Screen) -> None:
        self.screen = screen

    def summaries(self, connection: Connection, archived: bool = False) -> list[StrategySummary]:
        rows = self.rows(connection)
        return [self.summary(row) for row in rows if archived or not row.archived]

    def detail(self, connection: Connection, name: str) -> StrategyDefinition:
        rows = self.rows(connection, name)
        if not rows:
            raise NotFoundError(name)
        row = rows[0]
        history = connection.execute(
            sa.select(strategy_definition_versions)
            .where(strategy_definition_versions.c.name == name)
            .order_by(strategy_definition_versions.c.version.desc())
        ).all()
        spec = StrategySpec.model_validate(row.spec)
        specs = [StrategySpec.model_validate(item.spec) for item in history]
        earlier = [*specs[1:], None]
        return StrategyDefinition(
            **self.summary(row).model_dump(),
            draft=StrategyDraft.of(spec),
            spec=spec,
            history=[
                DefinitionVersion(
                    version=item.version,
                    created=item.created,
                    note=item.note,
                    spec_hash=item.spec_hash,
                    changes=changes(before, current),
                )
                for item, current, before in zip(history, specs, earlier, strict=True)
            ],
        )

    def version(self, connection: Connection, name: str, version: int) -> StrategySpec:
        found = connection.execute(
            sa.select(strategy_definition_versions.c.spec).where(
                strategy_definition_versions.c.name == name,
                strategy_definition_versions.c.version == version,
            )
        ).scalar_one_or_none()
        if found is None:
            raise NotFoundError(f"{name} v{version}")
        return StrategySpec.model_validate(found)

    def create(
        self,
        connection: Connection,
        draft: StrategyDraft,
        note: str | None,
        source: Source = "user",
    ) -> StrategyDefinition:
        self.check(draft)
        if self.exists(connection, draft.name):
            raise ConflictError(f"A strategy named {draft.name} already exists")
        now = datetime.now(UTC)
        connection.execute(
            sa.insert(strategy_definitions).values(
                name=draft.name, current_version=1, source=source, created=now, updated=now
            )
        )
        self.add_version(connection, draft, 1, note, now)
        return self.detail(connection, draft.name)

    def update(
        self, connection: Connection, name: str, draft: StrategyDraft, note: str | None
    ) -> StrategyDefinition:
        """A new version; a strategy trading the old one keeps it until the lifecycle moves."""
        if draft.name != name:
            raise RefusedError("A strategy cannot be renamed; duplicate it instead")
        current = self.detail(connection, name)
        if current.archived:
            raise RefusedError(f"{name} is archived; restore it first")
        self.check(draft)
        if spec_hash(draft.to_spec()) == spec_hash(current.spec):
            raise ConflictError("Nothing changed, so no new version was made")
        now = datetime.now(UTC)
        version = current.versions + 1
        self.add_version(connection, draft, version, note, now)
        connection.execute(
            sa.update(strategy_definitions)
            .where(strategy_definitions.c.name == name)
            .values(current_version=version, updated=now)
        )
        return self.detail(connection, name)

    def archive(
        self, connection: Connection, name: str, selected: Collection[str]
    ) -> StrategyDefinition:
        """Take a strategy out of use; refused while it trades or the autopilot has chosen it."""
        current = self.detail(connection, name)
        if current.stage in RUNNING:
            raise RefusedError(
                f"{name} is at {current.stage}; move it back to research before archiving it"
            )
        if name in selected:
            raise RefusedError(f"The autopilot has {name} selected; take it out there first")
        self.set_archived(connection, name, True)
        return self.detail(connection, name)

    def restore(self, connection: Connection, name: str) -> StrategyDefinition:
        self.detail(connection, name)
        self.set_archived(connection, name, False)
        return self.detail(connection, name)

    def export(self, connection: Connection, names: Iterable[str]) -> StrategyFile:
        found = []
        for name in names:
            current = self.detail(connection, name)
            found.append(
                ExportedStrategy(
                    draft=current.draft,
                    version=current.version,
                    evidence=self.evidence(connection, name),
                )
            )
        if not found:
            raise NotFoundError("Nothing to export")
        return StrategyFile(exported=datetime.now(UTC), strategies=found)

    def import_file(
        self, connection: Connection, file: StrategyFile, copy: bool = False
    ) -> ImportResult:
        """Add every strategy of a file, or none: each lands as a new strategy in research."""
        taken = set(self.names(connection))
        renamed: dict[str, str] = {}
        drafts = []
        for item in file.strategies:
            draft = item.draft
            if draft.name in taken:
                if not copy:
                    raise ConflictError(f"A strategy named {draft.name} already exists")
                fresh = self.free_name(draft.name, taken)
                renamed[draft.name] = fresh
                draft = draft.model_copy(update={"name": fresh})
            taken.add(draft.name)
            drafts.append(draft)
        for draft in drafts:
            self.create(connection, draft, "ImportResult", source="import")
        return ImportResult(names=[draft.name for draft in drafts], renamed=renamed)

    def same_rule_as(
        self, connection: Connection, spec: StrategySpec, other_than: str | None = None
    ) -> list[str]:
        """Every strategy, archived ones too, whose current version is this very rule."""
        found = rule_hash(spec)
        return [
            row.name
            for row in self.rows(connection)
            if row.name != other_than and rule_hash(StrategySpec.model_validate(row.spec)) == found
        ]

    def try_out(self, connection: Connection, draft: StrategyDraft) -> bool:
        """A draft about to be backtested: checked as a save is, and counted as a trial."""
        self.check(draft)
        return record_trial(connection, draft.to_spec(), "preview", draft.name)

    def inspect(
        self,
        connection: Connection,
        universe: list[str],
        draft: StrategyDraft | None,
        days_of: DaysOf,
        summaries: list[ExecutionSummary],
    ) -> DraftCheck:
        """Each symbol's verdict and history, the rule's twins and what trading it costs."""
        symbols = [symbol.strip().upper() for symbol in universe if symbol.strip()]
        spec = None if draft is None else draft.to_spec()
        needs = MIN_DAYS + (signals.warmup_days(spec) if spec else 0)
        checked = []
        for symbol in symbols:
            verdict = self.screen(symbol)
            checked.append(
                SymbolCheck(
                    symbol=symbol,
                    allowed=verdict.allowed,
                    reason=verdict.reason,
                    days=days_of(symbol),
                )
            )
        warnings = [
            f"{item.symbol} has {item.days} days of prices; the backtest needs {needs}"
            for item in checked
            if item.days is not None and item.days < needs
        ]
        warnings += [
            f"{item.symbol}: no price history found" for item in checked if item.days is None
        ]
        twins = (
            []
            if spec is None
            else self.same_rule_as(connection, spec, draft.name if draft else None)
        )
        if twins:
            warnings.append(f"The same rule as {', '.join(twins)}; saving it adds no new trial")
        measured = measured_for(symbols, summaries) if symbols else None
        if draft is not None and measured is not None and measured.cost_bps > draft.fee_bps:
            warnings.append(
                f"Your {measured.orders} {measured.asset} fills cost {measured.cost_bps:g} bps "
                f"on average, more than the {draft.fee_bps:g} bps assumed"
            )
        return DraftCheck(
            symbols=checked,
            needs_days=needs,
            same_rule_as=twins,
            measured_cost=measured,
            problems=[],
            warnings=warnings,
        )

    def check(self, draft: StrategyDraft) -> None:
        refused = refused_symbols(draft, self.screen)
        if refused:
            raise RefusedError("Fails the Shariah screen: " + "; ".join(refused))

    @staticmethod
    def free_name(name: str, taken: Collection[str]) -> str:
        for number in range(2, COPY_LIMIT):
            candidate = f"{name[:44]}-{number}"
            if candidate not in taken:
                return candidate
        raise ConflictError(f"No free name left for a copy of {name}")

    @staticmethod
    def add_version(
        connection: Connection,
        draft: StrategyDraft,
        version: int,
        note: str | None,
        now: datetime,
    ) -> None:
        spec = draft.to_spec()
        record_trial(connection, spec, "version", draft.name)
        connection.execute(
            sa.insert(strategy_definition_versions).values(
                name=draft.name,
                version=version,
                spec=spec.model_dump(mode="json"),
                spec_hash=spec_hash(spec),
                note=note,
                created=now,
            )
        )

    @staticmethod
    def set_archived(connection: Connection, name: str, archived: bool) -> None:
        connection.execute(
            sa.update(strategy_definitions)
            .where(strategy_definitions.c.name == name)
            .values(archived=archived, updated=datetime.now(UTC))
        )

    @staticmethod
    def exists(connection: Connection, name: str) -> bool:
        found = connection.execute(
            sa.select(strategy_definitions.c.name).where(strategy_definitions.c.name == name)
        ).first()
        return found is not None

    @staticmethod
    def names(connection: Connection) -> list[str]:
        return list(connection.execute(sa.select(strategy_definitions.c.name)).scalars())

    @staticmethod
    def evidence(connection: Connection, name: str) -> dict[str, Any] | None:
        """The metrics the lifecycle recorded for the version it holds, if any."""
        found = connection.execute(
            sa.select(strategies.c.stage, strategy_versions.c.metrics)
            .join(strategy_versions, strategies.c.version_id == strategy_versions.c.id)
            .where(strategies.c.name == name)
        ).one_or_none()
        if found is None or not found.metrics:
            return None
        return {"stage": found.stage, **found.metrics}

    @staticmethod
    def rows(connection: Connection, name: str | None = None) -> list[sa.Row[Any]]:
        counts = (
            sa.select(
                strategy_definition_versions.c.name,
                sa.func.count().label("versions"),
            )
            .group_by(strategy_definition_versions.c.name)
            .subquery()
        )
        query = (
            sa.select(
                strategy_definitions,
                strategy_definition_versions.c.spec,
                counts.c.versions,
                strategies.c.stage,
            )
            .join(
                strategy_definition_versions,
                (strategy_definition_versions.c.name == strategy_definitions.c.name)
                & (
                    strategy_definition_versions.c.version == strategy_definitions.c.current_version
                ),
            )
            .join(counts, counts.c.name == strategy_definitions.c.name)
            .outerjoin(strategies, strategies.c.name == strategy_definitions.c.name)
        )
        if name is not None:
            query = query.where(strategy_definitions.c.name == name)
        found = connection.execute(query).all()
        return sorted(found, key=lambda row: listed_order(row.name, row.created))

    @staticmethod
    def summary(row: sa.Row[Any]) -> StrategySummary:
        spec = StrategySpec.model_validate(row.spec)
        return StrategySummary(
            name=row.name,
            title=spec.title,
            signal=spec.signal,
            universe=spec.universe,
            version=row.current_version,
            versions=row.versions,
            archived=row.archived,
            source=row.source,
            stage=row.stage,
            created=row.created,
            updated=row.updated,
        )
