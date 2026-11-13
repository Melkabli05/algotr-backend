import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from app.core.database import transaction
from app.core.tables import strategy_definition_versions, strategy_definitions, strategy_trials
from app.features.systematic.schemas import StrategySpec

logger = logging.getLogger(__name__)

CRYPTO_NOTE = "Spot crypto only. Scholarly opinion is divided; check it is acceptable to you."
ETF_NOTE = "Verify each ETF's Shariah screening and purification with its issuer."

TEMPLATES: dict[str, StrategySpec] = {
    spec.name: spec
    for spec in (
        StrategySpec(
            name="btc-trend-200",
            title="Bitcoin 200-day trend",
            description="Hold Bitcoin while it closes above its 200-day average, otherwise cash.",
            universe=["BTC/USD"],
            signal="sma",
            window=200,
            fee_bps=25,
            periods_per_year=365,
            fills="close",
            note=CRYPTO_NOTE,
        ),
        StrategySpec(
            name="crypto-donchian",
            title="Crypto Donchian ensemble",
            description=(
                "Nine Donchian breakout models (5 to 360 days) vote on each coin; "
                "the basket is sized to 25% annual volatility and never above 1x."
            ),
            universe=["BTC/USD", "ETH/USD"],
            signal="donchian",
            lookbacks=[5, 10, 20, 30, 60, 90, 150, 250, 360],
            target_vol=0.25,
            fee_bps=25,
            periods_per_year=365,
            fills="close",
            note=CRYPTO_NOTE,
        ),
        StrategySpec(
            name="halal-etf-trend",
            title="Shariah ETF 200-day trend",
            description="Hold each Shariah-screened US equity ETF while above its 200-day average.",
            universe=["SPUS", "HLAL"],
            signal="sma",
            window=200,
            fee_bps=5,
            periods_per_year=252,
            proxy=True,
            note=ETF_NOTE,
        ),
        StrategySpec(
            name="halal-multi-trend",
            title="Shariah ETF multi-speed trend, volatility targeted",
            description=(
                "Hold each Shariah-screened US equity ETF in proportion to the share of five "
                "trend windows (50 to 250 days) it is above, sized to 12% annual volatility and "
                "never above 1x, moving a quarter of the way to each target weight every day."
            ),
            universe=["SPUS", "HLAL"],
            signal="multitrend",
            lookbacks=[50, 100, 150, 200, 250],
            target_vol=0.12,
            vol_window=60,
            trade_rate=0.25,
            fee_bps=5,
            periods_per_year=252,
            proxy=True,
            note=ETF_NOTE,
        ),
        StrategySpec(
            name="halal-etf-rotation",
            title="Shariah ETF dual momentum",
            description=(
                "Hold the two best Shariah ETFs of the last six months, "
                "but only those that gained; otherwise cash."
            ),
            universe=["SPUS", "HLAL", "UMMA", "SPSK"],
            signal="momentum",
            window=126,
            top_n=2,
            fee_bps=5,
            periods_per_year=252,
            note=ETF_NOTE,
        ),
    )
}


def is_crypto(spec: StrategySpec) -> bool:
    return any("/" in symbol for symbol in spec.universe)


def listed_order(name: str, created: datetime) -> tuple[int, datetime, str]:
    """The built-in templates first, in their own order, then the rest as they were made."""
    order = list(TEMPLATES)
    return (order.index(name) if name in order else len(order)), created, name


RETURN_FIELDS = (
    "universe",
    "signal",
    "window",
    "lookbacks",
    "target_vol",
    "vol_window",
    "band",
    "trade_rate",
    "top_n",
    "fee_bps",
    "fills",
    "periods_per_year",
)


def spec_hash(spec: StrategySpec) -> str:
    return hashlib.sha256(spec.model_dump_json().encode()).hexdigest()


def rule_hash(spec: StrategySpec) -> str:
    """What makes two specs one rule: the fields that change its returns, not its wording."""
    found = spec.model_dump(include=set(RETURN_FIELDS), mode="json")
    found["universe"] = sorted(found["universe"])
    found["lookbacks"] = sorted(set(found["lookbacks"]))
    if spec.signal not in ("donchian", "multitrend"):
        found.pop("lookbacks")
    if spec.signal not in ("sma", "momentum"):
        found.pop("window")
    if spec.signal != "momentum":
        found.pop("top_n")
    if spec.target_vol is None:
        found.pop("vol_window")
    return hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()


def record_trial(
    connection: Connection, spec: StrategySpec, origin: str, strategy: str | None
) -> bool:
    """Count a rule once, however often it is saved or tested: true when it is new."""
    found = connection.execute(
        insert(strategy_trials)
        .values(rule_hash=rule_hash(spec), origin=origin, strategy=strategy)
        .on_conflict_do_nothing()
        .returning(strategy_trials.c.rule_hash)
    ).first()
    return found is not None


def backfill_trials(connection: Connection) -> int:
    """Every saved version is a trial; versions saved before trials were kept are added."""
    rows = connection.execute(
        sa.select(strategy_definition_versions.c.name, strategy_definition_versions.c.spec)
    ).all()
    return sum(
        record_trial(connection, StrategySpec.model_validate(row.spec), "version", row.name)
        for row in rows
    )


def seed(connection: Connection, templates: Mapping[str, StrategySpec] = TEMPLATES) -> list[str]:
    """Add each built-in template the database has never had, as version 1 of a strategy.

    A template the user archived or changed is left as it is, since its name is already known.
    """
    known: set[str] = set(connection.execute(sa.select(strategy_definitions.c.name)).scalars())
    added = [name for name in templates if name not in known]
    for name in added:
        spec = templates[name]
        record_trial(connection, spec, "version", name)
        connection.execute(
            insert(strategy_definitions).values(name=name, current_version=1, source="template")
        )
        connection.execute(
            sa.insert(strategy_definition_versions).values(
                name=name,
                version=1,
                spec=spec.model_dump(mode="json"),
                spec_hash=spec_hash(spec),
                note="Built-in template",
            )
        )
    return added


class StrategyRegistry(Mapping[str, StrategySpec]):
    """The strategies the platform runs, read from the database: each one's current version,
    archived ones left out. Until it is first loaded it holds the built-in templates."""

    def __init__(self, templates: Mapping[str, StrategySpec]) -> None:
        self.templates = dict(templates)
        self.specs: dict[str, StrategySpec] = dict(templates)
        self.tried = len(templates)

    def __getitem__(self, name: str) -> StrategySpec:
        return self.specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def trials(self) -> int:
        """Every distinct rule ever saved or backtested as a draft, archived ones included."""
        return self.tried

    def load(self, connection: Connection) -> None:
        rows = connection.execute(
            sa.select(
                strategy_definitions.c.name,
                strategy_definitions.c.created,
                strategy_definition_versions.c.spec,
            )
            .join(
                strategy_definition_versions,
                (strategy_definition_versions.c.name == strategy_definitions.c.name)
                & (
                    strategy_definition_versions.c.version == strategy_definitions.c.current_version
                ),
            )
            .where(strategy_definitions.c.archived.is_(False))
        ).all()
        self.specs = {
            row.name: StrategySpec.model_validate(row.spec)
            for row in sorted(rows, key=lambda row: listed_order(row.name, row.created))
        }
        self.tried = connection.execute(
            sa.select(sa.func.count()).select_from(strategy_trials)
        ).scalar_one()

    def refresh(self) -> bool:
        """Read the database again; on failure the strategies already known stay in use."""
        try:
            with transaction() as connection:
                self.load(connection)
        except Exception:
            logger.exception("Could not read the strategies; keeping the %d known", len(self))
            return False
        return True

    def reset(self) -> None:
        self.specs, self.tried = dict(self.templates), len(self.templates)


STRATEGIES = StrategyRegistry(TEMPLATES)


def open_registry() -> bool:
    """Seed the templates the database lacks, then read every strategy from it."""
    try:
        with transaction() as connection:
            added = seed(connection)
            backfill_trials(connection)
    except Exception:
        logger.exception("Could not seed the strategy templates")
        return False
    if added:
        logger.info("Added the built-in strategies %s", ", ".join(added))
    return STRATEGIES.refresh()
