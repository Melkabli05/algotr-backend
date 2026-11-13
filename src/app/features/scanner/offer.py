from datetime import datetime, timedelta

from app.core.sleeve import Criterion, Holding, Sleeve
from app.features.scanner import lists
from app.features.scanner.market import Market
from app.features.scanner.schemas import Evaluation, Refresh

MIN_SESSIONS = 40
EVALUATION_DAYS = 30
FRESH = timedelta(hours=36)
OLD_SESSION_DAYS = 5


def criteria(
    evaluation: Evaluation | None, variant: str, sessions: int, intact: bool, now: datetime
) -> list[Criterion]:
    """The record a variant needs before the autopilot may trade it without an override."""
    found: list[Criterion] = []
    if evaluation is None:
        found.append(
            Criterion(
                name="The variants have been evaluated",
                passed=False,
                detail="run it on the Evaluation tab",
            )
        )
    else:
        age = (now.date() - evaluation.end).days
        found.append(
            Criterion(
                name=f"Evaluated in the last {EVALUATION_DAYS} days",
                passed=age <= EVALUATION_DAYS,
                detail=f"through {evaluation.end:%b} {evaluation.end.day}",
            )
        )
        result = next((item for item in evaluation.variants if item.label == variant), None)
        if result is None:
            found.append(Criterion(name="This variant was evaluated", passed=False))
        else:
            found.extend(Criterion(name=check.name, passed=check.passed) for check in result.checks)
    found.append(
        Criterion(
            name=f"At least {MIN_SESSIONS} sessions logged",
            passed=sessions >= MIN_SESSIONS,
            detail=f"{sessions} logged",
        )
    )
    found.append(Criterion(name="Audit log intact", passed=intact))
    return found


def problems(found: Market, last: Refresh | None, now: datetime) -> list[str]:
    """Why the holdings cannot be acted on right now: old data, an old list, no list."""
    session = found.analysis.found.close.index[-1].date()
    reasons = []
    if last is None or last.status != "ok":
        reasons.append("The last data refresh did not finish")
    elif now - last.started > FRESH:
        reasons.append("The data was last refreshed more than 36 hours ago")
    if (now.date() - session).days > OLD_SESSION_DAYS:
        reasons.append(f"The last session in the data is {(now.date() - session).days} days old")
    if not found.analysis.members:
        reasons.append("No current Shariah list")
    reasons.extend(
        f"The {name} list is more than {lists.STALE_AFTER.days} days old" for name in found.stale
    )
    return reasons


def build(
    variant: str,
    found: Market,
    blocked: dict[str, str],
    gate: list[Criterion],
    trouble: list[str],
) -> Sleeve | None:
    """What `variant` holds on its schedule (monthly ranking, weekly drop), with the news vetoes."""
    weights = found.analysis.scheduled.get(variant)
    if weights is None:
        return None
    close = found.analysis.found.close.iloc[-1]
    return Sleeve(
        variant=variant,
        session=found.analysis.found.close.index[-1].date(),
        holdings=[
            Holding(
                symbol=symbol,
                weight=weight,
                price=float(close[symbol]),
                name=found.names.get(symbol),
                blocked=blocked.get(symbol),
            )
            for symbol, weight in sorted(weights.items(), key=lambda item: -item[1])
        ],
        criteria=gate,
        problems=trouble,
    )
