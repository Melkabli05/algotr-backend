import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
import sqlalchemy as sa
from pydantic import BaseModel
from scipy.stats import ks_2samp
from sqlalchemy.engine import Connection

from app.core import breakers
from app.core.tables import model_drift
from app.features.models import health

logger = logging.getLogger(__name__)

Kind = Literal["technical", "statistical"]
Severity = Literal["low", "medium", "high"]
TECHNICAL_LIMIT = 0.001
PSI_MEDIUM, PSI_HIGH = 0.1, 0.25
KS_P = 0.01
SHAP_SHIFT = 0.3
RECENT = 30
INPUT_RECENT = 60
INPUT_BASELINE = 250
PSI_BINS = 5
MIN_DECISIONS = 10
SHARPE_WINDOWS = (30, 90, 365)
BAND_Z = 1.96
CALIBRATION_LIMIT = 0.15
MIN_CALIBRATED = 30
CALIBRATION_BINS = 5
BLOCK_WINDOW = timedelta(days=2)
RANK = {"low": 0, "medium": 1, "high": 2}


class DriftFinding(BaseModel):
    subject: str
    kind: Kind
    metric: str
    value: float | None
    threshold: float | None
    drift_severity: Severity
    detail: str


def psi(baseline: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of `recent` against `baseline`, on the baseline's deciles."""
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(baseline, edges)[0] / len(baseline)
    actual = np.histogram(recent, edges)[0] / len(recent)
    expected, actual = np.clip(expected, 1e-6, None), np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def technical(subject: str, gap: float | None) -> DriftFinding | None:
    """The same rule replayed on the same history must reproduce what it decided live."""
    if gap is None:
        return None
    severity: Severity = "high" if gap > TECHNICAL_LIMIT else "low"
    return DriftFinding(
        subject=subject,
        kind="technical",
        metric="replay_divergence",
        value=gap,
        threshold=TECHNICAL_LIMIT,
        drift_severity=severity,
        detail=f"Live decisions and their replay differ by {gap:.3%}",
    )


def statistical(
    subject: str, returns: pd.Series, periods: int, sharpe_floor: float
) -> list[DriftFinding]:
    """Rolling Sharpe of the strategy's or model's own returns over the last 30 days."""
    values = returns.dropna().to_numpy()
    if len(values) < RECENT:
        return []
    recent = values[-RECENT:]
    spread = recent.std()
    sharpe = float(recent.mean() / spread * np.sqrt(periods)) if spread > 0 else 0.0
    return [
        DriftFinding(
            subject=subject,
            kind="statistical",
            metric="rolling_sharpe_30d",
            value=sharpe,
            threshold=sharpe_floor,
            drift_severity="medium" if sharpe < sharpe_floor else "low",
            detail=f"Sharpe over the last 30 days {sharpe:.2f}",
        )
    ]


def sharpe_band(
    subject: str, returns: pd.Series, periods: int, expected: float
) -> list[DriftFinding]:
    """Rolling Sharpe over 30, 90 and 365 observations against the Sharpe the subject was tested
    at: decayed only when even the top of its 95% band falls short (Lo, 2002, for the error)."""
    values = returns.dropna().to_numpy()
    found = []
    for window in SHARPE_WINDOWS:
        if len(values) < window:
            continue
        recent = values[-window:]
        spread = recent.std()
        daily = recent.mean() / spread if spread > 0 else 0.0
        sharpe = float(daily * np.sqrt(periods))
        error = float(np.sqrt((1 + 0.5 * daily**2) / window) * np.sqrt(periods))
        decayed = sharpe + BAND_Z * error < expected
        found.append(
            DriftFinding(
                subject=subject,
                kind="statistical",
                metric=f"sharpe_band_{window}",
                value=sharpe,
                threshold=expected,
                drift_severity="medium" if decayed else "low",
                detail=(
                    f"Sharpe over {window} days {sharpe:.2f} (95% band up to "
                    f"{sharpe + BAND_Z * error:.2f}) against the {expected:.2f} it was tested at"
                ),
            )
        )
    return found


def calibration(
    subject: str, confidences: Sequence[float], correct: Sequence[bool]
) -> DriftFinding | None:
    """Whether stated confidence matches the hit rate: the expected calibration error over
    equal-width bins, judged once there are enough decisions."""
    if len(confidences) < MIN_CALIBRATED:
        return None
    stated = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    hits = np.asarray(correct, dtype=float)
    bins = np.minimum((stated * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    error = sum(
        abs(hits[bins == b].mean() - stated[bins == b].mean()) * (bins == b).mean()
        for b in range(CALIBRATION_BINS)
        if (bins == b).any()
    )
    return DriftFinding(
        subject=subject,
        kind="statistical",
        metric="calibration_error",
        value=float(error),
        threshold=CALIBRATION_LIMIT,
        drift_severity="medium" if error > CALIBRATION_LIMIT else "low",
        detail=(
            f"Its stated confidence and its hit rate differ by {error:.0%} over "
            f"{len(stated)} decisions"
        ),
    )


def input_drift(subject: str, inputs: pd.Series) -> list[DriftFinding]:
    """PSI and a K-S test of the market inputs the subject decides on: recent against the year
    before. Measured on the inputs, not on the strategy's returns, whose distribution moves
    whenever a rule switches between holding and cash."""
    values = inputs.dropna().to_numpy()
    if len(values) < INPUT_RECENT + INPUT_BASELINE // 2:
        return []
    recent = values[-INPUT_RECENT:]
    baseline = values[-(INPUT_RECENT + INPUT_BASELINE) : -INPUT_RECENT]
    index = psi(baseline, recent, bins=PSI_BINS)
    level: Severity = "high" if index > PSI_HIGH else "medium" if index > PSI_MEDIUM else "low"
    test = ks_2samp(recent, baseline)
    return [
        DriftFinding(
            subject=subject,
            kind="statistical",
            metric="psi",
            value=index,
            threshold=PSI_HIGH,
            drift_severity=level,
            detail=f"PSI of its market's recent daily returns against the year before {index:.3f}",
        ),
        DriftFinding(
            subject=subject,
            kind="statistical",
            metric="ks",
            value=float(test.pvalue),
            threshold=KS_P,
            drift_severity="medium" if test.pvalue < KS_P else "low",
            detail=f"K-S test of its market's recent against earlier returns: p {test.pvalue:.4f}",
        ),
    ]


def shap_shares(decisions: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Each decision's share of total |SHAP| per feature group."""
    rows = []
    for decision in decisions:
        values = decision.get("values", {})
        totals = {group: sum(abs(v) for v in assets.values()) for group, assets in values.items()}
        whole = sum(totals.values())
        if whole > 0:
            rows.append({group: total / whole for group, total in totals.items()})
    return pd.DataFrame(rows).fillna(0.0)


def shap_drift(subject: str, decisions: Sequence[Mapping[str, Any]]) -> DriftFinding | None:
    """How far the features a model leans on have moved: the total variation distance between
    the recent decisions' attribution shares and the earlier ones'."""
    shares = shap_shares(decisions)
    if len(shares) < 2 * MIN_DECISIONS:
        return None
    recent = shares.iloc[-MIN_DECISIONS:].mean()
    earlier = shares.iloc[:-MIN_DECISIONS].mean()
    distance = float((recent - earlier).abs().sum() / 2)
    return DriftFinding(
        subject=subject,
        kind="statistical",
        metric="shap_feature_drift",
        value=distance,
        threshold=SHAP_SHIFT,
        drift_severity="high" if distance > SHAP_SHIFT else "low",
        detail=f"The features the model relies on moved by {distance:.0%}",
    )


def adwin(subject: str, changes: int) -> DriftFinding:
    return DriftFinding(
        subject=subject,
        kind="statistical",
        metric="adwin",
        value=float(changes),
        threshold=0.0,
        drift_severity="medium" if changes else "low",
        detail=f"ADWIN found {changes} change(s) in the returns",
    )


def write(connection: Connection, findings: Iterable[DriftFinding], now: datetime) -> None:
    rows = [{"time": now, **finding.model_dump()} for finding in findings]
    if rows:
        connection.execute(sa.insert(model_drift), rows)


def latest(connection: Connection, since: datetime) -> dict[str, tuple[Severity, str]]:
    """Each subject's worst drift since `since`, with the reason."""
    rows = connection.execute(
        sa.select(model_drift.c.subject, model_drift.c.drift_severity, model_drift.c.detail).where(
            model_drift.c.time >= since
        )
    )
    found: dict[str, tuple[Severity, str]] = {}
    for row in rows:
        current = found.get(row.subject)
        if current is None or RANK[row.drift_severity] > RANK[current[0]]:
            found[row.subject] = (row.drift_severity, row.detail)
    return found


def blocked(connection: Connection, now: datetime) -> dict[str, str]:
    """Subjects with high drift in the last two days: their orders are blocked."""
    return {
        subject: detail
        for subject, (severity, detail) in latest(connection, now - BLOCK_WINDOW).items()
        if severity == "high"
    }


def reduced(connection: Connection, now: datetime) -> set[str]:
    return {
        subject
        for subject, (severity, _) in latest(connection, now - BLOCK_WINDOW).items()
        if severity == "medium"
    }


@dataclass
class DriftSources:
    """Where the nightly job reads what it measures."""

    deployed: Callable[[], list[str]]
    replay_gap: Callable[[str], float | None]
    returns: Callable[[str], tuple[pd.Series, int] | None]
    inputs: Callable[[str], pd.Series | None]
    model_returns: Callable[[], dict[str, tuple[pd.Series, int]]]
    model_decisions: Callable[[], dict[str, list[Mapping[str, Any]]]]
    adwin_changes: Callable[[], dict[str, int]]
    retrain: Callable[[str], object] | None = None
    alert: Callable[[str, str], object] | None = None
    expected_sharpe: Callable[[str], float | None] | None = None
    calibrations: Callable[[], dict[str, tuple[list[float], list[bool]]]] | None = None


def nightly(
    connection: Connection, sources: DriftSources, sharpe_floor: float, now: datetime
) -> list[DriftFinding]:
    """Measure every deployed strategy and model, keep the results and act on them."""
    findings: list[DriftFinding] = []
    for name in sources.deployed():
        found = technical(name, sources.replay_gap(name))
        if found is not None:
            findings.append(found)
        series = sources.returns(name)
        if series is not None:
            findings.extend(statistical(name, series[0], series[1], sharpe_floor))
            expected = None if sources.expected_sharpe is None else sources.expected_sharpe(name)
            if expected is not None:
                findings.extend(sharpe_band(name, series[0], series[1], expected))
        market = sources.inputs(name)
        if market is not None:
            findings.extend(input_drift(name, market))
    for name, (returns, periods) in sources.model_returns().items():
        findings.extend(statistical(name, returns, periods, sharpe_floor))
        market = sources.inputs(name)
        if market is not None:
            findings.extend(input_drift(name, market))
    for name, decisions in sources.model_decisions().items():
        found = shap_drift(name, decisions)
        if found is not None:
            findings.append(found)
    findings.extend(adwin(name, count) for name, count in sources.adwin_changes().items())
    if sources.calibrations is not None:
        for name, (stated, correct) in sources.calibrations().items():
            found = calibration(name, stated, correct)
            if found is not None:
                findings.append(found)
    write(connection, findings, now)
    health.write(connection, health.classify(findings, now))
    respond(connection, findings, sources, now)
    return findings


def respond(
    connection: Connection, findings: list[DriftFinding], sources: DriftSources, now: datetime
) -> None:
    """Technical drift halts the strategy for code review; statistical drift asks for a retrain,
    once per subject and never for one whose pipeline is broken, and the arbiter halves it."""
    broken = {f.subject for f in findings if f.kind == "technical" and f.drift_severity == "high"}
    asked: set[str] = set(broken)
    for finding in findings:
        if finding.drift_severity == "low":
            continue
        if finding.kind == "technical":
            switch = breakers.strategy_switch(finding.subject)
            reason = f"Technical drift, code review required: {finding.detail}"
            breakers.set_switch(connection, switch, "strategy", True, reason, "drift", now)
        elif sources.retrain is not None and finding.subject not in asked:
            asked.add(finding.subject)
            sources.retrain(finding.subject)
        if sources.alert is not None:
            sources.alert(f"{finding.subject}: {finding.kind} drift", finding.detail)


def today() -> date:
    return datetime.now(UTC).date()
