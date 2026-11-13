import asyncio
import json
import logging
import smtplib
import ssl
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from statistics import fmean, pstdev
from threading import Lock
from typing import Any, Literal

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel, Field, computed_field

from app.core import metrics

logger = logging.getLogger(__name__)

PATH = Path("data/alerts.json")
LOG = Path("data/alerts.jsonl")
KEPT = 500
SCAN_SECONDS = 30
BEAT_SECONDS = 10
WATCH_SECONDS = 15
LOOP_SILENCE = 60
CONNECTION_GRACE = timedelta(seconds=30)
ANOMALY_EVERY = timedelta(minutes=10)
ANOMALY_Z = 3.0
MIN_DAYS = 20
LIMIT_NEAR = 0.75
LIMIT_CLEAR = 0.6
REJECTION_WINDOW = timedelta(minutes=10)
REJECTION_SPIKE = 3
Kind = Literal[
    "fill",
    "rejected",
    "risk",
    "kill_switch",
    "run_failed",
    "drift",
    "system",
    "connection",
    "reconcile",
    "watchdog",
    "anomaly",
    "limit_near",
    "price",
    "rejections",
    "settings",
    "model",
]
Channel = Literal["in_app", "email", "webhook"]
Tier = Literal["P0", "P1", "P2"]
Severity = Literal["critical", "warning", "anomaly", "info"]
TIERS: dict[Kind, Tier] = {
    "kill_switch": "P0",
    "run_failed": "P0",
    "rejected": "P0",
    "connection": "P0",
    "reconcile": "P0",
    "watchdog": "P0",
    "anomaly": "P0",
    "rejections": "P0",
    "risk": "P1",
    "drift": "P1",
    "system": "P1",
    "limit_near": "P1",
    "price": "P1",
    "fill": "P2",
    "settings": "P2",
    "model": "P2",
}
KINDS: tuple[Kind, ...] = tuple(TIERS)
UNMUTABLE: tuple[Kind, ...] = tuple(kind for kind, tier in TIERS.items() if tier == "P0")
TITLES: dict[Kind, str] = {
    "fill": "Order filled",
    "rejected": "Order rejected or failed",
    "risk": "Order refused by a risk check",
    "kill_switch": "Kill switch",
    "run_failed": "Autopilot run failed",
    "drift": "Drift detected",
    "system": "System check failed",
    "connection": "Connection to Alpaca lost",
    "reconcile": "Ledger and Alpaca disagree",
    "watchdog": "Heartbeat missed",
    "anomaly": "Unusual P&L detected",
    "limit_near": "Risk limit at 75% or more",
    "price": "Price alert",
    "rejections": "Order rejections spiking",
    "settings": "Configuration changed",
    "model": "Model deployed",
}


def severity_of(kind: Kind) -> Severity:
    if kind == "anomaly":
        return "anomaly"
    levels: dict[Tier, Severity] = {"P0": "critical", "P1": "warning", "P2": "info"}
    return levels[TIERS[kind]]


class Rule(BaseModel):
    in_app: bool = True
    email: bool = False
    webhook: bool = False


def default_rule(kind: Kind) -> Rule:
    """P0 everywhere, P1 in the app and by email, P2 in the app only."""
    tier = TIERS[kind]
    return Rule(in_app=True, email=tier in ("P0", "P1"), webhook=tier == "P0")


class PriceAlert(BaseModel):
    id: str
    symbol: str
    above: bool
    price: float = Field(gt=0)
    created: datetime


class AlertSettings(BaseModel):
    rules: dict[Kind, Rule] = Field(
        default_factory=lambda: {kind: default_rule(kind) for kind in KINDS}
    )
    webhook_url: str | None = None
    muted: dict[str, datetime] = {}
    prices: list[PriceAlert] = []


class Mail(BaseModel):
    """Where email alerts go; set in the environment, never in the terminal."""

    host: str | None = None
    port: int = 587
    user: str | None = None
    password: str | None = None
    sender: str | None = None
    to: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self.host and self.sender and self.to)


class Alert(BaseModel):
    time: datetime
    kind: Kind
    title: str
    detail: str | None = None
    symbol: str | None = None
    tier: Tier = "P1"
    severity: Severity = "warning"
    channels: dict[str, str] = {}
    acknowledged: datetime | None = None
    cleared: datetime | None = None

    @computed_field
    @property
    def id(self) -> str:
        return f"{self.time.isoformat()}|{self.kind}"


def effective(settings: AlertSettings, kind: Kind) -> Rule:
    """The rule in force: a critical kind goes to every channel, whatever the settings say."""
    if kind in UNMUTABLE:
        return Rule(in_app=True, email=True, webhook=True)
    return settings.rules.get(kind, default_rule(kind))


def muted(settings: AlertSettings, kind: Kind, symbol: str | None, now: datetime) -> bool:
    """A kind muted for one symbol until a time; critical kinds are never muted."""
    if kind in UNMUTABLE or symbol is None:
        return False
    until = settings.muted.get(f"{kind}:{symbol}")
    return until is not None and until > now


def z_score(today: float, history: list[float]) -> float | None:
    """How many standard deviations today's return is above the mean of the days before."""
    if len(history) < MIN_DAYS:
        return None
    mean = fmean(history)
    spread = pstdev(history)
    return None if spread <= 0 else (today - mean) / spread


def from_audit(kind: str, payload: dict[str, Any]) -> tuple[Kind, str] | None:
    """The alert an audit entry raises, if any."""
    if kind == "order":
        outcome = payload.get("outcome")
        request = payload.get("request") or {}
        what = " ".join(str(request[key]) for key in ("side", "qty", "symbol") if key in request)
        detail = ": ".join(part for part in (what, str(payload.get("reason") or "")) if part)
        if outcome == "refused":
            return "risk", detail
        if outcome == "failed":
            return "rejected", detail
    if kind == "run":
        run = payload.get("run") or {}
        if run.get("status") == "halted":
            return "kill_switch", str(run.get("detail") or "positions closed and autopilot halted")
        if run.get("status") == "error":
            return "run_failed", str(run.get("detail") or "")
    if kind == "drift":
        return "drift", json.dumps(payload)[:300]
    if kind == "settings-change":
        return "settings", f"{payload.get('area', 'settings')} changed"
    if kind == "ai" and payload.get("event") == "trained":
        model = payload.get("model") or {}
        name = str(model.get("algorithm", "")).upper()
        return "model", f"{name} retrained on data to {model.get('last', '')}"
    return None


def from_update(
    event: str, symbol: str, side: str, qty: float | None, price: float | None
) -> tuple[Kind, str] | None:
    """The alert a broker order update raises, if any."""
    if event in ("fill", "partial_fill"):
        return "fill", f"{side} {qty or ''} {symbol} at {price or ''}".strip()
    if event == "rejected":
        return "rejected", f"{side} {qty or ''} {symbol} rejected by the broker"
    return None


class AlertCenter:
    """Turns events into alerts and delivers each by the channels its rule names."""

    def __init__(
        self,
        mail: Mail,
        path: Path = PATH,
        log_path: Path = LOG,
        client: httpx.Client | None = None,
        smtp: Callable[[Mail, EmailMessage], None] | None = None,
    ) -> None:
        self.mail = mail
        self.path = path
        self.log_path = log_path
        self.client = client or httpx.Client(timeout=10)
        self.smtp = smtp or send_mail
        self.lock = Lock()
        self.seen = 0
        self.failing: set[str] = set()
        self.scheduler = AsyncIOScheduler(timezone=UTC)
        self.audit: Any = None
        self.status: Callable[[], Any] | None = None
        self.problems: Callable[[], list[str]] | None = None
        self.pager: Callable[[Alert], str] | None = None
        self.ledger: Callable[[], Any] | None = None
        self.daily_returns: Callable[[], list[float]] | None = None
        self.overdue: Callable[[datetime], list[str]] | None = None
        self.since: dict[str, datetime] = {}
        self.open_path = path.with_name("alerts_open.json")
        self.open: set[str] = self._load_open()
        self.anomaly_day: str | None = None
        self.anomaly_checked: datetime | None = None
        self.beat = time.monotonic()
        self.watchdog: threading.Thread | None = None
        self.running = False

    def settings(self) -> AlertSettings:
        try:
            return AlertSettings.model_validate_json(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return AlertSettings()

    def save(self, settings: AlertSettings) -> AlertSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(settings.model_dump_json(indent=2))
        return settings

    def recent(self, limit: int = 100) -> list[Alert]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text().splitlines()[-limit:]
        return [Alert.model_validate_json(line) for line in reversed(lines)]

    def raise_alert(
        self, kind: Kind, detail: str | None, force: bool = False, symbol: str | None = None
    ) -> Alert:
        settings = self.settings()
        now = datetime.now(UTC)
        rule = Rule(in_app=True, email=True, webhook=True) if force else effective(settings, kind)
        if not force and muted(settings, kind, symbol, now):
            rule = Rule(in_app=False)
        alert = Alert(
            time=now,
            kind=kind,
            title=TITLES[kind],
            detail=detail,
            symbol=symbol,
            tier=TIERS[kind],
            severity=severity_of(kind),
        )
        metrics.ALERTS.labels(kind, alert.tier).inc()
        if alert.tier == "P0" and self.pager is not None:
            alert.channels["pagerduty"] = self.pager(alert)
        if rule.webhook and settings.webhook_url:
            alert.channels["webhook"] = self._webhook(settings.webhook_url, alert)
        if rule.email and self.mail.ready:
            alert.channels["email"] = self._email(alert)
        if rule.in_app:
            alert.channels["in_app"] = "shown"
            with self.lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as handle:
                    handle.write(alert.model_dump_json(exclude={"id"}) + "\n")
                lines = self.log_path.read_text().splitlines()
                if len(lines) > KEPT:
                    self.log_path.write_text("\n".join(lines[-KEPT:]) + "\n")
        return alert

    def _webhook(self, url: str, alert: Alert) -> str:
        try:
            response = self.client.post(
                url,
                json={
                    "text": f"{alert.title}: {alert.detail or ''}",
                    **alert.model_dump(mode="json"),
                },
            )
            return "sent" if response.is_success else f"failed: HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            logger.warning("Webhook alert failed: %s", exc)
            return f"failed: {type(exc).__name__}"

    def _email(self, alert: Alert) -> str:
        message = EmailMessage()
        message["Subject"] = f"[algotr] {alert.title}"
        message["From"] = str(self.mail.sender)
        message["To"] = str(self.mail.to)
        message.set_content(f"{alert.title}\n\n{alert.detail or ''}\n\n{alert.time.isoformat()}")
        try:
            self.smtp(self.mail, message)
            return "sent"
        except (OSError, smtplib.SMTPException) as exc:
            logger.warning("Email alert failed: %s", exc)
            return f"failed: {type(exc).__name__}"

    def scan(self, entries: Iterable[Any]) -> list[Alert]:
        """Alerts for audit entries not seen before; the first scan only marks the place."""
        found = list(entries)
        first = self.seen == 0
        raised = []
        for entry in found:
            if entry.seq <= self.seen:
                continue
            self.seen = max(self.seen, entry.seq)
            if first:
                continue
            made = from_audit(entry.kind, entry.payload)
            if made is not None:
                raised.append(self.raise_alert(*made))
        return raised

    def acknowledge(self, alert_id: str) -> Alert | None:
        """Mark one alert as seen by the operator; P0 alerts stay up until this is done."""
        found = None
        with self.lock:
            if not self.log_path.exists():
                return None
            lines = self.log_path.read_text().splitlines()
            for index, line in enumerate(lines):
                alert = Alert.model_validate_json(line)
                if alert.id == alert_id and alert.acknowledged is None:
                    alert.acknowledged = datetime.now(UTC)
                    lines[index] = alert.model_dump_json(exclude={"id"})
                    found = alert
            if found is not None:
                self.log_path.write_text("\n".join(lines) + "\n")
        return found

    def resolve(self, kind: Kind, still: set[str], now: datetime) -> list[Alert]:
        """Mark `kind` alerts whose condition no longer holds as cleared; the record stays."""
        found: list[Alert] = []
        with self.lock:
            if not self.log_path.exists():
                return found
            lines = self.log_path.read_text().splitlines()
            for index, line in enumerate(lines):
                alert = Alert.model_validate_json(line)
                waiting = alert.acknowledged is None and alert.cleared is None
                if alert.kind == kind and waiting and alert.detail not in still:
                    alert.cleared = now
                    lines[index] = alert.model_dump_json(exclude={"id"})
                    found.append(alert)
            if found:
                self.log_path.write_text("\n".join(lines) + "\n")
        return found

    def check_rejections(self, now: datetime) -> Alert | None:
        """Three or more rejected or failed orders within ten minutes is a spike."""
        recent = [
            alert
            for alert in self.recent(KEPT)
            if alert.kind == "rejected" and now - alert.time <= REJECTION_WINDOW
        ]
        return self.once(
            "rejections",
            len(recent) >= REJECTION_SPIKE,
            "rejections",
            f"{len(recent)} orders rejected or failed in ten minutes; check the broker and limits",
        )

    def check_status(self, checks: Iterable[Any]) -> list[Alert]:
        """An alert when a status check turns bad, once until it recovers."""
        raised = []
        bad = {check.name: check.detail for check in checks if check.level == "error"}
        for name, detail in bad.items():
            if name not in self.failing:
                raised.append(self.raise_alert("system", f"{name}: {detail}"))
        self.failing = set(bad)
        return raised

    def on_update(self, event: Any) -> None:
        made = from_update(
            event.event, event.symbol, event.side, event.qty, event.filled_avg_price or event.price
        )
        if made is not None:
            self.raise_alert(*made)

    def _load_open(self) -> set[str]:
        """Conditions already reported, so a restart does not report them again."""
        try:
            return set(json.loads(self.open_path.read_text()))
        except (FileNotFoundError, ValueError):
            return set()

    def _save_open(self) -> None:
        self.open_path.parent.mkdir(parents=True, exist_ok=True)
        self.open_path.write_text(json.dumps(sorted(self.open)))

    def close(self, key: str) -> None:
        if key in self.open:
            self.open.discard(key)
            self._save_open()

    def once(
        self, key: str, active: bool, kind: Kind, detail: str, symbol: str | None = None
    ) -> Alert | None:
        """Raise when `key` turns active, and not again until it has cleared."""
        if not active:
            self.close(key)
            return None
        if key in self.open:
            return None
        self.open.add(key)
        self._save_open()
        return self.raise_alert(kind, detail, symbol=symbol)

    def check_connection(self, problems: Iterable[str], now: datetime) -> list[Alert]:
        """A connection problem that lasts 30 seconds raises one alert until it clears."""
        current = set(problems)
        for gone in [key for key in self.since if key not in current]:
            del self.since[gone]
        raised = []
        for problem in current:
            first = self.since.setdefault(problem, now)
            lasting = now - first >= CONNECTION_GRACE
            found = self.once(f"connection:{problem}", lasting, "connection", problem)
            if found:
                raised.append(found)
        for key in [key for key in self.open if key.startswith("connection:")]:
            if key.removeprefix("connection:") not in current:
                self.close(key)
        self.resolve("connection", current, now)
        return raised

    def check_ledger(self, found: Any) -> Alert | None:
        if found is None:
            return None
        symbols = [item.symbol for item in found.differences if item.kind == "quantity"]
        return self.once(
            "reconcile",
            not found.ok,
            "reconcile",
            f"Quantities differ for {', '.join(symbols)}; open Reconciliation",
        )

    def check_anomaly(self, returns: list[float], now: datetime) -> Alert | None:
        """Today's gain more than three deviations above the last 30 days: likely a fault."""
        if len(returns) < 2:
            return None
        today, history = returns[-1], returns[:-1][-30:]
        score = z_score(today, history)
        day = now.date().isoformat()
        if score is None or score <= ANOMALY_Z or self.anomaly_day == day:
            return None
        self.anomaly_day = day
        return self.raise_alert(
            "anomaly",
            f"Today's return ({today:+.1%}) is {score:.1f} sigma above the 30-day mean; "
            "verify state reconciliation before continuing",
        )

    def on_risk(self, snapshot: Any) -> list[Alert]:
        """A limit at 75% of its threshold raises once, again only after falling below 60%."""
        raised = []
        for item in snapshot.limits:
            if item.used is None:
                continue
            key = f"limit:{item.name}"
            if item.used < LIMIT_CLEAR:
                self.close(key)
                continue
            if item.used < LIMIT_NEAR:
                continue
            found = self.once(
                key,
                True,
                "limit_near",
                f"{item.name} at {item.used:.0%} of its limit",
            )
            if found:
                raised.append(found)
        return raised

    def on_quote(self, event: Any) -> None:
        """Fire and remove the price alerts the quote's mid has crossed."""
        if getattr(event, "type", None) != "quote":
            return
        settings = self.settings()
        mine = [item for item in settings.prices if item.symbol == event.symbol]
        if not mine or not event.bid_price or not event.ask_price:
            return
        mid = (event.bid_price + event.ask_price) / 2
        hit = [item for item in mine if (mid >= item.price) == item.above]
        if not hit:
            return
        with self.lock:
            fresh = self.settings()
            fresh.prices = [item for item in fresh.prices if item.id not in {h.id for h in hit}]
            self.save(fresh)
        for item in hit:
            word = "above" if item.above else "below"
            self.raise_alert(
                "price",
                f"{item.symbol} at {mid:,.4g}, {word} {item.price:,.4g}",
                symbol=item.symbol,
            )

    def heartbeat(self) -> None:
        self.beat = time.monotonic()

    def watch(self) -> None:
        """Runs on its own thread: the event loop must beat; sleep of the machine is skipped."""
        last = time.monotonic()
        while self.running:
            time.sleep(WATCH_SECONDS)
            now = time.monotonic()
            slept = now - last > WATCH_SECONDS * 3
            last = now
            if slept:
                self.beat = now
                continue
            silent = now - self.beat
            self.once(
                "watchdog:loop",
                silent > LOOP_SILENCE,
                "watchdog",
                f"The event loop has not run for {silent:.0f}s; streams and jobs are stalled",
            )

    def tick(self) -> None:
        now = datetime.now(UTC)
        steps: list[Callable[[], Any]] = []
        if (audit := self.audit) is not None:
            steps.append(lambda: self.scan(audit.entries()))
        if (status := self.status) is not None:
            read_status = status
            steps.append(lambda: self.check_status(read_status().checks))
        if (problems := self.problems) is not None:
            read_problems = problems
            steps.append(lambda: self.check_connection(read_problems(), now))
        if (ledger := self.ledger) is not None:
            read_ledger = ledger
            steps.append(lambda: self.check_ledger(read_ledger()))
        if (overdue := self.overdue) is not None:
            read_overdue = overdue
            steps.append(lambda: self.check_overdue(read_overdue(now)))
        due = self.anomaly_checked is None or now - self.anomaly_checked >= ANOMALY_EVERY
        if (returns := self.daily_returns) is not None and due:
            self.anomaly_checked = now
            read_returns = returns
            steps.append(lambda: self.check_anomaly(read_returns(), now))
        steps.append(lambda: self.check_rejections(now))
        for step in steps:
            try:
                step()
            except Exception:
                logger.exception("An alert check failed")

    def check_overdue(self, jobs: Iterable[str]) -> list[Alert]:
        """A scheduled job far past its run time: the loop or the scheduler has stopped."""
        current = {f"watchdog:{job}" for job in jobs}
        for key in [key for key in self.open if key.startswith("watchdog:job ")]:
            if key not in current:
                self.close(key)
        raised = []
        for key in current:
            found = self.once(key, True, "watchdog", f"{key.removeprefix('watchdog:')} is overdue")
            if found:
                raised.append(found)
        return raised

    async def _tick(self) -> None:
        await asyncio.to_thread(self.tick)

    def start(self) -> None:
        self.scheduler.add_job(
            self._tick,
            "interval",
            seconds=SCAN_SECONDS,
            id="alerts",
            next_run_time=datetime.now(UTC),
            misfire_grace_time=None,
        )
        self.scheduler.add_job(self.heartbeat, "interval", seconds=BEAT_SECONDS, id="heartbeat")
        self.scheduler.start()
        self.running = True
        self.watchdog = threading.Thread(target=self.watch, daemon=True, name="alert-watchdog")
        self.watchdog.start()

    def stop(self) -> None:
        self.running = False
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)


def send_mail(mail: Mail, message: EmailMessage) -> None:
    """Send by SMTP: implicit TLS on port 465, STARTTLS otherwise."""
    context = ssl.create_default_context()
    if mail.port == 465:
        with smtplib.SMTP_SSL(str(mail.host), mail.port, context=context, timeout=20) as server:
            if mail.user:
                server.login(mail.user, mail.password or "")
            server.send_message(message)
        return
    with smtplib.SMTP(str(mail.host), mail.port, timeout=20) as server:
        server.starttls(context=context)
        if mail.user:
            server.login(mail.user, mail.password or "")
        server.send_message(message)
