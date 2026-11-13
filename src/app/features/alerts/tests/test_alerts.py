from collections.abc import Mapping
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import httpx

from app.features.alerts.service import AlertCenter, AlertSettings, Mail, Rule, from_audit

MAIL = Mail(host="smtp.example.com", sender="desk@example.com", to="me@example.com")


def center(
    tmp_path: Path, sent: list[EmailMessage], posted: list[httpx.Request], mail: Mail = MAIL
) -> AlertCenter:
    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request)
        return httpx.Response(200)

    def smtp(_: Mail, message: EmailMessage) -> None:
        sent.append(message)

    return AlertCenter(
        mail,
        path=tmp_path / "alerts.json",
        log_path=tmp_path / "alerts.jsonl",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        smtp=smtp,
    )


def entry(seq: int, kind: str, payload: Mapping[str, object]) -> SimpleNamespace:
    return SimpleNamespace(seq=seq, kind=kind, payload=payload)


def test_a_muted_fill_is_not_delivered(tmp_path: Path) -> None:
    sent: list[EmailMessage] = []
    posted: list[httpx.Request] = []
    found = center(tmp_path, sent, posted)
    found.save(AlertSettings(rules={"fill": Rule(in_app=False)}, webhook_url="https://hook.test"))
    alert = found.raise_alert("fill", "buy 1 SPUS")
    assert alert.channels == {}
    assert found.recent() == []


def test_a_kill_switch_reaches_every_channel_even_when_muted(tmp_path: Path) -> None:
    sent: list[EmailMessage] = []
    posted: list[httpx.Request] = []
    found = center(tmp_path, sent, posted)
    muted = Rule(in_app=False, email=False, webhook=False)
    found.save(AlertSettings(rules={"kill_switch": muted}, webhook_url="https://hook.test"))
    alert = found.raise_alert("kill_switch", "flattened")
    assert alert.channels == {"webhook": "sent", "email": "sent", "in_app": "shown"}
    assert len(sent) == 1 and len(posted) == 1
    assert found.recent()[0].kind == "kill_switch"


def test_email_is_skipped_without_smtp_settings(tmp_path: Path) -> None:
    sent: list[EmailMessage] = []
    found = center(tmp_path, sent, [], mail=Mail())
    alert = found.raise_alert("rejected", "sell 1 AAPL")
    assert "email" not in alert.channels and sent == []


def test_the_first_scan_only_marks_the_place(tmp_path: Path) -> None:
    found = center(tmp_path, [], [])
    failed = {"outcome": "failed", "request": {"side": "buy", "qty": 1, "symbol": "AAPL"}}
    assert found.scan([entry(1, "order", failed)]) == []
    raised = found.scan([entry(1, "order", failed), entry(2, "order", failed)])
    assert [alert.kind for alert in raised] == ["rejected"]
    assert raised[0].detail is not None and raised[0].detail.startswith("buy 1 AAPL")


def test_audit_entries_map_to_alerts() -> None:
    halted = from_audit("run", {"run": {"status": "halted"}})
    assert halted is not None and halted[0] == "kill_switch"
    assert from_audit("run", {"run": {"status": "error", "detail": "x"}}) == ("run_failed", "x")
    assert from_audit("run", {"run": {"status": "ok"}}) is None
    assert from_audit("order", {"outcome": "refused", "reason": "cap"}) == ("risk", "cap")


def test_a_status_check_alerts_once_until_it_recovers(tmp_path: Path) -> None:
    found = center(tmp_path, [], [])
    bad = [SimpleNamespace(name="Ledger", level="error", detail="stale")]
    good = [SimpleNamespace(name="Ledger", level="ok", detail="")]
    assert len(found.check_status(bad)) == 1
    assert found.check_status(bad) == []
    assert found.check_status(good) == []
    assert len(found.check_status(bad)) == 1


def test_tiers_set_the_default_channels() -> None:
    from app.features.alerts.service import default_rule

    assert default_rule("watchdog") == Rule(in_app=True, email=True, webhook=True)
    assert default_rule("limit_near") == Rule(in_app=True, email=True, webhook=False)
    assert default_rule("fill") == Rule(in_app=True, email=False, webhook=False)


def test_a_symbol_mute_silences_one_kind_but_never_a_critical_one(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    found = center(tmp_path, [], [])
    later = datetime.now(UTC) + timedelta(hours=1)
    found.save(AlertSettings(muted={"price:AAPL": later, "rejected:AAPL": later}))
    assert found.raise_alert("price", "AAPL crossed", symbol="AAPL").channels == {}
    assert "in_app" in found.raise_alert("price", "MSFT crossed", symbol="MSFT").channels
    assert "in_app" in found.raise_alert("rejected", "AAPL", symbol="AAPL").channels


def test_a_connection_problem_alerts_after_30_seconds_and_once(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    found = center(tmp_path, [], [])
    start = datetime(2026, 11, 7, tzinfo=UTC)
    assert found.check_connection(["down"], start) == []
    assert len(found.check_connection(["down"], start + timedelta(seconds=31))) == 1
    assert found.check_connection(["down"], start + timedelta(seconds=60)) == []
    assert found.check_connection([], start + timedelta(seconds=90)) == []
    assert found.check_connection(["down"], start + timedelta(seconds=91)) == []


def test_a_connection_alert_clears_once_its_problem_is_gone_and_stays_in_the_record(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime, timedelta

    found = center(tmp_path, [], [])
    start = datetime(2026, 11, 7, tzinfo=UTC)
    found.check_connection(["account down", "crypto down"], start)
    found.check_connection(["account down", "crypto down"], start + timedelta(seconds=31))

    found.check_connection(["crypto down"], start + timedelta(seconds=60))

    alerts = {alert.detail: alert for alert in found.recent()}
    assert alerts["account down"].cleared == start + timedelta(seconds=60)
    assert alerts["crypto down"].cleared is None
    assert alerts["account down"].acknowledged is None


def test_a_day_far_above_the_month_is_flagged_once(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    found = center(tmp_path, [], [])
    calm = [0.001 * ((-1) ** day) for day in range(30)]
    now = datetime(2026, 11, 7, tzinfo=UTC)
    assert found.check_anomaly([*calm, 0.0015], now) is None
    alert = found.check_anomaly([*calm, 0.05], now)
    assert alert is not None and alert.severity == "anomaly"
    assert found.check_anomaly([*calm, 0.05], now) is None


def test_a_limit_near_its_threshold_alerts_until_it_falls_back(tmp_path: Path) -> None:
    found = center(tmp_path, [], [])

    def snap(used: float) -> SimpleNamespace:
        return SimpleNamespace(limits=[SimpleNamespace(name="Sector", used=used)])

    assert len(found.on_risk(snap(0.8))) == 1
    assert found.on_risk(snap(0.7)) == []
    assert found.on_risk(snap(0.9)) == []
    assert found.on_risk(snap(0.5)) == []
    assert len(found.on_risk(snap(0.9))) == 1


def test_a_price_alert_fires_once_when_the_mid_crosses(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from app.features.alerts.service import PriceAlert

    found = center(tmp_path, [], [])
    level = PriceAlert(id="a", symbol="BTC/USD", above=True, price=100, created=datetime.now(UTC))
    found.save(AlertSettings(prices=[level]))
    quote = SimpleNamespace(type="quote", symbol="BTC/USD", bid_price=99, ask_price=100)
    found.on_quote(quote)
    assert found.recent() == []
    found.on_quote(SimpleNamespace(type="quote", symbol="BTC/USD", bid_price=101, ask_price=102))
    assert [alert.kind for alert in found.recent()] == ["price"]
    assert found.settings().prices == []


def test_a_reported_condition_is_not_reported_again_after_a_restart(tmp_path: Path) -> None:
    def snap(used: float) -> SimpleNamespace:
        return SimpleNamespace(limits=[SimpleNamespace(name="Position size", used=used)])

    assert len(center(tmp_path, [], []).on_risk(snap(1.9))) == 1
    assert center(tmp_path, [], []).on_risk(snap(1.9)) == []


def test_a_critical_alert_stays_until_acknowledged(tmp_path: Path) -> None:
    found = center(tmp_path, [], [])
    alert = found.raise_alert("kill_switch", "flattened")
    assert found.recent()[0].acknowledged is None
    assert found.acknowledge(alert.id) is not None
    assert found.recent()[0].acknowledged is not None
    assert found.acknowledge(alert.id) is None


def test_three_rejections_in_ten_minutes_are_a_spike(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    found = center(tmp_path, [], [])
    for _ in range(2):
        found.raise_alert("rejected", "sell 1 X")
    assert found.check_rejections(datetime.now(UTC)) is None
    found.raise_alert("rejected", "sell 1 X")
    spike = found.check_rejections(datetime.now(UTC))
    assert spike is not None and spike.tier == "P0"


def test_settings_and_model_entries_are_quiet_alerts() -> None:
    assert from_audit("settings-change", {"area": "limits"}) == ("settings", "limits changed")
    made = from_audit(
        "ai", {"event": "trained", "model": {"algorithm": "ppo", "last": "2026-11-03"}}
    )
    assert made == ("model", "PPO retrained on data to 2026-11-03")
