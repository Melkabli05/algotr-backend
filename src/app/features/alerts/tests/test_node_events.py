from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.features.alerts.node_events import NodeEventAlerts, classify
from app.features.alerts.pager import pager


def order(status: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "type": "order",
        "client_order_id": "CMD-1",
        "instrument_id": "SPUS.ALPACA",
        "side": "buy",
        "quantity": "10",
        "status": status,
        "reason": reason,
    }


def test_a_refused_order_alerts_once_however_often_the_node_writes_it() -> None:
    seen: set[str] = set()
    found = classify(order("rejected", "insufficient buying power"), lambda _: None, seen)
    assert found == ("rejected", "SPUS buy 10 rejected: insufficient buying power", "SPUS")
    assert classify(order("rejected"), lambda _: None, seen) is None
    assert classify(order("accepted"), lambda _: None, seen) is None


def test_a_fill_far_from_the_last_price_is_an_anomaly() -> None:
    fill = {"type": "fill", "instrument_id": "SPUS.ALPACA", "price": "53"}
    found = classify(fill, lambda _: 50.0, set())
    assert found is not None and found[0] == "anomaly" and "6.0% from the last 50" in found[1]
    assert classify({**fill, "price": "50.4"}, lambda _: 50.0, set()) is None


def test_critical_reconciliation_and_a_stopped_node_alert() -> None:
    drift = {"type": "reconciliation", "severity": "critical", "detail": "lost order"}
    assert classify(drift, lambda _: None, set()) == ("reconcile", "lost order", None)
    stopped = {"type": "status", "name": "node", "state": "stopped"}
    assert classify(stopped, lambda _: None, set()) == (
        "connection",
        "The trading node stopped",
        None,
    )


def test_each_alerting_event_raises_its_alert_as_it_arrives() -> None:
    raised: list[tuple[Any, ...]] = []
    alerts = NodeEventAlerts(
        lambda: None,  # type: ignore[arg-type, return-value]
        lambda kind, detail, symbol=None: raised.append((kind, detail, symbol)),
        lambda _: None,
    )
    alerts.handle(order("denied", "halted"))
    assert raised == [("rejected", "SPUS buy 10 denied: halted", "SPUS")]


class Events:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[dict[str, Any]] = []

    def trigger(self, summary: str, source: str, **kwargs: Any) -> str:
        if self.fail:
            raise RuntimeError("down")
        self.sent.append({"summary": summary, "source": source, **kwargs})
        return "key"


def test_a_critical_alert_pages_through_pagerduty(tmp_path: Path) -> None:
    from app.features.alerts.service import AlertCenter, Mail

    events = Events()
    center = AlertCenter(
        Mail(), path=tmp_path / "settings.json", log_path=tmp_path / "alerts.jsonl"
    )
    center.pager = pager("routing-key", events)
    paged = center.raise_alert("connection", "Alpaca down")
    quiet = center.raise_alert("drift", "a little")
    assert paged.channels["pagerduty"] == "sent"
    assert "pagerduty" not in quiet.channels
    assert events.sent[0]["dedup_key"] == paged.id
    center.pager = pager("routing-key", Events(fail=True))
    assert center.raise_alert("connection", "again").channels["pagerduty"] == "failed: RuntimeError"


def test_prometheus_can_scrape_the_api(client: TestClient) -> None:
    body = client.get("/metrics").text
    assert "algotr_alerts_total" in body or "algotr_breaker_state" in body or "http_request" in body
