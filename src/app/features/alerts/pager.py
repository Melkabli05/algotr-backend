import logging
from collections.abc import Callable
from typing import Any

import pagerduty

logger = logging.getLogger(__name__)

SOURCE = "algotr"


def pager(routing_key: str, client: Any = None) -> Callable[[Any], str]:
    """Pages for critical alerts through PagerDuty's Events API v2, one incident per alert."""
    events = client or pagerduty.EventsApiV2Client(routing_key)

    def send(alert: Any) -> str:
        try:
            events.trigger(
                f"{alert.title}: {alert.detail or ''}"[:1024],
                SOURCE,
                dedup_key=alert.id,
                severity="critical",
                custom_details={"kind": alert.kind, "symbol": alert.symbol},
            )
        except Exception as exc:
            logger.warning("PagerDuty page failed: %s", exc)
            return f"failed: {type(exc).__name__}"
        return "sent"

    return send
