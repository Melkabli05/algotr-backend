import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.features.alerts.service import (
    KINDS,
    TIERS,
    TITLES,
    UNMUTABLE,
    Alert,
    AlertCenter,
    AlertSettings,
    Kind,
    PriceAlert,
)

router = APIRouter(prefix="/alerts", tags=["alerts"])


def get_center(request: Request) -> AlertCenter:
    found: AlertCenter = request.app.state.alerts
    return found


Center = Annotated[AlertCenter, Depends(get_center)]


class Mute(BaseModel):
    kind: Kind
    symbol: str
    minutes: int = Field(default=60, ge=1, le=24 * 60)


class NewPrice(BaseModel):
    symbol: str
    above: bool
    price: float = Field(gt=0)


class SettingsView(BaseModel):
    settings: AlertSettings
    kinds: dict[str, str]
    tiers: dict[str, str]
    unmutable: list[str]
    email_ready: bool
    email_to: str | None


@router.get("")
def read_alerts(center: Center, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> list[Alert]:
    """The alerts shown in the terminal, newest first, with how each was delivered."""
    return center.recent(limit)


@router.get("/settings")
def read_settings(center: Center) -> SettingsView:
    return SettingsView(
        settings=center.settings(),
        kinds={kind: TITLES[kind] for kind in KINDS},
        tiers={kind: TIERS[kind] for kind in KINDS},
        unmutable=list(UNMUTABLE),
        email_ready=center.mail.ready,
        email_to=center.mail.to,
    )


@router.put("/settings")
def put_settings(request: Request, center: Center, settings: AlertSettings) -> SettingsView:
    """Save which channels each kind of alert uses; critical kinds stay on every channel."""
    before = center.settings()
    center.save(settings)
    request.app.state.audit.record(
        "settings-change",
        {"area": "alerts", "before": before.model_dump(), "after": settings.model_dump()},
    )
    return read_settings(center)


@router.post("/test")
def send_test(center: Center) -> Alert:
    """Send one test alert by every configured channel."""
    return center.raise_alert("system", "Test alert from the terminal", force=True)


@router.post("/mute")
def mute(center: Center, body: Mute) -> SettingsView:
    """Silence one kind of alert for one symbol for a while; critical kinds cannot be muted."""
    if body.kind in UNMUTABLE:
        raise HTTPException(status_code=422, detail="Critical alerts cannot be muted")
    with center.lock:
        settings = center.settings()
        now = datetime.now(UTC)
        settings.muted = {key: until for key, until in settings.muted.items() if until > now}
        settings.muted[f"{body.kind}:{body.symbol}"] = now + timedelta(minutes=body.minutes)
        center.save(settings)
    return read_settings(center)


@router.post("/price")
def add_price(center: Center, body: NewPrice) -> PriceAlert:
    """Alert once when the quote's mid crosses `price`, from below if `above`, else from above."""
    found = PriceAlert(
        id=uuid.uuid4().hex[:10],
        symbol=body.symbol.upper(),
        above=body.above,
        price=body.price,
        created=datetime.now(UTC),
    )
    with center.lock:
        settings = center.settings()
        settings.prices.append(found)
        center.save(settings)
    return found


@router.delete("/price/{alert_id}")
def delete_price(center: Center, alert_id: str) -> SettingsView:
    with center.lock:
        settings = center.settings()
        settings.prices = [item for item in settings.prices if item.id != alert_id]
        center.save(settings)
    return read_settings(center)


@router.post("/{alert_id}/ack")
def acknowledge(center: Center, alert_id: str) -> Alert:
    """Acknowledge an alert; critical ones stay on screen until this is done."""
    found = center.acknowledge(alert_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No such unacknowledged alert")
    return found
