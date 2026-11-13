from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.features.drift.monitor import DriftMonitor
from app.features.drift.schemas import DriftStatus

router = APIRouter(prefix="/drift", tags=["drift"])


def get_monitor(request: Request) -> DriftMonitor:
    monitor: DriftMonitor = request.app.state.drift
    return monitor


Monitor = Annotated[DriftMonitor, Depends(get_monitor)]


@router.get("/status")
def read_status(monitor: Monitor) -> DriftStatus:
    """Each stream's changes since its baseline, the size cut in force and the events recorded."""
    return monitor.status()


@router.post("/check")
def run_check(monitor: Monitor) -> DriftStatus:
    """Scan every stream now and act on any new event."""
    monitor.refresh()
    return monitor.status()
