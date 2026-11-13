import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.core import sidecar


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> None:
        found["job"] = json.loads(Path(command[2]).read_text())
        found["timeout"] = kwargs["timeout"]
        found["files"] = {
            name: Path(path).read_text() for name, path in found["job"]["files"].items()
        }
        Path(command[3]).write_text(
            json.dumps({"status": "ok", "seconds": 1.0, "task": "t", "data": {"fine": True}})
        )

    monkeypatch.setattr(subprocess, "run", run)
    return found


def test_the_job_carries_the_task_the_parameters_and_the_prices_as_files(
    seen: dict[str, Any],
) -> None:
    outcome = sidecar.run_job("t", {"a": 1}, {"data": pd.DataFrame({"x": [1, 2]})})
    assert outcome.status == "ok" and outcome.data == {"fine": True}
    assert seen["job"]["task"] == "t" and seen["job"]["params"] == {"a": 1}
    assert seen["files"]["data"] == "x\n1\n2\n"


def test_the_sidecars_own_time_limit_follows_the_timeout_it_is_given(seen: dict[str, Any]) -> None:
    sidecar.run_job("t", {}, {}, timeout=7200)
    assert seen["timeout"] == 7200 and seen["job"]["limit"] == 7170


def test_a_short_timeout_still_leaves_the_sidecar_a_minute(seen: dict[str, Any]) -> None:
    sidecar.run_job("t", {}, {}, timeout=10)
    assert seen["job"]["limit"] == sidecar.MIN_LIMIT


def test_the_default_timeout_keeps_the_sidecar_inside_it(seen: dict[str, Any]) -> None:
    sidecar.run_job("t", {}, {})
    assert seen["job"]["limit"] == sidecar.TIMEOUT_SECONDS - sidecar.LIMIT_MARGIN
