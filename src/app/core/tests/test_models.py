import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from app.core import models, sidecar


def done(folder: Path) -> Path:
    folder.mkdir(parents=True)
    (folder / "revision.json").write_text("{}")
    return folder


class Recorder:
    def __init__(self, results: list[bool] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.results = results or []

    async def __call__(self, command: list[str]) -> bool:
        self.commands.append(command)
        return self.results.pop(0) if self.results else True


def test_a_model_is_missing_until_its_revision_file_is_written(tmp_path: Path) -> None:
    folders = {"finbert": done(tmp_path / "finbert"), "kronos": tmp_path / "kronos"}

    assert models.missing(folders) == ["kronos"]


def test_nothing_runs_when_every_model_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Recorder()
    monkeypatch.setattr(models, "run", run)
    folders = {"finbert": done(tmp_path / "finbert"), "kronos": done(tmp_path / "kronos")}

    asyncio.run(models.ensure(folders))

    assert run.commands == []


def test_only_the_missing_model_is_fetched_with_the_sidecar_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Recorder()
    monkeypatch.setattr(models, "run", run)
    monkeypatch.setattr(sidecar, "available", lambda: True)
    folders = {"finbert": done(tmp_path / "finbert"), "kronos": tmp_path / "kronos"}

    asyncio.run(models.ensure(folders))

    assert len(run.commands) == 1
    python, script, target = run.commands[0]
    assert python == str(sidecar.sidecar_python())
    assert Path(script).name == "fetch_kronos.py"
    assert target == str(tmp_path / "kronos")


def test_the_sidecar_environment_is_built_first_when_it_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Recorder()
    monkeypatch.setattr(models, "run", run)
    monkeypatch.setattr(sidecar, "available", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/uv")

    asyncio.run(models.ensure({"finbert": tmp_path / "finbert"}))

    steps = [" ".join(command[:2]) for command in run.commands]
    fetch = f"{sidecar.sidecar_python()} {sidecar.SIDECAR / 'fetch_finbert.py'}"
    assert steps == ["/bin/uv venv", "/bin/uv pip", fetch]


def test_nothing_is_fetched_without_uv_to_build_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Recorder()
    monkeypatch.setattr(models, "run", run)
    monkeypatch.setattr(sidecar, "available", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    asyncio.run(models.ensure({"finbert": tmp_path / "finbert"}))

    assert run.commands == []


def test_no_model_is_fetched_when_building_the_environment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Recorder([False])
    monkeypatch.setattr(models, "run", run)
    monkeypatch.setattr(sidecar, "available", lambda: False)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/uv")

    asyncio.run(models.ensure({"finbert": tmp_path / "finbert"}))

    assert len(run.commands) == 1


def test_a_command_reports_whether_it_succeeded() -> None:
    assert asyncio.run(models.run([sys.executable, "-c", "print('fetched')"])) is True
    assert asyncio.run(models.run([sys.executable, "-c", "raise SystemExit(3)"])) is False
