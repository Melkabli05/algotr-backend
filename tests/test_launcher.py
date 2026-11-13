import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dry_run(task: str) -> str:
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "poethepoet", "--dry-run", task],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout + done.stderr


def test_start_builds_the_terminal_then_runs_the_api_that_serves_it() -> None:
    plan = dry_run("start").splitlines()

    assert [" ".join(line.split()[2:5]) for line in plan] == [
        "pnpm --dir frontend",
        "uvicorn app.main:app --port",
    ]
    assert (ROOT / "frontend" / "package.json").exists()


def test_finrl_setup_creates_the_sidecar_environment_then_installs_into_it() -> None:
    plan = dry_run("finrl-setup").splitlines()

    assert [" ".join(line.split()[2:4]) for line in plan] == ["uv venv", "uv pip"]
    assert (ROOT / "finrl_sidecar" / "requirements.txt").exists()
