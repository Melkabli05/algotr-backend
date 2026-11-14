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


def test_finrl_setup_creates_the_sidecar_environment_then_installs_into_it() -> None:
    plan = dry_run("finrl-setup").splitlines()

    assert [" ".join(line.split()[2:4]) for line in plan] == ["uv venv", "uv pip"]
    assert (ROOT / "finrl_sidecar" / "requirements.txt").exists()


def test_setup_fetches_both_models() -> None:
    plan = dry_run("setup").splitlines()

    assert [" ".join(line.split()[2:4]) for line in plan][-1] == (
        "finrl_sidecar/.venv/bin/python finrl_sidecar/fetch_kronos.py"
    )
    assert any("fetch_finbert.py" in line for line in plan)
