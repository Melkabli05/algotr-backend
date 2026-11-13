import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core import sidecar
from app.features.systematic import agent_lab

COST = 0.0005
ASSETS = ["HLAL", "SPUS"]
pytestmark = pytest.mark.skipif(
    not sidecar.available(), reason="The FinRL sidecar is not set up: uv run poe finrl-setup"
)


def prices(days: int = 520) -> pd.DataFrame:
    """Two ETFs on random walks, in the long layout the sidecar reads."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2023-01-02", periods=days)
    frames = []
    for offset, tic in enumerate(ASSETS):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, days)))
        frames.append(
            pd.DataFrame(
                {
                    "date": index.strftime("%Y-%m-%d"),
                    "tic": tic,
                    "open": close * (1 + rng.normal(0, 0.002, days)),
                    "high": close * 1.004,
                    "low": close * 0.996,
                    "close": close,
                    "volume": 1_000_000 + 10_000 * offset,
                }
            )
        )
    return pd.concat(frames).reset_index(drop=True)


def ask(task: str, params: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
    outcome = sidecar.run_job(task, params, {"data": data}, timeout=600)
    assert outcome.status == "ok" and outcome.data is not None, outcome.error
    return outcome.data


def returns_of(data: pd.DataFrame) -> pd.DataFrame:
    close = data.pivot(index="date", columns="tic", values="close")
    return close.pct_change()


def recomputed(found: dict[str, Any], moves: pd.DataFrame, held: np.ndarray) -> list[float]:
    """The net return of each day from the weights alone: gross, minus the cost of rebalancing."""
    nets = []
    for day, chosen in zip(found["dates"], found.get("applied") or found["weights"], strict=True):
        weights = np.asarray(chosen)
        gross = float(weights[:-1] @ moves.loc[day, found["assets"]].to_numpy())
        nets.append(gross - COST * float(np.abs(weights - held).sum()))
        grown = weights * np.append(1 + moves.loc[day, found["assets"]].to_numpy(), 1.0)
        held = grown / grown.sum()
    return nets


@pytest.fixture(scope="module")
def walked() -> tuple[dict[str, Any], pd.DataFrame]:
    data = prices()
    days = sorted(data["date"].unique())
    params = {
        "algorithm": "ppo",
        "seed": 1,
        "cost": COST,
        "timesteps": 300,
        "lookback": 300,
        "retrain_dates": [days[380], days[400], days[420]],
    }
    return ask("ai_walk_forward", params, data), data


def test_the_agent_chooses_weights_that_are_long_only_and_fully_allocated(
    walked: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    found, _ = walked
    weights = np.array(found["weights"])
    assert weights.shape == (len(found["dates"]), 3) and found["assets"] == ASSETS
    assert (weights >= 0).all() and weights.sum(axis=1) == pytest.approx(1.0)


def test_each_months_trading_starts_on_the_retrain_day_and_runs_to_the_end_of_the_data(
    walked: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    found, data = walked
    days = sorted(data["date"].unique())
    assert found["dates"][0] == days[380] and found["dates"][-1] == days[-1]
    assert len(found["dates"]) == len(days) - 380


def test_every_return_is_the_weights_times_the_days_moves_less_the_cost_of_rebalancing(
    walked: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    found, data = walked
    expected = recomputed(found, returns_of(data), np.array([0.0, 0.0, 1.0]))
    assert found["returns"] == pytest.approx(expected, abs=1e-9)


def test_the_weights_held_at_the_end_are_the_drifted_ones(
    walked: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    found, data = walked
    moves = returns_of(data).loc[found["dates"][-1], ASSETS].to_numpy()
    grown = np.array(found["weights"][-1]) * np.append(1 + moves, 1.0)
    assert found["held"] == pytest.approx((grown / grown.sum()).tolist())
    assert found["in_sample"] is not None


def test_a_trained_agent_is_kept_and_then_followed_day_by_day(tmp_path: Path) -> None:
    data = prices()
    days = sorted(data["date"].unique())
    paths = {"model_path": str(tmp_path / "ppo.zip"), "scaler_path": str(tmp_path / "ppo.json")}
    base = {"algorithm": "ppo", "cost": COST}
    trained = ask("ai_train", {**base, "seed": 1, "timesteps": 300, "lookback": 300, **paths}, data)
    assert trained["days"] == 300 and trained["last"] == days[-1] and trained["assets"] == ASSETS
    scaler = json.loads(Path(paths["scaler_path"]).read_text())
    assert len(scaler["columns"]) == 12 * len(ASSETS) and set(scaler) == {"mean", "std", "columns"}

    start = {**base, **paths, "since": None, "target": None, "held": None}
    first = ask("ai_shadow", start, data)
    assert first["dates"] == [] and first["last"] == days[-1]
    assert first["target"] == pytest.approx(first["target"]) and sum(
        first["target"]
    ) == pytest.approx(1)

    since = days[-6]
    following = ask(
        "ai_shadow",
        {**base, **paths, "since": since, "target": [0.2, 0.3, 0.5], "held": [0.1, 0.1, 0.8]},
        data,
    )
    assert following["dates"] == days[-5:] and following["last"] == days[-1]
    assert following["applied"][0] == pytest.approx([0.2, 0.3, 0.5])
    expected = recomputed(following, returns_of(data), np.array([0.1, 0.1, 0.8]))
    assert following["returns"] == pytest.approx(expected, abs=1e-9)
    assert sum(following["target"]) == pytest.approx(1) and len(following["held"]) == 3
    sure = following["confidence"]
    assert set(sure) == set(days[-5:])
    assert all(
        len(found) == 3 and all(0 <= share <= 1 for share in found) for found in sure.values()
    )


def test_each_decision_is_explained_by_the_indicators_it_saw_and_the_weights_held(
    tmp_path: Path,
) -> None:
    data = prices()
    days = sorted(data["date"].unique())
    paths = {"model_path": str(tmp_path / "ppo.zip"), "scaler_path": str(tmp_path / "ppo.json")}
    base = {"algorithm": "ppo", "cost": COST}
    earlier = data[data["date"] <= days[-31]]
    params = {**base, "seed": 1, "timesteps": 300, "lookback": 300, **paths}
    trained = ask("ai_train", params, earlier)
    following = ask(
        "ai_shadow",
        {**base, **paths, "since": days[-6], "target": [0.2, 0.3, 0.5], "held": [0.1, 0.1, 0.8]},
        data,
    )
    decisions = following["decisions"]
    assert [item["date"] for item in decisions] == days[-5:]
    assert all(len(item["observation"]) == 12 * len(ASSETS) + 3 for item in decisions)
    assert decisions[-1]["observation"][-3:] == pytest.approx(following["held"])

    window = {"first": trained["first"], "last": trained["last"]}
    observations = [item["observation"] for item in decisions]
    found = ask(
        "ai_explain",
        {**base, **paths, **window, "observations": observations, "sample": 4},
        data,
    )
    assert found["groups"][-1] == "held" and len(found["groups"]) == 13
    assert found["assets"] == [*ASSETS, "cash"] and found["sample"] == 4
    assert len(found["columns"]) == 12 * len(ASSETS) + 3
    assert found["columns"][:2] == ["ret_1:HLAL", "ret_1:SPUS"]
    assert found["columns"][-3:] == ["held:HLAL", "held:SPUS", "held:cash"]
    assert found["background"] == {**window, "days": trained["days"] - 1, "rows": 64, "seed": 0}
    for item in found["decisions"]:
        values = np.array(item["values"])
        assert values.shape == (13, 3) and item["residual"] < 1e-5
        assert values.sum(axis=0) + item["base"] == pytest.approx(item["weights"], abs=1e-5)
        assert np.abs(values.sum(axis=1)).max() < 1e-9
    assert found["decisions"][-1]["weights"] == pytest.approx(following["target"], abs=1e-5)
    assert np.array(found["overall"]).shape == (13, 3) and (np.array(found["overall"]) >= 0).all()

    alone = ask("ai_explain", {**base, **paths, **window, "observations": [], "sample": 0}, data)
    assert alone["decisions"] == [] and alone["overall"] is None and alone["sample"] == 0


def test_the_sidecars_own_unit_tests_pass() -> None:
    found = ask("selftest", {}, pd.DataFrame())
    assert found["ran"] >= 10 and found["problems"] == []


def test_the_agent_is_deterministic_for_a_seed(walked: tuple[dict[str, Any], pd.DataFrame]) -> None:
    found, data = walked
    days = sorted(data["date"].unique())
    params = {
        "algorithm": "ppo",
        "seed": 1,
        "cost": COST,
        "timesteps": 300,
        "lookback": 300,
        "retrain_dates": [days[380], days[400], days[420]],
    }
    again = ask("ai_walk_forward", params, data)
    assert again["returns"] == pytest.approx(found["returns"])


KRONOS = {
    "model_dir": str(agent_lab.KRONOS_DIR.resolve()),
    "since": "2024-12-02",
    "seen_days": 6,
    "context": 120,
    "samples": 1,
    "seed": 1,
    "cost": COST,
}


@pytest.mark.skipif(
    not (agent_lab.KRONOS_DIR / "revision.json").exists(),
    reason="Kronos is not set up: uv run poe kronos-setup",
)
def test_kronos_trades_the_days_after_the_date_it_was_given_and_repeats_with_its_seed() -> None:
    data = prices()
    first = ask("kronos_walk_forward", KRONOS, data)
    again = ask("kronos_walk_forward", KRONOS, data)
    days = sorted(data["date"].unique())
    assert first["dates"][0] == "2024-12-02" and first["dates"][-1] == days[-1]
    assert len(first["dates"]) == len(first["returns"]) and first["in_sample"] is not None
    assert first["assets"] == ["HLAL", "SPUS"]
    assert first["returns"] == again["returns"]
