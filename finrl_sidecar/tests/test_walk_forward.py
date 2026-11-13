import unittest
from unittest import mock

import numpy as np
import pandas as pd
import runner
import weights_env


class PurgedWalkForward(unittest.TestCase):
    def run_with(self, purge: int, embargo: int) -> list[tuple[int, int]]:
        days = pd.bdate_range("2024-01-01", periods=400)
        features = pd.DataFrame({"f": np.arange(400.0)}, index=days)
        returns = pd.DataFrame({"A": np.zeros(400)}, index=days)

        def fit(algorithm: str, train: np.ndarray, moves: np.ndarray, params: dict, seed: int):
            return object()

        def roll_out(agent, shown, moves, cost, held):
            return [0.0] * (len(moves) - 1), [held.tolist()] * (len(moves) - 1), held

        windows: list[tuple[int, int]] = []
        real_standardized = runner.standardized

        def standardized(frame: pd.DataFrame, mean: pd.Series, std: pd.Series) -> np.ndarray:
            if len(frame) == 100:
                windows.append((int(frame["f"].iloc[0]), int(frame["f"].iloc[-1])))
            return real_standardized(frame, mean, std)

        job = {
            "params": {
                "algorithm": "ppo",
                "seed": 1,
                "cost": 0.0,
                "lookback": 100,
                "purge": purge,
                "embargo": embargo,
                "retrain_dates": [days[200].strftime("%Y-%m-%d"), days[300].strftime("%Y-%m-%d")],
            }
        }
        with (
            mock.patch.object(runner, "prepare_agents", return_value=(features, returns)),
            mock.patch.object(runner, "fit_agent", side_effect=fit),
            mock.patch.object(runner, "standardized", side_effect=standardized),
            mock.patch.object(weights_env, "roll_out", side_effect=roll_out),
        ):
            result = runner.task_ai_walk_forward(job)
        self.assertEqual(result["dates"][0], days[200].strftime("%Y-%m-%d"))
        return windows

    def test_training_stops_the_purge_and_embargo_before_the_month_it_trades(self) -> None:
        self.assertEqual(self.run_with(1, 5), [(94, 193), (194, 293)])

    def test_without_them_training_runs_up_to_the_day_before(self) -> None:
        self.assertEqual(self.run_with(0, 0), [(100, 199), (200, 299)])


if __name__ == "__main__":
    unittest.main()
