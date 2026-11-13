import unittest

import pandas as pd
from runner import validation_sharpe


class ValidationSharpe(unittest.TestCase):
    def test_the_chosen_model_is_the_best_validation_score_of_each_window(self) -> None:
        summary = pd.DataFrame(
            {
                "Iter": [1, 2],
                "Val Start": ["a", "b"],
                "Val End": ["c", "d"],
                "Model Used": ["ppo", "sac"],
                "A2C Sharpe": [0.1, 0.2],
                "PPO Sharpe": [0.9, 0.3],
                "DDPG Sharpe": [0.0, 0.0],
                "SAC Sharpe": [0.5, 0.7],
                "TD3 Sharpe": [0.2, 0.1],
            }
        )
        self.assertAlmostEqual(validation_sharpe(summary), 0.8)

    def test_no_scores_gives_none(self) -> None:
        empty = pd.DataFrame(columns=["a", "b", "c", "d", "e"])
        self.assertIsNone(validation_sharpe(empty))


if __name__ == "__main__":
    unittest.main()
