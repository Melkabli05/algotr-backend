import unittest

import numpy as np
from weights_env import REWARD_SCALE, WeightsEnv


class Reward(unittest.TestCase):
    def test_the_days_log_return_is_scaled_so_the_entropy_bonus_does_not_drown_it(self) -> None:
        env = WeightsEnv(np.zeros((3, 2)), np.array([[0.0], [0.01], [0.0]]), cost=0.0)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.array([1.0, -1.0]))
        self.assertAlmostEqual(reward, REWARD_SCALE * np.log1p(info["net"]))
        self.assertTrue(0.5 < reward < 1.0)


if __name__ == "__main__":
    unittest.main()
