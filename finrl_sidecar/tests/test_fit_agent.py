import unittest

import numpy as np
from runner import fit_agent


class FitAgent(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.features = rng.normal(size=(300, 4))
        self.returns = rng.normal(0.0004, 0.01, size=(300, 2))

    def params(self, **extra: float) -> dict:
        return {"cost": 0.0005, "timesteps": 600, **extra}

    def test_sac_updates_once_every_train_freq_steps(self) -> None:
        agent = fit_agent("sac", self.features, self.returns, self.params(train_freq=4), 1)
        self.assertEqual(agent.train_freq.frequency, 4)

    def test_without_the_setting_sac_keeps_the_library_default(self) -> None:
        agent = fit_agent("sac", self.features, self.returns, self.params(), 1)
        self.assertEqual(agent.train_freq.frequency, 1)

    def test_the_other_agents_ignore_it(self) -> None:
        agent = fit_agent("ppo", self.features, self.returns, self.params(train_freq=4), 1)
        self.assertEqual(agent.n_steps, 512)


if __name__ == "__main__":
    unittest.main()
