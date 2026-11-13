import unittest

import numpy as np
from weights_env import agreement


class Agreement(unittest.TestCase):
    def test_a_move_is_as_sure_as_the_share_of_samples_that_make_it(self) -> None:
        held = np.array([0.5, 0.5])
        chosen = np.array([0.7, 0.3])
        sampled = np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.55, 0.45]])
        self.assertEqual(agreement(sampled, chosen, held).tolist(), [0.75, 0.75])

    def test_a_hold_is_matched_by_samples_that_also_stay_put(self) -> None:
        held = np.array([0.5, 0.5])
        chosen = np.array([0.51, 0.49])
        sampled = np.array([[0.505, 0.495], [0.7, 0.3]])
        self.assertEqual(agreement(sampled, chosen, held).tolist(), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
