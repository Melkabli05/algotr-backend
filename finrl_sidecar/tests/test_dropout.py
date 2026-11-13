import unittest
from types import SimpleNamespace

import numpy as np
import torch
from runner import dropout_spread


class Tiny(torch.nn.Module):
    """Three-class logits through a dropout layer: unsure under dropout, steady without it."""

    def __init__(self) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.drop(values))


class DropoutSpreadTest(unittest.TestCase):
    def test_the_spread_is_measured_with_dropout_on_and_the_model_left_in_eval(self) -> None:
        torch.manual_seed(0)
        model = Tiny().eval()
        encoded = {"values": torch.tensor([[3.0, -3.0, 0.0], [0.0, 0.0, 0.0]])}
        spread = dropout_spread(model, encoded, 20, 0, 1)
        self.assertEqual(spread.shape, (2,))
        self.assertGreater(spread[0], 0.05)
        self.assertAlmostEqual(float(spread[1]), 0.0, places=6)
        self.assertFalse(model.training)
        self.assertTrue(np.all(np.isfinite(spread)))


if __name__ == "__main__":
    unittest.main()
