import itertools
import math
import unittest
from collections.abc import Callable

import explain
import numpy as np
from weights_env import SCALE, WeightsEnv, softmax

GROUPS = [[0, 1], [2], [3, 4, 5], [6], [7, 8]]
Function = Callable[[np.ndarray], np.ndarray]


def network(seed: int = 3) -> Function:
    rng = np.random.default_rng(seed)
    first, second = rng.normal(size=(9, 8)), rng.normal(size=(8, 3))
    return lambda rows: softmax(np.tanh(rows @ first) @ second)


def brute_force(
    function: Function, instance: np.ndarray, background: np.ndarray, groups: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    players = len(groups)

    def value(coalition: tuple[int, ...]) -> np.ndarray:
        kept = np.zeros(instance.size, dtype=bool)
        for player in coalition:
            kept[groups[player]] = True
        return function(np.where(kept, instance, background)).mean(axis=0)

    values = np.zeros((players, len(value(()))))
    for player in range(players):
        others = [other for other in range(players) if other != player]
        for size in range(players):
            share = math.factorial(size) * math.factorial(players - size - 1)
            for subset in itertools.combinations(others, size):
                values[player] += (
                    share / math.factorial(players) * (value((*subset, player)) - value(subset))
                )
    return values, value(())


class Layout(unittest.TestCase):
    def test_an_indicator_is_one_group_across_the_assets_and_the_held_weights_are_one(self) -> None:
        names, groups = explain.layout(["ret_1:A", "ret_1:B", "macd:A", "macd:B"], 2)
        self.assertEqual(names, ["ret_1", "macd", "held"])
        self.assertEqual(groups, [[0, 1], [2, 3], [4, 5, 6]])


class Values(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.instance = rng.normal(size=9)
        self.background = rng.normal(size=(12, 9))

    def test_they_equal_a_brute_force_enumeration_of_every_coalition(self) -> None:
        function = network()
        found, base = explain.grouped_shap(function, self.instance, self.background, GROUPS)
        expected, expected_base = brute_force(function, self.instance, self.background, GROUPS)
        np.testing.assert_allclose(found, expected, atol=1e-9)
        np.testing.assert_allclose(base, expected_base, atol=1e-9)

    def test_for_a_linear_function_each_group_gets_its_weights_times_the_distance_from_the_mean(
        self,
    ) -> None:
        slopes = np.random.default_rng(8).normal(size=(9, 3))
        found, base = explain.grouped_shap(
            lambda rows: rows @ slopes, self.instance, self.background, GROUPS
        )
        gap = (self.instance - self.background.mean(axis=0))[:, None] * slopes
        np.testing.assert_allclose(found, [gap[members].sum(axis=0) for members in GROUPS])
        np.testing.assert_allclose(base, self.background.mean(axis=0) @ slopes)

    def test_the_values_and_the_base_add_up_to_what_the_function_gives(self) -> None:
        function = network()
        found, base = explain.grouped_shap(function, self.instance, self.background, GROUPS)
        np.testing.assert_allclose(found.sum(axis=0) + base, function(self.instance[None])[0])

    def test_weights_that_sum_to_one_give_values_that_sum_to_nothing_across_them(self) -> None:
        found, _ = explain.grouped_shap(network(), self.instance, self.background, GROUPS)
        np.testing.assert_allclose(found.sum(axis=1), 0, atol=1e-12)

    def test_a_group_the_function_ignores_gets_nothing(self) -> None:
        def blind(rows: np.ndarray) -> np.ndarray:
            return network()(rows * np.array([1, 1, 1, 1, 1, 1, 0, 1, 1]))

        found, _ = explain.grouped_shap(blind, self.instance, self.background, GROUPS)
        np.testing.assert_allclose(found[3], 0, atol=1e-12)

    def test_the_same_inputs_give_the_same_numbers(self) -> None:
        first = explain.grouped_shap(network(), self.instance, self.background, GROUPS)
        second = explain.grouped_shap(network(), self.instance, self.background, GROUPS)
        np.testing.assert_array_equal(first[0], second[0])


class Residual(unittest.TestCase):
    def test_it_reports_the_worst_miss_of_any_weight(self) -> None:
        function = network()

        def drifting(rows: np.ndarray) -> np.ndarray:
            return function(rows) + (np.array([0.1, 0.0, 0.0]) if len(rows) == 1 else 0)

        rng = np.random.default_rng(5)
        found = explain.explain(drifting, rng.normal(size=(2, 9)), rng.normal(size=(12, 9)), GROUPS)
        self.assertEqual(len(found), 2)
        for item in found:
            self.assertAlmostEqual(item["residual"], 0.1, places=9)


class Samples(unittest.TestCase):
    def test_the_background_is_the_same_rows_each_time_and_never_repeats_one(self) -> None:
        pool = np.arange(500, dtype=float).reshape(100, 5)
        first, second = explain.background_of(pool, 20), explain.background_of(pool, 20)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first[:, 0])), 20)
        self.assertEqual(len(explain.background_of(pool[:7], 20)), 7)

    def test_an_even_sample_runs_from_the_first_row_to_the_last(self) -> None:
        pool = np.arange(100, dtype=float)[:, None]
        spread = explain.evenly(pool, 5)[:, 0]
        self.assertEqual(spread[0], 0)
        self.assertEqual(spread[-1], 99)
        self.assertEqual(len(spread), 5)


class Agent(unittest.TestCase):
    def test_a_trained_policy_is_explained_to_the_weights_it_chooses(self) -> None:
        from stable_baselines3 import PPO

        rng = np.random.default_rng(2)
        features, moves = rng.normal(size=(60, 4)), rng.normal(0, 0.01, size=(60, 2))
        env = WeightsEnv(features, moves, 0.0005, episode_days=20)
        agent = PPO("MlpPolicy", env, seed=1, device="cpu", n_steps=16, batch_size=16)
        agent.learn(total_timesteps=32)
        weights = explain.weights_of(agent)
        row = env.reset()[0]
        action, _ = agent.predict(row, deterministic=True)
        np.testing.assert_allclose(weights(row[None])[0], softmax(SCALE * action.astype(float)))
        names, groups = explain.layout(["a:X", "a:Y", "b:X", "b:Y"], 2)
        pool = np.array([env.reset()[0] for _ in range(10)])
        found = explain.explain(weights, pool[:3], pool, groups)
        self.assertEqual(names, ["a", "b", "held"])
        for each in found:
            self.assertAlmostEqual(sum(each["weights"]), 1.0, places=6)
            self.assertLess(each["residual"], 1e-5)


if __name__ == "__main__":
    unittest.main()
