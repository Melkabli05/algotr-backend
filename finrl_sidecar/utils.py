"""The one helper FinRL's tuner imports from a top-level `utils` it does not ship."""


def linear_schedule(initial_value: float):
    """A learning rate that falls linearly to zero (the rl-baselines3-zoo definition)."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule
