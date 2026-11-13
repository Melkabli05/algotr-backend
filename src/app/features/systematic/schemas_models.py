from datetime import date

from pydantic import BaseModel


class AgentModel(BaseModel):
    """One trained agent: when, on what, how, and how it did in sample."""

    algorithm: str
    trained: date
    first: date
    last: date
    days: int
    seed: int
    timesteps: int
    in_sample_sharpe: float
    version: int = 1
    explained: bool = False
