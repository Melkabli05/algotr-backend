from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from app.core.settings import get_settings


@lru_cache
def engine() -> Engine:
    return sa.create_engine(get_settings().database_url, pool_pre_ping=True, future=True)


@contextmanager
def transaction() -> Iterator[Connection]:
    """One connection inside one transaction, committed on success and rolled back on error."""
    with engine().begin() as connection:
        yield connection
