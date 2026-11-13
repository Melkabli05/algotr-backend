from collections.abc import Iterator

import pytest

from app.quant import history


@pytest.fixture(autouse=True)
def fresh_cache() -> Iterator[None]:
    history.cached_prices.cache_clear()
    yield
    history.cached_prices.cache_clear()
