from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(("period", "timeframe"), [("1D", "5Min"), ("1W", "1H"), ("1M", "1D")])
def test_history_picks_timeframe_and_drops_empty_equity(
    client: TestClient, trading: MagicMock, period: str, timeframe: str
) -> None:
    trading.get_portfolio_history.return_value = {
        "timestamp": [1, 2, 3],
        "equity": [None, 0.0, 100000.0],
    }
    points = client.get("/portfolio/history", params={"period": period}).json()
    request = trading.get_portfolio_history.call_args.args[0]
    assert (request.period, request.timeframe) == (period, timeframe)
    assert points == [{"time": "1970-01-01T00:00:03Z", "equity": 100000.0}]


def test_period_is_validated(client: TestClient) -> None:
    assert client.get("/portfolio/history", params={"period": "2Y"}).status_code == 422
