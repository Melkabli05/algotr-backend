from unittest.mock import MagicMock

import pytest
import requests
from alpaca.common.exceptions import APIError
from fastapi.testclient import TestClient


def broker_error(status: int, body: str) -> APIError:
    response = requests.Response()
    response.status_code = status
    return APIError(body, requests.HTTPError(response=response))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_broker_rejection_is_a_422_with_the_broker_message(
    client: TestClient, trading: MagicMock, status: int
) -> None:
    body = '{"code": 40310000, "message": "insufficient buying power"}'
    trading.get_all_positions.side_effect = broker_error(status, body)
    result = client.get("/positions")
    assert result.status_code == 422
    assert result.json() == {"detail": "insufficient buying power"}


def test_unparseable_broker_error_falls_back_to_status(
    client: TestClient, trading: MagicMock
) -> None:
    trading.get_all_positions.side_effect = broker_error(404, "404 page not found")
    result = client.get("/positions")
    assert result.status_code == 422
    assert "404" in result.json()["detail"]


def test_broker_server_error_becomes_502(client: TestClient, trading: MagicMock) -> None:
    trading.get_all_positions.side_effect = broker_error(500, '{"message": "boom"}')
    result = client.get("/positions")
    assert result.status_code == 502
    assert result.json() == {"detail": "boom"}


def test_broker_unreachable_becomes_502(client: TestClient, trading: MagicMock) -> None:
    trading.get_all_positions.side_effect = requests.ConnectionError("boom")
    result = client.get("/positions")
    assert result.status_code == 502
    assert "unreachable" in result.json()["detail"]
