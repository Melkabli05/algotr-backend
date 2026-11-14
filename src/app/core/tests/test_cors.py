from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.cors import allow_origins

ORIGIN = "https://terminal.example.com"


def app_with(origins: list[str]) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    allow_origins(app, origins)
    return app


def preflight(app: FastAPI, origin: str) -> dict[str, str]:
    headers = {"Origin": origin, "Access-Control-Request-Method": "GET"}
    return dict(TestClient(app).options("/health", headers=headers).headers)


def test_a_listed_origin_may_call_the_api() -> None:
    headers = preflight(app_with([ORIGIN]), ORIGIN)

    assert headers["access-control-allow-origin"] == ORIGIN


def test_an_unlisted_origin_is_not_allowed() -> None:
    assert "access-control-allow-origin" not in preflight(app_with([ORIGIN]), "https://evil.test")


def test_no_origin_is_allowed_when_none_are_listed() -> None:
    assert "access-control-allow-origin" not in preflight(app_with([]), ORIGIN)
