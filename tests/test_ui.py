from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import ui


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<div id=root></div>")
    (tmp_path / "assets" / "app.js").write_text("run()")
    (tmp_path.parent / "secret.txt").write_text("no")
    monkeypatch.setattr(ui, "BUILD", tmp_path)
    return tmp_path


def test_files_are_served_and_every_route_gets_the_page(client: TestClient, build: Path) -> None:
    assert client.get("/ui/assets/app.js").text == "run()"
    assert client.get("/ui/positions").text == "<div id=root></div>"
    assert client.get("/ui", follow_redirects=False).headers["location"] == "/ui/"


def test_nothing_outside_the_build_is_served(client: TestClient, build: Path) -> None:
    assert client.get("/ui/..%2Fsecret.txt").text == "<div id=root></div>"


def test_without_a_build_the_page_says_how_to_make_one(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ui, "BUILD", tmp_path / "missing")
    assert client.get("/ui/").status_code == 404
