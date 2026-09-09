"""The client bundle is served from an attacker-controlled path, so it is fenced."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import app as app_module

TRAVERSALS = [
    "/../../../../.env",
    "/../../../../../../etc/passwd",
    "/..%2f..%2f..%2f..%2f.env",
    "/%2e%2e%2f%2e%2e%2f.env",
    "/assets/../../../../.env",
    "/./../../../../.env",
]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A throwaway bundle directory with a secret sitting just outside it."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>COMPASS</title>")
    (root / "assets" / "app.js").write_text("console.log('app')")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-secret-value")
    monkeypatch.setattr(app_module, "WEB_CLIENT_DIR", root)
    return root


@pytest.mark.parametrize("path", TRAVERSALS)
def test_a_traversal_never_escapes_the_bundle(bundle, path):
    client = TestClient(app_module.create_app())
    response = client.get(path)
    assert response.status_code == 200
    # Falling back to the app shell is correct; leaking the secret is not.
    assert "sk-or-v1-secret-value" not in response.text
    assert "OPENROUTER_API_KEY" not in response.text


def test_real_bundle_files_are_still_served(bundle):
    client = TestClient(app_module.create_app())
    assert client.get("/assets/app.js").text == "console.log('app')"


def test_client_routes_fall_through_to_the_app_shell(bundle):
    client = TestClient(app_module.create_app())
    for route in ("/", "/runs", "/runs/run_123", "/ontology"):
        assert "COMPASS" in client.get(route).text


def test_api_routes_are_not_shadowed_by_the_catch_all(bundle):
    client = TestClient(app_module.create_app())
    assert client.get("/api/health").json()["status"] == "ok"


def test_a_missing_bundle_explains_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_CLIENT_DIR", tmp_path / "absent")
    client = TestClient(app_module.create_app())
    response = client.get("/")
    assert response.status_code == 503
    assert "npm" in response.json()["fix"]
    # The API must stay usable while the client is unbuilt.
    assert client.get("/api/health").status_code == 200
