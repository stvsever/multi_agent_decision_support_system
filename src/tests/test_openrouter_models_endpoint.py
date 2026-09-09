"""Contract for the model catalog endpoint and its pricing normalisation."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import catalog as catalog_module
from src.full_stack.backend.api.app import create_app


@pytest.fixture(autouse=True)
def clear_catalog_cache(tmp_path, monkeypatch):
    """Every test starts from an empty cache in a throwaway directory."""
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path))
    catalog_module._memory.clear()
    yield
    catalog_module._memory.clear()


@pytest.fixture
def client():
    return TestClient(create_app())


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload, recorder=None):
        self._payload = payload
        self._recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        if self._recorder is not None:
            self._recorder.append(url)
        return _FakeResponse(self._payload)


def _install(monkeypatch, payload, recorder=None):
    monkeypatch.setattr(
        catalog_module.httpx, "Client", lambda **kwargs: _FakeClient(payload, recorder)
    )


SAMPLE = {
    "data": [
        {
            "id": "deepseek/deepseek-v4-flash-0731",
            "name": "DeepSeek V4 Flash",
            "context_length": 1310720,
            "pricing": {"prompt": "0.000000065", "completion": "0.00000018"},
            "architecture": {"modality": "text->text"},
            "supported_parameters": ["reasoning", "structured_outputs", "tools"],
            "top_provider": {"max_completion_tokens": 943718},
        },
        {
            "id": "openai/text-embedding-3-large",
            "name": "Text Embedding 3 Large",
            "context_length": 8191,
            "pricing": {"prompt": "0.00000013", "completion": "0"},
            "architecture": {"modality": "text->vector"},
            "supported_parameters": [],
        },
    ]
}


def test_pricing_is_normalised_to_usd_per_million_tokens(client, monkeypatch):
    _install(monkeypatch, SAMPLE)
    body = client.get("/api/catalog/models").json()
    rows = {row["id"]: row for row in body["models"]}

    default = rows["deepseek/deepseek-v4-flash-0731"]
    assert default["prompt_usd_per_mtok"] == pytest.approx(0.065)
    assert default["completion_usd_per_mtok"] == pytest.approx(0.18)
    assert default["context_length"] == 1310720
    assert default["provider"] == "deepseek"
    assert default["supports_structured_output"] is True
    assert default["is_free"] is False


def test_embedding_models_are_flagged(client, monkeypatch):
    _install(monkeypatch, SAMPLE)
    body = client.get("/api/catalog/models", params={"embedding": True}).json()
    assert [row["id"] for row in body["models"]] == ["openai/text-embedding-3-large"]
    assert body["models"][0]["is_embedding"] is True


def test_chat_filter_excludes_embedding_models(client, monkeypatch):
    _install(monkeypatch, SAMPLE)
    body = client.get("/api/catalog/models", params={"embedding": False}).json()
    assert [row["id"] for row in body["models"]] == ["deepseek/deepseek-v4-flash-0731"]


def test_search_and_provider_filters(client, monkeypatch):
    _install(monkeypatch, SAMPLE)
    assert client.get("/api/catalog/models", params={"search": "deepseek"}).json()["total"] == 1
    assert client.get("/api/catalog/models", params={"provider": "openai"}).json()["total"] == 1
    assert client.get("/api/catalog/models", params={"min_context": 100000}).json()["total"] == 1


def test_catalog_hits_the_configured_models_endpoint(client, monkeypatch):
    calls: list[str] = []
    _install(monkeypatch, SAMPLE, calls)
    client.get("/api/catalog/models")
    assert calls and calls[0].endswith("/models")


def test_a_network_failure_degrades_to_a_clear_error(client, monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(catalog_module.httpx, "Client", _boom)
    response = client.get("/api/catalog/models")
    assert response.status_code == 502
    assert "catalog" in response.json()["detail"].lower()


def test_unknown_model_detail_is_a_404(client, monkeypatch):
    _install(monkeypatch, SAMPLE)
    assert client.get("/api/catalog/models/does/not-exist").status_code == 404
