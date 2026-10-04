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
    assert not any(row["is_embedding"] for row in body["models"])
    # Decision models are not embeddings, so the role pickers still offer them.
    assert [row["id"] for row in body["models"] if row["kind"] == "llm"] == ["deepseek/deepseek-v4-flash-0731"]


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


# --- structured decision models ----------------------------------------------


def test_every_row_says_what_kind_of_model_it_is_and_which_roles_it_serves(client, monkeypatch):
    _install(monkeypatch, SAMPLE)
    rows = {row["id"]: row for row in client.get("/api/catalog/models").json()["models"]}

    llm = rows["deepseek/deepseek-v4-flash-0731"]
    assert llm["kind"] == "llm"
    assert set(llm["roles"]) == {"orchestrator", "integrator", "predictor", "critic", "communicator", "tool"}

    embedding = rows["openai/text-embedding-3-large"]
    assert embedding["kind"] == "embedding"
    assert embedding["roles"] == []


def test_registry_decision_models_are_listed_when_the_provider_omits_them(client, monkeypatch):
    _install(monkeypatch, SAMPLE)
    body = client.get("/api/catalog/models").json()
    rows = {row["id"]: row for row in body["models"]}

    jev = rows["typesafe/jev-1.13"]
    assert jev["kind"] == "decision"
    assert jev["roles"] == ["predictor"]
    assert jev["context_length"] == 32_000
    assert jev["prompt_usd_per_mtok"] == pytest.approx(0.042)
    # Input-only pricing: the output is free, never unpriced.
    assert jev["completion_usd_per_mtok"] == 0
    assert jev["is_free"] is False
    assert "typesafe" in body["providers"]


def test_a_provider_row_for_a_decision_model_is_not_duplicated_and_gets_its_limits(client, monkeypatch):
    payload = {
        "data": SAMPLE["data"]
        + [
            {
                "id": "typesafe/jev-1.13",
                "name": "TypeSafe: Jev 1.13",
                "context_length": 64000,
                "pricing": {"prompt": "0.000000042"},
                "architecture": {"modality": "text->text"},
            }
        ]
    }
    _install(monkeypatch, payload)
    rows = [row for row in client.get("/api/catalog/models").json()["models"] if row["id"] == "typesafe/jev-1.13"]
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "TypeSafe: Jev 1.13"
    assert row["kind"] == "decision"
    # The state limit bounds the model, not the request window the provider quotes.
    assert row["context_length"] == 32_000
    assert row["completion_usd_per_mtok"] == 0


def test_a_cache_written_before_decision_models_is_classified_on_read(client, monkeypatch):
    """A row cached by an older build has no kind; readers must not see that."""
    import time

    catalog_module._memory["catalog"] = {
        "fetched_at": time.time(),
        "models": [{"id": "deepseek/deepseek-v4-flash-0731", "provider": "deepseek", "is_embedding": False}],
    }
    rows = {row["id"]: row for row in client.get("/api/catalog/models").json()["models"]}
    assert rows["deepseek/deepseek-v4-flash-0731"]["kind"] == "llm"
    assert rows["typesafe/jev-1.13"]["kind"] == "decision"


def test_a_decision_model_detail_is_known_without_the_network(client, monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(catalog_module.httpx, "Client", _boom)
    row = client.get("/api/catalog/models/typesafe/jev-1.13").json()
    assert row["kind"] == "decision"
    assert row["context_length"] == 32_000


def test_capabilities_list_decision_models_and_orchestration_modes(client):
    caps = client.get("/api/capabilities").json()
    assert caps["orchestration_modes"] == ["auto", "always", "never"]
    models = {spec["model_id"]: spec for spec in caps["decision_models"]}
    jev = models["typesafe/jev-1.13"]
    assert jev["state_context_tokens"] == 32_000
    assert jev["request_context_tokens"] == 64_000
    assert jev["roles"] == ["predictor"]
    assert jev["output_price_per_million"] == 0
