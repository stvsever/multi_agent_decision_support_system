"""Contract for the HuggingFace lookup used by the local inference backend."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import hf_catalog
from src.full_stack.backend.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def _fake_get_json(mapping):
    # Longest fragment first: "config.json" is a substring of
    # "tokenizer_config.json", so a naive scan would match the wrong file.
    ordered = sorted(mapping.items(), key=lambda item: -len(item[0]))

    def _get(url, timeout=15.0):
        for fragment, payload in ordered:
            if fragment in url:
                return payload
        raise RuntimeError(f"unexpected url: {url}")

    return _get


def test_search_returns_normalised_rows(client, monkeypatch):
    monkeypatch.setattr(
        hf_catalog,
        "_get_json",
        _fake_get_json(
            {
                "api/models?": [
                    {"modelId": "Qwen/Qwen3-14B-AWQ", "downloads": 5000, "likes": 40, "pipeline_tag": "text-generation", "tags": ["awq"]},
                    {"id": "no-model-id-key", "downloads": 1},
                ]
            }
        ),
    )
    body = client.get("/api/hf/models", params={"q": "qwen"}).json()
    assert [row["id"] for row in body["models"]] == ["Qwen/Qwen3-14B-AWQ", "no-model-id-key"]
    assert body["models"][0]["is_embedding"] is False


def test_embedding_task_filters_to_feature_extraction(client, monkeypatch):
    seen: list[str] = []

    def _get(url, timeout=15.0):
        seen.append(url)
        return [{"modelId": "BAAI/bge-large-en", "pipeline_tag": "feature-extraction"}]

    monkeypatch.setattr(hf_catalog, "_get_json", _get)
    body = client.get("/api/hf/models", params={"task": "embedding"}).json()
    assert "pipeline_tag=feature-extraction" in seen[0]
    assert body["models"][0]["is_embedding"] is True


def test_tokenizer_length_wins_over_the_architectural_maximum(client, monkeypatch):
    """A tokenizer's model_max_length reflects what the model actually serves."""
    monkeypatch.setattr(
        hf_catalog,
        "_get_json",
        _fake_get_json(
            {
                "api/models/": {"pipeline_tag": "text-generation", "downloads": 1},
                "config.json": {"max_position_embeddings": 131072, "model_type": "qwen3"},
                "tokenizer_config.json": {"model_max_length": 32768},
            }
        ),
    )
    body = client.get("/api/hf/model/Qwen/Qwen3-14B").json()
    assert body["context_length"] == 32768
    assert body["architectural_context_length"] == 131072
    assert body["model_type"] == "qwen3"


def test_missing_config_leaves_the_context_window_unknown(client, monkeypatch):
    def _get(url, timeout=15.0):
        if "api/models/" in url:
            return {"pipeline_tag": "text-generation"}
        raise RuntimeError("404")

    monkeypatch.setattr(hf_catalog, "_get_json", _get)
    assert client.get("/api/hf/model/some/model").json()["context_length"] is None


def test_sentinel_context_values_are_rejected(client, monkeypatch):
    """Some repositories publish 1e30 to mean "unbounded"; that is not a length."""
    monkeypatch.setattr(
        hf_catalog,
        "_get_json",
        _fake_get_json(
            {
                "api/models/": {"pipeline_tag": "text-generation"},
                "config.json": {"max_position_embeddings": 4096},
                "tokenizer_config.json": {"model_max_length": 1000000000000000019884624838656},
            }
        ),
    )
    assert client.get("/api/hf/model/some/model").json()["context_length"] == 4096


def test_search_reports_a_reachability_failure_without_raising(client, monkeypatch):
    def _boom(url, timeout=15.0):
        raise RuntimeError("offline")

    monkeypatch.setattr(hf_catalog, "_get_json", _boom)
    body = client.get("/api/hf/models").json()
    assert body["models"] == []
    assert "HuggingFace" in body["error"]
