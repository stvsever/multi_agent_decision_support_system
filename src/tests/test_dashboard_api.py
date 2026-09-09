"""End to end contract for the dashboard service."""

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import catalog as catalog_module
from src.full_stack.backend.api import config_store
from src.full_stack.backend.api.app import create_app

REPO = Path(__file__).resolve().parent.parent.parent
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never read or write the developer's real ~/.compass during a test."""
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path))
    config_store._cached = None
    catalog_module._memory.clear()
    yield
    config_store._cached = None
    catalog_module._memory.clear()


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def offline_catalog(monkeypatch):
    monkeypatch.setattr(
        catalog_module,
        "pricing_index",
        lambda: {
            "deepseek/deepseek-v4-flash-0731": {
                "prompt_usd_per_mtok": 0.065,
                "completion_usd_per_mtok": 0.18,
            }
        },
    )
    from src.full_stack.backend.api import cost as cost_module

    monkeypatch.setattr(cost_module, "pricing_index", catalog_module.pricing_index)


def test_health_and_capabilities(client):
    assert client.get("/api/health").json()["status"] == "ok"
    caps = client.get("/api/capabilities").json()
    assert caps["default_model"] == "deepseek/deepseek-v4-flash-0731"
    assert len(caps["stages"]) == 7
    assert {t["name"] for t in caps["tools"]}
    assert {p["value"] for p in caps["prediction_types"]} == {
        "binary",
        "multiclass",
        "regression_univariate",
        "regression_multivariate",
        "hierarchical",
    }
    # Only the Executor lacks a model of its own.
    assert [a["role"] for a in caps["agents"] if not a["has_model"]] == ["executor"]


def test_settings_round_trip_and_patch_is_partial(client):
    original = client.get("/api/settings").json()["config"]
    assert original["models"]["reasoning_effort"] == "off"

    patched = client.patch("/api/settings", json={"engine": {"max_iterations": 7}}).json()["config"]
    assert patched["engine"]["max_iterations"] == 7
    # A patch of one section leaves the others untouched.
    assert patched["models"]["default_model"] == original["models"]["default_model"]
    assert patched["appearance"]["accent"] == original["appearance"]["accent"]

    assert client.post("/api/settings/reset").json()["config"]["engine"]["max_iterations"] == 3


def test_a_stored_key_is_never_returned_to_the_client(client):
    secret = "sk-or-v1-abcdef0123456789abcdef0123456789"
    body = client.put("/api/settings/credentials", json={"provider": "openrouter", "api_key": secret}).json()
    assert body["credential"]["configured"] is True
    assert body["credential"]["source"] == "stored"
    assert secret not in json.dumps(body)
    assert body["credential"]["masked"].endswith("6789")

    everything = client.get("/api/settings").text
    assert secret not in everything


def test_participants_are_discovered_and_validated(client):
    body = client.get("/api/datasets/participants").json()
    assert body["count"] >= 5
    subject = next(p for p in body["participants"] if p["id"] == "SUBJ_001_PSEUDO")
    assert subject["valid"] is True
    assert subject["input_tokens"] == 8680
    assert len(subject["files"]) == 4
    assert all(f["valid"] for f in subject["files"])


def test_an_incomplete_folder_reports_which_file_is_missing(client, tmp_path):
    folder = tmp_path / "half"
    folder.mkdir()
    (folder / "data_overview.json").write_text('{"participant_id": "half"}')
    body = client.get("/api/datasets/participant", params={"directory": str(folder)}).json()
    assert body["valid"] is False
    assert "multimodal_data.json" in body["missing"]
    assert "hierarchical_deviation_map.json" in body["missing"]


def test_malformed_json_is_reported_as_such_not_as_missing(client, tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    for name in ("data_overview.json", "multimodal_data.json", "hierarchical_deviation_map.json"):
        (folder / name).write_text("{not json")
    (folder / "non_numerical_data.txt").write_text("text")
    body = client.get("/api/datasets/participant", params={"directory": str(folder)}).json()
    overview = next(f for f in body["files"] if f["file"] == "data_overview.json")
    assert overview["present"] is True and overview["valid"] is False
    assert "Invalid JSON" in overview["issue"]


def test_ontology_endpoint_returns_a_navigable_tree(client):
    body = client.get("/api/datasets/ontology", params={"directory": str(SUBJ_001)}).json()
    assert body["summary"]["domain_count"] == 5
    assert body["extremes"]
    assert body["domains"][0]["children"]


def test_estimate_is_free_and_priced(client, offline_catalog):
    body = client.post(
        "/api/runs/estimate",
        json={
            "participant_dirs": [str(SUBJ_001)],
            "task": {"prediction_type": "binary", "target_label": "T", "control_label": "C"},
        },
    ).json()
    assert body["totals"]["count"] == 1
    assert body["totals"]["fully_priced"] is True
    assert body["totals"]["usd"] > 0
    assert body["totals"]["usd_low"] < body["totals"]["usd"] < body["totals"]["usd_high"]
    assert set(body["effective_models"]) == {
        "orchestrator",
        "integrator",
        "predictor",
        "critic",
        "communicator",
        "tool",
    }


def test_the_hard_stop_refuses_the_run_before_any_provider_call(client, offline_catalog):
    client.patch("/api/settings", json={"cost": {"block_above_usd": 0.000001}})
    response = client.post(
        "/api/runs",
        json={
            "participant_dir": str(SUBJ_001),
            "task": {"prediction_type": "binary", "target_label": "T", "control_label": "C"},
        },
    )
    assert response.status_code == 400
    assert "hard stop" in response.json()["detail"]


def test_an_invalid_task_is_rejected_with_the_engine_contract(client, offline_catalog):
    response = client.post(
        "/api/runs",
        json={
            "participant_dir": str(SUBJ_001),
            "task": {"prediction_type": "multiclass", "target_label": "T", "class_labels": ["a", "b"]},
        },
    )
    assert response.status_code in (400, 422)


def test_a_missing_participant_directory_is_a_404(client):
    response = client.post(
        "/api/runs",
        json={"participant_dir": "/nope/not/here", "task": {"prediction_type": "binary", "target_label": "T"}},
    )
    assert response.status_code == 404


def test_reports_bundle_reads_artifacts_from_disk(client):
    body = client.get("/api/reports", params={"participant_dir": str(SUBJ_001)}).json()
    assert body["participant_id"] == "SUBJ_001_PSEUDO"
    keys = {a["key"] for a in body["artifacts"]}
    assert {"performance_report", "report_markdown", "deep_phenotype"} <= keys


@pytest.fixture
def restore_prompt_file():
    """The prompt endpoints edit real files, so put them back whatever happens."""
    path = REPO / "src/full_stack/backend/agents/prompts/critic_prompt.txt"
    original = path.read_text()
    yield path
    path.write_text(original)


def test_prompts_can_be_listed_read_edited_and_restored(client, restore_prompt_file):
    rows = client.get("/api/prompts").json()["prompts"]
    assert any(r["name"] == "orchestrator_prompt.txt" for r in rows)

    before = client.get("/api/prompts/agent/critic_prompt.txt").json()
    assert before["content"]

    edited = client.put(
        "/api/prompts/agent/critic_prompt.txt", json={"content": before["content"] + "\n# marker\n"}
    ).json()
    assert edited["modified"] is True

    restored = client.post("/api/prompts/agent/critic_prompt.txt/restore").json()
    assert restored["content"] == before["content"]
    assert restored["modified"] is False


def test_an_empty_prompt_is_rejected(client, restore_prompt_file):
    assert client.put("/api/prompts/agent/critic_prompt.txt", json={"content": "  "}).status_code == 400


def test_unknown_prompt_is_a_404(client):
    assert client.get("/api/prompts/agent/nope.txt").status_code == 404


# --- error contract ----------------------------------------------------------


def test_an_unmatched_api_path_is_a_404_not_the_app_shell(client):
    """The SPA catch-all must not answer for the API namespace."""
    for path in ("/api/nope", "/api/settings/typo", "/api/batches/x/y"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert "text/html" not in response.headers.get("content-type", "")


@pytest.mark.parametrize(
    "patch,expected",
    [
        ({"appearance": {"theme": "neon"}}, "theme"),
        ({"engine": {"max_iterations": 999}}, "max_iterations"),
        ({"cost": {"warn_above_usd": -5}}, "warn_above_usd"),
        ({"local": {"gpu_memory_utilization": 4}}, "gpu_memory_utilization"),
    ],
)
def test_an_out_of_range_setting_is_a_422_with_the_offending_field(client, patch, expected):
    response = client.patch("/api/settings", json=patch)
    assert response.status_code == 422
    assert expected in response.json()["detail"]


def test_a_rejected_setting_leaves_the_stored_config_untouched(client):
    before = client.get("/api/settings").json()["config"]["appearance"]["theme"]
    client.patch("/api/settings", json={"appearance": {"theme": "neon"}})
    assert client.get("/api/settings").json()["config"]["appearance"]["theme"] == before


def test_an_invalid_override_on_estimate_is_a_422(client, offline_catalog):
    response = client.post(
        "/api/runs/estimate",
        json={"participant_dirs": [], "overrides": {"engine": {"max_iterations": 999}}},
    )
    assert response.status_code == 422
    assert "max_iterations" in response.json()["detail"]


def test_audit_rejects_an_invalid_task_the_same_way_a_run_does(client, offline_catalog):
    body = {
        "participant_dir": str(SUBJ_001),
        "task": {"prediction_type": "multiclass", "target_label": "Dx", "class_labels": ["A", "B"]},
    }
    audit = client.post("/api/runs/audit", json=body)
    run = client.post("/api/runs", json=body)
    assert audit.status_code == run.status_code == 400
    assert "class_labels" in audit.json()["detail"]


@pytest.mark.parametrize(
    "endpoint,params",
    [
        ("/api/datasets/participant", {"directory": "~nosuchuser12345/data"}),
        ("/api/datasets/ontology", {"directory": "~nosuchuser12345/data"}),
        ("/api/datasets/participant/file", {"directory": "~nosuchuser12345/data", "key": "data_overview"}),
        ("/api/reports", {"participant_dir": "~nosuchuser12345/data"}),
    ],
)
def test_an_unresolvable_home_path_is_a_400_not_a_crash(client, endpoint, params):
    """`~unknownuser` makes Path.expanduser raise rather than pass the text through."""
    response = client.get(endpoint, params=params)
    assert response.status_code == 400
    assert "usable path" in response.json()["detail"]


def test_a_batch_that_cannot_be_built_leaves_no_orphan_runs(client, offline_catalog):
    before = len(client.get("/api/runs").json()["runs"])
    response = client.post(
        "/api/batches",
        json={
            "participant_dirs": [str(SUBJ_001), "/definitely/not/here"],
            "task": {"prediction_type": "binary", "target_label": "T", "control_label": "C"},
        },
    )
    assert response.status_code == 404
    # The first participant was accepted before the second failed; it must be
    # rolled back rather than left queued against a batch that never existed.
    assert len(client.get("/api/runs").json()["runs"]) == before
