"""Run supervision: worker isolation, event fan-out, and batch scheduling."""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import config_store
from src.full_stack.backend.api import run_manager as rm
from src.full_stack.backend.api.engine_bridge import (
    build_task_spec,
    describe_task,
    merge_overrides,
    normalise_instructions,
)
from src.full_stack.backend.api.schemas import (
    DashboardConfig,
    RunOverrides,
    RunRequest,
    TaskNodeInput,
    TaskSpecInput,
)

REPO = Path(__file__).resolve().parent.parent.parent
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path))
    config_store._cached = None
    yield
    config_store._cached = None


# --- task translation --------------------------------------------------------


def test_flat_task_families_translate_to_the_engine_spec():
    spec = build_task_spec(
        TaskSpecInput(prediction_type="binary", target_label="DEPRESSION", control_label="HEALTHY")
    )
    assert spec.root.mode.value == "binary_classification"
    assert spec.root.class_labels == ["DEPRESSION", "HEALTHY"]
    assert spec.legacy_target_control() == ("DEPRESSION", "HEALTHY")

    multi = build_task_spec(
        TaskSpecInput(prediction_type="multiclass", target_label="subtype", class_labels=["a", "b", "c"])
    )
    assert multi.root.mode.value == "multiclass_classification"

    uni = build_task_spec(
        TaskSpecInput(prediction_type="regression_univariate", target_label="iq", regression_outputs=["total_iq"])
    )
    assert uni.root.regression_outputs == ["total_iq"]


def test_a_hierarchical_task_keeps_mixed_child_modes():
    task = TaskSpecInput(
        prediction_type="hierarchical",
        root=TaskNodeInput(
            node_id="root",
            display_name="Profile",
            mode="multivariate_regression",
            regression_outputs=["p1", "p2"],
            children=[
                TaskNodeInput(
                    node_id="dx",
                    display_name="Diagnosis",
                    mode="binary_classification",
                    class_labels=["CASE", "CONTROL"],
                )
            ],
        ),
    )
    summary = describe_task(build_task_spec(task))
    assert summary["is_hierarchical"] is True
    assert summary["node_count"] == 2
    assert [n["mode"] for n in summary["nodes"]] == ["multivariate_regression", "binary_classification"]


def test_an_invalid_task_is_rejected_by_the_engine_contract():
    with pytest.raises(Exception):
        build_task_spec(TaskSpecInput(prediction_type="multiclass", target_label="t", class_labels=["a", "b"]))
    with pytest.raises(ValueError):
        build_task_spec(TaskSpecInput(prediction_type="hierarchical", root=None))


# --- configuration merge -----------------------------------------------------


def test_run_overrides_are_merged_without_mutating_the_saved_config():
    base = DashboardConfig()
    merged = merge_overrides(
        base,
        RunOverrides(engine={"max_iterations": 1}, models={"reasoning_effort": "high"}),
    )
    assert merged.engine.max_iterations == 1
    assert merged.models.reasoning_effort == "high"
    # Untouched sections survive, and the original is unchanged.
    assert merged.models.default_model == base.models.default_model
    assert base.engine.max_iterations == 3


def test_instruction_slots_are_always_complete():
    filled = normalise_instructions({"global": " study specific ", "predictor": "x"})
    assert filled["global"] == "study specific"
    assert filled["critic"] == ""
    assert len(filled) == 8


def test_worker_environment_carries_the_concurrency_and_key(monkeypatch):
    config_store.set_credential("openrouter", "sk-or-test-key")
    config = DashboardConfig()
    config.engine.executor_max_workers = 5
    env = config_store.worker_environment(config)
    assert env["COMPASS_EXECUTOR_MAX_WORKERS"] == "5"
    assert env["OPENROUTER_API_KEY"] == "sk-or-test-key"
    assert env["PYTHONUNBUFFERED"] == "1"


# --- run records -------------------------------------------------------------


def _record(monkeypatch, **overrides) -> rm.RunRecord:
    """A record without a worker process, so the reducer can be driven directly."""
    monkeypatch.setattr(rm, "estimate_run", lambda **kwargs: {"usd": 0.01, "total_tokens": 1000, "lines": []})
    manager = rm.RunManager()
    request = RunRequest(
        participant_dir=str(SUBJ_001),
        task=TaskSpecInput(prediction_type="binary", target_label="T", control_label="C"),
        **overrides,
    )
    return manager.create_run(request, autostart=False)


def test_a_record_starts_queued_with_a_full_stage_list(monkeypatch):
    record = _record(monkeypatch)
    assert record.status == "queued"
    assert record.state["current_stage"] == -1
    assert len(record.state["stages"]) == 7
    assert record.summary()["participant_id"] == "SUBJ_001_PSEUDO"
    assert (record.dir / "meta.json").exists()


def test_worker_messages_drive_the_record_and_reach_subscribers(monkeypatch):
    record = _record(monkeypatch)
    seen: list = []
    record.subscribe(seen.append)

    record._handle(
        {
            "t": "event",
            "event": {"id": 1, "time": "00:00:00", "type": "PLAN", "data": {"steps": 2, "plan": {
                "plan_id": "P", "steps": [
                    {"step_id": 1, "tool_name": "UnimodalCompressor", "description": "a", "depends_on": []},
                    {"step_id": 2, "tool_name": "HypothesisGenerator", "description": "b", "depends_on": [1]},
                ]}}},
            "state": {"current_stage": 1, "steps": [], "stages": list(rm.STAGE_NAMES)},
        }
    )
    assert record.graph is not None
    assert record.graph["meta"]["step_count"] == 2
    assert record.state["current_stage"] == 1

    first = next(m for m in seen if m["type"] == "event")
    assert first["graph"]["meta"]["depth"] == 2


def test_usage_messages_become_a_priced_running_cost(monkeypatch):
    from src.full_stack.backend.api import run_manager

    monkeypatch.setattr(
        run_manager,
        "actual_cost_from_usage",
        lambda usage, cached_only=False: {"usd": 0.42, "total_tokens": 100, "lines": [], "fully_priced": True},
    )
    record = _record(monkeypatch)
    record._handle({"t": "usage", "by_model": {"m": {"prompt": 60, "completion": 40}}, "totals": {"calls": 1}})
    assert record.cost["usd"] == 0.42
    assert record.usage["by_model"]["m"]["prompt"] == 60


def test_a_worker_error_is_kept_verbatim(monkeypatch):
    record = _record(monkeypatch)
    record._handle({"t": "error", "error": "RuntimeError: connectivity check failed", "traceback": "tb"})
    assert "connectivity check failed" in record.error
    assert record.traceback == "tb"


def test_the_result_drops_the_untransportable_internal_context(monkeypatch):
    record = _record(monkeypatch)
    record._handle({"t": "result", "result": {"prediction": "CASE", "verdict": "SATISFACTORY"}})
    assert record.result["prediction"] == "CASE"
    assert "internal_context" not in record.result


def test_detail_filters_events_by_the_last_seen_id(monkeypatch):
    record = _record(monkeypatch)
    for index in (1, 2, 3):
        record._handle(
            {"t": "event", "event": {"id": index, "time": "t", "type": "STATUS", "data": {}}, "state": record.state}
        )
    assert [e["id"] for e in record.detail(since_event=1)["events"]] == [2, 3]
    assert len(record.detail()["events"]) == 3


def test_a_broken_subscriber_cannot_stall_the_run(monkeypatch):
    """A disconnected reader raises on delivery; the run must not notice."""
    record = _record(monkeypatch)

    def explode(_message):
        raise RuntimeError("client went away")

    record.subscribe(explode)
    healthy: list = []
    record.subscribe(healthy.append)

    record._handle({"t": "event", "event": {"id": 9, "time": "t", "type": "STATUS", "data": {}}, "state": {}})
    assert record.state == {}  # the reducer still ran
    assert healthy and healthy[-1]["type"] == "event"


def test_a_terminal_status_is_final(monkeypatch):
    """The pump thread and an HTTP cancel can both reach the transition."""
    record = _record(monkeypatch)
    assert record._set_status("succeeded", finished_at="now") is True
    assert record._set_status("failed") is False
    assert record.status == "succeeded"


def test_a_malformed_worker_message_does_not_stop_the_drain(monkeypatch):
    """If the pump stopped reading, the pipe would fill and the worker would wedge."""
    record = _record(monkeypatch)
    record._handle({"t": "usage"})  # missing fields
    record._handle({"t": "event"})  # missing event and state
    record._handle({"t": "unknown-kind"})
    assert record.status == "queued"


def test_cancelling_a_record_with_no_process_marks_it_cancelled(monkeypatch):
    record = _record(monkeypatch)
    assert record.cancel() is True
    assert record.status == "cancelled"


def test_the_worker_job_file_carries_everything_the_subprocess_needs(monkeypatch, tmp_path):
    """The worker is a separate process, so the job file is the whole contract."""
    record = _record(monkeypatch)
    job = {
        "run_id": record.id,
        "participant_dir": str(record.participant_dir),
        "config": record.config.model_dump(mode="json"),
        "task": record.request.task.model_dump(mode="json"),
        "generate_deep_phenotype": record.request.generate_deep_phenotype,
        "audit": record.audit,
    }
    # Everything must survive a JSON round trip.
    restored = json.loads(json.dumps(job))
    assert DashboardConfig.model_validate(restored["config"]).models.default_model
    assert TaskSpecInput.model_validate(restored["task"]).prediction_type == "binary"


def test_run_listing_is_newest_first_and_filterable_by_batch(monkeypatch):
    monkeypatch.setattr(rm, "estimate_run", lambda **kwargs: {"usd": 0.01, "total_tokens": 1, "lines": []})
    manager = rm.RunManager()
    request = RunRequest(
        participant_dir=str(SUBJ_001),
        task=TaskSpecInput(prediction_type="binary", target_label="T", control_label="C"),
    )
    first = manager.create_run(request, autostart=False)
    time.sleep(0.01)
    second = manager.create_run(request, batch_id="B1", autostart=False)

    ids = [r["id"] for r in manager.list_runs()]
    assert ids.index(second.id) < ids.index(first.id)
    assert [r["id"] for r in manager.list_runs(batch_id="B1")] == [second.id]


# --- history rehydration -----------------------------------------------------


def _archive(root: Path, run_id: str, status: str = "succeeded", **extra) -> Path:
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": run_id,
        "label": "SUBJ_001_PSEUDO",
        "participant_id": "SUBJ_001_PSEUDO",
        "participant_dir": str(SUBJ_001),
        "status": status,
        "audit": False,
        "batch_id": None,
        "created_at": f"2026-09-09T00:0{run_id[-1]}:00.000+00:00",
        "started_at": None,
        "finished_at": None,
        "error": None,
        "task": {"root_mode": "binary_classification"},
        "estimate": {"usd": 0.01, "total_tokens": 100},
        "cost": {"usd": 0.02, "total_tokens": 200},
        "stage": 6,
        "progress": 1,
        "max_steps": 1,
        "verdict": "SATISFACTORY",
        "state": {"stages": list(rm.STAGE_NAMES)},
        **extra,
    }
    (directory / "meta.json").write_text(json.dumps(payload))
    return directory


def test_finished_runs_survive_a_restart(tmp_path):
    _archive(tmp_path, "run_1")
    _archive(tmp_path, "run_2")
    manager = rm.RunManager()
    rows = manager.list_runs()
    assert {r["id"] for r in rows} == {"run_1", "run_2"}
    assert all(r["archived"] for r in rows)
    assert manager.get("run_1").cost["usd"] == 0.02


def test_a_run_recorded_as_in_flight_is_reported_as_cancelled(tmp_path):
    _archive(tmp_path, "run_3", status="running")
    manager = rm.RunManager()
    recovered = manager.get("run_3")
    assert recovered.status == "cancelled"
    assert "restarted" in recovered.summary()["error"]


def test_archived_events_are_replayed_from_the_ndjson_log(tmp_path):
    directory = _archive(tmp_path, "run_4")
    (directory / "events.ndjson").write_text(
        "\n".join(
            json.dumps({"t": "event", "event": {"id": i, "time": "t", "type": "STATUS", "data": {}}})
            for i in (1, 2, 3)
        )
    )
    manager = rm.RunManager()
    detail = manager.get("run_4").detail()
    assert [e["id"] for e in detail["events"]] == [1, 2, 3]
    assert [e["id"] for e in manager.get("run_4").detail(since_event=2)["events"]] == [3]


def test_a_corrupt_archive_is_skipped_rather_than_fatal(tmp_path):
    (tmp_path / "runs" / "run_bad").mkdir(parents=True)
    (tmp_path / "runs" / "run_bad" / "meta.json").write_text("{not json")
    _archive(tmp_path, "run_5")
    manager = rm.RunManager()
    assert [r["id"] for r in manager.list_runs()] == ["run_5"]


def test_rehydration_can_be_disabled_for_a_clean_registry(tmp_path):
    _archive(tmp_path, "run_6")
    assert rm.RunManager(rehydrate=False).list_runs() == []


# --- budget precedence -------------------------------------------------------


def test_explicit_budgets_survive_the_pipeline_preamble(monkeypatch):
    """
    `run_compass_pipeline` re-derives budgets at the start of every run, so an
    explicit choice has to be remembered or it is silently discarded.
    """
    import main as compass_main
    from src.full_stack.backend.api.engine_bridge import apply_config_to_settings
    from src.full_stack.backend.config.settings import reload_settings

    config = DashboardConfig()
    config.token_budget.max_agent_input_tokens = 250_000
    config.token_budget.max_agent_output_tokens = 40_000
    config.token_budget.critic_budget = 777_000

    reload_settings()
    settings = apply_config_to_settings(config)
    # Exactly what the pipeline does before it starts work.
    compass_main._sync_component_token_budgets(settings)
    compass_main._sync_role_token_limits_with_budgets(settings)

    assert settings.token_budget.max_agent_input_tokens == 250_000
    assert settings.token_budget.max_agent_output_tokens == 40_000
    assert settings.token_budget.critic_budget == 777_000
    reload_settings()


def test_role_ceilings_are_not_widened_by_the_preamble(monkeypatch):
    """The derived default is half the context window, which is far too wide."""
    import main as compass_main
    from src.full_stack.backend.api.engine_bridge import apply_config_to_settings
    from src.full_stack.backend.config.settings import reload_settings

    reload_settings()
    settings = apply_config_to_settings(DashboardConfig())
    compass_main._sync_component_token_budgets(settings)
    compass_main._sync_role_token_limits_with_budgets(settings)

    assert settings.models.orchestrator_max_tokens == 8_000
    assert settings.models.predictor_max_tokens == 12_000
    assert settings.models.tool_max_tokens == 4_000
    reload_settings()


def test_a_zero_means_derive_not_override(monkeypatch):
    import main as compass_main
    from src.full_stack.backend.api.engine_bridge import apply_config_to_settings
    from src.full_stack.backend.config.settings import reload_settings

    reload_settings()
    settings = apply_config_to_settings(DashboardConfig())
    compass_main._sync_role_token_limits_with_budgets(settings)
    # Nothing explicit was asked for, so the derived values stand.
    assert settings.token_budget.max_agent_input_tokens > 0
    assert settings.token_budget.explicit_limits == {}
    reload_settings()


def test_the_context_window_resolves_from_the_catalog_when_not_overridden(monkeypatch):
    """Every derived budget scales off this, so a stale default distorts them all."""
    from src.full_stack.backend.api import engine_bridge
    from src.full_stack.backend.config.settings import reload_settings

    monkeypatch.setattr(engine_bridge, "_catalog_context_window", lambda model_id: 128_000)
    reload_settings()
    settings = engine_bridge.apply_config_to_settings(DashboardConfig())
    assert settings.models.public_max_context_tokens == 128_000
    reload_settings()


def test_an_explicit_context_window_wins_over_the_catalog(monkeypatch):
    from src.full_stack.backend.api import engine_bridge
    from src.full_stack.backend.config.settings import reload_settings

    monkeypatch.setattr(engine_bridge, "_catalog_context_window", lambda model_id: 128_000)
    config = DashboardConfig()
    config.models.context_window = 32_000
    reload_settings()
    settings = engine_bridge.apply_config_to_settings(config)
    assert settings.models.public_max_context_tokens == 32_000
    reload_settings()


def test_role_model_overrides_go_through_the_engine_helper(monkeypatch):
    from src.full_stack.backend.api.engine_bridge import apply_config_to_settings
    from src.full_stack.backend.config.settings import reload_settings

    config = DashboardConfig()
    config.models.default_model = "vendor/base"
    config.models.role_models.critic = "vendor/sharper"
    reload_settings()
    settings = apply_config_to_settings(config)
    assert settings.models.critic_model == "vendor/sharper"
    assert settings.models.predictor_model == "vendor/base"
    reload_settings()
