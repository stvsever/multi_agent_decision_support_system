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


def test_the_context_window_resolves_from_the_catalog_when_not_overridden(monkeypatch, tmp_path):
    """Every derived budget scales off this, so a stale default distorts them all."""
    from src.full_stack.backend.api import engine_bridge
    from src.full_stack.backend.config.settings import reload_settings

    monkeypatch.setenv("COMPASS_HOME", str(tmp_path))
    (tmp_path / "model_catalog.json").write_text(
        json.dumps({"models": [{"id": DashboardConfig().models.default_model, "context_length": 128_000}]})
    )
    monkeypatch.setattr(engine_bridge, "_catalog_context_window", lambda model_id: 128_000)
    reload_settings()
    settings = engine_bridge.apply_config_to_settings(DashboardConfig())
    # Nothing configured: the engine resolves the window per role from the catalog.
    assert settings.models.public_max_context_tokens == 0
    assert settings.effective_context_window(settings.models.predictor_model, role="predictor") == 128_000
    assert settings.token_budget.max_agent_input_tokens == int(128_000 * 0.95)
    reload_settings()


def test_an_explicit_context_window_covers_unknown_models_and_role_overrides_force(monkeypatch):
    from src.full_stack.backend.api import engine_bridge
    from src.full_stack.backend.config.settings import reload_settings

    monkeypatch.setattr(engine_bridge, "_catalog_context_window", lambda model_id: 128_000)
    config = DashboardConfig()
    config.models.context_window = 32_000
    config.models.role_models.critic = "acme/unlisted-model"
    config.models.role_context_windows.predictor = 64_000
    reload_settings()
    settings = engine_bridge.apply_config_to_settings(config)
    assert settings.models.public_max_context_tokens == 32_000
    assert settings.effective_context_window("acme/unlisted-model", role="critic") == 32_000
    assert settings.effective_context_window(settings.models.predictor_model, role="predictor") == 64_000
    assert settings.models.role_context_windows == {"predictor": 64_000}
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


# --- evidence routing ----------------------------------------------------------


def _route_event(event_id, route="direct", iteration=1, **extra):
    data = {
        "route": route,
        "mode": "auto",
        "input_tokens": 9_000,
        "budget_tokens": 120_000,
        "threshold_source": "predictor_budget",
        "predictor_kind": "llm",
        "reason": "The record fits.",
        "escalated": False,
        "iteration": iteration,
        **extra,
    }
    state = {"current_stage": 4, "iteration": iteration, "route": data, "routes": [data]}
    return {"t": "event", "event": {"id": event_id, "time": "t", "type": "ROUTE", "data": data}, "state": state}


def _plan_event(event_id, iteration=1, state=None):
    plan = {
        "plan_id": f"P{iteration}",
        "iteration": iteration,
        "steps": [{"step_id": 1, "tool_name": "UnimodalCompressor", "description": "a", "depends_on": []}],
    }
    return {
        "t": "event",
        "event": {"id": event_id, "time": "t", "type": "PLAN", "data": {"steps": 1, "plan": plan}},
        # The worker's event store carries the latest route forward in its state.
        "state": {**(state or {}), "current_stage": 1, "iteration": iteration},
    }


def test_a_direct_route_draws_the_short_graph_and_reaches_subscribers(monkeypatch):
    record = _record(monkeypatch)
    seen: list = []
    record.subscribe(seen.append)

    record._handle(_route_event(1))
    graph = record.graph
    assert graph["meta"]["route"] == "direct"
    ids = [n["id"] for n in graph["nodes"]]
    assert ids == ["source:record", "agent:predictor", "agent:critic", "agent:communicator"]
    assert next(m for m in seen if m["type"] == "event")["graph"]["meta"]["route"] == "direct"
    assert record.summary()["input_route"] == "direct"
    assert record.summary()["predictor_kind"] == "llm"


def test_an_escalation_plan_replaces_the_direct_graph(monkeypatch):
    record = _record(monkeypatch)
    record._handle(_route_event(1, iteration=1))
    record._handle(_route_event(2, route="orchestrated", iteration=2, escalated=True, predictor_kind="decision"))
    # The orchestrated route has no graph until its plan arrives.
    assert record.graph["meta"]["route"] == "direct"
    record._handle(_plan_event(3, iteration=2, state=record.state))
    assert record.graph["meta"]["route"] == "orchestrated"
    assert record.graph["plan_id"] == "P2"
    assert record.summary()["input_route"] == "orchestrated"
    assert record.summary()["predictor_kind"] == "decision"


def test_a_plan_already_drawn_for_the_attempt_is_not_replaced(monkeypatch):
    record = _record(monkeypatch)
    record._handle(_plan_event(1, iteration=1))
    record._handle(_route_event(2, route="direct", iteration=1))
    assert record.graph["meta"]["route"] == "orchestrated"
    # The next attempt's direct route does draw its own graph.
    record._handle(_route_event(3, route="direct", iteration=2))
    assert record.graph["meta"]["route"] == "direct"
    assert record.graph["iteration"] == 2


def test_an_orchestrated_route_event_leaves_the_graph_to_the_plan(monkeypatch):
    record = _record(monkeypatch)
    seen: list = []
    record.subscribe(seen.append)
    record._handle(_route_event(1, route="orchestrated"))
    assert record.graph is None
    assert "graph" not in next(m for m in seen if m["type"] == "event")


def test_the_result_names_the_selected_route_over_the_latest_event(monkeypatch):
    record = _record(monkeypatch)
    assert record.summary()["input_route"] is None
    assert record.summary()["predictor_kind"] is None
    record._handle(_route_event(1, route="orchestrated", iteration=2))
    record._handle({"t": "result", "result": {"input_route": "direct", "predictor_kind": "decision"}})
    summary = record.summary()
    assert (summary["input_route"], summary["predictor_kind"]) == ("direct", "decision")
    assert record.detail()["input_route"] == "direct"


def test_archived_runs_keep_their_route(tmp_path):
    _archive(tmp_path, "run_7", input_route="direct", predictor_kind="decision")
    # A record written before routing existed still reads its result.
    _archive(tmp_path, "run_8", result={"input_route": "orchestrated", "predictor_kind": "llm"})
    manager = rm.RunManager()
    rows = {r["id"]: r for r in manager.list_runs()}
    assert (rows["run_7"]["input_route"], rows["run_7"]["predictor_kind"]) == ("direct", "decision")
    assert (rows["run_8"]["input_route"], rows["run_8"]["predictor_kind"]) == ("orchestrated", "llm")
    assert manager.get("run_7").detail()["input_route"] == "direct"


def test_the_estimate_for_a_new_run_says_which_route_it_assumed(monkeypatch):
    from src.full_stack.backend.api import cost as cost_module
    from src.full_stack.backend.api import engine_bridge

    monkeypatch.setattr(cost_module, "pricing_index", lambda cached_only=False: {})
    monkeypatch.setattr(engine_bridge, "get_credential", lambda provider: "test-key")
    manager = rm.RunManager(rehydrate=False)
    request = RunRequest(
        participant_dir=str(SUBJ_001),
        task=TaskSpecInput(prediction_type="binary", target_label="T", control_label="C"),
        overrides=RunOverrides(models={"role_models": {"predictor": "typesafe/jev-1.13"}}),
    )
    record = manager.create_run(request, autostart=False)
    assert record.estimate["route"] == "direct"
    assert record.estimate["predictor_kind"] == "decision"


def test_the_worker_passes_routing_to_the_pipeline():
    """The worker is a subprocess, so check its call against the pipeline signature."""
    import inspect

    import main as compass_main
    from src.full_stack.backend.api import worker

    parameters = inspect.signature(compass_main.run_compass_pipeline).parameters
    source = inspect.getsource(worker.main)
    for name in ("orchestration_mode", "orchestration_threshold_tokens"):
        assert name in parameters
        assert f"{name}=config.engine.{name}" in source
