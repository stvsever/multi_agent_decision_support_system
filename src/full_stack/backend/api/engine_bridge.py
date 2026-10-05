"""
Translation layer between the dashboard's configuration and the engine.

The engine reads a process-global ``Settings`` singleton, so every run executes
in its own worker process and this module is what stamps that process with the
run's configuration before the pipeline starts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..config.settings import LLMBackend, get_settings
from ..data.models.prediction_task import (
    PredictionMode,
    PredictionTaskNode,
    PredictionTaskSpec,
    build_task_spec_from_flat_args,
)
from ..agents.decision import is_decision_model
from .config_store import get_credential, resolve_predictor_kind, resolve_role
from .schemas import (
    AGENT_ROLES,
    INSTRUCTION_SLOTS,
    DashboardConfig,
    RunOverrides,
    TaskSpecInput,
    decision_model_conflicts,
)

_ROLE_TO_SETTINGS = {
    "orchestrator": "orchestrator",
    "integrator": "integrator",
    "predictor": "predictor",
    "critic": "critic",
    "communicator": "communicator",
    "tool": "tool",
}


def merge_overrides(config: DashboardConfig, overrides: Optional[RunOverrides]) -> DashboardConfig:
    """Apply per-run deltas on top of the saved configuration."""
    if overrides is None:
        return config
    delta = {k: v for k, v in overrides.model_dump(exclude_unset=True).items() if v is not None}
    if not delta:
        return config

    base = config.model_dump(mode="json")

    def deep(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(dst)
        for key, value in src.items():
            out[key] = deep(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        return out

    return DashboardConfig.model_validate(deep(base, delta))


class ConfigurationProblem(ValueError):
    """A configuration that cannot start a run, phrased for the person who set it."""


def run_blockers(config: DashboardConfig, audit: bool = False) -> List[str]:
    """
    Everything about this configuration that would stop a run from starting.

    The worker discovers these the hard way, minutes in and with a traceback,
    so the service checks them before it spawns anything. A structural audit
    calls no provider and loads no model, so none of this applies to it.
    """
    if audit:
        return []

    problems: List[str] = []
    backend = config.connection.backend

    if not config.models.default_model.strip():
        problems.append("Choose a default model before starting a run.")

    if backend == "local":
        if not config.local.model_name.strip():
            problems.append("Choose a local model before switching the backend to Local.")
    elif not get_credential("openrouter"):
        problems.append(
            "Add an OpenRouter key in Settings before starting a run, or switch the backend to Local."
        )

    # Validation already refuses these, but a config can be edited in place
    # after it was validated, and the engine would refuse it minutes later.
    conflicts = decision_model_conflicts(config)
    if conflicts:
        problems.append(
            "Structured decision models can only serve the Predictor role. "
            f"Choose a conventional LLM for: {', '.join(conflicts)}."
        )

    if resolve_predictor_kind(config) == "decision":
        model = resolve_role(config, "predictor")
        if config.decision.provider == "typesafe":
            if not get_credential("typesafe"):
                problems.append(
                    f"The Predictor {model} is called through TypeSafe: add a TypeSafe key "
                    "(TYPESAFE_API_KEY) or switch the decision provider to OpenRouter."
                )
        elif backend == "local" and not get_credential("openrouter"):
            # On OpenRouter the missing key is already reported above.
            problems.append(
                f"The Predictor {model} is called through OpenRouter even on the Local backend: "
                "add an OpenRouter key, or switch the decision provider to TypeSafe."
            )

    return problems


def ensure_runnable(config: DashboardConfig, audit: bool = False) -> None:
    problems = run_blockers(config, audit=audit)
    if problems:
        raise ConfigurationProblem(" ".join(problems))


def normalise_instructions(raw: Optional[Dict[str, str]]) -> Dict[str, str]:
    source = raw or {}
    return {slot: str(source.get(slot) or "").strip() for slot in INSTRUCTION_SLOTS}


def build_task_spec(task: TaskSpecInput) -> PredictionTaskSpec:
    """Turn the dashboard's task definition into the engine's canonical spec."""
    if task.prediction_type == "hierarchical":
        if task.root is None:
            raise ValueError("A hierarchical task needs a root node.")
        return PredictionTaskSpec(root=_to_engine_node(task.root))
    spec = build_task_spec_from_flat_args(
        prediction_type=task.prediction_type,
        target_label=task.target_label,
        control_label=task.control_label,
        class_labels=list(task.class_labels or []),
        regression_outputs=list(task.regression_outputs or []),
    )
    scales = {k: v for k, v in (task.output_scales or {}).items() if k in spec.root.regression_outputs and v}
    if scales:
        payload = spec.model_dump()
        payload["root"]["output_scales"] = scales
        spec = PredictionTaskSpec(**payload)
    return spec


def _to_engine_node(node: Any) -> PredictionTaskNode:
    return PredictionTaskNode(
        node_id=node.node_id,
        display_name=node.display_name,
        mode=PredictionMode(node.mode),
        class_labels=list(node.class_labels or []),
        regression_outputs=list(node.regression_outputs or []),
        unit_by_output=dict(node.unit_by_output or {}),
        output_scales={key: dict(value or {}) for key, value in (node.output_scales or {}).items()},
        class_definitions=dict(node.class_definitions or {}),
        required=bool(node.required),
        children=[_to_engine_node(child) for child in node.children or []],
    )


def describe_task(spec: PredictionTaskSpec) -> Dict[str, Any]:
    """A compact, display-ready summary of what the run is predicting."""
    nodes = spec.root.walk()
    return {
        "root_mode": spec.root.mode.value,
        "display_name": spec.root.display_name,
        "node_count": len(nodes),
        "is_hierarchical": len(nodes) > 1,
        "class_labels": list(spec.root.class_labels or []),
        "regression_outputs": list(spec.root.regression_outputs or []),
        "nodes": [
            {
                "node_id": n.node_id,
                "display_name": n.display_name,
                "mode": n.mode.value,
                "class_labels": list(n.class_labels or []),
                "regression_outputs": list(n.regression_outputs or []),
                "required": n.required,
                "child_ids": [c.node_id for c in n.children],
            }
            for n in nodes
        ],
    }


def _catalog_context_window(model_id: str) -> Optional[int]:
    """Context length from the model catalog; None when it cannot be reached."""
    try:
        from .catalog import get_model

        row = get_model(model_id) or {}
    except Exception:
        return None
    value = row.get("context_length")
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def apply_config_to_settings(config: DashboardConfig, settings: Any = None, *, process_env: bool = True) -> Any:
    """
    Stamp the engine's settings with the run configuration.

    Without `settings` this stamps the process-wide singleton (a run worker);
    with a fresh `Settings()` it builds an isolated copy (the cost estimate's
    offline preview), and `process_env=False` keeps the process environment
    untouched. Mirrors the token-budget derivation the CLI performs so a
    dashboard run and an equivalent CLI run behave identically.
    """
    import main as compass_main  # imported lazily; pulls in the whole engine

    settings = settings if settings is not None else get_settings()

    backend_name = config.connection.backend
    settings.models.backend = LLMBackend.LOCAL if backend_name == "local" else LLMBackend.OPENROUTER

    settings.openrouter_api_key = get_credential("openrouter")
    settings.typesafe_api_key = get_credential("typesafe")
    settings.openrouter_base_url = config.connection.openrouter_base_url
    settings.openrouter_site_url = config.connection.openrouter_site_url
    settings.openrouter_app_name = config.connection.openrouter_app_name

    settings.request_timeout_seconds = float(config.connection.request_timeout_seconds)
    settings.retry.max_retries = config.connection.max_retries
    settings.retry.retry_delay_seconds = config.connection.retry_delay_seconds
    settings.retry.auto_repair_enabled = config.engine.auto_repair_enabled
    settings.retry.max_critic_iterations = max(1, config.engine.max_iterations - 1)

    settings.detailed_tool_logging = config.engine.detailed_tool_logging
    settings.verbose_logging = config.engine.verbose

    effort = config.models.reasoning_effort
    settings.reasoning_effort = "" if effort == "provider_default" else effort

    default_model = config.models.default_model
    settings.models.public_model_name = default_model
    settings.models.embedding_model = config.models.embedding_model
    # Context windows resolve per role in the engine (Settings.context_window_resolution):
    # per-role override, served length (local), catalog cache, built-in table, then
    # this configured window for a model nothing else knows, then a generic fallback.
    settings.models.public_max_context_tokens = int(config.models.context_window or 0)
    settings.models.role_context_windows = {
        role: int(value)
        for role, value in config.models.role_context_windows.model_dump().items()
        if int(value or 0) > 0
    }
    if process_env and settings.models.backend != LLMBackend.LOCAL:
        # A run worker refreshes the catalog cache the engine reads windows from.
        _catalog_context_window(default_model)

    if settings.models.backend == LLMBackend.LOCAL:
        local = config.local
        settings.models.local_model_name = local.model_name
        settings.models.local_max_tokens = local.max_tokens
        settings.models.local_backend_type = local.engine
        settings.models.local_dtype = local.dtype
        settings.models.local_quantization = local.quantization or None
        settings.models.local_kv_cache_dtype = local.kv_cache_dtype or None
        settings.models.local_attn_implementation = local.attn_implementation
        settings.models.local_tensor_parallel_size = local.tensor_parallel_size
        settings.models.local_pipeline_parallel_size = local.pipeline_parallel_size
        settings.models.local_gpu_memory_utilization = local.gpu_memory_utilization
        settings.models.local_max_model_len = local.max_model_len or local.max_tokens
        settings.models.local_enforce_eager = local.enforce_eager
        settings.models.local_trust_remote_code = local.trust_remote_code
        for role in AGENT_ROLES:
            setattr(settings.models, f"{_ROLE_TO_SETTINGS[role]}_model", local.model_name)
        # A structured decision model is a remote service the local runtime
        # cannot serve, so a Predictor set to one keeps it rather than being
        # silently replaced; its own client calls OpenRouter or TypeSafe.
        chosen_predictor = str(config.models.role_models.predictor or "").strip()
        if is_decision_model(chosen_predictor):
            settings.models.predictor_model = chosen_predictor
    else:
        for role in AGENT_ROLES:
            chosen = str(getattr(config.models.role_models, role, "") or "").strip() or default_model
            setattr(settings.models, f"{_ROLE_TO_SETTINGS[role]}_model", chosen)

    for role in AGENT_ROLES:
        temperature = getattr(config.models.role_temperatures, role, None)
        if temperature is not None:
            setattr(settings.models, f"{_ROLE_TO_SETTINGS[role]}_temperature", float(temperature))

    settings.orchestration.mode = config.engine.orchestration_mode
    settings.orchestration.threshold_tokens = int(config.engine.orchestration_threshold_tokens)

    decision = config.decision
    settings.decision.provider = decision.provider
    settings.decision.choice_orders = int(decision.choice_orders)
    settings.decision.score_levels = int(decision.score_levels)
    settings.decision.regression_refine = bool(decision.regression_refine)
    settings.decision.stability_threshold = float(decision.stability_threshold)
    settings.decision.regression_stability_threshold = float(decision.regression_stability_threshold)
    settings.decision.sufficiency_threshold = float(decision.sufficiency_threshold)
    settings.decision.compiler_model = decision.compiler_model.strip()

    budget = config.token_budget
    # A zero means "derive it"; anything else is an explicit choice that the
    # engine records so its own start-of-run derivation cannot discard it.
    settings.token_budget.explicit_limits.clear()
    settings.token_budget.explicit_component_budgets.clear()
    settings.token_budget.explicit_role_max_tokens.clear()

    overrides = {
        "max_agent_input": budget.max_agent_input_tokens or None,
        "max_agent_output": budget.max_agent_output_tokens or None,
        "max_tool_input": budget.max_tool_input_tokens or None,
        "max_tool_output": budget.max_tool_output_tokens or None,
    }
    compass_main._apply_token_budget_defaults(settings, {k: v for k, v in overrides.items() if v})
    if budget.total_budget:
        settings.token_budget.total_budget = int(budget.total_budget)

    for field in (
        "orchestrator_budget",
        "executor_budget_per_step",
        "fusion_budget",
        "integrator_budget",
        "predictor_budget",
        "critic_budget",
        "communicator_budget",
    ):
        value = int(getattr(budget, field, 0) or 0)
        if value:
            settings.token_budget.explicit_component_budgets[field] = value

    role_max_tokens = {
        role: int(getattr(config.models.role_max_tokens, role, 0) or 0) for role in AGENT_ROLES
    }
    role_max_tokens = {k: v for k, v in role_max_tokens.items() if v}
    compass_main._apply_role_max_token_overrides(settings, role_max_tokens)
    compass_main._sync_component_token_budgets(settings)
    compass_main._sync_role_token_limits_with_budgets(settings, role_max_tokens)

    if process_env:
        os.environ["COMPASS_EXECUTOR_MAX_WORKERS"] = str(config.engine.executor_max_workers)
    return settings


def build_engine_settings(config: DashboardConfig) -> Any:
    """An isolated engine Settings object for this configuration (no process side effects)."""
    from ..config.settings import Settings

    return apply_config_to_settings(config, Settings(), process_env=False)


def effective_settings_snapshot(config: DashboardConfig) -> Dict[str, Any]:
    """What the engine will actually use, for the "effective configuration" panel."""
    return {
        "backend": config.connection.backend,
        "models": {
            role: (str(getattr(config.models.role_models, role, "") or "").strip() or config.models.default_model)
            for role in AGENT_ROLES
        },
        "temperatures": {role: getattr(config.models.role_temperatures, role) for role in AGENT_ROLES},
        "max_output_tokens": {
            role: (int(getattr(config.models.role_max_tokens, role, 0) or 0) or None) for role in AGENT_ROLES
        },
        "executor_max_workers": config.engine.executor_max_workers,
        "max_iterations": config.engine.max_iterations,
        "total_budget": config.token_budget.total_budget,
        "embedding_model": config.models.embedding_model,
        "orchestration_mode": config.engine.orchestration_mode,
        "orchestration_threshold_tokens": config.engine.orchestration_threshold_tokens,
        "predictor_kind": resolve_predictor_kind(config),
        "decision": {
            **config.decision.model_dump(mode="json"),
            # Blank means the Orchestrator writes the question book.
            "compiler_model": config.decision.compiler_model.strip() or resolve_role(config, "orchestrator"),
        },
        **_resolved_windows(config),
    }


def _resolved_windows(config: DashboardConfig) -> Dict[str, Any]:
    """
    Each role's context window as the engine resolves it (and the source that
    decided it), plus the Predictor's input budget. Empty when the engine
    cannot be built here, so the settings screen never fails on it.
    """
    try:
        from ..config.settings import llm_predictor_input_budget

        settings = build_engine_settings(config)
        windows = {role: window.as_dict() for role, window in settings.role_context_windows().items()}
        budget = None if resolve_predictor_kind(config) == "decision" else llm_predictor_input_budget(settings)
        return {"context_windows": windows, "predictor_input_budget": budget}
    except Exception:
        return {"context_windows": {}, "predictor_input_budget": None}
