#!/usr/bin/env python3
"""
COMPASS Multi-Agent System

Clinical Ontology-driven Multi-modal Predictive Agentic Support System

Main entry point for running the COMPASS pipeline on participant data.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Default control condition (non-target comparator)
DEFAULT_CONTROL_CONDITION = "non-target comparator phenotype profile"
AGENT_INSTRUCTION_KEYS = (
    "global",
    "orchestrator",
    "executor",
    "tools",
    "integrator",
    "predictor",
    "critic",
    "communicator",
)

# Keep the repository root importable when this file is launched directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.full_stack.backend.config.settings import get_settings, LLMBackend
from src.full_stack.backend.utils.core.data_loader import DataLoader
from src.full_stack.backend.utils.core.token_manager import TokenManager
from src.full_stack.backend.utils.llm_client import get_llm_client, is_context_length_error, reset_llm_client
from src.full_stack.backend.utils.core.fusion_layer import FusionLayer, FusionResult
from src.full_stack.backend.utils.core.predictor_input_assembler import PredictorInputAssembler
from src.full_stack.backend.utils.core.explainability_feature_space import build_feature_space
from src.full_stack.backend.utils.core.explainability_runner import run_explainability_methods
from src.full_stack.backend.utils.token_packer import count_tokens
from src.full_stack.backend.agents.orchestrator import Orchestrator
from src.full_stack.backend.agents.executor import Executor
from src.full_stack.backend.agents.predictor import Predictor
from src.full_stack.backend.agents.critic import Critic
from src.full_stack.backend.agents.communicator import Communicator
from src.full_stack.backend.utils.compass_logging.execution_logger import ExecutionLogger
from src.full_stack.backend.utils.compass_logging.decision_trace import DecisionTrace
from src.full_stack.backend.utils.compass_logging.patient_report import PatientReportGenerator
from src.full_stack.backend.data.models.prediction_result import Verdict
from src.full_stack.backend.data.models.prediction_task import (
    PredictionTaskSpec,
    PredictionMode,
    build_binary_task_spec,
    build_task_spec_from_flat_args,
    parse_csv_list,
)
from src.full_stack.backend.runtime.event_bus import get_ui
from src.full_stack.backend.data.models.execution_plan import ExecutionPlan
from src.full_stack.backend.agents.decision import enforce_role_models, get_decision_model_spec, is_decision_model
from src.full_stack.backend.agents.decision.predictor import DecisionPredictor
from src.full_stack.backend.agents.decision.quality import route_can_help as decision_route_can_help
from src.full_stack.backend.utils.core.input_routing import (
    build_direct_executor_output,
    decide_route,
    normalize_mode as normalize_orchestration_mode,
)


def _resolve_output_dir(participant_dir: Path, participant_id: str, settings) -> Path:
    pseudo_inputs = settings.paths.base_dir / "data" / "pseudo_data" / "inputs"
    pseudo_outputs = settings.paths.base_dir / "data" / "pseudo_data" / "outputs"
    try:
        if pseudo_inputs in participant_dir.resolve().parents:
            return pseudo_outputs / f"participant_{participant_id}"
    except Exception:
        pass
    return settings.paths.output_dir / f"participant_{participant_id}"




def _build_report_context_note(final_evaluation, selected_iteration: int, selection_reason: str) -> str:
    if not final_evaluation or final_evaluation.verdict == Verdict.SATISFACTORY:
        return ""
    return (
        "WARNING: Selected final verdict remains UNSATISFACTORY. "
        f"Selected iteration: {selected_iteration}. "
        f"Selection basis: {selection_reason}. "
        "Interpret deep phenotype content as exploratory and not production-final."
    )


def _append_execution_log_entry(output_dir: Path, participant_id: str, entry: Dict[str, Any]) -> None:
    log_path = output_dir / f"execution_log_{participant_id}.json"
    payload = []
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    payload = loaded
        except Exception:
            payload = []
    payload.append(entry)
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)


def _clamp_role_token_limits(settings) -> None:
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    for role in role_names:
        model_attr = f"{role}_model"
        max_attr = f"{role}_max_tokens"
        model_name = getattr(settings.models, model_attr, "")
        configured = int(getattr(settings.models, max_attr, 0) or 0)
        safe_cap = int(settings.auto_output_token_limit(model_name=model_name, role=role))
        if configured <= 0:
            setattr(settings.models, max_attr, safe_cap)
        else:
            setattr(settings.models, max_attr, min(configured, safe_cap))


def _apply_role_model_overrides(settings, role_models: Dict[str, Any]) -> None:
    if not isinstance(role_models, dict):
        return
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    for role in role_names:
        value = role_models.get(role)
        if value:
            setattr(settings.models, f"{role}_model", str(value))


def _apply_role_max_token_overrides(settings, role_max_tokens: Dict[str, Any]) -> None:
    if not isinstance(role_max_tokens, dict):
        return
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    for role in role_names:
        value = role_max_tokens.get(role)
        if value in (None, "", 0, "0"):
            continue
        setattr(settings.models, f"{role}_max_tokens", int(value))
        settings.token_budget.explicit_role_max_tokens[role] = int(value)


def _sync_role_token_limits_with_budgets(settings, role_max_tokens: Optional[Dict[str, Any]] = None) -> None:
    """
    Keep per-role generation caps aligned with global budget controls.

    Precedence:
    1) Explicit per-role max token overrides (if provided by UI)
    2) Global `max_agent_output_tokens` / `max_tool_output_tokens`
    3) Model-safe cap via `_clamp_role_token_limits`
    """
    role_names = ("orchestrator", "critic", "integrator", "predictor", "communicator", "tool")
    # Remembered choices count as explicit, so the pipeline's own preamble does
    # not overwrite a ceiling the caller set before the run started.
    remembered = dict(getattr(settings.token_budget, "explicit_role_max_tokens", {}) or {})
    explicit_roles = set()
    merged: Dict[str, Any] = {**remembered, **(role_max_tokens if isinstance(role_max_tokens, dict) else {})}
    for role in role_names:
        value = merged.get(role)
        if value in (None, "", 0, "0"):
            continue
        explicit_roles.add(role)
        setattr(settings.models, f"{role}_max_tokens", int(value))
        settings.token_budget.explicit_role_max_tokens[role] = int(value)

    max_agent_output = max(1, int(getattr(settings.token_budget, "max_agent_output_tokens", 16000) or 16000))
    max_tool_output = max(1, int(getattr(settings.token_budget, "max_tool_output_tokens", 8000) or 8000))

    for role in ("orchestrator", "critic", "integrator", "predictor", "communicator"):
        if role not in explicit_roles:
            setattr(settings.models, f"{role}_max_tokens", max_agent_output)
    if "tool" not in explicit_roles:
        settings.models.tool_max_tokens = max_tool_output

    _clamp_role_token_limits(settings)

def _compute_token_budget_defaults(settings, context_window: int) -> Dict[str, int]:
    ctx = max(1, int(context_window or 0))
    backend_value = getattr(settings.models.backend, "value", settings.models.backend)
    is_local = (
        settings.models.backend == LLMBackend.LOCAL
        or str(backend_value).lower() == "local"
    )
    if is_local:
        # Local-hosting profile:
        # - fixed output caps (agent=16K, tool=8K)
        # - tool input defaults to the same as agent input (capped for safety).
        agent_output = 16000
        tool_output = 8000
        headroom = 2048

        agent_input_cap = max(4096, ctx - agent_output - headroom)
        tool_input_cap = max(4096, ctx - tool_output - headroom)

        agent_input = max(4096, min(int(ctx * 0.75), agent_input_cap))
        tool_input = max(4096, min(agent_input, tool_input_cap))
        return {
            "max_agent_input": agent_input,
            "max_agent_output": agent_output,
            "max_tool_input": tool_input,
            "max_tool_output": tool_output,
        }
    return {
        "max_agent_input": int(ctx * 0.95),
        "max_agent_output": int(ctx * 0.25),
        "max_tool_input": int(ctx * 0.40),
        "max_tool_output": int(ctx * 0.25),
    }


_LIMIT_FIELDS = {
    "max_agent_input": "max_agent_input_tokens",
    "max_agent_output": "max_agent_output_tokens",
    "max_tool_input": "max_tool_input_tokens",
    "max_tool_output": "max_tool_output_tokens",
}


def _apply_token_budget_defaults(settings, overrides: Dict[str, Any]) -> None:
    """
    Derive the four global limits, keeping any value the caller set explicitly.

    An explicit value is both applied and remembered, because the pipeline
    re-derives these on every start and would otherwise discard the choice.
    """
    remembered = dict(getattr(settings.token_budget, "explicit_limits", {}) or {})
    merged = {**remembered, **{k: v for k, v in (overrides or {}).items() if v not in (None, "", 0)}}
    defaults = _compute_token_budget_defaults(settings, settings.effective_context_window())
    for key, field_name in _LIMIT_FIELDS.items():
        value = merged.get(key)
        if value in (None, "", 0):
            setattr(settings.token_budget, field_name, defaults[key])
        else:
            setattr(settings.token_budget, field_name, int(value))
            settings.token_budget.explicit_limits[key] = int(value)


def _sync_component_token_budgets(settings) -> None:
    """
    Synchronize per-component budgets with dynamic token limits.

    Why: component budgets were historically static (e.g., critic_budget=50k), which
    can produce misleading CRITICAL percentages when users select large-context models.
    """
    max_agent_input = int(getattr(settings.token_budget, "max_agent_input_tokens", 30000) or 30000)
    max_agent_output = int(getattr(settings.token_budget, "max_agent_output_tokens", 16000) or 16000)
    max_tool_input = int(getattr(settings.token_budget, "max_tool_input_tokens", 30000) or 30000)
    max_tool_output = int(getattr(settings.token_budget, "max_tool_output_tokens", 8000) or 8000)

    # Agent-level budgets scale with agent context + generation capacity.
    settings.token_budget.orchestrator_budget = max(50000, int(max_agent_input * 0.80) + max_agent_output)
    settings.token_budget.predictor_budget = max(100000, int(max_agent_input * 0.95) + max_agent_output)
    settings.token_budget.critic_budget = max(50000, int(max_agent_input * 0.95) + max_agent_output)
    settings.token_budget.communicator_budget = max(50000, int(max_agent_input * 0.75) + max_agent_output)

    # Integration/execution budgets scale with tool payload sizes.
    tool_fusion_floor = int(max(max_agent_input * 0.85, max_tool_input + max_tool_output))
    settings.token_budget.fusion_budget = max(90000, tool_fusion_floor)
    settings.token_budget.integrator_budget = max(90000, tool_fusion_floor)
    settings.token_budget.executor_budget_per_step = max(30000, max_tool_input + max_tool_output)

    # A component budget the caller set explicitly wins over the derived value.
    for name, value in (getattr(settings.token_budget, "explicit_component_budgets", {}) or {}).items():
        if hasattr(settings.token_budget, name) and value:
            setattr(settings.token_budget, name, int(value))


def _parse_xai_methods(raw_methods: Optional[str]) -> List[str]:
    """Parse comma-separated XAI methods with support for `all` alias."""
    if raw_methods is None:
        return []
    text = str(raw_methods).strip().lower()
    if not text:
        return []
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if not parts:
        return []
    valid = {"external", "internal", "hybrid"}
    if "all" in parts:
        parts = ["external", "internal", "hybrid"]
    invalid = sorted([p for p in parts if p not in valid])
    if invalid:
        raise ValueError(f"Invalid --xai_methods entries: {', '.join(invalid)}")
    # preserve order, remove duplicates
    dedup: List[str] = []
    for p in parts:
        if p not in dedup:
            dedup.append(p)
    return dedup


def _apply_explainability_overrides(settings, args: argparse.Namespace) -> None:
    methods = _parse_xai_methods(getattr(args, "xai_methods", None))
    settings.explainability.methods = methods
    settings.explainability.enabled = bool(methods)
    settings.explainability.run_full_validation = bool(getattr(args, "xai_full_validation", False))

    if getattr(args, "xai_external_k", None) is not None:
        settings.explainability.external_k = max(1, int(args.xai_external_k))
    if getattr(args, "xai_external_runs", None) is not None:
        settings.explainability.external_runs = max(1, int(args.xai_external_runs))
    if getattr(args, "xai_external_adaptive", None) is not None:
        settings.explainability.external_adaptive = bool(args.xai_external_adaptive)

    if getattr(args, "xai_internal_model", None):
        settings.explainability.internal_model = str(args.xai_internal_model)
    if getattr(args, "xai_internal_steps", None) is not None:
        settings.explainability.internal_steps = max(1, int(args.xai_internal_steps))
    if getattr(args, "xai_internal_baseline", None):
        settings.explainability.internal_baseline_mode = str(args.xai_internal_baseline)
    if getattr(args, "xai_internal_span_mode", None):
        settings.explainability.internal_span_mode = str(args.xai_internal_span_mode)

    if getattr(args, "xai_hybrid_model", None):
        settings.explainability.hybrid_model = str(args.xai_hybrid_model)
    if getattr(args, "xai_hybrid_repeats", None) is not None:
        settings.explainability.hybrid_repeats = max(1, int(args.xai_hybrid_repeats))
    if getattr(args, "xai_hybrid_temperature", None) is not None:
        settings.explainability.hybrid_temperature = float(args.xai_hybrid_temperature)


def _parse_task_spec_payload(raw_payload: Any) -> PredictionTaskSpec:
    if isinstance(raw_payload, PredictionTaskSpec):
        return raw_payload
    if isinstance(raw_payload, dict):
        return PredictionTaskSpec(**raw_payload)
    if isinstance(raw_payload, str):
        text = raw_payload.strip()
        if not text:
            raise ValueError("prediction task spec payload is empty")
        return PredictionTaskSpec(**json.loads(text))
    raise ValueError(f"Unsupported prediction task spec payload type: {type(raw_payload).__name__}")


def _resolve_prediction_task_spec(
    *,
    prediction_type: Optional[str],
    target_label: str,
    control_label: str,
    class_labels: Optional[List[str]] = None,
    regression_outputs: Optional[List[str]] = None,
    task_spec_json: Optional[str] = None,
    task_spec_file: Optional[str] = None,
    task_spec_payload: Optional[Dict[str, Any]] = None,
) -> PredictionTaskSpec:
    if task_spec_payload is not None:
        return _parse_task_spec_payload(task_spec_payload)

    if task_spec_json and str(task_spec_json).strip():
        return _parse_task_spec_payload(task_spec_json)

    if task_spec_file and str(task_spec_file).strip():
        path = Path(str(task_spec_file)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Task spec file not found: {path}")
        with open(path, "r") as f:
            payload = json.load(f)
        return PredictionTaskSpec(**payload)

    mode = str(prediction_type or "binary").strip().lower()
    if mode == "hierarchical":
        raise ValueError("prediction_type=hierarchical requires --task_spec_json or --task_spec_file")

    return build_task_spec_from_flat_args(
        prediction_type=mode,
        target_label=target_label,
        control_label=control_label,
        class_labels=list(class_labels or []),
        regression_outputs=list(regression_outputs or []),
    )


def _task_spec_to_legacy_labels(task_spec: PredictionTaskSpec) -> Tuple[str, str]:
    target_label, control_label = task_spec.legacy_target_control()
    target = str(target_label or "Target Phenotype").strip() or "Target Phenotype"
    root_mode = task_spec.root.mode
    if root_mode == PredictionMode.BINARY_CLASSIFICATION:
        control = str(control_label or DEFAULT_CONTROL_CONDITION).strip() or DEFAULT_CONTROL_CONDITION
    else:
        control = str(control_label or "").strip()
    return target, control


def _align_binary_task_spec_labels(
    task_spec: PredictionTaskSpec,
    *,
    target_label: str,
    control_label: str,
) -> PredictionTaskSpec:
    """
    Keep binary class labels aligned with explicit runtime labels.

    This prevents stale UI payload defaults (e.g., CASE/CONTROL from cached JS)
    from overriding user-entered binary labels.
    """
    if task_spec.root.mode != PredictionMode.BINARY_CLASSIFICATION:
        return task_spec
    target = str(target_label or "").strip()
    control = str(control_label or "").strip()
    if not target or not control:
        return task_spec
    current = list(task_spec.root.class_labels or [])
    if len(current) == 2 and current[0] == target and current[1] == control:
        return task_spec
    payload = task_spec.model_dump() if hasattr(task_spec, "model_dump") else task_spec.dict()
    payload.setdefault("root", {})
    payload["root"]["display_name"] = target
    payload["root"]["class_labels"] = [target, control]
    return PredictionTaskSpec(**payload)


def _normalize_agent_instructions(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out = {key: "" for key in AGENT_INSTRUCTION_KEYS}
    if not isinstance(raw, dict):
        return out
    for key in AGENT_INSTRUCTION_KEYS:
        out[key] = str(raw.get(key) or "").strip()
    return out


def _combine_instruction(global_instruction: str, scoped_instruction: str) -> str:
    global_text = str(global_instruction or "").strip()
    scoped_text = str(scoped_instruction or "").strip()
    if global_text and scoped_text:
        return f"{global_text}\n\n{scoped_text}"
    return scoped_text or global_text


def _prediction_primary_label(prediction: Any) -> str:
    root = getattr(prediction, "root_prediction", None)
    if root is not None:
        mode = getattr(root, "mode", None)
        if mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            cls = getattr(root, "classification", None)
            label = str(getattr(cls, "predicted_label", "") or "").strip()
            if label:
                return label
            return str(getattr(root, "node_id", "classification_output"))
        reg = getattr(root, "regression", None)
        values = getattr(reg, "values", {}) if reg is not None else {}
        if isinstance(values, dict) and values:
            if len(values) == 1:
                key, value = next(iter(values.items()))
                try:
                    return f"{key}: {float(value):.3f}"
                except Exception:
                    return f"{key}: {value}"
            rows = []
            for key, value in list(values.items())[:3]:
                try:
                    rows.append(f"{key}: {float(value):.3f}")
                except Exception:
                    rows.append(f"{key}: {value}")
            if len(values) > 3:
                rows.append(f"+{len(values) - 3} more")
            return "; ".join(rows)
        return str(getattr(root, "node_id", "regression_output"))
    if getattr(prediction, "binary_classification", None) is not None:
        return str(prediction.binary_classification.value)
    return "NON_BINARY"


def _prediction_display_probability(prediction: Any) -> Optional[float]:
    root = getattr(prediction, "root_prediction", None)
    if root is not None:
        mode = getattr(root, "mode", None)
        if mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            cls = getattr(root, "classification", None)
            probs = getattr(cls, "probabilities", {}) if cls is not None else {}
            predicted_label = str(getattr(cls, "predicted_label", "") or "").strip()
            if isinstance(probs, dict) and probs:
                if predicted_label in probs:
                    try:
                        return max(0.0, min(1.0, float(probs[predicted_label])))
                    except Exception:
                        pass
                try:
                    top_prob = max(float(v) for v in probs.values())
                    return max(0.0, min(1.0, top_prob))
                except Exception:
                    pass
    score = getattr(prediction, "probability_score", None)
    if isinstance(score, (int, float)):
        return float(score)
    if root is None:
        return None
    confidence_score = getattr(root, "confidence_score", None)
    if isinstance(confidence_score, (int, float)):
        return max(0.0, min(1.0, float(confidence_score)))
    return None


def _decision_orchestrator_note(settings, prediction_task_spec: PredictionTaskSpec) -> str:
    """Planning guidance when the Predictor is a structured decision model."""
    spec = get_decision_model_spec(settings.models.predictor_model)
    budget = spec.state_budget_cl100k(ratio=float(settings.decision.tokenizer_ratio)) if spec else 26000
    label = spec.label if spec else settings.models.predictor_model
    nodes = ", ".join(str(n.display_name or n.node_id) for n in prediction_task_spec.root.walk())
    return (
        f"The Predictor in this run is a structured decision model ({label}). It writes no text: it reads one "
        f"evidence state of at most about {budget:,} tokens and answers typed questions with calibrated "
        f"probabilities for: {nodes}. Long records full of unrelated detail lower its accuracy. Plan steps that "
        "distil the record into compact, decision-relevant evidence for every output: a phenotype "
        "representation, a feature synthesis of the findings that bear on the outputs with their values, and "
        "compressions of large domains. Keep explicit scores and values; drop detail that does not bear on the outputs."
    )


def _direct_plan(
    *,
    participant_id: str,
    target_condition: str,
    control_condition: str,
    prediction_task_spec: PredictionTaskSpec,
    domains: List[str],
    reason: str,
    iteration: int,
) -> ExecutionPlan:
    """An empty plan that records a direct attempt in the logs and reports."""
    return ExecutionPlan(
        plan_id=f"direct-{iteration}",
        participant_id=participant_id,
        target_condition=target_condition,
        control_condition=control_condition,
        prediction_task_spec=prediction_task_spec,
        priority_domains=list(domains),
        fusion_strategy="direct",
        user_facing_explanation="The full record fits the Predictor input, so it was passed on without orchestration.",
        reasoning=reason,
        steps=[],
        iteration=iteration,
    )


def _run_orchestrated_attempt(
    *,
    orchestrator: Orchestrator,
    executor: Executor,
    participant_data,
    target_condition: str,
    control_condition: str,
    prediction_task_spec: PredictionTaskSpec,
    previous_feedback: Optional[str],
    iteration: int,
    runtime_agent_instructions: Dict[str, str],
    exec_logger: ExecutionLogger,
    decision_trace: DecisionTrace,
    interactive_ui: bool,
    settings,
) -> Tuple[Any, Dict[str, Any]]:
    """Orchestrator plans, the Executor runs the tools and the Integrator fuses."""
    ui = get_ui()
    if interactive_ui:
        ui.set_status("Orchestrator creating execution plan...", stage=1, iteration=iteration)
    print(f"\n[3/5] Orchestrator creating execution plan...")
    print(
        f"[Orchestrator] Runtime config: model={settings.models.orchestrator_model}, "
        f"max_tokens={settings.models.orchestrator_max_tokens}, temp={settings.models.orchestrator_temperature}, "
        f"agent_in={settings.token_budget.max_agent_input_tokens}, "
        f"agent_out={settings.token_budget.max_agent_output_tokens}, "
        f"tool_in={settings.token_budget.max_tool_input_tokens}, "
        f"tool_out={settings.token_budget.max_tool_output_tokens}"
    )
    orchestrator_started = time.time()
    plan = orchestrator.execute(
        participant_data=participant_data,
        target_condition=target_condition,
        control_condition=control_condition,
        prediction_task_spec=prediction_task_spec,
        previous_feedback=previous_feedback,
        iteration=iteration,
    )
    orchestrator_elapsed = int((time.time() - orchestrator_started) * 1000)
    print(
        f"[Orchestrator] Plan created in {orchestrator_elapsed}ms "
        f"(steps={plan.total_steps}, est_tokens={plan.total_estimated_tokens})"
    )
    exec_logger.log_orchestrator(
        {"plan_id": plan.plan_id, "total_steps": plan.total_steps, "priority_domains": plan.priority_domains}
    )
    decision_trace.record_orchestrator_plan(
        domains=plan.priority_domains,
        num_steps=plan.total_steps,
        reasoning=plan.reasoning[:500],
    )

    print(f"\n[4/5] Executor processing plan...")
    executor_started = time.time()
    executor_output = executor.execute(
        plan=plan,
        participant_data=participant_data,
        target_condition=target_condition,
        control_condition=control_condition,
        prediction_task_spec=prediction_task_spec,
        agent_instructions=runtime_agent_instructions,
    )
    executor_output["route"] = "orchestrated"
    print(f"[Executor] Completed in {int((time.time() - executor_started) * 1000)}ms")

    exec_result = executor_output.get("execution_result")
    if exec_result and hasattr(exec_result, "step_statuses"):
        for step_id, status in exec_result.step_statuses.items():
            exec_logger.log_executor_step(
                step_id=step_id,
                tool_name=status.get("tool_name", "unknown"),
                status=status.get("status", "UNKNOWN"),
                tokens=status.get("tokens", 0),
            )

    # The FUSION event sets the status to "Fusion complete", so the caller sets
    # the prediction status after this returns.
    if interactive_ui and "predictor_input" in executor_output:
        ui.on_fusion_complete(executor_output["predictor_input"])
    return plan, executor_output


def _generate_deep_phenotype_report(
    *,
    communicator: Communicator,
    prediction: Any,
    evaluation: Any,
    executor_output: Dict[str, Any],
    data_overview: Dict[str, Any],
    execution_summary: Dict[str, Any],
    control_condition: str,
    report_context_note: str,
    base_output_dir: Path,
    participant_id: str,
    user_focus_modalities: str = "",
    user_general_instruction: str = "",
    trigger_source: str = "manual",
    ui_step_id: Optional[int] = None,
    interactive_ui: bool = False,
) -> Dict[str, Any]:
    ui = get_ui()
    if interactive_ui:
        ui.set_status("Generating deep phenotype report...", stage=6)
        ui.on_step_start(
            step_id=ui_step_id or 930,
            tool_name="Communicator Agent",
            description="Generating deep phenotype report...",
            stage=6,
        )
    try:
        deep_report = communicator.execute(
            prediction=prediction,
            evaluation=evaluation,
            executor_output=executor_output,
            data_overview=data_overview,
            execution_summary=execution_summary,
            report_context_note=report_context_note,
            control_condition=control_condition,
            user_focus_modalities=user_focus_modalities,
            user_general_instruction=user_general_instruction,
            status_callback=(lambda msg: ui.set_status(msg, stage=6)) if interactive_ui else None,
        )
        deep_path = base_output_dir / "deep_phenotype.md"
        with open(deep_path, "w") as f:
            f.write(deep_report)

        metadata = dict(getattr(communicator, "last_run_metadata", {}) or {})
        metadata.update(
            {
                "trigger_source": trigger_source,
                "user_focus_modalities_present": bool(str(user_focus_modalities or "").strip()),
                "user_general_instruction_present": bool(str(user_general_instruction or "").strip()),
                "output_path": str(deep_path),
            }
        )
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "COMMUNICATOR",
                "timestamp": datetime.now().isoformat(),
                "data": metadata,
            },
        )
        perf_path = base_output_dir / f"performance_report_{participant_id}.json"
        if perf_path.exists():
            try:
                with open(perf_path, "r") as f:
                    perf = json.load(f)
                perf["deep_phenotype"] = {
                    "generated": True,
                    "path": str(deep_path),
                    "trigger_source": trigger_source,
                    "metadata": metadata,
                }
                with open(perf_path, "w") as f:
                    json.dump(perf, f, indent=2)
            except Exception:
                pass

        if interactive_ui:
            ui.on_step_complete(
                step_id=ui_step_id or 930,
                tokens=0,
                duration_ms=0,
                preview="Deep phenotype report generated.",
            )
        print(f"[Communicator] Saved to: {deep_path}")
        return {"success": True, "path": str(deep_path), "metadata": metadata}
    except Exception as e:
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "COMMUNICATOR",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "trigger_source": trigger_source,
                    "success": False,
                    "error": str(e),
                    "user_focus_modalities_present": bool(str(user_focus_modalities or "").strip()),
                    "user_general_instruction_present": bool(str(user_general_instruction or "").strip()),
                },
            },
        )
        if interactive_ui:
            ui.on_step_failed(step_id=ui_step_id or 930, error=str(e))
        return {"success": False, "error": str(e), "metadata": {}}


def _generate_xai_explainability_report(
    *,
    communicator: Communicator,
    xai_result: Dict[str, Any],
    prediction: Any,
    evaluation: Any,
    execution_summary: Dict[str, Any],
    target_condition: str,
    control_condition: str,
    base_output_dir: Path,
    participant_id: str,
    trigger_source: str = "cli",
    ui_step_id: Optional[int] = None,
    interactive_ui: bool = False,
) -> Dict[str, Any]:
    ui = get_ui()
    methods = dict((xai_result or {}).get("methods") or {})
    successful_methods = sorted(
        [name for name, payload in methods.items() if (payload or {}).get("status") == "success"]
    )

    if not successful_methods:
        reason = "No successful explainability method outputs available; skipping XAI report."
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "XAI_REPORT",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "trigger_source": trigger_source,
                    "success": False,
                    "skipped": True,
                    "reason": reason,
                    "methods_requested": list((xai_result or {}).get("methods_requested") or []),
                },
            },
        )
        return {"success": False, "skipped": True, "reason": reason, "metadata": {}}

    if interactive_ui:
        ui.set_status("Generating explainability report...", stage=6)
        ui.on_step_start(
            step_id=ui_step_id or 995,
            tool_name="Communicator Agent",
            description="Generating explainability report...",
            stage=6,
        )

    try:
        xai_report = communicator.execute_xai_report(
            xai_result=xai_result,
            prediction=prediction,
            evaluation=evaluation,
            execution_summary=execution_summary,
            target_condition=target_condition,
            control_condition=control_condition,
            status_callback=(lambda msg: ui.set_status(msg, stage=6)) if interactive_ui else None,
        )
        xai_path = base_output_dir / "xai_explainability_report.md"
        with open(xai_path, "w") as f:
            f.write(xai_report)

        metadata = dict(getattr(communicator, "last_run_metadata", {}) or {})
        metadata.update(
            {
                "trigger_source": trigger_source,
                "output_path": str(xai_path),
                "methods_requested": list((xai_result or {}).get("methods_requested") or []),
                "methods_successful": successful_methods,
            }
        )
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "XAI_REPORT",
                "timestamp": datetime.now().isoformat(),
                "data": metadata,
            },
        )

        if interactive_ui:
            ui.on_step_complete(
                step_id=ui_step_id or 995,
                tokens=0,
                duration_ms=0,
                preview="XAI explainability report generated.",
            )
        print(f"[Communicator] XAI report saved to: {xai_path}")
        return {"success": True, "path": str(xai_path), "metadata": metadata}
    except Exception as e:
        _append_execution_log_entry(
            base_output_dir,
            participant_id,
            {
                "type": "XAI_REPORT",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "trigger_source": trigger_source,
                    "success": False,
                    "error": str(e),
                    "methods_requested": list((xai_result or {}).get("methods_requested") or []),
                    "methods_successful": successful_methods,
                },
            },
        )
        if interactive_ui:
            ui.on_step_failed(step_id=ui_step_id or 995, error=str(e))
        return {"success": False, "error": str(e), "metadata": {}}


def run_compass_pipeline(
    participant_dir: Path,
    target_condition: str,
    control_condition: str = DEFAULT_CONTROL_CONDITION,
    prediction_task_spec: Optional[PredictionTaskSpec] = None,
    agent_instructions: Optional[Dict[str, str]] = None,
    max_iterations: int = 3,
    verbose: bool = True,
    interactive_ui: bool = False,
    generate_deep_phenotype: bool = False,
    generate_xai_report: bool = False,
    deep_report_focus_modalities: str = "",
    deep_report_general_instruction: str = "",
    orchestration_mode: Optional[str] = None,
    orchestration_threshold_tokens: Optional[int] = None,
) -> dict:
    """
    Run the complete COMPASS pipeline for a participant.
    
    Args:
        participant_dir: Path to participant data directory
        target_condition: Target phenotype string (legacy compatibility)
        control_condition: Control comparator string (legacy compatibility)
        prediction_task_spec: Canonical task specification (hierarchical/mixed)
        max_iterations: Maximum actor-critic iterations
        verbose: Enable verbose output
        orchestration_mode: "auto" (default from settings): orchestrate only when the
            record does not fit the Predictor input; "always"; or "never".
        orchestration_threshold_tokens: replace the Predictor input budget used by
            "auto" with a fixed token count (it cannot exceed the budget).
    
    Returns:
        Dictionary with prediction result and metadata
    """
    settings = get_settings()
    _sync_component_token_budgets(settings)
    _sync_role_token_limits_with_budgets(settings)
    start_time = datetime.now()

    if prediction_task_spec is None:
        prediction_task_spec = build_binary_task_spec(
            target_label=target_condition,
            control_label=control_condition,
        )
    target_condition, control_condition = _task_spec_to_legacy_labels(prediction_task_spec)
    runtime_agent_instructions = _normalize_agent_instructions(agent_instructions)
    
    # Initialize UI
    ui = get_ui(enabled=interactive_ui)
    
    # Initialize components
    if interactive_ui:
        ui.on_pipeline_start(
            participant_id=participant_dir.name,
            target=target_condition,
            control=control_condition,
            prediction_spec=(
                prediction_task_spec.model_dump()
                if hasattr(prediction_task_spec, "model_dump")
                else prediction_task_spec.dict()
            ),
            participant_dir=str(participant_dir),
            max_iterations=max_iterations
        )
    else:
        print("\n" + "=" * 70)
        print("  COMPASS - Clinical Ontology-driven Multi-modal Predictive Agentic Support System")
        print("=" * 70)
    
    # Load participant data
    if interactive_ui: ui.set_status("Loading Participant Data...", stage=0)
    print(f"\n[1/5] Loading participant data from: {participant_dir}")
    data_loader = DataLoader()
    participant_data = data_loader.load(participant_dir)
    participant_id_raw = str(getattr(participant_data, "participant_id", "") or "").strip()
    if not participant_id_raw or participant_id_raw.lower() == "unknown":
        participant_id = participant_dir.name
        try:
            participant_data.participant_id = participant_id
        except Exception:
            pass
    else:
        participant_id = participant_id_raw

    # Build explainability feature space once from loaded artifacts.
    multimodal_for_xai = getattr(participant_data.multimodal_data, "features", {}) or {}
    try:
        raw_multimodal_path = (participant_data.raw_files or {}).get("multimodal_data")
        if raw_multimodal_path and Path(raw_multimodal_path).exists():
            with open(raw_multimodal_path, "r") as f:
                multimodal_for_xai = json.load(f)
    except Exception:
        multimodal_for_xai = getattr(participant_data.multimodal_data, "features", {}) or {}

    non_numerical_for_xai = str(
        getattr(participant_data.non_numerical_data, "raw_text", "") or ""
    )
    feature_space = build_feature_space(
        multimodal_for_xai,
        non_numerical_text=non_numerical_for_xai,
    )
    
    # Initialize logging
    exec_logger = ExecutionLogger(participant_id, verbose=verbose)
    decision_trace = DecisionTrace(participant_id)
    exec_logger.log_pipeline_start(
        target_condition=target_condition,
        control_condition=control_condition,
        prediction_task_mode=prediction_task_spec.root.mode.value if prediction_task_spec is not None else None,
    )
    
    # Initialize token manager
    token_manager = TokenManager()
    
    # Preflight connectivity check (public API backend)
    if settings.models.backend in (LLMBackend.OPENAI, LLMBackend.OPENROUTER):
        provider_label = "OpenRouter" if settings.models.backend == LLMBackend.OPENROUTER else "OpenAI"
        if interactive_ui:
            ui.set_status(f"Checking {provider_label} connectivity...", stage=0)
        llm_client = get_llm_client()
        try:
            llm_client.ping()
        except Exception as e:
            key_name = "OPENROUTER_API_KEY" if settings.models.backend == LLMBackend.OPENROUTER else "OPENAI_API_KEY"
            raise RuntimeError(
                f"{provider_label} connectivity check failed. Verify network access and {key_name}."
            ) from e
    elif settings.models.backend == LLMBackend.LOCAL:
        if interactive_ui:
            ui.set_status("Initializing local model...", stage=0)
        try:
            get_llm_client()
        except Exception as e:
            print(f"[Init][Local] Initialization failed: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"Local backend initialization failed: {e}"
            ) from e

    # Initialize agents
    if interactive_ui: ui.set_status("Initializing Agents...", stage=0)
    print(f"\n[2/5] Initializing COMPASS agents...")
    enforce_role_models(settings)
    windows = settings.role_context_windows()
    print(
        "[Init] Context windows: "
        + ", ".join(f"{role} {window.tokens:,} ({window.source})" for role, window in windows.items())
    )
    predictor_kind = "decision" if is_decision_model(settings.models.predictor_model) else "llm"
    orchestrator = Orchestrator(token_manager=token_manager)
    executor = Executor(token_manager=token_manager)
    if predictor_kind == "decision":
        predictor = DecisionPredictor(token_manager=token_manager)
        spec = get_decision_model_spec(settings.models.predictor_model)
        print(
            f"[Init] Predictor: {spec.label if spec else settings.models.predictor_model} (structured decision "
            f"model, state limit {spec.state_context_tokens if spec else 0:,} tokens); companion LLM for the other "
            f"roles: {settings.models.orchestrator_model}"
        )
    else:
        predictor = Predictor(token_manager=token_manager)
    critic = Critic(token_manager=token_manager)
    communicator = Communicator(token_manager=token_manager)
    orchestrator_instruction = _combine_instruction(
        runtime_agent_instructions.get("global", ""), runtime_agent_instructions.get("orchestrator", "")
    )
    if predictor_kind == "decision":
        orchestrator_instruction = _combine_instruction(
            orchestrator_instruction, _decision_orchestrator_note(settings, prediction_task_spec)
        )
    orchestrator.set_runtime_instruction(orchestrator_instruction)
    predictor.set_runtime_instruction(
        _combine_instruction(runtime_agent_instructions.get("global", ""), runtime_agent_instructions.get("predictor", ""))
    )
    critic.set_runtime_instruction(
        _combine_instruction(runtime_agent_instructions.get("global", ""), runtime_agent_instructions.get("critic", ""))
    )
    communicator.set_runtime_instruction(
        _combine_instruction(runtime_agent_instructions.get("global", ""), runtime_agent_instructions.get("communicator", ""))
    )
    executor.integrator.set_runtime_instruction(
        _combine_instruction(runtime_agent_instructions.get("global", ""), runtime_agent_instructions.get("integrator", ""))
    )
    
    # Evidence route. The direct executor output needs no model call, so the
    # decision is made up front: a record that fits the Predictor's input goes
    # straight to it; a larger one goes through the orchestration workflow.
    mode = normalize_orchestration_mode(
        orchestration_mode if orchestration_mode is not None else settings.orchestration.mode
    )
    threshold_override = int(
        orchestration_threshold_tokens
        if orchestration_threshold_tokens is not None
        else (settings.orchestration.threshold_tokens or 0)
    )
    direct_output = build_direct_executor_output(
        participant_data,
        target_condition=target_condition,
        control_condition=control_condition,
        prediction_task_spec=prediction_task_spec,
        agent_instructions=runtime_agent_instructions,
        model_hint=settings.models.predictor_model or "gpt-5",
    )
    if interactive_ui:
        ui.set_status("Measuring the record against the Predictor input...", stage=0)
    try:
        measurement = predictor.measure_direct(
            dict(direct_output),
            target_condition=target_condition,
            control_condition=control_condition,
            prediction_task_spec=prediction_task_spec,
        )
    except Exception as exc:
        # Without a measurement, auto mode orchestrates; never mode still goes direct.
        print(f"[Route] Could not measure the direct input ({type(exc).__name__}: {exc}); assuming it does not fit.")
        measurement = None
    initial_route = decide_route(
        mode=mode,
        measurement=measurement,
        threshold_override=threshold_override,
        predictor_kind=predictor_kind,
    )
    # Whether the record fits the Predictor itself (a lower routing threshold
    # decides the route, not whether a direct prompt may be cut).
    direct_output["direct_fits"] = bool(
        measurement is not None
        and int(measurement.get("input_tokens") or 0) <= int(measurement.get("budget_tokens") or 0)
    )
    print(f"\n[Route] {initial_route.route.upper()}: {initial_route.reason}")
    current_route = initial_route.route
    route_reason = initial_route.reason
    route_cause = "measurement"
    route_history: List[Dict[str, Any]] = []

    # Main loop: [Orchestrator -> Executor ->] Predictor -> Critic
    iteration = 1
    previous_feedback = None
    final_prediction = None
    final_evaluation = None
    final_executor_output = None
    final_plan = None
    attempts: List[Dict[str, Any]] = []

    while iteration <= max_iterations:
        print(f"\n{'='*70}")
        print(f"  ITERATION {iteration}/{max_iterations} ({current_route} route)")
        print(f"{'='*70}")

        attempt_route = initial_route.to_dict()
        attempt_route["route"] = current_route
        attempt_route["escalated"] = current_route != initial_route.route
        attempt_route["reason"] = route_reason
        attempt_route["cause"] = route_cause
        attempt_route["iteration"] = iteration
        route_history.append(attempt_route)
        if interactive_ui:
            ui.on_route_decision(attempt_route, iteration=iteration)

        if current_route == "direct":
            plan = _direct_plan(
                participant_id=participant_id,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=prediction_task_spec,
                domains=list(direct_output.get("domains_processed") or []),
                reason=attempt_route["reason"],
                iteration=iteration,
            )
            executor_output = dict(direct_output)
            final_executor_output = executor_output
            final_plan = plan
            exec_logger.log_orchestrator(
                {"plan_id": plan.plan_id, "total_steps": 0, "priority_domains": plan.priority_domains}
            )
            decision_trace.record_orchestrator_plan(
                domains=plan.priority_domains, num_steps=0, reasoning=plan.reasoning[:500]
            )
            print(f"\n[3/5] Orchestration skipped ({attempt_route['reason']})")
        else:
            plan, executor_output = _run_orchestrated_attempt(
                orchestrator=orchestrator,
                executor=executor,
                participant_data=participant_data,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=prediction_task_spec,
                previous_feedback=previous_feedback,
                iteration=iteration,
                runtime_agent_instructions=runtime_agent_instructions,
                exec_logger=exec_logger,
                decision_trace=decision_trace,
                interactive_ui=interactive_ui,
                settings=settings,
            )
            final_executor_output = executor_output
            final_plan = plan
            # The parsed deviation tree carries the score kinds the renderers need.
            executor_output.setdefault("deviation_tree", direct_output.get("deviation_tree"))

        # Step 5: Predictor makes prediction
        if interactive_ui:
            ui.set_status(
                "Predictor generating phenotype outputs..."
                if predictor_kind == "llm"
                else "Decision model answering typed questions...",
                stage=4,
                iteration=iteration,
            )
        if interactive_ui:
            ui.on_step_start(
                step_id=910 + iteration,
                tool_name="Predictor Agent",
                description=(
                    "Evaluating integrated evidence for final phenotype prediction..."
                    if predictor_kind == "llm"
                    else f"Structured decision on {settings.models.predictor_model}..."
                ),
                stage=4,
            )

        try:
            prediction = predictor.execute(
                executor_output=executor_output,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=prediction_task_spec,
                iteration=iteration
            )
        except Exception as exc:
            if current_route != "direct" or mode == "never" or not is_context_length_error(exc):
                raise
            # The provider rejected the direct prompt as too long (its real window
            # is smaller than the configured one): orchestrate this attempt instead.
            print(f"[Route] Direct prompt rejected as too long ({exc}); orchestrating this attempt.")
            if interactive_ui:
                ui.on_step_failed(step_id=910 + iteration, error=f"Direct prompt rejected as too long: {exc}")
            current_route = "orchestrated"
            route_reason = "The provider rejected the direct prompt as too long; orchestrating."
            route_cause = "context_length"
            route_history[-1].update(route="orchestrated", reason=route_reason, cause=route_cause, escalated=True)
            # Critic feedback meant for a direct revision goes to the Orchestrator now.
            if getattr(predictor, "revision_feedback", ""):
                previous_feedback = previous_feedback or predictor.revision_feedback
                predictor.set_revision_feedback("")
            if interactive_ui:
                ui.on_route_decision(route_history[-1], iteration=iteration)
            plan, executor_output = _run_orchestrated_attempt(
                orchestrator=orchestrator,
                executor=executor,
                participant_data=participant_data,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=prediction_task_spec,
                previous_feedback=previous_feedback,
                iteration=iteration,
                runtime_agent_instructions=runtime_agent_instructions,
                exec_logger=exec_logger,
                decision_trace=decision_trace,
                interactive_ui=interactive_ui,
                settings=settings,
            )
            final_executor_output = executor_output
            final_plan = plan
            executor_output.setdefault("deviation_tree", direct_output.get("deviation_tree"))
            prediction = predictor.execute(
                executor_output=executor_output,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=prediction_task_spec,
                iteration=iteration
            )
        print(f"[Predictor] Completed for iteration {iteration}")

        dataflow_summary = _build_dataflow_summary(
            executor_output=executor_output,
            target_condition=target_condition,
            control_condition=control_condition,
            prediction_task_spec=prediction_task_spec,
            iteration=iteration,
            agent_instructions=runtime_agent_instructions,
        )
        executor_output["dataflow_summary"] = dataflow_summary
        exec_logger.log_dataflow_summary(dataflow_summary, iteration=iteration)
        
        primary_label = _prediction_primary_label(prediction)
        display_probability = _prediction_display_probability(prediction)
        exec_logger.log_predictor({
            "classification": primary_label,
            "probability": display_probability,
            "prediction_task_mode": (
                prediction_task_spec.root.mode.value if prediction_task_spec is not None else None
            ),
        })
        
        if interactive_ui:
            ui.on_prediction(
                classification=primary_label,
                probability=display_probability,
                confidence=prediction.confidence_level.value,
                prediction_payload=(
                    prediction.model_dump() if hasattr(prediction, "model_dump") else prediction.dict()
                ),
            )
        
        decision_trace.record_prediction(
            classification=primary_label,
            probability=display_probability or 0.0,
            key_findings=[f.finding for f in prediction.key_findings[:3]],
            reasoning=prediction.clinical_summary[:500]
        )
        
        # Step 6: Critic evaluates
        if interactive_ui: ui.set_status("Critic Evaluating...", stage=5)
        print(f"\n[6/6] Critic evaluating prediction...")
        # Pass FULL data overview as dictionary (User Requirement)
        data_overview_dict = participant_data.data_overview.model_dump()
        
        evaluation = critic.execute(
            prediction=prediction,
            executor_output=executor_output,
            data_overview=data_overview_dict,
            hierarchical_deviation=participant_data.hierarchical_deviation.model_dump(),
            non_numerical_data=participant_data.non_numerical_data.raw_text,
            control_condition=control_condition,
            prediction_task_spec=prediction_task_spec,
        )
        print(
            f"[Critic] Verdict={evaluation.verdict.value} "
            f"confidence={evaluation.confidence_in_verdict}"
        )
        
        exec_logger.log_critic({
            "verdict": evaluation.verdict.value,
            "confidence": evaluation.confidence_in_verdict
        })
        
        decision_trace.record_critic_verdict(
            verdict=evaluation.verdict.value,
            checklist_passed=evaluation.checklist.pass_count,
            checklist_total=evaluation.checklist.total_count,
            reasoning=evaluation.reasoning[:500]
        )

        if interactive_ui:
            if hasattr(evaluation.checklist, "model_dump"):
                checklist_data = evaluation.checklist.model_dump()
            elif hasattr(evaluation.checklist, "dict"):
                checklist_data = evaluation.checklist.dict()
            else:
                checklist_data = {}
            checklist_data["pass_count"] = evaluation.checklist.pass_count
            checklist_data["total_count"] = evaluation.checklist.total_count
            
            improvements = []
            for s in evaluation.improvement_suggestions[:5]:
                if hasattr(s, "model_dump"):
                    improvements.append(s.model_dump())
                elif hasattr(s, "dict"):
                    improvements.append(s.dict())
                else:
                    improvements.append({
                        "issue": getattr(s, "issue", ""),
                        "suggestion": getattr(s, "suggestion", ""),
                        "priority": getattr(s, "priority", "")
                    })
            
            ui.on_critic_verdict(
                verdict=evaluation.verdict.value,
                confidence=evaluation.confidence_in_verdict,
                checklist_passed=evaluation.checklist.pass_count,
                checklist_total=evaluation.checklist.total_count,
                summary=evaluation.concise_summary or evaluation.reasoning[:240],
                checklist=checklist_data,
                weaknesses=evaluation.weaknesses[:5],
                improvement_suggestions=improvements,
                domains_missed=evaluation.domains_missed[:5],
                composite_score=evaluation.composite_score,
                score_breakdown=evaluation.score_breakdown,
                iteration=iteration,
                fallback_used=getattr(evaluation, "fallback_used", False),
                fallback_reason=getattr(evaluation, "fallback_reason", ""),
                fallback_recommendation=getattr(evaluation, "fallback_recommendation", "")
            )
        
        final_prediction = prediction
        final_evaluation = evaluation
        attempts.append(
            {
                "iteration": iteration,
                "prediction": prediction,
                "evaluation": evaluation,
                "executor_output": executor_output,
                "plan": plan,
                "route": current_route,
            }
        )
        
        # Check if satisfactory / decide on re-orchestration
        if evaluation.verdict == Verdict.SATISFACTORY:
            print(f"\n✓ Prediction deemed SATISFACTORY by Critic")
            break

        print(f"\n✗ Prediction deemed UNSATISFACTORY by Critic")
        if iteration >= max_iterations:
            # Final attempt reached; do not increment `iteration` (keeps accurate count for reports/UI).
            break

        feedback = _format_feedback(evaluation)
        if current_route == "orchestrated":
            print(f"  Re-orchestrating with critic feedback...")
            previous_feedback = feedback
        elif predictor_kind == "llm":
            # Actor-critic on the direct route: the same complete record, with
            # the critic's feedback appended to the Predictor's next prompt.
            print(f"  Revising the direct prediction with critic feedback...")
            predictor.set_revision_feedback(feedback)
        elif mode == "never":
            # A decision model answers the same state the same way; without
            # orchestration there is no other evidence route to try.
            print("  Orchestration is disabled, so a decision model has no other evidence route; stopping.")
            break
        elif not decision_route_can_help(evaluation):
            # The only problem is one the evidence route cannot change (an output
            # without a measurement scale): orchestrating would only add cost.
            print("  The rejection is not about the evidence (missing output scale); stopping.")
            break
        else:
            # Decision critic rejected the direct attempt: escalate to the
            # orchestrated route so the workflow distils the record.
            print(f"  Escalating to the orchestrated route with critic feedback...")
            current_route = "orchestrated"
            route_reason = "The critic rejected the direct attempt; distilling the record with the orchestration workflow."
            route_cause = "critic"
            previous_feedback = feedback
        iteration += 1

    if hasattr(predictor, "set_revision_feedback"):
        predictor.set_revision_feedback("")

    selected_attempt, selection_reason = _select_best_attempt(attempts)
    if not selected_attempt:
        raise RuntimeError("No prediction attempts were generated by the pipeline.")

    selected_iteration = int(selected_attempt["iteration"])
    final_prediction = selected_attempt["prediction"]
    final_evaluation = selected_attempt["evaluation"]
    final_executor_output = selected_attempt["executor_output"]
    final_plan = selected_attempt["plan"]
    coverage_summary = (
        final_executor_output.get("coverage_summary")
        or (final_executor_output.get("coverage_ledger") or {}).get("summary")
        or {}
    )
    
    # Prepare output directory and token usage
    token_usage = token_manager.get_detailed_usage()
    base_output_dir = _resolve_output_dir(participant_dir, participant_id, settings)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    explainability_result = _run_explainability_for_selected_attempt(
        settings=settings,
        participant_id=participant_id,
        target_condition=target_condition,
        control_condition=control_condition,
        prediction_task_spec=prediction_task_spec,
        selected_attempt=selected_attempt,
        feature_space=feature_space,
        output_dir=base_output_dir,
        exec_logger=exec_logger,
    )

    # Generate final report (standard outputs first)
    print(f"\n{'='*70}")
    print(f"  GENERATING FINAL REPORT")
    print(f"{'='*70}")
    
    report_generator = PatientReportGenerator()
    
    # Collect detailed logs from Logger or Trace? 
    # Actually, we need to collect them from the executor results if they are stored there.
    # But executor returns the result of the LAST iteration.
    # We should rely on `exec_logger` to track them across all steps?
    # For now, let's grab them from the final executor_output if available, or just empty list.
    detailed_logs_collection = []
    # (Implementation Note: Ideally we'd aggregate them properly. Basic placeholder for now.)
    
    execution_summary = {
        "iterations": len(attempts),
        "selected_iteration": selected_iteration,
        "selection_reason": selection_reason,
        "coverage_summary": coverage_summary,
        "dataflow_summary": (final_executor_output or {}).get("dataflow_summary", {}),
        "target_condition": target_condition,
        "control_condition": control_condition,
        "prediction_task_spec": (
            prediction_task_spec.model_dump()
            if prediction_task_spec is not None and hasattr(prediction_task_spec, "model_dump")
            else (prediction_task_spec.dict() if prediction_task_spec is not None else None)
        ),
        "agent_instructions": runtime_agent_instructions,
        "tokens_used": token_usage.get("total_tokens", 0),
        "domains_processed": (final_plan.priority_domains if final_plan else plan.priority_domains),
        "detailed_logs": detailed_logs_collection, # We need to populate this
        "explainability": {
            "enabled": bool(explainability_result.get("enabled")),
            "status": explainability_result.get("status"),
            "methods_requested": explainability_result.get("methods_requested") or [],
            "artifact_path": explainability_result.get("artifact_path"),
        },
    }
    
    report = report_generator.generate(
        participant_id=participant_id,
        prediction=final_prediction,
        evaluation=final_evaluation,
        execution_summary=execution_summary,
        decision_trace=decision_trace.get_trace()
    )
    
    # Save outputs to configured output directory
    
    report_generator.save(report, base_output_dir)
    report_generator.save_markdown(report, base_output_dir)
    exec_logger.save_structured_log(base_output_dir / f"execution_log_{participant_id}.json")

    # Log completion duration early so standard reports can include it
    duration_so_far = (datetime.now() - start_time).total_seconds()

    # Generate Performance Report
    final_binary_label = _prediction_primary_label(final_prediction)
    final_probability = final_prediction.probability_score
    final_display_probability = _prediction_display_probability(final_prediction)
    performance_report = {
        "participant_id": participant_id,
        "target_condition": target_condition,
        "control_condition": control_condition,
        "prediction_task_spec": (
            prediction_task_spec.model_dump()
            if prediction_task_spec is not None and hasattr(prediction_task_spec, "model_dump")
            else (prediction_task_spec.dict() if prediction_task_spec is not None else None)
        ),
        "agent_instructions": runtime_agent_instructions,
        "execution_timestamp": start_time.isoformat(),
        "total_duration_seconds": round(duration_so_far, 2),
        "iterations": len(attempts),
        "selected_iteration": selected_iteration,
        "selection_reason": selection_reason,
        "coverage_summary": coverage_summary,
        "dataflow_summary": (final_executor_output or {}).get("dataflow_summary", {}),
        "prediction_result": {
            "classification": final_binary_label,
            "probability": round(float(final_probability), 4) if final_probability is not None else None,
            "root_confidence": round(float(final_display_probability), 4) if final_display_probability is not None else None,
            "confidence": final_prediction.confidence_level.value,
            "primary_output_kind": final_prediction.primary_output_kind,
            "root_prediction": (
                final_prediction.root_prediction.model_dump()
                if final_prediction.root_prediction is not None and hasattr(final_prediction.root_prediction, "model_dump")
                else (final_prediction.root_prediction.dict() if final_prediction.root_prediction is not None else None)
            ),
        },
        "control_condition": control_condition,
        "critic_verdict": final_evaluation.verdict.value,
        # The full evaluation, not just the verdict: the score breakdown, the
        # checklist, and the improvement suggestions are the reviewable part.
        "critic_evaluation": (
            final_evaluation.model_dump(mode="json")
            if hasattr(final_evaluation, "model_dump")
            else final_evaluation.dict()
        ),
        "token_usage": {
            "total_tokens": token_usage.get("total_tokens", 0),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "calls": token_usage.get("calls", [])
        },
        "plan_summary": {
            "plan_id": final_plan.plan_id if final_plan else plan.plan_id,
            "total_steps": final_plan.total_steps if final_plan else plan.total_steps,
            "priority_domains": final_plan.priority_domains if final_plan else plan.priority_domains,
        },
        "routing": {
            "mode": mode,
            "initial": initial_route.to_dict(),
            "attempts": route_history,
            "selected_route": selected_attempt.get("route"),
        },
        "predictor": {
            "kind": predictor_kind,
            "model": settings.models.predictor_model,
            "companion_model": settings.models.orchestrator_model if predictor_kind == "decision" else None,
        },
        "decision": dict(getattr(final_prediction, "decision_report", {}) or {}),
        "explainability": explainability_result,
    }
    
    # Save performance report as JSON
    import json
    performance_report_path = base_output_dir / f"performance_report_{participant_id}.json"
    with open(performance_report_path, 'w') as f:
        json.dump(performance_report, f, indent=2)

    data_overview_dict = participant_data.data_overview.model_dump()
    report_context_note = _build_report_context_note(
        final_evaluation=final_evaluation,
        selected_iteration=selected_iteration,
        selection_reason=selection_reason,
    )
    xai_report_result = {"success": False, "metadata": {}, "path": None, "skipped": False}
    if generate_xai_report:
        xai_report_result = _generate_xai_explainability_report(
            communicator=communicator,
            xai_result=explainability_result,
            prediction=final_prediction,
            evaluation=final_evaluation,
            execution_summary=execution_summary,
            target_condition=target_condition,
            control_condition=control_condition,
            base_output_dir=base_output_dir,
            participant_id=participant_id,
            trigger_source="cli",
            ui_step_id=980 + iteration,
            interactive_ui=interactive_ui,
        )

    deep_report_result = {"success": False, "metadata": {}, "path": None}
    if generate_deep_phenotype and final_prediction and final_evaluation and final_executor_output:
        deep_report_result = _generate_deep_phenotype_report(
            communicator=communicator,
            prediction=final_prediction,
            evaluation=final_evaluation,
            executor_output=final_executor_output,
            data_overview=data_overview_dict,
            execution_summary=execution_summary,
            control_condition=control_condition,
            report_context_note=report_context_note,
            base_output_dir=base_output_dir,
            participant_id=participant_id,
            user_focus_modalities=deep_report_focus_modalities,
            user_general_instruction=deep_report_general_instruction,
            trigger_source="cli",
            ui_step_id=930 + iteration,
            interactive_ui=interactive_ui,
        )
    performance_report["xai_report"] = {
        "generated": bool(xai_report_result.get("success")),
        "path": xai_report_result.get("path"),
        "trigger_source": "cli" if generate_xai_report else None,
        "skipped": bool(xai_report_result.get("skipped", False)),
        "reason": xai_report_result.get("reason"),
        "metadata": xai_report_result.get("metadata") or {},
    }
    performance_report["deep_phenotype"] = {
        "generated": bool(deep_report_result.get("success")),
        "path": deep_report_result.get("path"),
        "trigger_source": "cli" if generate_deep_phenotype else None,
        "metadata": deep_report_result.get("metadata") or {},
    }
    with open(performance_report_path, 'w') as f:
        json.dump(performance_report, f, indent=2)

    # Log completion
    duration = (datetime.now() - start_time).total_seconds()
    if abs(duration - duration_so_far) > 0.05:
        performance_report["total_duration_seconds"] = round(duration, 2)
        with open(performance_report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)

    exec_logger.log_pipeline_end(
        success=True,
        summary={
            "prediction": final_binary_label,
            "probability": final_display_probability,
            "iterations": len(attempts),
            "duration_seconds": duration
        }
    )

    if interactive_ui:
        ui.on_pipeline_complete(
            result=final_binary_label,
            probability=final_display_probability,
            iterations=len(attempts),
            total_duration_secs=duration,
            total_tokens=token_usage.get("total_tokens", 0),
            prediction_payload=(
                final_prediction.model_dump() if hasattr(final_prediction, "model_dump") else final_prediction.dict()
            ),
        )
    
    print(f"\n{'='*70}")
    print(f"  COMPASS PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Participant: {participant_id}")
    print(f"  Prediction: {final_binary_label}")
    if final_prediction.binary_classification is not None and final_probability is not None:
        print(f"  Probability: {final_probability:.1%}")
    elif final_display_probability is not None:
        print(f"  Root Confidence: {final_display_probability:.1%}")
    else:
        print(f"  Probability: N/A (non-binary primary output)")
    print(f"  Iterations: {len(attempts)} (selected iteration {selected_iteration})")
    print(f"  Selection reason: {selection_reason}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Total Tokens: {token_usage.get('total_tokens', 0)}")
    print(f"  Output: {base_output_dir}")
    print(f"{'='*70}\n")
    
    return {
        "participant_id": participant_id,
        "prediction": final_binary_label,
        "probability": final_display_probability,
        "binary_probability": final_probability,
        "root_confidence": final_display_probability,
        "confidence": final_prediction.confidence_level.value,
        "verdict": final_evaluation.verdict.value,
        "iterations": len(attempts),
        "selected_iteration": selected_iteration,
        "selection_reason": selection_reason,
        "prediction_task_spec": (
            prediction_task_spec.model_dump()
            if prediction_task_spec is not None and hasattr(prediction_task_spec, "model_dump")
            else (prediction_task_spec.dict() if prediction_task_spec is not None else None)
        ),
        "control_condition": control_condition,
        "coverage_summary": coverage_summary,
        "input_route": selected_attempt.get("route"),
        "predictor_kind": predictor_kind,
        "routing": performance_report.get("routing"),
        "duration_seconds": duration,
        "output_dir": str(base_output_dir),
        "report": report,
        "deep_phenotype_generated": bool(deep_report_result.get("success")),
        "deep_phenotype_path": deep_report_result.get("path"),
        "xai_report_generated": bool(xai_report_result.get("success")),
        "xai_report_path": xai_report_result.get("path"),
        "explainability": explainability_result,
        "internal_context": {
            "participant_id": participant_id,
            "prediction": final_prediction,
            "evaluation": final_evaluation,
            "executor_output": final_executor_output,
            "data_overview": data_overview_dict,
            "execution_summary": execution_summary,
            "control_condition": control_condition,
            "prediction_task_spec": (
                prediction_task_spec.model_dump()
                if prediction_task_spec is not None and hasattr(prediction_task_spec, "model_dump")
                else (prediction_task_spec.dict() if prediction_task_spec is not None else None)
            ),
            "agent_instructions": runtime_agent_instructions,
            "report_context_note": report_context_note,
            "base_output_dir": str(base_output_dir),
            "explainability": explainability_result,
            "xai_report": xai_report_result,
        },
    }


def run_dataflow_audit(
    participant_dir: Path,
    target_condition: str,
    control_condition: str = DEFAULT_CONTROL_CONDITION,
    prediction_task_spec: Optional[PredictionTaskSpec] = None,
    verbose: bool = True,
) -> dict:
    """
    Offline dataflow audit: build predictor payload and chunking without LLM calls.
    """
    settings = get_settings()
    if prediction_task_spec is None:
        prediction_task_spec = build_binary_task_spec(
            target_label=target_condition,
            control_label=control_condition,
        )
    target_condition, control_condition = _task_spec_to_legacy_labels(prediction_task_spec)
    data_loader = DataLoader()
    participant_data = data_loader.load(participant_dir)

    # Audit mode is intentionally offline. Bypass agent constructors because
    # they initialize the configured LLM client even though these serialization
    # and coverage helpers do not perform model calls.
    executor = Executor.__new__(Executor)
    context = executor._build_context(
        participant_data,
        target_condition,
        control_condition,
        prediction_task_spec=prediction_task_spec,
    )

    fusion_layer = FusionLayer.__new__(FusionLayer)
    pass_through = FusionResult(
        fused_narrative="Audit pass-through",
        domain_summaries={},
        key_findings=[],
        cross_modal_patterns=[],
        evidence_summary={"for_case": [], "for_control": []},
        tokens_used=0,
        source_outputs=[],
        skipped_fusion=True,
        raw_multimodal_data=context.get("multimodal_data") or {},
        raw_processed_multimodal_data=None,
        raw_step_outputs={},
        context_fill_report={"audit": True},
    )
    predictor_input = fusion_layer.compress_for_predictor(
        fusion_result=pass_through,
        hierarchical_deviation=context.get("hierarchical_deviation") or {},
        non_numerical_data=context.get("non_numerical_data") or "",
    )

    coverage_ledger = executor._build_coverage_ledger(
        multimodal_data=context.get("multimodal_data") or {},
        step_outputs={},
        predictor_input=predictor_input,
    )
    predictor_input["coverage_ledger"] = coverage_ledger

    max_tool_input = int(getattr(settings.token_budget, "max_tool_input_tokens", 30000) or 30000)
    chunk_budget = max(30000, min(60000, int(max_tool_input * 2.0)))
    assembler = PredictorInputAssembler(max_chunk_tokens=chunk_budget, model_hint=settings.models.tool_model)
    executor_stub = {
        "step_outputs": {},
        "data_overview": context.get("data_overview") or {},
        "hierarchical_deviation": context.get("hierarchical_deviation") or {},
        "non_numerical_data": context.get("non_numerical_data") or "",
    }
    sections = assembler.build_sections(
        executor_output=executor_stub,
        predictor_input=predictor_input,
        coverage_ledger=coverage_ledger,
    )
    core_names = {
        "non_numerical_data_raw",
        "hierarchical_deviation_raw",
        "data_overview",
        "phenotype_representation",
        "feature_synthesizer",
        "differential_diagnosis",
    }
    def _is_core(name: str) -> bool:
        base = name.split("#", 1)[0]
        return base in core_names
    chunk_sections = [s for s in sections if not _is_core(s.name)]
    chunks = assembler.build_chunks(chunk_sections)

    section_stats = []
    for sec in sections:
        section_stats.append(
            {
                "name": sec.name,
                "tokens": count_tokens(sec.text, model_hint=settings.models.tool_model),
                "feature_key_count": len(sec.feature_keys),
            }
        )

    chunk_stats = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_text = assembler.chunk_to_text(chunk, idx, len(chunks))
        chunk_stats.append(
            {
                "chunk_index": idx,
                "sections": [s.name for s in chunk],
                "tokens": count_tokens(chunk_text, model_hint=settings.models.tool_model),
            }
        )

    try:
        payload_tokens = len(
            fusion_layer.encoder.encode(json.dumps(predictor_input, default=str))
        )
    except Exception:
        payload_tokens = count_tokens(str(predictor_input), model_hint=settings.models.tool_model)

    report = {
        "participant_id": participant_data.participant_id,
        "target_condition": target_condition,
        "control_condition": control_condition,
        "prediction_task_root_mode": (
            prediction_task_spec.root.mode.value if prediction_task_spec is not None else None
        ),
        "prediction_task_node_count": (
            len(prediction_task_spec.node_index()) if prediction_task_spec is not None else 0
        ),
        "coverage_summary": coverage_ledger.get("summary", {}),
        "predictor_payload_tokens": payload_tokens,
        "chunk_budget_tokens": chunk_budget,
        "chunk_count": len(chunks),
        "section_stats": section_stats,
        "chunk_stats": chunk_stats,
        "predictor_input_mode": predictor_input.get("mode"),
    }
    report["assertions"] = {
        "predictor_input_mode_present": bool(str(report.get("predictor_input_mode") or "").strip()),
        "chunk_count_non_negative": int(report.get("chunk_count") or 0) >= 0,
        "coverage_summary_present": isinstance(report.get("coverage_summary"), dict),
        "task_mode_present": bool(str(report.get("prediction_task_root_mode") or "").strip()),
    }
    report["assertions_ok"] = all(bool(v) for v in report["assertions"].values())

    output_dir = _resolve_output_dir(participant_dir, participant_data.participant_id, settings)
    output_path = output_dir / f"dataflow_audit_{participant_data.participant_id}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        fallback_dir = settings.paths.logs_dir
        fallback_dir.mkdir(parents=True, exist_ok=True)
        output_path = fallback_dir / f"dataflow_audit_{participant_data.participant_id}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    if verbose:
        print(f"[Audit] Dataflow audit saved: {output_path}")
        print(f"[Audit] Payload tokens: {payload_tokens}")
        print(f"[Audit] Chunk count: {len(chunks)} (budget {chunk_budget})")

    return report


def _format_feedback(evaluation) -> str:
    """Format critic evaluation as feedback for re-orchestration."""
    lines = [
        f"Previous prediction was deemed {evaluation.verdict.value}.",
        "",
        "Weaknesses identified:",
    ]
    
    for weakness in evaluation.weaknesses[:3]:
        lines.append(f"- {weakness}")
    
    lines.append("")
    lines.append("Suggested improvements:")
    
    for sugg in evaluation.high_priority_issues[:3]:
        lines.append(f"- [{sugg.priority.value}] {sugg.issue}: {sugg.suggestion}")
    
    if evaluation.domains_missed:
        lines.append("")
        lines.append(f"Domains missed: {', '.join(evaluation.domains_missed)}")
    
    return "\n".join(lines)


def _build_dataflow_summary(
    *,
    executor_output: Dict[str, Any],
    target_condition: str,
    control_condition: str,
    prediction_task_spec: Optional[PredictionTaskSpec],
    iteration: int,
    agent_instructions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    predictor_input = executor_output.get("predictor_input") or {}
    coverage_ledger = executor_output.get("coverage_ledger") or {}
    coverage_summary = (
        executor_output.get("coverage_summary")
        or coverage_ledger.get("summary")
        or {}
    )
    context_fill_report = predictor_input.get("context_fill_report") or {}
    chunk_evidence = executor_output.get("chunk_evidence") or []
    predictor_chunk_count = int(
        executor_output.get("predictor_chunk_count") or len(chunk_evidence) or 0
    )
    chunk_evidence_count = len(chunk_evidence)

    processed_raw_included = context_fill_report.get("processed_raw_full_included")
    processed_raw_present = bool(predictor_input.get("multimodal_processed_raw_low_priority"))

    missing_count = (
        coverage_summary.get("missing_feature_count")
        or coverage_summary.get("missing_count")
        or 0
    )
    invariant_ok = coverage_summary.get("invariant_ok")
    if invariant_ok is None:
        invariant_ok = int(missing_count) == 0

    assertions = {
        "invariant_ok": bool(invariant_ok),
        "missing_feature_count_zero": int(missing_count) == 0,
        "chunk_evidence_matches_count": (
            predictor_chunk_count == 0 or chunk_evidence_count == predictor_chunk_count
        ),
        "processed_raw_flag_consistent": (
            processed_raw_included is None or processed_raw_included == processed_raw_present
        ),
    }

    payload_estimate = context_fill_report.get("predictor_payload_estimate") or {}
    coverage_block = {
        "summary": coverage_summary,
        "forced_raw_count": len(coverage_ledger.get("forced_raw_features") or []),
    }
    chunking_block = {
        "predictor_chunk_count": predictor_chunk_count,
        "chunk_evidence_count": chunk_evidence_count,
        "chunked_two_pass_required": payload_estimate.get("chunked_two_pass_required"),
        "single_chunk_limit": payload_estimate.get("single_chunk_limit"),
        "chunking_skipped": bool(executor_output.get("chunking_skipped")),
        "chunking_reason": executor_output.get("chunking_reason"),
    }
    context_block = {
        "processed_raw_full_included": processed_raw_included,
        "rag_added_count": context_fill_report.get("added_count")
            or context_fill_report.get("top_added_count"),
        "predictor_payload_estimate": payload_estimate,
        "coverage_snapshot": context_fill_report.get("coverage"),
        "embedding_store": context_fill_report.get("embedding_store"),
    }

    return {
        "iteration": iteration,
        "target_condition": target_condition,
        "control_condition": control_condition,
        "prediction_task_root_mode": (
            prediction_task_spec.root.mode.value if prediction_task_spec is not None else None
        ),
        "agent_instruction_flags": {
            key: bool(str(value or "").strip())
            for key, value in _normalize_agent_instructions(agent_instructions).items()
        },
        "predictor_input_mode": predictor_input.get("mode"),
        "coverage": coverage_block,
        "chunking": chunking_block,
        "context_fill": context_block,
        "assertions": assertions,
    }


def _run_explainability_for_selected_attempt(
    *,
    settings,
    participant_id: str,
    target_condition: str,
    control_condition: str,
    prediction_task_spec: Optional[PredictionTaskSpec] = None,
    selected_attempt: Dict[str, Any],
    feature_space: Dict[str, Any],
    output_dir: Path,
    exec_logger: ExecutionLogger,
) -> Dict[str, Any]:
    if not bool(getattr(settings.explainability, "enabled", False)):
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "Explainability disabled.",
            "methods_requested": [],
            "methods": {},
        }
    if getattr(selected_attempt.get("prediction"), "predictor_kind", "llm") == "decision":
        result = {
            "enabled": True,
            "status": "skipped",
            "reason": "XAI attribution methods need a text-generating Predictor; a structured decision model reports per-order answers and probabilities instead.",
            "methods_requested": list(getattr(settings.explainability, "methods", []) or []),
            "methods": {},
        }
        try:
            exec_logger.log_explainability(result)
        except Exception:
            pass
        return result
    if prediction_task_spec is not None and not prediction_task_spec.is_pure_binary_root():
        result = {
            "enabled": True,
            "status": "skipped",
            "reason": "XAI currently supports binary classification only.",
            "methods_requested": list(getattr(settings.explainability, "methods", []) or []),
            "methods": {},
            "task_mode": prediction_task_spec.root.mode.value,
        }
        try:
            exec_logger.log_explainability(result)
        except Exception:
            pass
        return result

    try:
        result = run_explainability_methods(
            settings=settings,
            participant_id=participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            selected_attempt=selected_attempt,
            feature_space=feature_space,
            output_dir=output_dir,
        )
    except Exception as exc:
        result = {
            "enabled": True,
            "status": "failed",
            "reason": f"Runner failure: {exc}",
            "methods_requested": list(getattr(settings.explainability, "methods", []) or []),
            "methods": {},
        }

    try:
        exec_logger.log_explainability(result)
    except Exception:
        pass
    return result


def _select_best_attempt(attempts: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Select final attempt:
    1) Any SATISFACTORY verdict wins (highest composite/checklist among them).
    2) Otherwise choose highest critic composite/checklist score.
    """
    if not attempts:
        return None, "No attempts available"

    def _quality_tuple(item: Dict[str, Any]) -> Tuple[float, float, float]:
        evaluation = item.get("evaluation")
        if evaluation is None:
            return (0.0, 0.0, 0.0)
        composite = float(getattr(evaluation, "composite_score", 0.0) or 0.0)
        checklist = float(getattr(getattr(evaluation, "checklist", None), "pass_count", 0) or 0.0)
        confidence = float(getattr(evaluation, "confidence_in_verdict", 0.0) or 0.0)
        return (composite, checklist, confidence)

    satisfactory = [a for a in attempts if getattr(a.get("evaluation"), "verdict", None) == Verdict.SATISFACTORY]
    if satisfactory:
        selected = max(satisfactory, key=_quality_tuple)
        iteration = selected.get("iteration", "?")
        return selected, f"Satisfactory verdict available; chose strongest satisfactory attempt (iteration {iteration})."

    selected = max(attempts, key=_quality_tuple)
    score = _quality_tuple(selected)
    iteration = selected.get("iteration", "?")
    return (
        selected,
        "No satisfactory verdict; selected best unsatisfactory attempt by critic composite/checklist "
        f"(iteration {iteration}, composite={score[0]:.3f}, checklist={score[1]:.0f}).",
    )


DEFAULT_COMPANION_MODEL = "deepseek/deepseek-v4-flash-0731"


def _ask_companion_model(decision_model: str) -> str:
    """Ask which conventional LLM runs the other roles (terminal only)."""
    prompt = (
        f"\n{decision_model} is a structured decision model: it can only serve the Predictor.\n"
        "The Orchestrator, tools, Integrator, Critic and Communicator need a conventional LLM.\n"
        f"Companion LLM [{DEFAULT_COMPANION_MODEL}]: "
    )
    if not sys.stdin or not sys.stdin.isatty():
        print(f"[Init] {decision_model} serves the Predictor; other roles use {DEFAULT_COMPANION_MODEL}.")
        return DEFAULT_COMPANION_MODEL
    try:
        answer = input(prompt).strip()
    except EOFError:
        answer = ""
    return answer or DEFAULT_COMPANION_MODEL


def _apply_predictor_choice(settings, args: argparse.Namespace, *, interactive: bool = True) -> None:
    """Resolve the Predictor model, the companion LLM and the routing options from the CLI."""
    predictor_model = str(getattr(args, "predictor_model", None) or "").strip()
    companion = str(getattr(args, "companion_model", None) or "").strip()
    public = str(getattr(args, "public_model", "") or "").strip()
    if settings.models.backend != LLMBackend.LOCAL and is_decision_model(public):
        predictor_model = predictor_model or public
        if not companion:
            companion = _ask_companion_model(public) if interactive else DEFAULT_COMPANION_MODEL
    if companion:
        if is_decision_model(companion):
            raise ValueError(f"--companion_model must be a conventional LLM, not the decision model {companion}.")
        for role in ("orchestrator", "critic", "integrator", "communicator", "tool"):
            setattr(settings.models, f"{role}_model", companion)
        settings.models.public_model_name = companion
        if is_decision_model(settings.models.predictor_model):
            settings.models.predictor_model = companion
    if predictor_model:
        settings.models.predictor_model = predictor_model
    enforce_role_models(settings)

    if getattr(args, "orchestration", None):
        settings.orchestration.mode = normalize_orchestration_mode(args.orchestration)
    if getattr(args, "orchestration_threshold", None) is not None:
        settings.orchestration.threshold_tokens = max(0, int(args.orchestration_threshold))
    if getattr(args, "decision_provider", None):
        settings.decision.provider = str(args.decision_provider)
    if getattr(args, "decision_choice_orders", None) is not None:
        settings.decision.choice_orders = max(1, int(args.decision_choice_orders))
    if getattr(args, "decision_score_levels", None) is not None:
        settings.decision.score_levels = max(2, min(10, int(args.decision_score_levels)))
    if getattr(args, "decision_refine", None) is not None:
        settings.decision.regression_refine = bool(args.decision_refine)
    if getattr(args, "decision_compiler_model", None):
        settings.decision.compiler_model = str(args.decision_compiler_model).strip()
    if getattr(args, "decision_stability_threshold", None) is not None:
        settings.decision.stability_threshold = float(args.decision_stability_threshold)
    if getattr(args, "decision_regression_stability_threshold", None) is not None:
        settings.decision.regression_stability_threshold = float(args.decision_regression_stability_threshold)
    if getattr(args, "decision_sufficiency_threshold", None) is not None:
        settings.decision.sufficiency_threshold = float(args.decision_sufficiency_threshold)


#: Accepted ranges of the numeric CLI options, (minimum, maximum); None is open.
_CLI_RANGES = {
    "iterations": (1, 20),
    "orchestration_threshold": (0, None),
    "public_max_context_tokens": (0, None),
    "context_window": (1024, None),
    "max_tokens": (1024, None),
    "local_max_model_len": (0, None),
    "local_tensor_parallel": (1, None),
    "local_pipeline_parallel": (1, None),
    "decision_choice_orders": (1, 6),
    "decision_score_levels": (2, 10),
    "decision_stability_threshold": (0.05, 0.6),
    "decision_regression_stability_threshold": (0.1, 2.0),
    "decision_sufficiency_threshold": (0.0, 0.9),
    "total_budget": (1, None),
    "max_agent_input": (1, None),
    "max_agent_output": (1, None),
    "max_tool_input": (1, None),
    "max_tool_output": (1, None),
}


def _cli_role_context_windows(args: argparse.Namespace) -> Dict[str, int]:
    """--context_window (every role) plus --role_context_window pairs, the latter winning per role."""
    from src.full_stack.backend.config.settings import ALL_ROLES_KEY, parse_role_context_windows

    windows: Dict[str, int] = {}
    if getattr(args, "context_window", None):
        windows[ALL_ROLES_KEY] = int(args.context_window)
    windows.update(parse_role_context_windows(getattr(args, "role_context_window", None)))
    return windows


def validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse an inconsistent command line with a message that names the flag (exit code 2)."""
    for name, (low, high) in _CLI_RANGES.items():
        value = getattr(args, name, None)
        if value is None:
            continue
        if name == "public_max_context_tokens" and 0 < value < 1024:
            parser.error(f"--{name} must be 0 (not configured) or at least 1024 tokens; got {value}.")
        if (low is not None and value < low) or (high is not None and value > high):
            bounds = f"between {low} and {high}" if high is not None else f"at least {low}"
            parser.error(f"--{name} must be {bounds}; got {value}.")
    if getattr(args, "local_max_model_len", 0) and 0 < args.local_max_model_len < 1024:
        parser.error(f"--local_max_model_len must be 0 (use --max_tokens) or at least 1024; got {args.local_max_model_len}.")
    try:
        _cli_role_context_windows(args)
    except ValueError as exc:
        parser.error(f"--role_context_window: {exc}")

    companion = str(getattr(args, "companion_model", None) or "").strip()
    if companion and is_decision_model(companion):
        parser.error(f"--companion_model must be a conventional LLM, not the decision model {companion}.")
    compiler = str(getattr(args, "decision_compiler_model", None) or "").strip()
    if compiler and is_decision_model(compiler):
        parser.error(f"--decision_compiler_model must be a conventional LLM, not the decision model {compiler}.")
    predictor = str(getattr(args, "predictor_model", None) or "").strip()
    if getattr(args, "backend", "openrouter") == "local":
        local_model = str(getattr(args, "model", "") or "").strip()
        if is_decision_model(local_model):
            parser.error(
                f"--model {local_model} is a structured decision model; on --backend local --model is the "
                "conventional LLM every role runs on. Put the decision model in --predictor_model."
            )
        if predictor and not is_decision_model(predictor) and predictor != local_model:
            parser.error(
                f"--predictor_model {predictor} cannot run on --backend local, which serves only --model "
                f"{local_model} in process. Use a structured decision model (served remotely), or drop the flag."
            )
        if companion:
            parser.error("--companion_model applies to hosted backends; on --backend local every LLM role uses --model.")
    elif predictor and is_decision_model(str(getattr(args, "public_model", "") or "")) and is_decision_model(predictor):
        public = str(args.public_model)
        if public.lower().lstrip("~") != predictor.lower().lstrip("~"):
            parser.error(f"--public_model {public} and --predictor_model {predictor} are two different decision models; name one.")


def cli_notes(args: argparse.Namespace, settings) -> List[str]:
    """Flags that are accepted but have no effect with this configuration."""
    notes: List[str] = []
    if int(settings.orchestration.threshold_tokens or 0) > 0 and settings.orchestration.mode != "auto":
        notes.append(
            f"--orchestration_threshold has no effect with --orchestration {settings.orchestration.mode} "
            "(only auto mode routes on a threshold)."
        )
    decision_flags = [
        f"--{name}"
        for name in (
            "decision_provider",
            "decision_choice_orders",
            "decision_score_levels",
            "decision_refine",
            "decision_compiler_model",
            "decision_stability_threshold",
            "decision_regression_stability_threshold",
            "decision_sufficiency_threshold",
        )
        if getattr(args, name, None) is not None
    ]
    if decision_flags and not is_decision_model(settings.models.predictor_model):
        notes.append(f"{', '.join(decision_flags)} only apply when the Predictor is a structured decision model.")
    if getattr(args, "backend", "") == "local" and int(getattr(args, "public_max_context_tokens", 0) or 0):
        notes.append("--public_max_context_tokens has no effect on --backend local (the served length is the window).")
    elif int(getattr(args, "public_max_context_tokens", 0) or 0):
        # Older versions applied it to the public model whatever its window; now
        # it only covers a model neither the catalog nor the built-in table knows.
        unused = all(
            window.source != "configured" for window in settings.role_context_windows().values()
        )
        if unused:
            notes.append(
                f"--public_max_context_tokens {int(args.public_max_context_tokens):,} has no effect: every role's "
                "model has a known window (catalog, built-in table or override); force one with --context_window "
                "or --role_context_window."
            )
    if getattr(args, "backend", "") == "local":
        served = int(getattr(args, "local_max_model_len", 0) or 0) or int(getattr(args, "max_tokens", 0) or 0)
        forced = {role: tokens for role, tokens in (settings.models.role_context_windows or {}).items() if tokens > served > 0}
        if forced:
            listed = ", ".join(f"{role}={tokens:,}" for role, tokens in sorted(forced.items()))
            notes.append(
                f"Context window override above the served length {served:,} ({listed}): the local server rejects "
                "longer prompts; lower the override or raise --local_max_model_len."
            )
    return notes


def apply_cli_settings(settings, args: argparse.Namespace, *, interactive: bool = True, print_notes: bool = True):
    """Stamp the engine settings with the command line, exactly as a run uses them."""
    settings.detailed_tool_logging = bool(getattr(args, "detailed_log", False))
    if getattr(args, "reasoning_effort", None) is not None:
        settings.reasoning_effort = "" if args.reasoning_effort == "provider_default" else args.reasoning_effort

    roles = ("orchestrator", "critic", "predictor", "integrator", "communicator", "tool")
    if args.backend == "local":
        settings.models.backend = LLMBackend.LOCAL
        settings.models.local_model_name = args.model
        settings.models.local_max_tokens = args.max_tokens
        settings.models.local_backend_type = args.local_engine
        settings.models.local_dtype = args.local_dtype
        if args.local_quant is not None:
            settings.models.local_quantization = args.local_quant
        settings.models.local_kv_cache_dtype = args.local_kv_cache_dtype
        settings.models.local_tensor_parallel_size = args.local_tensor_parallel
        settings.models.local_pipeline_parallel_size = args.local_pipeline_parallel
        settings.models.local_gpu_memory_utilization = args.local_gpu_mem_util
        settings.models.local_max_model_len = args.local_max_model_len
        if args.local_enforce_eager:
            settings.models.local_enforce_eager = True
        if args.local_trust_remote_code:
            settings.models.local_trust_remote_code = True
        settings.models.local_attn_implementation = args.local_attn
        for role in roles:
            setattr(settings.models, f"{role}_model", args.model)
        if not getattr(args, "check_config", False):
            print(f"[Init] Switching to LOCAL Backend with model: {args.model}")
    else:
        settings.models.backend = LLMBackend.OPENAI if args.backend == "openai" else LLMBackend.OPENROUTER
        settings.models.public_model_name = args.public_model
        settings.models.public_max_context_tokens = int(args.public_max_context_tokens or 0)
        settings.models.embedding_model = args.embedding_model
        for role in roles:
            setattr(settings.models, f"{role}_model", args.public_model)
    settings.models.role_context_windows = _cli_role_context_windows(args)

    _apply_predictor_choice(settings, args, interactive=interactive)

    # Token limits: derived from the resolved context window unless set explicitly.
    _apply_token_budget_defaults(
        settings,
        {
            "max_agent_input": getattr(args, "max_agent_input", None),
            "max_agent_output": getattr(args, "max_agent_output", None),
            "max_tool_input": getattr(args, "max_tool_input", None),
            "max_tool_output": getattr(args, "max_tool_output", None),
        },
    )
    if getattr(args, "total_budget", None):
        settings.token_budget.total_budget = args.total_budget
    _sync_component_token_budgets(settings)
    _sync_role_token_limits_with_budgets(settings)
    # Warnings go to stderr, so `--check_config --check_format json` keeps a clean stdout.
    # The configuration check lists them in its own report instead.
    if print_notes:
        for note in cli_notes(args, settings):
            print(f"[Init] Note: {note}", file=sys.stderr)
    return settings


def run_config_check(
    settings,
    *,
    participant_dir: Optional[Path] = None,
    prediction_task_spec: Optional[PredictionTaskSpec] = None,
    agent_instructions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Everything a run with these settings would use, checked offline: no model
    call, no network. With a participant directory, the record is measured the
    way the pipeline measures it before its first attempt, and the route the
    configured orchestration mode takes is reported.
    """
    from src.full_stack.backend.utils.core.route_preview import configuration_summary, measure_record

    import contextlib

    report = configuration_summary(settings)
    report["record"] = None
    if participant_dir:
        try:
            # The loaders narrate on stdout; keep stdout for the report itself.
            with contextlib.redirect_stdout(sys.stderr):
                report["record"] = measure_record(
                    Path(participant_dir),
                    settings=settings,
                    prediction_task_spec=prediction_task_spec,
                    agent_instructions=agent_instructions,
                )
        except Exception as exc:
            report["problems"].append(f"The record at {participant_dir} could not be measured: {type(exc).__name__}: {exc}")
    return report


def format_config_check(report: Dict[str, Any]) -> str:
    """The --check_config report as text."""
    lines = ["COMPASS configuration check (offline: no model was called)", ""]
    lines.append(f"Backend: {report['backend']}    endpoint: {report['endpoint']}")
    if report.get("local"):
        local = report["local"]
        lines.append(
            f"Local model: {local['model']} via {local['engine']}, served length "
            f"{local['max_model_len'] or local['max_tokens']:,} tokens"
        )
    lines.append("")
    lines.append(f"{'Role':<14}{'Model':<42}{'Context window':>16}  {'Source':<15}{'Max output':>11}")
    for role, row in report["roles"].items():
        lines.append(
            f"{role:<14}{str(row['model'])[:41]:<42}{row['context_window']:>16,}  {row['context_source']:<15}"
            f"{row['max_output_tokens']:>11,}"
        )
    lines.append("")
    predictor = report["predictor"]
    if predictor["kind"] == "decision":
        lines.append(
            f"Predictor: structured decision model {predictor['model']}: state limit "
            f"{predictor['state_limit_provider_tokens']:,} provider tokens = {predictor['state_limit_tokens']:,} cl100k "
            f"tokens at ratio {predictor['tokenizer_ratio']}"
        )
        decision = report.get("decision") or {}
        lines.append(
            f"  provider {decision.get('provider')}, choice orders {decision.get('choice_orders')}, score levels "
            f"{decision.get('score_levels')}, refine {decision.get('regression_refine')}, question book by "
            f"{decision.get('compiler_model')}"
        )
    else:
        lines.append(
            f"Predictor input budget: {predictor['budget_tokens']:,} tokens (window {predictor['context_window']:,} "
            f"from {predictor['context_source']}, minus {predictor['output_reserve']:,} output reserve, capped by "
            f"max_agent_input {predictor['max_agent_input_tokens']:,})"
        )
    orchestration = report["orchestration"]
    threshold = orchestration["threshold_tokens"]
    lines.append(
        f"Orchestration: {orchestration['mode']}"
        + (f", threshold {threshold:,} tokens" if threshold else ", threshold: the Predictor input budget")
    )
    keys = ", ".join(f"{k} {'present' if v else 'MISSING'}" for k, v in report["credentials"].items()) or "none needed"
    lines.append(f"API keys: {keys}")
    cache = report.get("catalog_cache") or {}
    if cache.get("present"):
        lines.append(f"Model catalog cache: {cache['models']} models, {cache['age_hours']} h old ({cache['path']})")
    elif report["backend"] != "local":
        lines.append(
            "Model catalog cache: none, so windows come from the built-in table (refresh it with --refresh_catalog "
            "or by opening the dashboard once)"
        )
    record = report.get("record")
    if record:
        lines.append("")
        lines.append(
            f"Record {record['participant_id']}: {record['input_tokens']:,} {record['unit']} against a budget of "
            f"{record['budget_tokens']:,} -> {record['route'].upper()}"
        )
        lines.append(f"  {record['route_reason']}")
        if record.get("predictor_kind") == "decision":
            lines.append(
                f"  {record['questions_round_one']} questions in {record['requests_round_one']} request(s) "
                f"({', '.join(str(v) for v in record['question_tokens_per_request'])} question tokens); zoomed "
                f"regression round: {'yes' if record['round_two_expected'] else 'no'}"
            )
    lines.append("")
    if report.get("notes"):
        lines.append("Notes (accepted, but without effect or risky):")
        lines.extend(f"  - {note}" for note in report["notes"])
    if report["problems"]:
        lines.append("Problems (a run would fail to start):")
        lines.extend(f"  - {problem}" for problem in report["problems"])
    else:
        lines.append("No problems found.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line of main.py (kept separate so tests and tools can inspect it)."""
    parser = argparse.ArgumentParser(
        description="COMPASS Multi-Agent System for Deep Phenotype Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py /path/to/participant_001 --prediction_type binary --target_label CASE --control_label CONTROL
  python main.py /path/to/participant_001 --prediction_type binary --target_label CASE --control_label CONTROL --predictor_model typesafe/jev-1.13
  python main.py /path/to/participant_001 --prediction_type binary --target_label CASE --control_label CONTROL --orchestration always
  python main.py /path/to/participant_001 --prediction_type multiclass --target_label personality_type --class_labels A,B,C,D
  python main.py /path/to/participant_001 --prediction_type regression_univariate --target_label total_iq --regression_output total_iq
  python main.py /path/to/participant_001 --prediction_type regression_multivariate --target_label personality_traits --regression_outputs openness,conscientiousness,extraversion,agreeableness,neuroticism
  python main.py /path/to/participant_001 --prediction_type hierarchical --task_spec_file /path/to/task_spec.json

Check a configuration offline (no model call, no spend): per-role models and context windows,
the Predictor input budget, missing API keys, and the route this record would take:
  python main.py /path/to/participant_001 --predictor_model typesafe/jev-1.13 --companion_model deepseek/deepseek-v4-flash-0731 --check_config

Context windows resolve per role: decision-model state limit, --role_context_window/--context_window,
the served length on --backend local, the cached OpenRouter catalog, the built-in table,
--public_max_context_tokens, then 128000.
  python main.py /path/to/participant_001 --role_context_window predictor=128000,tool=32000
  python main.py /path/to/participant_001 --backend local --model Qwen/Qwen3-14B-AWQ --local_max_model_len 32768
  OPENROUTER_BASE_URL=http://node:8000/v1 python main.py /path/to/participant_001 --public_model Qwen/Qwen3-14B-AWQ --context_window 32768
        """
    )
    
    parser.add_argument(
        "participant_dir",
        type=Path,
        nargs='?',
        help="Path to participant data directory (Optional if using --ui)"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="target_phenotype",
        help="Legacy target label alias (kept for compatibility; maps to --target_label)"
    )
    parser.add_argument(
        "--control", "-c",
        type=str,
        default=DEFAULT_CONTROL_CONDITION,
        help="Legacy control label alias (kept for compatibility; maps to --control_label)"
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        choices=["binary", "multiclass", "regression_univariate", "regression_multivariate", "hierarchical"],
        default="binary",
        help="Prediction task family"
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        help="Canonical target label (default: --target)"
    )
    parser.add_argument(
        "--control_label",
        type=str,
        default=None,
        help="Canonical control/comparator label (default: --control)"
    )
    parser.add_argument(
        "--class_labels",
        type=str,
        default="",
        help="Comma-separated class labels (required for multiclass)"
    )
    parser.add_argument(
        "--regression_outputs",
        type=str,
        default="",
        help="Regression output names (comma-separated for multivariate; for univariate prefer --regression_output)"
    )
    parser.add_argument(
        "--regression_output",
        type=str,
        default="",
        help="Single regression output name (recommended for univariate regression)"
    )
    parser.add_argument(
        "--task_spec_file",
        type=str,
        default="",
        help="Path to canonical prediction task specification JSON file"
    )
    parser.add_argument(
        "--task_spec_json",
        type=str,
        default="",
        help="Inline canonical prediction task specification JSON string"
    )
    parser.add_argument(
        "--global_instruction",
        type=str,
        default="",
        help="Optional runtime instruction applied to all agents"
    )
    parser.add_argument(
        "--orchestrator_instruction",
        type=str,
        default="",
        help="Optional runtime instruction for Orchestrator agent"
    )
    parser.add_argument(
        "--executor_instruction",
        type=str,
        default="",
        help="Optional runtime instruction for Executor behavior"
    )
    parser.add_argument(
        "--tools_instruction",
        type=str,
        default="",
        help="Optional runtime instruction for all execution tools"
    )
    parser.add_argument(
        "--integrator_instruction",
        type=str,
        default="",
        help="Optional runtime instruction for Integrator/Fusion"
    )
    parser.add_argument(
        "--predictor_instruction",
        type=str,
        default="",
        help="Optional runtime instruction for Predictor agent"
    )
    parser.add_argument(
        "--critic_instruction",
        type=str,
        default="",
        help="Optional runtime instruction for Critic agent"
    )
    parser.add_argument(
        "--communicator_instruction",
        type=str,
        default="",
        help="Optional runtime instruction for Communicator agent"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Maximum orchestration iterations (default: 3)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Enable interactive UI mode with detailed step-by-step monitoring"
    )

    parser.add_argument(
        "--detailed_log", "-d",
        action="store_true",
        help="Enable full raw I/O logging for all tool calls"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run offline dataflow audit without LLM calls"
    )
    parser.add_argument(
        "--check_config",
        action="store_true",
        help=(
            "Check the configuration offline and exit: per-role models, context windows and their source, "
            "output ceilings, the Predictor input budget, missing API keys, and (with participant_dir) the "
            "measured record and the route it takes. No model call, no network, no spend. Exit code 1 when "
            "a run would fail to start."
        ),
    )
    parser.add_argument(
        "--refresh_catalog",
        action="store_true",
        help=(
            "Download the OpenRouter model list (a free, unauthenticated GET) into the catalog cache the context "
            "windows are read from (~/.compass/model_catalog.json, or $COMPASS_HOME) before anything else."
        ),
    )
    parser.add_argument(
        "--check_format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output of --check_config: text (default) or json.",
    )
    parser.add_argument(
        "--generate_deep_phenotype",
        action="store_true",
        help="Generate deep phenotype report at pipeline completion (manual opt-in)"
    )
    parser.add_argument(
        "--generate_xai_report",
        action="store_true",
        help="Generate communicator explainability report from XAI outputs at pipeline completion"
    )

    # --- LOCAL LLM ARGUMENTS ---
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["openrouter", "openai", "local"],
        default="openrouter",
        help="Choose LLM backend: 'openrouter' (default), 'openai', or 'local'"
    )
    parser.add_argument(
        "--public_model",
        type=str,
        default="deepseek/deepseek-v4-flash-0731",
        help="Model name for public API backend (default: deepseek/deepseek-v4-flash-0731)"
    )
    parser.add_argument(
        "--public_max_context_tokens",
        type=int,
        default=0,
        help=(
            "Context window for a hosted model that neither the cached OpenRouter catalog nor the built-in "
            "table knows (default 0: such a model gets 128000). It does not override a known model; use "
            "--context_window or --role_context_window for that."
        ),
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=None,
        help=(
            "Force this context window (tokens) for every LLM role, whatever the catalog says; for example the "
            "max model length of a self-hosted vLLM server. A structured decision model keeps its state limit."
        ),
    )
    parser.add_argument(
        "--role_context_window",
        action="append",
        default=None,
        metavar="ROLE=TOKENS",
        help=(
            "Force the context window of one role (orchestrator, critic, predictor, integrator, communicator, "
            "tool, or all). Comma-separated or repeated, for example predictor=128000,tool=32000. Wins over "
            "--context_window for that role."
        ),
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model for retrieval/rag (default: text-embedding-3-large)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "Name/Path of local model (default: Qwen/Qwen2.5-0.5B-Instruct). Only used if --backend local; "
            "for a model on a hosted or self-hosted OpenAI-compatible endpoint use --public_model."
        )
    )
    
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=32768,
        help=(
            "Context window of the local model (default: 32768). Only used if --backend local; "
            "--local_max_model_len, when set, is the served length and wins."
        )
    )

    parser.add_argument(
        "--local_engine",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="Local backend engine preference (auto|vllm|transformers)"
    )
    parser.add_argument(
        "--local_dtype",
        type=str,
        default="auto",
        help="Local dtype (auto|float16|bfloat16|float32|fp8)"
    )
    parser.add_argument(
        "--local_quant",
        type=str,
        default=None,
        help="Local quantization (e.g., awq|gptq|4bit|8bit|fp8)"
    )
    parser.add_argument(
        "--local_kv_cache_dtype",
        type=str,
        default=None,
        help="vLLM KV cache dtype (e.g., fp8_e4m3|fp8_e5m2)"
    )
    parser.add_argument(
        "--local_tensor_parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default 1)"
    )
    parser.add_argument(
        "--local_pipeline_parallel",
        type=int,
        default=1,
        help="Pipeline parallel size for vLLM (default 1)"
    )
    parser.add_argument(
        "--local_gpu_mem_util",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default 0.9)"
    )
    parser.add_argument(
        "--local_max_model_len",
        type=int,
        default=0,
        help=(
            "Max model length vLLM serves (0 = --max_tokens). It is also the context window every local "
            "role works with."
        )
    )
    parser.add_argument(
        "--local_enforce_eager",
        action="store_true",
        help="Force vLLM eager execution"
    )
    parser.add_argument(
        "--local_trust_remote_code",
        action="store_true",
        help="Trust remote code for local model"
    )
    parser.add_argument(
        "--local_attn",
        type=str,
        default="auto",
        help="Transformers attention implementation (auto|flash_attention_2|sdpa|eager)"
    )
    
    # --- PREDICTOR AND ORCHESTRATION ---
    parser.add_argument(
        "--predictor_model",
        type=str,
        default=None,
        help=(
            "Model for the Predictor role only. May be a conventional LLM or a structured decision model "
            "such as typesafe/jev-1.13 (every other role then keeps --public_model)."
        ),
    )
    parser.add_argument(
        "--companion_model",
        type=str,
        default=None,
        help=(
            "Conventional LLM for every role except the Predictor. Needed when --public_model is a decision "
            "model; asked for interactively when missing on a terminal, else defaults to deepseek/deepseek-v4-flash-0731."
        ),
    )
    parser.add_argument(
        "--orchestration",
        type=str,
        default=None,
        choices=["auto", "always", "never"],
        help=(
            "auto (default): run the multi-agent orchestration only when the participant record does not fit "
            "the Predictor input; always: run it on every attempt; never: always predict directly."
        ),
    )
    parser.add_argument(
        "--orchestration_threshold",
        type=int,
        default=None,
        help=(
            "Token count above which auto mode orchestrates (default 0: the Predictor input budget, which is "
            "also the maximum). Only used with --orchestration auto."
        ),
    )
    parser.add_argument(
        "--decision_provider",
        type=str,
        default=None,
        choices=["openrouter", "typesafe"],
        help="Route for decision model calls: openrouter (OPENROUTER_API_KEY, default) or typesafe (TYPESAFE_API_KEY).",
    )
    parser.add_argument(
        "--decision_choice_orders",
        type=int,
        default=None,
        help=(
            "Decision models: option orders asked and averaged for multiclass nodes (default 3). "
            "Binary nodes always get both orders plus a Noul per label."
        ),
    )
    parser.add_argument(
        "--decision_score_levels",
        type=int,
        default=None,
        help="Decision models: levels per regression Score, at most 10 (default 10).",
    )
    parser.add_argument(
        "--decision_refine",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decision models: zoomed second Score pass for continuous outputs (default on).",
    )
    parser.add_argument(
        "--decision_compiler_model",
        type=str,
        default=None,
        help="Decision models: conventional LLM that compiles the question book once per task (default: the Orchestrator's model).",
    )
    parser.add_argument(
        "--decision_stability_threshold",
        type=float,
        default=None,
        help="Decision models: largest probability shift between presentation orders the critic accepts, 0.05 to 0.6 (default 0.25).",
    )
    parser.add_argument(
        "--decision_regression_stability_threshold",
        type=float,
        default=None,
        help="Decision models: largest ascending/descending gap in reference SDs the critic accepts, 0.1 to 2.0 (default 0.5).",
    )
    parser.add_argument(
        "--decision_sufficiency_threshold",
        type=float,
        default=None,
        help="Decision models: evidence sufficiency below which the critic rejects an attempt, 0 to 0.9 (default 0: report only).",
    )

    # --- TOKEN CONTROLS ---
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["provider_default", "off", "low", "medium", "high"],
        help="Reasoning effort for providers that support it (default: off). Reasoning tokens are billed as output and count against the output ceiling."
    )
    parser.add_argument(
        "--total_budget",
        type=int,
        help="Override total token budget"
    )
    parser.add_argument(
        "--max_agent_input",
        type=int,
        help="Max limit for agent input context (Prompt)"
    )
    parser.add_argument(
        "--max_agent_output",
        type=int,
        help="Max tokens for agent generation"
    )
    parser.add_argument(
        "--max_tool_input",
        type=int,
        help="Max limit for tool input size"
    )
    parser.add_argument(
        "--max_tool_output",
        type=int,
        help="Max limit for tool output size"
    )
    # --- XAI CONTROLS ---
    parser.add_argument(
        "--xai_methods",
        type=str,
        default="",
        help="Comma-separated methods: external,internal,hybrid (or 'all')"
    )
    parser.add_argument(
        "--xai_external_k",
        type=int,
        default=None,
        help="aHFR-TokenSHAP permutations (default from settings)"
    )
    parser.add_argument(
        "--xai_external_runs",
        type=int,
        default=None,
        help="aHFR-TokenSHAP repeat runs (default from settings)"
    )
    parser.add_argument(
        "--xai_external_adaptive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable adaptive sampling for external aHFR-TokenSHAP"
    )
    parser.add_argument(
        "--xai_internal_model",
        type=str,
        default=None,
        help="Internal IGA model (local HF model id/path)"
    )
    parser.add_argument(
        "--xai_internal_steps",
        type=int,
        default=None,
        help="Internal IGA integration steps"
    )
    parser.add_argument(
        "--xai_internal_baseline",
        type=str,
        default=None,
        help="Internal IGA baseline mode (mask|prompt|eos|zero)"
    )
    parser.add_argument(
        "--xai_internal_span_mode",
        type=str,
        default=None,
        help="Internal IGA span mode (value|line)"
    )
    parser.add_argument(
        "--xai_hybrid_model",
        type=str,
        default=None,
        help="Hybrid LLM-select model (default from settings)"
    )
    parser.add_argument(
        "--xai_hybrid_repeats",
        type=int,
        default=None,
        help="Hybrid LLM-select repeats"
    )
    parser.add_argument(
        "--xai_hybrid_temperature",
        type=float,
        default=None,
        help="Hybrid LLM-select temperature"
    )
    parser.add_argument(
        "--xai_full_validation",
        action="store_true",
        help="Enable stricter explainability validation checks"
    )
    # ---------------------------
    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    validate_cli_args(parser, args)
    
    # Validate participant directory
    if not args.ui and not args.participant_dir and not args.check_config:
        print("Error: participant_dir is required when not using --ui mode.")
        parser.print_help()
        sys.exit(1)

    if args.participant_dir and not args.participant_dir.exists():
        print(f"Error: Participant directory not found: {args.participant_dir}")
        sys.exit(1)

    if str(args.task_spec_json or "").strip() and str(args.task_spec_file or "").strip():
        print("Error: --task_spec_json and --task_spec_file are mutually exclusive.")
        sys.exit(1)

    cli_target_label = str(args.target_label or args.target or "target_phenotype").strip()
    cli_control_label = str(args.control_label or args.control or DEFAULT_CONTROL_CONDITION).strip()
    cli_class_labels = parse_csv_list(args.class_labels)
    cli_regression_outputs = parse_csv_list(args.regression_outputs)
    cli_regression_output = str(args.regression_output or "").strip()
    if cli_regression_output:
        if "," in cli_regression_output:
            print("Error: --regression_output accepts exactly one output name without commas. Use --regression_outputs for multivariate mode.")
            sys.exit(1)
        if cli_regression_outputs and cli_regression_outputs != [cli_regression_output]:
            print("Error: --regression_output conflicts with --regression_outputs. Use one form or provide matching values.")
            sys.exit(1)
        cli_regression_outputs = [cli_regression_output]

    try:
        cli_prediction_task_spec = _resolve_prediction_task_spec(
            prediction_type=args.prediction_type,
            target_label=cli_target_label,
            control_label=cli_control_label,
            class_labels=cli_class_labels,
            regression_outputs=cli_regression_outputs,
            task_spec_json=str(args.task_spec_json or ""),
            task_spec_file=str(args.task_spec_file or ""),
        )
        cli_prediction_task_spec = _align_binary_task_spec_labels(
            cli_prediction_task_spec,
            target_label=cli_target_label,
            control_label=cli_control_label,
        )
    except Exception as e:
        print(f"Error resolving prediction task specification: {e}")
        sys.exit(1)
    cli_agent_instructions = _normalize_agent_instructions(
        {
            "global": args.global_instruction,
            "orchestrator": args.orchestrator_instruction,
            "executor": args.executor_instruction,
            "tools": args.tools_instruction,
            "integrator": args.integrator_instruction,
            "predictor": args.predictor_instruction,
            "critic": args.critic_instruction,
            "communicator": args.communicator_instruction,
        }
    )
    
    if getattr(args, "refresh_catalog", False):
        try:
            from src.full_stack.backend.api.catalog import fetch_catalog

            catalog = fetch_catalog(force=True)
            print(
                f"[Init] Model catalog refreshed: {len(catalog.get('models') or [])} models"
                + (" (stale: the provider could not be reached)" if catalog.get("stale") else ""),
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[Init] Could not refresh the model catalog ({exc}); using the cached one or the built-in table.", file=sys.stderr)

    if args.check_config:
        settings = apply_cli_settings(get_settings(), args, interactive=False, print_notes=False)
        report = run_config_check(
            settings,
            participant_dir=args.participant_dir,
            prediction_task_spec=cli_prediction_task_spec,
            agent_instructions=cli_agent_instructions,
        )
        report["notes"] = cli_notes(args, settings)
        if args.check_format == "json":
            print(json.dumps(report, indent=2, default=str))
        else:
            print(format_config_check(report))
        sys.exit(1 if report["problems"] else 0)

    # Run pipeline
    try:
        if args.audit:
            run_dataflow_audit(
                participant_dir=args.participant_dir,
                target_condition=cli_target_label,
                control_condition=cli_control_label,
                prediction_task_spec=cli_prediction_task_spec,
                verbose=not args.quiet,
            )
            sys.exit(0)
        if args.ui:
            # The dashboard is a separate service: a FastAPI backend that supervises
            # runs as isolated worker processes and serves the built web client.
            from src.full_stack.backend.api.server import serve_dashboard

            serve_dashboard(
                host=os.getenv("COMPASS_UI_HOST", "127.0.0.1"),
                port=int(os.getenv("COMPASS_UI_PORT", "5005")),
                open_browser=not args.quiet,
            )
            sys.exit(0)
        else:
            settings = apply_cli_settings(get_settings(), args)

            _apply_explainability_overrides(settings, args)

            reset_llm_client()

            # Run standard CLI
            target_condition, control_condition = _task_spec_to_legacy_labels(cli_prediction_task_spec)
            result = run_compass_pipeline(
                participant_dir=args.participant_dir,
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=cli_prediction_task_spec,
                agent_instructions=cli_agent_instructions,
                max_iterations=args.iterations,
                verbose=not args.quiet,
                interactive_ui=False,
                generate_deep_phenotype=bool(args.generate_deep_phenotype),
                generate_xai_report=bool(args.generate_xai_report),
            )
        
        # Exit with appropriate code
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError running COMPASS pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
