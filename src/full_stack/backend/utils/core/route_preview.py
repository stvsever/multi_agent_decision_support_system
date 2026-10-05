"""
Offline preview of a run: what the engine will do, without calling any model.

    configuration_summary(settings)
        every role's model, context window (and the source that decided it) and
        output ceiling, the Predictor's input budget, the routing mode, which
        API keys the configuration needs and whether they are present (never
        their values), and the problems that would stop a run.

    measure_record(participant_dir, settings=..., prediction_task_spec=...)
        the engine's own measurement of the direct Predictor input for one
        participant (the same DataLoader, direct executor output and
        `measure_direct` the pipeline runs before its first attempt) and the
        route `auto`, `always` or `never` takes for it.

The CLI (`main.py --check_config`) and the dashboard cost estimate both call
these functions, so neither keeps its own copy of the engine's arithmetic.

Offline means: no LLM, no decision-model request, no network. A structured
decision model's question book is normally compiled once per task by the
companion LLM; the preview uses the deterministic book of the same shape
(`decision.questions.deterministic_book`), so its question token counts can
differ from a compiled book by a few hundred tokens.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ...config import settings as settings_module
from ...config.settings import (
    CONTEXT_WINDOW_ROLES,
    LLMBackend,
    Settings,
    catalog_cache_status,
    llm_predictor_input_budget,
    predictor_output_reserve,
)
from ...data.models.prediction_task import PredictionTaskSpec, build_binary_task_spec

_SWAP_LOCK = threading.RLock()


class _OfflineLLM:
    """Stands in for the LLM client: measuring never sends a request."""

    class _Tracker:
        def add(self, *_: Any, **__: Any) -> None:
            return None

    token_tracker = _Tracker()

    def call(self, *_: Any, **__: Any):  # pragma: no cover - a call is a bug
        raise RuntimeError("route preview never calls a model")


class _OfflineDecisionClient:
    calls: List[Dict[str, Any]] = []

    def ask(self, **_: Any):  # pragma: no cover - a call is a bug
        raise RuntimeError("route preview never calls the decision model")


@contextmanager
def using_settings(settings: Settings) -> Iterator[Settings]:
    """
    Make `settings` the engine's settings object for the duration of the block.

    The agents read the process-wide settings singleton when they are built, so
    a preview for a configuration other than the running one swaps it in under
    a lock and restores the previous object afterwards.
    """
    with _SWAP_LOCK:
        previous = settings_module._settings_instance
        settings_module._settings_instance = settings
        try:
            yield settings
        finally:
            settings_module._settings_instance = previous


def predictor_kind(settings: Settings) -> str:
    from ...agents.decision import is_decision_model

    return "decision" if is_decision_model(settings.models.predictor_model) else "llm"


def _combine(global_text: str, scoped_text: str) -> str:
    global_text, scoped_text = str(global_text or "").strip(), str(scoped_text or "").strip()
    if global_text and scoped_text:
        return f"{global_text}\n\n{scoped_text}"
    return scoped_text or global_text


def _decision_spec(settings: Settings):
    from ...agents.decision import get_decision_model_spec

    return get_decision_model_spec(settings.models.predictor_model)


def predictor_budget_summary(settings: Settings) -> Dict[str, Any]:
    """The Predictor's input budget without a task: exact for an LLM, the upper bound for a decision model."""
    if predictor_kind(settings) == "decision":
        spec = _decision_spec(settings)
        ratio = float(settings.decision.tokenizer_ratio or 1.2)
        state = spec.state_budget_cl100k(ratio=ratio) if spec else 0
        return {
            "kind": "decision",
            "model": settings.models.predictor_model,
            "state_limit_provider_tokens": spec.state_context_tokens if spec else 0,
            "request_limit_provider_tokens": spec.request_context_tokens if spec else 0,
            "tokenizer_ratio": ratio,
            "state_limit_tokens": state,
            "budget_tokens": None,
            "note": "The state budget is the state limit minus the longest question of the task; measure a record to see it.",
        }
    completion = int(settings.models.predictor_max_tokens or 4096)
    window = settings.context_window_resolution(settings.models.predictor_model, role="predictor")
    return {
        "kind": "llm",
        "model": settings.models.predictor_model,
        "context_window": int(window.tokens),
        "context_source": window.source,
        "output_reserve": predictor_output_reserve(completion),
        "max_agent_input_tokens": int(settings.token_budget.max_agent_input_tokens or 0),
        "budget_tokens": llm_predictor_input_budget(settings),
    }


def required_credentials(settings: Settings) -> Dict[str, bool]:
    """API keys this configuration needs, mapped to whether one is present."""
    needed: Dict[str, bool] = {}
    if settings.models.backend == LLMBackend.OPENROUTER:
        needed["OPENROUTER_API_KEY"] = bool(settings.openrouter_api_key)
    elif settings.models.backend == LLMBackend.OPENAI:
        needed["OPENAI_API_KEY"] = bool(settings.openai_api_key)
    if predictor_kind(settings) == "decision":
        if str(settings.decision.provider or "openrouter").lower() == "typesafe":
            needed["TYPESAFE_API_KEY"] = bool(settings.typesafe_api_key)
        else:
            needed["OPENROUTER_API_KEY"] = bool(settings.openrouter_api_key)
    return needed


def configuration_problems(settings: Settings) -> List[str]:
    """Everything about these settings that would stop a run (missing keys, misplaced decision models)."""
    from ...agents.decision import is_decision_model

    problems: List[str] = []
    for key, present in required_credentials(settings).items():
        if not present:
            problems.append(f"{key} is not set (environment or .env).")
    for role in CONTEXT_WINDOW_ROLES:
        if role == "predictor":
            continue
        model = getattr(settings.models, f"{role}_model", "")
        if is_decision_model(model):
            problems.append(
                f"The {role} is set to the structured decision model {model}; only the Predictor may use one "
                "(pass --companion_model for the other roles)."
            )
    compiler = str(settings.decision.compiler_model or "").strip()
    if compiler and is_decision_model(compiler):
        problems.append(f"The question book compiler {compiler} must be a conventional LLM.")
    if (
        predictor_kind(settings) == "decision"
        and str(settings.decision.provider or "openrouter").lower() == "openrouter"
        and settings.models.backend == LLMBackend.OPENROUTER
        and "openrouter.ai" not in str(settings.openrouter_base_url or "").lower()
        and not os.getenv("COMPASS_DECISION_ENDPOINT", "").strip()
    ):
        problems.append(
            f"The decision model would be called at {settings.openrouter_base_url} (a self-hosted endpoint cannot "
            "serve it). Use --decision_provider typesafe, or set COMPASS_DECISION_ENDPOINT to the OpenRouter "
            "decisions URL."
        )
    return problems


def configuration_summary(settings: Settings) -> Dict[str, Any]:
    """What a run with these settings will use, per role, before anything is sent."""
    roles: Dict[str, Dict[str, Any]] = {}
    for role in CONTEXT_WINDOW_ROLES:
        model = getattr(settings.models, f"{role}_model", "")
        window = settings.context_window_resolution(model, role=role)
        roles[role] = {
            "model": model,
            "context_window": int(window.tokens),
            "context_source": window.source,
            "max_output_tokens": int(getattr(settings.models, f"{role}_max_tokens", 0) or 0),
        }
    backend = getattr(settings.models.backend, "value", str(settings.models.backend))
    summary: Dict[str, Any] = {
        "backend": backend,
        "endpoint": (
            settings.openrouter_base_url
            if settings.models.backend == LLMBackend.OPENROUTER
            else ("https://api.openai.com/v1" if settings.models.backend == LLMBackend.OPENAI else "in-process")
        ),
        "predictor_kind": predictor_kind(settings),
        "roles": roles,
        "predictor": predictor_budget_summary(settings),
        "orchestration": {
            "mode": settings.orchestration.mode,
            "threshold_tokens": int(settings.orchestration.threshold_tokens or 0),
        },
        "token_budget": {
            "max_agent_input_tokens": int(settings.token_budget.max_agent_input_tokens or 0),
            "max_agent_output_tokens": int(settings.token_budget.max_agent_output_tokens or 0),
            "max_tool_input_tokens": int(settings.token_budget.max_tool_input_tokens or 0),
            "max_tool_output_tokens": int(settings.token_budget.max_tool_output_tokens or 0),
            "total_budget": int(settings.token_budget.total_budget or 0),
        },
        "credentials": required_credentials(settings),
        "catalog_cache": catalog_cache_status(),
        "problems": configuration_problems(settings),
    }
    if settings.models.backend == LLMBackend.LOCAL:
        summary["local"] = {
            "model": settings.models.local_model_name,
            "engine": settings.models.local_backend_type,
            "max_model_len": int(settings.models.local_max_model_len or 0),
            "max_tokens": int(settings.models.local_max_tokens or 0),
        }
    if predictor_kind(settings) == "decision":
        decision = settings.decision
        summary["decision"] = {
            "provider": decision.provider,
            "choice_orders": int(decision.choice_orders),
            "score_levels": int(decision.score_levels),
            "regression_refine": bool(decision.regression_refine),
            "compiler_model": str(decision.compiler_model or "").strip() or settings.models.orchestrator_model,
        }
    return summary


DEFAULT_CONTROL_CONDITION = "non-target comparator phenotype profile"


def legacy_labels(task_spec: PredictionTaskSpec):
    """Target and control strings the pipeline derives from a task (main._task_spec_to_legacy_labels)."""
    from ...data.models.prediction_task import PredictionMode

    target_label, control_label = task_spec.legacy_target_control()
    target = str(target_label or "Target Phenotype").strip() or "Target Phenotype"
    if task_spec.root.mode == PredictionMode.BINARY_CLASSIFICATION:
        control = str(control_label or DEFAULT_CONTROL_CONDITION).strip() or DEFAULT_CONTROL_CONDITION
    else:
        control = str(control_label or "").strip()
    return target, control


def _round_two_expected(task_spec: PredictionTaskSpec, book, settings: Settings) -> bool:
    """Whether a zoomed regression round follows (any continuous output with refinement on)."""
    if not bool(settings.decision.regression_refine):
        return False
    from ...agents.decision.aggregate import is_classification
    from ...agents.decision.scales import default_scale

    levels = int(settings.decision.score_levels)
    for node in task_spec.root.walk():
        if is_classification(node):
            continue
        node_book = book.nodes.get(node.node_id)
        for out in node.regression_outputs:
            scale = (node_book.scales.get(out) if node_book else None) or default_scale(out)
            if not scale.exact_integer_levels(levels):
                return True
    return False


def measure_record(
    participant_dir: Path,
    *,
    settings: Settings,
    prediction_task_spec: Optional[PredictionTaskSpec] = None,
    agent_instructions: Optional[Dict[str, str]] = None,
    target_condition: Optional[str] = None,
    control_condition: Optional[str] = None,
) -> Dict[str, Any]:
    """
    The engine's measurement of one record and the route it takes, offline.

    Returns the `measure_direct` result (input tokens, budget, unit), the route
    decision for the configured mode and threshold, and for a decision model the
    question load per request.
    """
    from ...agents.predictor import Predictor
    from ...agents.decision.predictor import DecisionPredictor
    from ...agents.decision.questions import deterministic_book, round_one_questions
    from .data_loader import DataLoader
    from .input_routing import build_direct_executor_output, decide_route, normalize_mode

    participant_dir = Path(participant_dir)
    instructions = {k: str(v or "").strip() for k, v in (agent_instructions or {}).items()}
    if prediction_task_spec is None:
        prediction_task_spec = build_binary_task_spec(
            target_label=target_condition or "target_phenotype",
            control_label=control_condition or "non-target comparator phenotype profile",
        )
    legacy_target, legacy_control = legacy_labels(prediction_task_spec)
    target_condition = legacy_target if target_condition is None else target_condition
    control_condition = legacy_control if control_condition is None else control_condition

    kind = predictor_kind(settings)
    with using_settings(settings):
        data = DataLoader().load(participant_dir)
        participant_id = str(getattr(data, "participant_id", "") or "").strip()
        if not participant_id or participant_id.lower() == "unknown":
            data.participant_id = participant_dir.name
        direct = build_direct_executor_output(
            data,
            target_condition=target_condition,
            control_condition=control_condition,
            prediction_task_spec=prediction_task_spec,
            agent_instructions=instructions,
            model_hint=settings.models.predictor_model or "gpt-5",
        )
        predictor_instruction = _combine(instructions.get("global", ""), instructions.get("predictor", ""))
        extra: Dict[str, Any] = {}
        if kind == "decision":
            predictor = DecisionPredictor(llm_client=_OfflineLLM())
            predictor.client = _OfflineDecisionClient()
            predictor.set_runtime_instruction(predictor_instruction)
            predictor.question_book = lambda spec: deterministic_book(spec, predictor.task_context())
            measurement = predictor.measure_direct(dict(direct), prediction_task_spec=prediction_task_spec)
            book = predictor.question_book(prediction_task_spec)
            questions = round_one_questions(
                prediction_task_spec,
                book,
                choice_orders=int(settings.decision.choice_orders),
                score_levels=int(settings.decision.score_levels),
            )
            parts = predictor._split(questions)
            sizes = [sum(predictor._question_tokens(part)) for part in parts]
            extra = {
                "question_tokens_per_request": sizes,
                "requests_round_one": len(parts),
                "round_two_expected": _round_two_expected(prediction_task_spec, book, settings),
                "questions_round_one": len(questions.items),
                "question_book": "deterministic (offline stand-in for the compiled book)",
            }
        else:
            predictor = Predictor(llm_client=_OfflineLLM())
            predictor.set_runtime_instruction(predictor_instruction)
            measurement = predictor.measure_direct(
                dict(direct),
                target_condition=target_condition,
                control_condition=control_condition,
                prediction_task_spec=prediction_task_spec,
            )

    mode = normalize_mode(settings.orchestration.mode)
    decision = decide_route(
        mode=mode,
        measurement=measurement,
        threshold_override=int(settings.orchestration.threshold_tokens or 0),
        predictor_kind=kind,
    )
    tokens = int(measurement.get("input_tokens") or 0)
    budget = int(measurement.get("budget_tokens") or 0)
    return {
        "participant_dir": str(participant_dir),
        "participant_id": str(data.participant_id),
        "predictor_kind": kind,
        "model": settings.models.predictor_model,
        "input_tokens": tokens,
        "budget_tokens": budget,
        "fits": tokens <= budget,
        "unit": measurement.get("unit"),
        "route": decision.route,
        "route_reason": decision.reason,
        "mode": decision.mode,
        "threshold_tokens": decision.budget_tokens,
        "threshold_source": decision.threshold_source,
        "domains": list(direct.get("domains_processed") or []),
        **extra,
    }
