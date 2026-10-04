"""
Evidence routing: does the participant record go straight to the Predictor, or
through the multi-agent orchestration workflow first?

    direct        the whole record (notes, deviation map, data overview, every
                  feature leaf) is handed to the Predictor unchanged. Chosen when
                  it fits the Predictor's input budget.
    orchestrated  Orchestrator plans tool steps, the Executor runs them, the
                  Integrator fuses and, when needed, extracts chunk evidence.
                  Chosen when the record does not fit, or always on request.

The budget is the Predictor's own: for an LLM, the model's input budget (context
window minus the output reserve, capped by `max_agent_input_tokens`); for a
structured decision model, its state limit converted to cl100k tokens.
`--orchestration_threshold` replaces it with a fixed token count.

Building the direct executor output needs no model call, so the route decision
is free and is made before any LLM work starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from ...config.settings import ORCHESTRATION_MODES
from ...data.models.prediction_task import PredictionTaskSpec
from .data_loader import ParticipantData
from .fusion_layer import FusionLayer, FusionResult
from .record_rendering import DIRECT_RECORD_HEADER, render_direct_record
from ..token_packer import count_tokens


@dataclass
class RouteDecision:
    route: str  # "direct" | "orchestrated"
    mode: str  # "auto" | "always" | "never"
    input_tokens: int
    budget_tokens: int
    threshold_source: str  # "predictor_budget" | "override"
    predictor_kind: str
    reason: str
    escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_mode(mode: Optional[str]) -> str:
    text = str(mode or "auto").strip().lower()
    aliases = {"true": "auto", "on": "auto", "default": "auto", "force": "always", "off": "never", "false": "never", "direct": "never"}
    text = aliases.get(text, text)
    if text not in ORCHESTRATION_MODES:
        raise ValueError(f"orchestration mode must be one of {', '.join(ORCHESTRATION_MODES)}; got '{mode}'")
    return text


def decide_route(
    *,
    mode: str,
    measurement: Optional[Dict[str, Any]],
    threshold_override: int = 0,
    predictor_kind: str = "llm",
) -> RouteDecision:
    mode = normalize_mode(mode)
    tokens = int((measurement or {}).get("input_tokens") or 0)
    budget = int((measurement or {}).get("budget_tokens") or 0)
    source = "predictor_budget"
    if int(threshold_override or 0) > 0:
        # An override can lower the threshold, but never above what the Predictor accepts.
        budget = min(int(threshold_override), budget) if budget > 0 else int(threshold_override)
        source = "override"
    fits = measurement is not None and tokens <= budget
    if mode == "always":
        route, reason = "orchestrated", "Orchestration requested for every attempt."
    elif mode == "never":
        route = "direct"
        reason = (
            "Orchestration disabled; the record fits the Predictor input."
            if fits
            else "Orchestration disabled; the record exceeds the Predictor input and will be packed to fit."
        )
    elif fits:
        route = "direct"
        reason = f"The record ({tokens:,} tokens) fits the Predictor input budget ({budget:,}); orchestration skipped."
    else:
        route = "orchestrated"
        reason = f"The record ({tokens:,} tokens) exceeds the Predictor input budget ({budget:,}); orchestrating."
    return RouteDecision(
        route=route,
        mode=mode,
        input_tokens=tokens,
        budget_tokens=budget,
        threshold_source=source,
        predictor_kind=predictor_kind,
        reason=reason,
    )


def build_direct_executor_output(
    participant_data: ParticipantData,
    *,
    target_condition: str,
    control_condition: str,
    prediction_task_spec: Optional[PredictionTaskSpec],
    agent_instructions: Optional[Dict[str, str]] = None,
    model_hint: str = "gpt-5",
) -> Dict[str, Any]:
    """
    The executor output a Predictor receives on the direct route: the complete
    record in the same pass-through layout the Integrator produces when the
    record fits, with no tool outputs and no chunk evidence.

    The Predictor reads it through the compact, lossless rendering in
    `direct_record` (see record_rendering): every leaf once, with its value
    and z, under its ontology path. That is about half the tokens of the
    generic JSON rendering, so more records fit and none is cut.
    """
    from ...agents.executor import Executor  # local import: agents import this package

    # Neither helper calls a model; skip their constructors (which create clients).
    executor = Executor.__new__(Executor)
    context = executor._build_context(
        participant_data,
        target_condition,
        control_condition,
        prediction_task_spec=prediction_task_spec,
        agent_instructions=agent_instructions,
    )
    fusion = FusionLayer.__new__(FusionLayer)
    flat = context.get("multimodal_data") or {}
    tree = {
        domain: (fusion._features_to_nested_tree(features) if isinstance(features, list) else features)
        for domain, features in flat.items()
    }
    pass_through = FusionResult(
        fused_narrative="Direct route: the complete participant record is passed to the Predictor.",
        domain_summaries={},
        key_findings=[],
        cross_modal_patterns=[],
        evidence_summary={"for_case": [], "for_control": []},
        tokens_used=0,
        source_outputs=[],
        skipped_fusion=True,
        raw_multimodal_data=tree,
        raw_processed_multimodal_data=None,
        raw_step_outputs={},
        context_fill_report={"route": "direct"},
    )
    predictor_input = fusion.compress_for_predictor(
        fusion_result=pass_through,
        hierarchical_deviation=context.get("hierarchical_deviation") or {},
        non_numerical_data=context.get("non_numerical_data") or "",
    )
    coverage_ledger = executor._build_coverage_ledger(
        multimodal_data=flat,
        step_outputs={},
        predictor_input=predictor_input,
    )
    predictor_input["coverage_ledger"] = coverage_ledger

    # The parsed tree (not its serialization) knows which scores are signed and
    # which are a group's mean absolute deviation.
    deviation_tree = participant_data.hierarchical_deviation
    direct_record = render_direct_record(
        multimodal=tree,
        deviation=deviation_tree if deviation_tree is not None else (context.get("hierarchical_deviation") or {}),
        overview=context.get("data_overview") or {},
        notes=context.get("non_numerical_data") or "",
    )
    non_core_text = f"## measurements\n{DIRECT_RECORD_HEADER}\n\n{direct_record['measurements']}"
    non_core_tokens = count_tokens(non_core_text, model_hint=model_hint)
    record_tokens = non_core_tokens + sum(
        count_tokens(direct_record[key], model_hint=model_hint)
        for key in ("clinical_record", "deviation_profile", "data_overview")
    )
    # A copy, so the estimate does not also land in fusion_result.context_fill_report.
    predictor_input["context_fill_report"] = dict(predictor_input.get("context_fill_report") or {})
    predictor_input["context_fill_report"]["predictor_payload_estimate"] = {
        "final_tokens": record_tokens,
        "strategy": "direct_route_compact_record",
        "chunked_two_pass_required": False,
    }

    overview = context.get("data_overview") or {}
    domains = [
        str(name)
        for name, cov in dict(overview.get("domain_coverage") or {}).items()
        if isinstance(cov, dict) and int(cov.get("present_leaves") or 0) > 0
    ] or list(flat.keys())

    return {
        "route": "direct",
        "execution_result": None,
        "fusion_result": pass_through,
        "predictor_input": predictor_input,
        "step_outputs": {},
        "chunk_evidence": [],
        "predictor_chunk_count": 0,
        "chunking_skipped": True,
        "chunking_reason": "direct_route",
        "non_core_context_text": non_core_text,
        "non_core_context_tokens": non_core_tokens,
        "direct_record": direct_record,
        "direct_fits": True,
        "deviation_tree": deviation_tree,
        "processed_raw_excluded": False,
        "data_overview": overview,
        "hierarchical_deviation": context.get("hierarchical_deviation") or {},
        "non_numerical_data": context.get("non_numerical_data") or "",
        "multimodal_data": flat,
        "plan_id": "direct",
        "participant_id": participant_data.participant_id,
        "target_condition": target_condition,
        "control_condition": control_condition,
        "prediction_task_spec": context.get("prediction_task_spec"),
        "domains_processed": domains,
        "total_tokens_used": 0,
        "coverage_ledger": coverage_ledger,
    }
