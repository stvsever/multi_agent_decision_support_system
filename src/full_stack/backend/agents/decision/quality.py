"""
Decision critic: the quality gate for structured decision outputs.

The LLM critic judges a written rationale. A decision model writes none, so its
critic reads what such a model does provide and what a methodologist would ask:

- schema:        every required node present, probabilities valid, values finite;
- stability:     do the answers hold when the options are shown in another order?
                 A calibrated judgement should not depend on presentation order.
                 Classification nodes use the largest probability shift between
                 two orders; regression outputs the gap between ascending and
                 descending level orders in reference standard deviations; each
                 against its own threshold;
- scale:         every regression output was placed on a real measurement scale
                 (from the task specification or the compiled question book),
                 not the generic fallback;
- sufficiency:   does the model itself judge the record informative enough
                 for this question (a Noul asked in the same request)?
- coverage:      how many feature leaves survived state packing?
- confidence:    reported, but never a reason to reject. A calibrated model must
                 be allowed to say "uncertain" for a genuinely ambiguous record.

An unstable answer, insufficient evidence or lost coverage are all properties of
the evidence the model was shown, so the remedy is a different evidence route,
not the same question again. The pipeline therefore answers a rejected direct
attempt by re-running it on the orchestrated route, where the multi-agent
workflow distils the record into a focused dossier (see main.py).
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...data.models.prediction_result import (
    CriticEvaluation,
    EvaluationChecklist,
    ImprovementPriority,
    ImprovementSuggestion,
    PredictionResult,
    Verdict,
)
from ...data.models.prediction_task import PredictionMode, PredictionTaskSpec


def evaluate_decision_prediction(
    prediction: PredictionResult,
    task_spec: PredictionTaskSpec,
    *,
    sufficiency_threshold: float = 0.0,
    coverage_threshold: float = 0.90,
) -> CriticEvaluation:
    report: Dict[str, Any] = dict(prediction.decision_report or {})
    quality: Dict[str, Any] = dict(report.get("quality") or {})
    spec_nodes = task_spec.node_index()
    pred_nodes = {n.node_id: n for n in (prediction.root_prediction.walk() if prediction.root_prediction else [])}

    missing = [nid for nid, node in spec_nodes.items() if node.required and nid not in pred_nodes]
    probs_ok, values_ok = True, True
    has_cls = has_reg = False
    for nid, node in spec_nodes.items():
        pred = pred_nodes.get(nid)
        if pred is None:
            continue
        if node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            has_cls = True
            cls = pred.classification
            if cls is None or not cls.probabilities or abs(sum(cls.probabilities.values()) - 1.0) > 0.05:
                probs_ok = False
            elif cls.predicted_label not in node.class_labels:
                probs_ok = False
        else:
            has_reg = True
            reg = pred.regression
            if reg is None:
                values_ok = False
                continue
            for out in node.regression_outputs:
                value = reg.values.get(out)
                if value is None or not math.isfinite(float(value)):
                    values_ok = False
    default_scales = [str(x) for x in (quality.get("default_scale_outputs") or [])]
    schema_ok = not missing and probs_ok and values_ok and not default_scales

    # Instabilities are reported as multiples of each node's own threshold.
    max_instability = float(quality.get("max_instability") or 0.0)
    mean_instability = float(quality.get("mean_instability") or 0.0)
    sufficiency: Optional[float] = quality.get("evidence_sufficiency")
    coverage = float(quality.get("feature_coverage") if quality.get("feature_coverage") is not None else 1.0)
    confidence = float(quality.get("mean_confidence") or 0.0)

    stable = max_instability <= 1.0
    gate_sufficiency = float(sufficiency_threshold or 0.0) > 0.0
    sufficient = sufficiency is None or not gate_sufficiency or float(sufficiency) >= sufficiency_threshold
    covered = coverage >= coverage_threshold

    active = ["has_required_outputs", "output_schema_valid"]
    if has_cls:
        active.append("classification_probabilities_valid")
    if has_reg:
        active.append("regression_values_valid")
    if len(spec_nodes) > 1:
        active.append("hierarchy_consistent")
    active += ["sufficient_coverage", "decision_stable"]
    if gate_sufficiency:
        active.append("evidence_sufficient")
    checklist = EvaluationChecklist(
        has_required_outputs=not missing,
        output_schema_valid=schema_ok,
        classification_probabilities_valid=probs_ok,
        regression_values_valid=values_ok,
        hierarchy_consistent=not missing,
        sufficient_coverage=covered,
        decision_stable=stable,
        evidence_sufficient=sufficient,
        has_binary_outcome=prediction.binary_classification is not None,
        valid_probability=prediction.probability_score is not None,
        active_checks=active,
    )

    stability_score = max(0.0, 1.0 - mean_instability / 2.0)
    sufficiency_score = float(sufficiency) if sufficiency is not None else 1.0
    breakdown = {
        "schema": 1.0 if schema_ok else 0.0,
        "stability": round(stability_score, 4),
        "evidence_sufficiency": round(sufficiency_score, 4),
        "coverage": round(coverage, 4),
        "confidence": round(confidence, 4),
    }
    composite = (
        0.35 * breakdown["schema"]
        + 0.25 * breakdown["stability"]
        + 0.20 * breakdown["evidence_sufficiency"]
        + 0.10 * breakdown["coverage"]
        + 0.10 * breakdown["confidence"]
    )
    satisfactory = schema_ok and stable and sufficient and covered
    verdict = Verdict.SATISFACTORY if satisfactory else Verdict.UNSATISFACTORY

    strengths: List[str] = []
    weaknesses: List[str] = []
    suggestions: List[ImprovementSuggestion] = []
    route = str(report.get("route") or prediction.input_route or "orchestrated")
    focus = str(task_spec.root.display_name or task_spec.root.node_id)

    if schema_ok:
        strengths.append("Every required node has a valid, mode-consistent decision output.")
    elif default_scales:
        weaknesses.append(
            "No measurement scale was available for "
            + ", ".join(default_scales[:6])
            + "; the estimate used a generic standardized scale and is not interpretable."
        )
        suggestions.append(
            ImprovementSuggestion(
                issue="Unknown output scale",
                suggestion=(
                    "Define output_scales (min, max, integer, unit) for these outputs in the task specification, "
                    "or make sure the companion LLM can compile the question book."
                ),
                priority=ImprovementPriority.HIGH,
            )
        )
    else:
        weaknesses.append(
            "Decision output is incomplete or invalid"
            + (f" (missing nodes: {', '.join(missing)})" if missing else "")
            + "."
        )
        suggestions.append(
            ImprovementSuggestion(
                issue="Invalid decision output",
                suggestion="Re-run the request; check that every task node produced questions and answers.",
                priority=ImprovementPriority.HIGH,
            )
        )
    if stable:
        strengths.append(
            f"Answers were stable across presentation orders (largest shift {max_instability:.2f} of its threshold)."
        )
    else:
        unstable_nodes = [
            nid
            for nid, s in dict(quality.get("per_node") or {}).items()
            if float(s.get("instability_ratio") or 0.0) > 1.0
        ]
        weaknesses.append(
            f"Answers shifted with presentation order (largest shift {max_instability:.2f} times its threshold)"
            + (f" for {', '.join(unstable_nodes[:4])}" if unstable_nodes else "")
            + "."
        )
        suggestions.append(
            ImprovementSuggestion(
                issue="Order-sensitive decision",
                suggestion=(
                    f"Distil the record into a focused evidence dossier for {focus}: keep the findings that bear on "
                    "it with their values, drop unrelated detail that distracts the decision model."
                ),
                priority=ImprovementPriority.HIGH,
            )
        )
    if sufficiency is not None:
        if float(sufficiency) >= 0.5:
            strengths.append(f"The decision model judged the evidence sufficient (p={float(sufficiency):.2f}).")
        elif sufficient:
            weaknesses.append(
                f"The decision model judged the evidence thin (p={float(sufficiency):.2f}); reported as uncertainty."
            )
        else:
            weaknesses.append(f"The decision model judged the evidence insufficient (p={float(sufficiency):.2f}).")
            suggestions.append(
                ImprovementSuggestion(
                    issue="Insufficient evidence in the state",
                    suggestion=(
                        f"Make the evidence for {focus} explicit: summarize each relevant domain with its key values "
                        "and deviations, and state which expected information is missing."
                    ),
                    priority=ImprovementPriority.HIGH,
                )
            )
    if covered:
        strengths.append(f"{coverage:.0%} of feature leaves reached the decision state.")
    else:
        weaknesses.append(f"Only {coverage:.0%} of feature leaves fit the decision state.")
        suggestions.append(
            ImprovementSuggestion(
                issue="Coverage lost in state packing",
                suggestion="Compress the large domains with the tools so every domain is represented within the state limit.",
                priority=ImprovementPriority.HIGH,
            )
        )
    if confidence < 0.45:
        weaknesses.append(f"Low decision confidence ({confidence:.2f}); reported as uncertainty, not as a failure.")
    placeholders = [str(x) for x in (quality.get("placeholder_items") or [])]
    if placeholders:
        weaknesses.append(
            "The question book still uses generic wording for "
            + ", ".join(placeholders[:6])
            + ("" if len(placeholders) <= 6 else f" and {len(placeholders) - 6} more")
            + "; add class_definitions or check the companion LLM."
        )
    guessed = []
    for node_id, node in dict((report.get("question_book") or {}).get("nodes") or {}).items():
        for out, scale in dict(node.get("scales") or {}).items():
            if isinstance(scale, dict) and scale.get("source") == "compiler":
                guessed.append(f"{out} ({scale.get('minimum'):g} to {scale.get('maximum'):g})")
    if guessed:
        weaknesses.append(
            "The range of " + ", ".join(guessed[:6]) + " was proposed by the companion LLM, not given in the task; "
            "set output_scales to the instrument's real range if it is wrong."
        )
    unassessed = [nid for nid, s in dict(quality.get("per_node") or {}).items() if s.get("unassessed")]
    if unassessed:
        weaknesses.append(
            "Stability could not be assessed for " + ", ".join(unassessed[:6]) + " (presentation orders went unanswered)."
        )

    reasoning = (
        f"Decision critic on the {route} route: schema={schema_ok}, max instability={max_instability:.2f} "
        f"of the threshold, evidence sufficiency="
        f"{'n/a' if sufficiency is None else f'{float(sufficiency):.2f}'} "
        f"({'gate ' + format(sufficiency_threshold, '.2f') if gate_sufficiency else 'reported only'}), "
        f"coverage={coverage:.2f}, mean confidence={confidence:.2f}."
    )
    if satisfactory:
        summary = f"Decision output accepted: stable across orders and supported by the evidence on the {route} route."
    elif route == "direct":
        summary = "Decision output rejected on the direct route; the next attempt distils the record with the multi-agent workflow."
    else:
        summary = "Decision output rejected; the next attempt re-orchestrates with this feedback."

    return CriticEvaluation(
        evaluation_id=str(uuid.uuid4())[:8],
        prediction_id=prediction.prediction_id,
        created_at=datetime.now(),
        verdict=verdict,
        confidence_in_verdict=min(0.95, max(0.4, checklist.pass_count / float(max(1, checklist.total_count)))),
        composite_score=float(round(max(0.0, min(1.0, composite)), 4)),
        score_breakdown=breakdown,
        checklist=checklist,
        strengths=strengths,
        weaknesses=weaknesses,
        improvement_suggestions=suggestions,
        domains_missed=[],
        reasoning=reasoning,
        concise_summary=summary,
    )


ROUTE_INDEPENDENT_ISSUES = ("Unknown output scale",)


def route_can_help(evaluation: CriticEvaluation) -> bool:
    """False when every high-priority issue is one another evidence route cannot fix."""
    high = [s for s in evaluation.improvement_suggestions if s.priority == ImprovementPriority.HIGH]
    if not high:
        return True
    return any(s.issue not in ROUTE_INDEPENDENT_ISSUES for s in high)
