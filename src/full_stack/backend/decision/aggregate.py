"""
From decision-model answers to COMPASS node predictions.

Classification nodes
    Every presentation order of a node's question (and, for binary nodes, the
    Noul) gives one probability distribution over the class labels. The node's
    distribution is their mean, kept at full float precision. Stability is the
    mean total variation distance of each order from that mean (0 means every
    order agreed exactly).

Regression outputs
    The ascending and descending coarse Scores are mapped back onto the same
    levels and averaged. When refined, the zoomed Score replaces the coarse
    window. The point estimate is the mean of the combined density; the
    standard deviation and quantiles are reported next to it. Stability is the
    gap between the ascending and descending means as a fraction of the range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.models.prediction_result import (
    ClassificationPrediction,
    ConfidenceLevel,
    NodePrediction,
    RegressionPrediction,
)
from ..data.models.prediction_task import PredictionMode, PredictionTaskNode
from .questions import AskedQuestion
from .scales import OutputScale, ScaleBin, best_window, combine_density, density_summary


def confidence_level(score: float) -> ConfidenceLevel:
    if score >= 0.75:
        return ConfidenceLevel.HIGH
    if score >= 0.45:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _normalize(dist: Dict[str, float], labels: Sequence[str]) -> Dict[str, float]:
    values = {label: max(0.0, float(dist.get(label, 0.0) or 0.0)) for label in labels}
    total = sum(values.values())
    if total <= 0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: v / total for label, v in values.items()}


def _tv(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _answer_probabilities(answer: Dict[str, Any]) -> Dict[str, float]:
    raw = answer.get("probabilities") or {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _confidence_of(answer: Dict[str, Any]) -> Optional[float]:
    try:
        value = answer.get("confidence")
        return None if value is None else max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- classification
@dataclass
class ClassificationAggregate:
    probabilities: Dict[str, float]
    predicted_label: str
    variants: List[Dict[str, Any]]
    stability: float  # mean TV distance from the ensemble mean
    confidence: float


def aggregate_classification(
    node: PredictionTaskNode,
    questions: Sequence[AskedQuestion],
    answers: Dict[str, Dict[str, Any]],
) -> ClassificationAggregate:
    labels = list(node.class_labels)
    variants: List[Dict[str, Any]] = []
    confidences: List[float] = []
    for q in questions:
        answer = answers.get(q.qid)
        if not isinstance(answer, dict):
            continue
        if q.kind == "binary_noul":
            try:
                p = max(0.0, min(1.0, float(answer.get("noul"))))
            except (TypeError, ValueError):
                continue
            first, second = q.order
            dist = {first: p, second: 1.0 - p}
            variants.append({"variant": q.variant, "probabilities": dist})
            confidences.append(abs(2.0 * p - 1.0))
        elif q.kind == "choice":
            dist = _normalize(_answer_probabilities(answer), labels)
            variants.append({"variant": q.variant, "order": list(q.order), "probabilities": dist, "choice": answer.get("choice")})
            conf = _confidence_of(answer)
            if conf is not None:
                confidences.append(conf)
    if not variants:
        raise ValueError(f"No usable answers for classification node '{node.node_id}'.")
    mean = {label: sum(v["probabilities"].get(label, 0.0) for v in variants) / len(variants) for label in labels}
    mean = _normalize(mean, labels)
    stability = sum(_tv(v["probabilities"], mean) for v in variants) / len(variants)
    predicted = max(labels, key=lambda label: (mean[label], -labels.index(label)))
    confidence = sum(confidences) / len(confidences) if confidences else abs(2.0 * mean[predicted] - 1.0)
    return ClassificationAggregate(
        probabilities=mean,
        predicted_label=predicted,
        variants=variants,
        stability=float(stability),
        confidence=float(confidence),
    )


# ------------------------------------------------------------------ regression
def score_distribution(q: AskedQuestion, answer: Dict[str, Any]) -> List[float]:
    """Probabilities per bin (in bin order), mapped back from the presented order."""
    n = len(q.bins)
    probs = [0.0] * n
    raw = _answer_probabilities(answer)
    for level_text, p in raw.items():
        try:
            level = int(float(level_text))
        except ValueError:
            continue
        if 0 <= level < len(q.order):
            probs[int(q.order[level])] += max(0.0, p)
    total = sum(probs)
    if total <= 0:
        # Fall back to the weighted score when probabilities are missing.
        try:
            s = float(answer.get("score"))
            level = min(n - 1, max(0, int(round(s))))
            probs[int(q.order[level])] = 1.0
            total = 1.0
        except (TypeError, ValueError):
            return [1.0 / n] * n
    return [p / total for p in probs]


def _mean_of(bins: Sequence[ScaleBin], probs: Sequence[float]) -> float:
    return float(sum(b.center * p for b, p in zip(bins, probs)))


@dataclass
class RegressionAggregate:
    value: float
    summary: Dict[str, float]
    coarse: List[float]
    fine: Optional[List[float]]
    window: Optional[Tuple[float, float]]
    stability: float  # |ascending mean - descending mean| / range
    confidence: float
    detail: Dict[str, Any] = field(default_factory=dict)


def coarse_ensemble(
    questions: Sequence[AskedQuestion], answers: Dict[str, Dict[str, Any]]
) -> Tuple[List[ScaleBin], List[float], List[Dict[str, Any]], List[float]]:
    bins: List[ScaleBin] = []
    runs: List[Dict[str, Any]] = []
    confidences: List[float] = []
    for q in questions:
        answer = answers.get(q.qid)
        if not isinstance(answer, dict):
            continue
        bins = list(q.bins)
        dist = score_distribution(q, answer)
        runs.append({"variant": q.variant, "probabilities": dist, "mean": _mean_of(bins, dist)})
        conf = _confidence_of(answer)
        if conf is not None:
            confidences.append(conf)
    if not runs:
        return bins, [], runs, confidences
    mean = [sum(r["probabilities"][i] for r in runs) / len(runs) for i in range(len(bins))]
    return bins, mean, runs, confidences


def refinement_window(scale: OutputScale, bins: Sequence[ScaleBin], probs: Sequence[float]) -> Optional[Tuple[int, int]]:
    if not bins or not probs or bins[0].exact:
        return None
    start, stop = best_window(probs, width=3 if len(bins) > 3 else len(bins))
    if stop - start <= 0:
        return None
    return start, stop


def aggregate_regression_output(
    scale: OutputScale,
    coarse_questions: Sequence[AskedQuestion],
    fine_questions: Sequence[AskedQuestion],
    answers: Dict[str, Dict[str, Any]],
    window: Optional[Tuple[int, int]],
) -> RegressionAggregate:
    bins, coarse, coarse_runs, confidences = coarse_ensemble(coarse_questions, answers)
    if not coarse:
        raise ValueError(f"No usable answers for regression output '{scale.output}'.")
    fine_bins, fine, fine_runs, fine_conf = coarse_ensemble(fine_questions, answers) if fine_questions else ([], [], [], [])
    window_bounds = None
    if window is not None and fine:
        pieces = combine_density(bins, coarse, window=window, fine_bins=fine_bins, fine_probs=fine)
        window_bounds = (bins[window[0]].lo, bins[window[1] - 1].hi)
    else:
        pieces = combine_density(bins, coarse)
    summary = density_summary(pieces)
    means = [r["mean"] for r in coarse_runs]
    stability = (max(means) - min(means)) / scale.span if len(means) > 1 else 0.0
    all_conf = confidences + fine_conf
    confidence = sum(all_conf) / len(all_conf) if all_conf else 0.5
    detail = {
        "scale": scale.to_dict(),
        "coarse_levels": [{"lo": b.lo, "hi": b.hi, "center": b.center, "p": p} for b, p in zip(bins, coarse)],
        "coarse_runs": [{"variant": r["variant"], "mean": r["mean"]} for r in coarse_runs],
        "refined": bool(window_bounds),
    }
    if window_bounds:
        detail["window"] = list(window_bounds)
        detail["fine_levels"] = [{"lo": b.lo, "hi": b.hi, "center": b.center, "p": p} for b, p in zip(fine_bins, fine)]
        detail["fine_runs"] = [{"variant": r["variant"], "mean": r["mean"]} for r in fine_runs]
    return RegressionAggregate(
        value=float(summary["mean"]),
        summary=summary,
        coarse=coarse,
        fine=fine or None,
        window=window_bounds,
        stability=float(min(1.0, max(0.0, stability))),
        confidence=float(confidence),
        detail=detail,
    )


# --------------------------------------------------------------------- nodes
def build_node_prediction(
    node: PredictionTaskNode,
    *,
    path: str,
    classification: Optional[ClassificationAggregate],
    regression: Dict[str, RegressionAggregate],
    children: List[NodePrediction],
    sufficiency: Optional[float],
) -> NodePrediction:
    details: Dict[str, Any] = {"source": "decision_model"}
    uncertainty: List[str] = []
    if classification is not None:
        stability = classification.stability
        confidence = classification.confidence
        details["variants"] = classification.variants
        details["stability_tv"] = stability
        cls = ClassificationPrediction(
            predicted_label=classification.predicted_label,
            probabilities=classification.probabilities,
        )
        reg = None
    else:
        stabilities = [agg.stability for agg in regression.values()]
        confidences = [agg.confidence for agg in regression.values()]
        stability = sum(stabilities) / len(stabilities) if stabilities else 0.0
        confidence = sum(confidences) / len(confidences) if confidences else 0.5
        details["outputs"] = {name: agg.detail for name, agg in regression.items()}
        details["stability_range_fraction"] = stability
        cls = None
        reg = RegressionPrediction(
            values={name: agg.value for name, agg in regression.items()},
            uncertainty={name: agg.summary for name, agg in regression.items()},
        )
    if sufficiency is not None:
        details["evidence_sufficiency"] = sufficiency
        if sufficiency < 0.5:
            uncertainty.append(f"Decision model judged the evidence insufficient (p={sufficiency:.2f}).")
    if stability > 0.2:
        uncertainty.append(f"Answer moved between presentation orders (instability {stability:.2f}).")
    return NodePrediction(
        node_id=node.node_id,
        path=path,
        mode=node.mode,
        classification=cls,
        regression=reg,
        confidence_level=confidence_level(confidence),
        confidence_score=max(0.0, min(1.0, confidence)),
        uncertainty_factors=uncertainty,
        children=children,
        decision_details=details,
    )


def is_classification(node: PredictionTaskNode) -> bool:
    return node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION)
