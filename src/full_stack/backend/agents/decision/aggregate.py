"""
From decision-model answers to COMPASS node predictions.

Classification nodes
    Every presentation of a node's question (each Choice order and, for binary
    nodes, each Noul) gives one probability distribution over the class labels.
    The node's distribution is their mean, kept at full float precision.
    Instability is the largest total variation distance between any two
    presentations (0 means they all agreed exactly).

Regression outputs
    The ascending and descending coarse Scores are mapped back onto the same
    levels and averaged. When refined, the zoomed Score replaces the coarse
    window. The point estimate is the mean of the combined density; the
    standard deviation and quantiles are reported next to it. Instability is
    the largest gap between the ascending and descending means (coarse or
    refined), in reference standard deviations (the scale's reference SD, or a
    sixth of its range when none is known).

A node answered in fewer presentations than were asked cannot show that it is
stable: it is flagged `unassessed`, and the critic treats it as unstable
whatever the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...data.models.prediction_result import (
    ClassificationPrediction,
    ConfidenceLevel,
    NodePrediction,
    RegressionPrediction,
)
from ...data.models.prediction_task import PredictionMode, PredictionTaskNode
from .questions import AskedQuestion
from .scales import OutputScale, ScaleBin, best_window, combine_density, density_summary


def confidence_level(score: float) -> ConfidenceLevel:
    if score >= 0.75:
        return ConfidenceLevel.HIGH
    if score >= 0.45:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _label_key(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def _match_labels(dist: Dict[str, float], labels: Sequence[str]) -> Optional[Dict[str, float]]:
    """Map answer keys onto the class labels (exact, then case and space insensitive); None when nothing matches."""
    lookup = {_label_key(k): v for k, v in dist.items()}
    values: Dict[str, float] = {}
    matched = 0
    for label in labels:
        if label in dist:
            raw = dist[label]
        elif _label_key(label) in lookup:
            raw = lookup[_label_key(label)]
        else:
            continue
        matched += 1
        values[label] = max(0.0, float(raw or 0.0))
    if matched == 0:
        return None
    for label in labels:
        values.setdefault(label, 0.0)
    total = sum(values.values())
    if total <= 0:
        return None
    return {label: values[label] / total for label in labels}


def _normalize(dist: Dict[str, float], labels: Sequence[str]) -> Dict[str, float]:
    matched = _match_labels(dist, labels)
    if matched is None:
        return {label: 1.0 / len(labels) for label in labels}
    return matched


def _tv(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _answer_probabilities(answer: Dict[str, Any]) -> Dict[str, float]:
    """Probabilities keyed as returned; a list is read as level-ordered ("0", "1", ...)."""
    raw = answer.get("probabilities") or {}
    if isinstance(raw, (list, tuple)):
        raw = {str(i): v for i, v in enumerate(raw)}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        number = _finite(value)
        if number is not None:
            out[str(key)] = number
    return out


def noul_value(answer: Any) -> Optional[float]:
    if not isinstance(answer, dict):
        return None
    p = _finite(answer.get("noul"))
    return None if p is None else max(0.0, min(1.0, p))


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
    stability: float  # largest TV distance between two presentations
    confidence: float
    asked: int = 0
    answered: int = 0
    unassessed: bool = False  # fewer presentations answered than needed to judge stability


def _orientation(q: AskedQuestion, labels: Sequence[str]) -> str:
    """Which label a binary presentation puts first ("first" or "second")."""
    return "first" if q.order and q.order[0] == labels[0] else "second"


def aggregate_classification(
    node: PredictionTaskNode,
    questions: Sequence[AskedQuestion],
    answers: Dict[str, Dict[str, Any]],
) -> ClassificationAggregate:
    labels = list(node.class_labels)
    binary = len(labels) == 2
    variants: List[Dict[str, Any]] = []
    confidences: List[float] = []
    nouls: Dict[str, float] = {}
    asked = [q for q in questions if q.kind in ("binary_noul", "choice")]
    for q in asked:
        answer = answers.get(q.qid)
        if q.kind == "binary_noul":
            p = noul_value(answer)
            if p is None:
                continue
            label, other = q.order
            dist = {label: p, other: 1.0 - p}
            nouls[label] = p
            variants.append({"variant": q.variant, "probabilities": dist, "kind": "noul", "orientation": _orientation(q, labels)})
            confidences.append(abs(2.0 * p - 1.0))
        elif isinstance(answer, dict):
            dist = _match_labels(_answer_probabilities(answer), labels)
            if dist is None:
                continue  # keys that match no label are a missing answer, not a uniform one
            variants.append(
                {
                    "variant": q.variant,
                    "order": list(q.order),
                    "probabilities": dist,
                    "choice": answer.get("choice"),
                    "kind": "choice",
                    "orientation": _orientation(q, labels) if binary else q.variant,
                }
            )
            conf = _confidence_of(answer)
            if conf is not None:
                confidences.append(conf)
    if not variants:
        # Nothing usable came back: abstain with a uniform distribution, flagged
        # so the critic rejects the attempt instead of the run failing.
        uniform = {label: 1.0 / len(labels) for label in labels}
        return ClassificationAggregate(
            probabilities=uniform,
            predicted_label=labels[0],
            variants=[],
            stability=1.0,
            confidence=0.0,
            asked=len(asked),
            answered=0,
            unassessed=True,
        )
    mean = {label: sum(v["probabilities"].get(label, 0.0) for v in variants) / len(variants) for label in labels}
    mean = _normalize(mean, labels)

    # Presentations compared for stability. The two Nouls of a binary node form
    # one presentation (their lean toward "yes" cancels in the mean and is
    # reported separately as yes_bias), compared with each Choice order.
    presentations = [v["probabilities"] for v in variants if v["kind"] == "choice"]
    noul_rows = [v for v in variants if v["kind"] == "noul"]
    yes_bias = None
    if len(noul_rows) == 2:
        pair = {label: (noul_rows[0]["probabilities"][label] + noul_rows[1]["probabilities"][label]) / 2.0 for label in labels}
        presentations.append(pair)
        yes_bias = (sum(nouls.values()) - 1.0) / 2.0
    else:
        presentations.extend(v["probabilities"] for v in noul_rows)

    # Presentations that were asked, counted the way they are compared.
    asked_presentations = sum(1 for q in asked if q.kind == "choice") + (
        1 if sum(1 for q in asked if q.kind == "binary_noul") >= 2 else sum(1 for q in asked if q.kind == "binary_noul")
    )
    unassessed = asked_presentations >= 2 and len(presentations) < 2
    if binary and not unassessed:
        asked_orientations = {_orientation(q, labels) for q in asked}
        answered_orientations = {v["orientation"] for v in variants}
        unassessed = len(asked_orientations) >= 2 and len(answered_orientations) < 2
    if unassessed:
        stability = 1.0
    else:
        stability = max(
            (_tv(a, b) for i, a in enumerate(presentations) for b in presentations[i + 1 :]),
            default=0.0,
        )
    predicted = max(labels, key=lambda label: (mean[label], -labels.index(label)))
    confidence = sum(confidences) / len(confidences) if confidences else abs(2.0 * mean[predicted] - 1.0)
    for row in variants:
        row.pop("kind", None)
    if yes_bias is not None:
        variants.append({"variant": "noul_pair", "yes_bias": yes_bias})
    return ClassificationAggregate(
        probabilities=mean,
        predicted_label=predicted,
        variants=variants,
        stability=float(stability),
        confidence=float(confidence),
        asked=len(asked),
        answered=len([v for v in variants if "probabilities" in v]),
        unassessed=unassessed,
    )


# ------------------------------------------------------------------ regression
def score_distribution(q: AskedQuestion, answer: Dict[str, Any]) -> Optional[List[float]]:
    """
    Probabilities per bin (in bin order), mapped back from the presented order.
    Levels may be keyed by index ("0", "1", ...) or by their criterion text. None
    when the answer carries no usable probability mass and no usable score.
    """
    n = len(q.bins)
    probs = [0.0] * n
    raw = _answer_probabilities(answer) if isinstance(answer, dict) else {}
    criteria = list((q.payload or {}).get("criteria") or [])
    by_text = {" ".join(str(c).split()).casefold(): i for i, c in enumerate(criteria)}
    for level_key, p in raw.items():
        level: Optional[int] = None
        number = _finite(level_key)
        if number is not None and float(number).is_integer():
            level = int(number)
        else:
            level = by_text.get(" ".join(str(level_key).split()).casefold())
        if level is not None and 0 <= level < len(q.order):
            probs[int(q.order[level])] += max(0.0, p)
    total = sum(probs)
    if total <= 0:
        score = _finite(answer.get("score")) if isinstance(answer, dict) else None
        if score is None:
            return None
        level = min(n - 1, max(0, int(round(score))))
        probs[int(q.order[level])] = 1.0
        total = 1.0
    return [p / total for p in probs]


def answer_usable(q: AskedQuestion, answer: Any, labels: Sequence[str] = ()) -> bool:
    """Whether an answer can be used for its question kind (otherwise it is re-asked)."""
    if not isinstance(answer, dict):
        return False
    if q.kind in ("binary_noul", "sufficiency"):
        return noul_value(answer) is not None
    if q.kind == "choice":
        return _match_labels(_answer_probabilities(answer), list(labels or q.order)) is not None
    if q.kind in ("score_coarse", "score_fine"):
        return score_distribution(q, answer) is not None
    return True


def _mean_of(bins: Sequence[ScaleBin], probs: Sequence[float]) -> float:
    return float(sum(b.center * p for b, p in zip(bins, probs)))


@dataclass
class RegressionAggregate:
    value: float
    summary: Dict[str, float]
    coarse: List[float]
    fine: Optional[List[float]]
    window: Optional[Tuple[float, float]]
    stability: float  # largest |ascending mean - descending mean|, in reference SDs
    confidence: float
    detail: Dict[str, Any] = field(default_factory=dict)
    unassessed: bool = False  # a level order went unanswered


def coarse_ensemble(
    questions: Sequence[AskedQuestion], answers: Dict[str, Dict[str, Any]]
) -> Tuple[List[ScaleBin], List[float], List[Dict[str, Any]], List[float]]:
    bins: List[ScaleBin] = []
    runs: List[Dict[str, Any]] = []
    confidences: List[float] = []
    for q in questions:
        bins = list(q.bins)
        answer = answers.get(q.qid)
        dist = score_distribution(q, answer) if isinstance(answer, dict) else None
        if dist is None:
            continue  # unusable answers are missing, never a uniform "stable" answer
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
    width = 3 if len(bins) > 3 else len(bins)
    start, stop = best_window(probs, width=width)
    if stop - start <= 0:
        return None
    # A flat answer has no region worth zooming into.
    if sum(probs[start:stop]) <= width / float(len(bins)) + 0.05:
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
        # Nothing usable came back: abstain over the whole range, flagged so the
        # critic rejects the attempt instead of the run failing.
        bins = list(coarse_questions[0].bins) if coarse_questions else scale.coarse_grid(10)
        coarse = [1.0 / len(bins)] * len(bins)
        summary = density_summary(combine_density(bins, coarse))
        return RegressionAggregate(
            value=float(summary["mean"]),
            summary=summary,
            coarse=coarse,
            fine=None,
            window=None,
            stability=1.0,
            confidence=0.0,
            detail={"scale": scale.to_dict(), "abstained": True, "refined": False},
            unassessed=True,
        )
    fine_bins, fine, fine_runs, fine_conf = coarse_ensemble(fine_questions, answers) if fine_questions else ([], [], [], [])
    window_bounds = None
    if window is not None and fine:
        pieces = combine_density(bins, coarse, window=window, fine_bins=fine_bins, fine_probs=fine)
        window_bounds = (bins[window[0]].lo, bins[window[1] - 1].hi)
    else:
        pieces = combine_density(bins, coarse)
    summary = density_summary(pieces)
    # Gaps are judged in reference SDs, but never against an SD narrower than half
    # a coarse level: the levels themselves cannot resolve finer differences.
    reference_sd = max(float(scale.reference_sd) if scale.reference_sd else scale.span / 6.0, scale.span / 20.0)
    gaps: List[float] = []
    for runs, asked in ((coarse_runs, coarse_questions), (fine_runs, fine_questions)):
        if not asked:
            continue
        if len(runs) < min(2, len(asked)):
            gaps.append(float("inf"))
            continue
        means = [r["mean"] for r in runs]
        gaps.append(max(means) - min(means))
    worst = max(gaps, default=0.0)
    unassessed = worst == float("inf")
    stability = 1.0 if unassessed else worst / max(1e-9, reference_sd)
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
        stability=float(max(0.0, stability)),
        confidence=float(confidence),
        detail=detail,
        unassessed=unassessed,
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
    stability_threshold: float = 0.25,
    regression_stability_threshold: float = 0.5,
) -> NodePrediction:
    details: Dict[str, Any] = {"source": "decision_model"}
    uncertainty: List[str] = []
    if classification is not None:
        stability = classification.stability
        unassessed = classification.unassessed
        confidence = classification.confidence
        details["variants"] = classification.variants
        details["instability_max_tv"] = stability
        details["presentations"] = {"asked": classification.asked, "answered": classification.answered}
        threshold = stability_threshold
        cls = ClassificationPrediction(
            predicted_label=classification.predicted_label,
            probabilities=classification.probabilities,
        )
        reg = None
    else:
        stabilities = [agg.stability for agg in regression.values()]
        confidences = [agg.confidence for agg in regression.values()]
        # The node is as unstable as its least stable output (what the critic reads).
        stability = max(stabilities) if stabilities else 0.0
        unassessed = any(agg.unassessed for agg in regression.values())
        confidence = sum(confidences) / len(confidences) if confidences else 0.5
        details["outputs"] = {name: agg.detail for name, agg in regression.items()}
        details["instability_reference_sd"] = stability
        threshold = regression_stability_threshold
        cls = None
        reg = RegressionPrediction(
            values={name: agg.value for name, agg in regression.items()},
            uncertainty={name: agg.summary for name, agg in regression.items()},
        )
    if sufficiency is not None:
        details["evidence_sufficiency"] = sufficiency
        if sufficiency < 0.5:
            uncertainty.append(f"Decision model judged the evidence insufficient (p={sufficiency:.2f}).")
    if unassessed:
        uncertainty.append("Not every presentation order was answered, so stability could not be assessed.")
    elif stability > threshold:
        uncertainty.append(f"Answer moved between presentation orders (instability {stability:.2f}).")
    details["instability"] = stability
    details["instability_threshold"] = threshold
    details["unassessed"] = unassessed
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
