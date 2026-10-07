"""
Compact, lossless text renderings of a participant record.

The input files are verbose as JSON: every multimodal leaf repeats its whole
ontology path, and the deviation map repeats each leaf's z-score next to the
leaf itself. Rendered with the generic serializers this roughly doubles the
token count of a record. The renderings here carry the same information once:

    measurements       every feature leaf as "label = value (z +1.23)", grouped
                       under its ontology path; the z appears only when the
                       record has a reference sample for that leaf
    deviation profile  every scored ontology group: a mean absolute deviation as
                       "mean |z|" (no direction), a signed score with its sign;
                       leaf scores already in the measurements are not repeated
    data overview      how the values are expressed, and leaf coverage per domain

Leaf values are preprocessed measurements on their native scale. A deviation
score is optional: it exists only where a reference sample (healthy controls,
a normative model, the rest of a cohort) was available, so a record can hold
deviation scores for every leaf, for some, or for none.

Both the direct route of the LLM Predictor and the state of a structured
decision model use them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .multimodal_coverage import flatten_features
from ..toon import json_to_toon

MeasurementRow = Tuple[str, str, Optional[float]]


def format_z(z: Any) -> Optional[str]:
    try:
        value = float(z)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return f"{value:+.2f}"


def label_states_value(label: str, value: Any) -> bool:
    """True when a leaf label already ends with its value ("Age: 58.2 years", "Sex = Female")."""
    text = str(value).strip()
    if not text:
        return True
    return re.search(r"(?:^|[:=]\s*)" + re.escape(text) + r"\s*$", str(label).strip()) is not None


def leaf_line(feat: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    label = str(feat.get("field_name") or feat.get("feature") or feat.get("feature_id") or "feature").strip()
    value = feat.get("value")
    z_text = format_z(feat.get("z_score"))
    z_val = float(feat["z_score"]) if z_text is not None else None
    parts = [label]
    show_value = value is not None and str(value).strip() != ""
    if show_value:
        # The loader falls back to the z-score as value; do not print it twice.
        try:
            if z_val is not None and abs(float(value) - z_val) < 1e-9:
                show_value = False
        except (TypeError, ValueError):
            pass
    if show_value and label_states_value(label, value):
        show_value = False
    if show_value:
        unit = str(feat.get("unit") or "").strip()
        parts.append(f"= {value}{(' ' + unit) if unit else ''}")
    if z_text is not None:
        parts.append(f"(z {z_text})")
    qualifiers = feat.get("qualifiers") or {}
    if isinstance(qualifiers, dict) and qualifiers:
        parts.append("[" + "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in qualifiers.items()) + "]")
    return " ".join(parts), z_val


def measurement_rows(multimodal: Any) -> List[MeasurementRow]:
    """(group path, leaf line, z) for every feature leaf, in input order."""
    rows: List[MeasurementRow] = []
    for _key, feat in flatten_features(multimodal or {}):
        group = _leaf_group(feat)
        line, z = leaf_line(feat)
        rows.append((group, line, z))
    return rows


def _leaf_group(feat: Dict[str, Any]) -> str:
    domain = str(feat.get("domain") or "").strip()
    path = [str(p) for p in (feat.get("path_in_hierarchy") or []) if str(p).strip()]
    return " > ".join([p for p in [domain, *path] if p]) or "Measurements"


_LEADING_NUMBER = re.compile(r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")


def _is_number(value: Any) -> bool:
    """A numeric measurement: a number, or text that starts with one ("2.41 mm")."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == value and value not in (float("inf"), float("-inf"))
    return bool(_LEADING_NUMBER.match(str(value)))


VALUE_MODES = ("deviation", "native", "mixed", "categorical", "empty")


def value_representation(multimodal: Any) -> Dict[str, Any]:
    """
    How the leaves of a record express their values.

    A leaf with a z carries a deviation from a reference sample; a numeric leaf
    without one is a preprocessed value on its native scale; anything else is
    categorical. The mode and the note tell every agent which of these it reads.
    """
    with_z = native = categorical = 0
    for _key, feat in flatten_features(multimodal or {}):
        if format_z(feat.get("z_score")) is not None:
            with_z += 1
        elif _is_number(feat.get("value")):
            native += 1
        elif feat.get("value") not in (None, ""):
            categorical += 1
    if with_z and native:
        mode = "mixed"
    elif with_z:
        mode = "deviation"
    elif native:
        mode = "native"
    elif categorical:
        mode = "categorical"
    else:
        mode = "empty"
    return {
        "mode": mode,
        "leaves_with_deviation": with_z,
        "numeric_leaves_without_deviation": native,
        "categorical_leaves": categorical,
        "note": describe_value_representation(mode, with_z, native),
    }


def describe_value_representation(mode: str, with_z: int = 0, native: int = 0) -> str:
    """One plain statement of how this record's values are expressed."""
    if mode == "deviation":
        return (
            "Every numeric leaf carries a deviation score z against a reference sample "
            "(for example healthy controls or a normative model); group scores in the "
            "deviation profile aggregate these deviations."
        )
    if mode == "native":
        return (
            "No leaf carries a deviation score: the values are preprocessed measurements on "
            "their native scale, without a reference sample, so there are no group scores. "
            "Interpret each value from its label, unit and qualifiers (such as a reference "
            "range), from the study context and from domain knowledge; never read a value "
            "as a z-score."
        )
    if mode == "mixed":
        return (
            f"{with_z} of {with_z + native} numeric leaves carry a deviation score z against a "
            f"reference sample; the other {native} are preprocessed measurements on their native "
            "scale, without a reference. Group scores aggregate only the deviation-scored leaves; "
            "never read a native value as a z-score."
        )
    if mode == "categorical":
        return "The record holds categorical values only; there are no deviation scores."
    return "The record holds no measured values."


def measurement_groups(multimodal: Any) -> set:
    """Ontology paths that directly hold feature leaves."""
    return {_leaf_group(feat) for _key, feat in flatten_features(multimodal or {})}


def render_measurements(rows: Sequence[MeasurementRow], *, min_abs_z: float = 0.0) -> Tuple[str, int]:
    """Group leaves by ontology path. Leaves without a z (native-scale or categorical) are always kept."""
    out: List[str] = []
    current = None
    kept = 0
    for group, line, z in rows:
        if min_abs_z > 0 and z is not None and abs(z) < min_abs_z:
            continue
        if group != current:
            out.append(f"[{group}]")
            current = group
        out.append(f"- {line}")
        kept += 1
    return "\n".join(out), kept


def _node_get(node: Any, key: str) -> Any:
    return node.get(key) if isinstance(node, dict) else getattr(node, key, None)


def render_deviation_profile(deviation: Any, *, max_depth: int = 99, leaf_groups: Optional[set] = None) -> str:
    """
    Every scored node of the deviation tree, by ontology path. A group's mean
    absolute deviation is printed as a magnitude ("mean |z| 1.39"); a signed
    score as "+1.39". Nodes that are feature leaves already listed in the
    measurements (their parent path holds leaves there) are left out.
    """
    leaf_groups = set(leaf_groups or ())
    lines: List[str] = []

    def walk(node: Any, trail: List[str], depth: int) -> None:
        children = list(_node_get(node, "children") or [])
        name = str(_node_get(node, "node_name") or _node_get(node, "node_id") or "").strip()
        here = trail + ([name] if name and name.upper() != "ROOT" else [])
        path = " > ".join(here)
        if here and depth <= max_depth:
            duplicate_leaf = not children and " > ".join(trail) in leaf_groups and path not in leaf_groups
            score = format_z(_node_get(node, "z_score"))
            if score is not None and not duplicate_leaf:
                if _node_get(node, "score_kind") == "mean_abs":
                    lines.append(f"{path}: mean |z| {abs(float(_node_get(node, 'z_score'))):.2f}")
                else:
                    lines.append(f"{path}: {score}")
        if depth < max_depth:
            for child in children:
                walk(child, here, depth + 1)

    root = _node_get(deviation, "root") if deviation is not None else None
    if root is not None:
        walk(root, [], 0)
    if not lines:
        summaries = _node_get(deviation, "domain_summaries") if deviation is not None else None
        if isinstance(summaries, dict):
            for domain, summary in summaries.items():
                score = summary.get("mean_abs_score") if isinstance(summary, dict) else None
                if isinstance(score, (int, float)):
                    lines.append(f"{domain}: mean |z| {abs(float(score)):.2f}")
    return "\n".join(lines)


def render_data_overview(overview: Dict[str, Any]) -> str:
    """The value representation, then one line per domain: leaves present out of total."""
    lines: List[str] = []
    note = str(((overview or {}).get("value_representation") or {}).get("note") or "").strip()
    if note:
        lines.append(f"Values: {note}")
    for name, cov in dict((overview or {}).get("domain_coverage") or {}).items():
        if not isinstance(cov, dict):
            continue
        present = int(cov.get("present_leaves") or 0)
        total = int(cov.get("total_leaves") or 0)
        pct = cov.get("coverage_percentage")
        pct_text = f" ({float(pct):.0f}%)" if isinstance(pct, (int, float)) else ""
        lines.append(f"{name}: {present}/{total} leaves present{pct_text}")
    return "\n".join(lines)


DIRECT_RECORD_HEADER = (
    "Every feature leaf of the participant record, grouped by ontology path, as "
    "'label = preprocessed value unit (z deviation from the reference sample, two decimals) "
    "[qualifiers]'. A z appears only where the record has a reference sample; a leaf without "
    "one is a value on its native scale or a categorical value. No leaf was summarised or left out."
)


def render_direct_record(
    *,
    multimodal: Any,
    deviation: Any,
    overview: Dict[str, Any],
    notes: str,
) -> Dict[str, str]:
    """The four compact sections of a complete record."""
    measurements, _kept = render_measurements(measurement_rows(multimodal))
    return {
        "clinical_record": str(notes or "").strip(),
        "deviation_profile": render_deviation_profile(deviation, leaf_groups=measurement_groups(multimodal)),
        "data_overview": render_data_overview(overview or {}),
        "measurements": measurements,
    }
