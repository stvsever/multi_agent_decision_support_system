"""
The state a decision model reads.

A decision model sees one JSON object (the state) and nothing else. COMPASS
builds it from the same evidence an LLM Predictor would get, as named sections:

    task_context          study context and target definitions (runtime instructions)
    clinical_record       free-text notes
    phenotype_synthesis   tool outputs of the orchestrated workflow (orchestrated route only)
    chunk_evidence        per-chunk evidence rows (orchestrated route, when chunking ran)
    measurements          every feature leaf: label, value, deviation z, grouped by ontology path
    deviation_profile     aggregate deviation of each ontology group (no leaf duplicates)

Each section has renderings from complete to compact. The packer starts from the
complete renderings and, only while the state exceeds the model's budget, steps
the least important section down one rendering at a time. Measurements are
compacted by deviation (|z| at least 0.5, then 1, then 1.5 and 2), never at
random. The report records which rendering each section ended at and how many
feature leaves the final state still holds, so the critic and the run record can
see any coverage loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..utils.core.multimodal_coverage import flatten_features
from ..utils.toon import json_to_toon

TokenCounter = Callable[[str], int]


@dataclass
class StateSection:
    key: str
    priority: int  # lower is more important
    variants: List[str]
    variant_labels: List[str]
    feature_counts: List[int] = field(default_factory=list)


@dataclass
class PackedState:
    state: Dict[str, str]
    tokens: int
    budget: int
    sections: List[Dict[str, Any]]
    features_total: int
    features_included: int
    truncated: bool

    @property
    def feature_coverage(self) -> float:
        if self.features_total <= 0:
            return 1.0
        return self.features_included / float(self.features_total)

    def report(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "budget_tokens": self.budget,
            "features_total": self.features_total,
            "features_included": self.features_included,
            "feature_coverage": round(self.feature_coverage, 4),
            "truncated": self.truncated,
            "sections": self.sections,
        }


# ---------------------------------------------------------------- renderers
def _fmt_z(z: Any) -> Optional[str]:
    try:
        value = float(z)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return f"{value:+.2f}"


def _leaf_line(feat: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    label = str(feat.get("field_name") or feat.get("feature") or feat.get("feature_id") or "feature").strip()
    value = feat.get("value")
    z_text = _fmt_z(feat.get("z_score"))
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
    if show_value and str(value).strip() in label:
        show_value = False
    if show_value:
        unit = str(feat.get("unit") or "").strip()
        parts.append(f"= {value}{(' ' + unit) if unit else ''}")
    if z_text is not None:
        parts.append(f"(z {z_text})")
    return " ".join(parts), z_val


def measurement_rows(multimodal: Any) -> List[Tuple[str, str, Optional[float]]]:
    """(group path, leaf line, z) for every feature leaf, in input order."""
    rows: List[Tuple[str, str, Optional[float]]] = []
    for _key, feat in flatten_features(multimodal or {}):
        domain = str(feat.get("domain") or "").strip()
        path = [str(p) for p in (feat.get("path_in_hierarchy") or []) if str(p).strip()]
        group = " > ".join([p for p in [domain, *path] if p]) or "Measurements"
        line, z = _leaf_line(feat)
        rows.append((group, line, z))
    return rows


def render_measurements(rows: Sequence[Tuple[str, str, Optional[float]]], *, min_abs_z: float = 0.0) -> Tuple[str, int]:
    """Group leaves by ontology path. Leaves without a z (categorical) are always kept."""
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


def render_deviation_profile(deviation: Dict[str, Any], *, max_depth: int = 99) -> str:
    """Aggregate deviation of internal ontology nodes (leaves live in `measurements`)."""
    lines: List[str] = []

    def walk(node: Dict[str, Any], trail: List[str], depth: int) -> None:
        children = node.get("children") or []
        name = str(node.get("node_name") or node.get("node_id") or "").strip()
        here = trail + ([name] if name and name.lower() != "root" else [])
        if children and here and depth <= max_depth:
            z = _fmt_z(node.get("z_score"))
            if z is not None:
                lines.append(f"{' > '.join(here)}: {z}")
        if depth < max_depth:
            for child in children:
                if isinstance(child, dict) and child.get("children"):
                    walk(child, here, depth + 1)

    root = deviation.get("root") if isinstance(deviation, dict) else None
    if isinstance(root, dict):
        walk(root, [], 0)
    if not lines and isinstance(deviation, dict) and deviation.get("domain_summaries"):
        return json_to_toon(deviation.get("domain_summaries"))
    return "\n".join(lines)


def _truncate_to_tokens(text: str, max_tokens: int, count: TokenCounter) -> str:
    if count(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "\n[truncated to fit the decision model state]"


# ------------------------------------------------------------------ sections
def build_sections(
    *,
    executor_output: Dict[str, Any],
    task_context: str,
    count: TokenCounter,
) -> Tuple[List[StateSection], int]:
    predictor_input = executor_output.get("predictor_input") or {}
    sections: List[StateSection] = []

    context = str(task_context or "").strip()
    if context:
        sections.append(
            StateSection(
                key="task_context",
                priority=0,
                variants=[context, _truncate_to_tokens(context, 1500, count)],
                variant_labels=["full", "first 1500 tokens"],
            )
        )

    notes = str(predictor_input.get("non_numerical_data_raw") or executor_output.get("non_numerical_data") or "").strip()
    if notes:
        sections.append(
            StateSection(
                key="clinical_record",
                priority=1,
                variants=[notes, _truncate_to_tokens(notes, max(400, count(notes) // 2), count)],
                variant_labels=["full", "first half"],
            )
        )

    # Orchestrated route: what the tools and the integrator produced.
    step_outputs = executor_output.get("step_outputs") or {}
    synthesis_parts: List[str] = []
    for sid in sorted(step_outputs.keys(), key=lambda s: int(s) if str(s).isdigit() else 0):
        out = step_outputs.get(sid)
        if not isinstance(out, dict):
            continue
        tool = str(out.get("tool_name") or (out.get("_step_meta") or {}).get("tool_name") or "tool")
        body = {k: v for k, v in out.items() if not str(k).startswith("_")}
        synthesis_parts.append(f"## {tool} (step {sid})\n{json_to_toon(body)}")
    if synthesis_parts:
        full = "\n\n".join(synthesis_parts)
        sections.append(
            StateSection(
                key="phenotype_synthesis",
                priority=2,
                variants=[full, _truncate_to_tokens(full, max(1500, count(full) // 2), count), _truncate_to_tokens(full, 3000, count)],
                variant_labels=["full", "first half", "first 3000 tokens"],
            )
        )

    chunk_rows = executor_output.get("chunk_evidence") or []
    if chunk_rows:
        rich = json_to_toon(chunk_rows)
        lean_rows = [
            {
                "chunk": f"{r.get('chunk_index', '?')}/{r.get('chunk_total', '?')}",
                "summary": str(r.get("summary") or "")[:400],
                "for_target": list(r.get("for_case") or [])[:4],
                "against_target": list(r.get("for_control") or [])[:4],
                "key_findings": [
                    {k: f.get(k) for k in ("domain", "finding", "direction", "z_score")}
                    for f in list(r.get("key_findings") or [])[:4]
                    if isinstance(f, dict)
                ],
            }
            for r in chunk_rows
            if isinstance(r, dict)
        ]
        lean = json_to_toon(lean_rows)
        sections.append(
            StateSection(
                key="chunk_evidence",
                priority=3,
                variants=[rich, lean, _truncate_to_tokens(lean, max(1500, count(lean) // 2), count)],
                variant_labels=["full", "compact rows", "compact rows, first half"],
            )
        )

    multimodal = predictor_input.get("multimodal_unprocessed_raw")
    if multimodal is None:
        multimodal = executor_output.get("multimodal_data") or {}
    rows = measurement_rows(multimodal)
    total_features = len(rows)
    if rows:
        variants, labels, counts = [], [], []
        for threshold, label in ((0.0, "all leaves"), (0.5, "|z| >= 0.5 and categorical"), (1.0, "|z| >= 1 and categorical"), (1.5, "|z| >= 1.5 and categorical"), (2.0, "|z| >= 2 and categorical")):
            text, kept = render_measurements(rows, min_abs_z=threshold)
            if variants and kept == counts[-1]:
                continue
            variants.append(text)
            labels.append(label)
            counts.append(kept)
        sections.append(
            StateSection(key="measurements", priority=4, variants=variants, variant_labels=labels, feature_counts=counts)
        )

    deviation = predictor_input.get("hierarchical_deviation_raw") or executor_output.get("hierarchical_deviation") or {}
    deviation_text = render_deviation_profile(deviation) if isinstance(deviation, dict) else ""
    if deviation_text.strip():
        sections.append(
            StateSection(
                key="deviation_profile",
                priority=5,
                variants=[deviation_text, render_deviation_profile(deviation, max_depth=2)],
                variant_labels=["all groups", "top two levels"],
            )
        )
    return sections, total_features


# --------------------------------------------------------------------- packer
def pack_state(
    sections: Sequence[StateSection],
    *,
    budget_tokens: int,
    count: TokenCounter,
    features_total: int,
) -> PackedState:
    level = {s.key: 0 for s in sections}
    dropped: set = set()

    def assemble() -> Dict[str, str]:
        return {
            s.key: s.variants[level[s.key]]
            for s in sections
            if s.key not in dropped and s.variants[level[s.key]].strip()
        }

    def size(state: Dict[str, str]) -> int:
        return count(json.dumps(state, ensure_ascii=False))

    state = assemble()
    tokens = size(state)
    by_priority = sorted(sections, key=lambda s: -s.priority)
    while tokens > budget_tokens:
        progressed = False
        for s in by_priority:  # least important first
            if s.key in dropped or level[s.key] >= len(s.variants) - 1:
                continue
            level[s.key] += 1
            progressed = True
            break
        if not progressed:
            # Every section is at its most compact rendering: drop whole sections,
            # least important first, but never the task context or the record.
            for s in by_priority:
                if s.key in dropped or s.key in ("task_context", "clinical_record", "measurements"):
                    continue
                dropped.add(s.key)
                progressed = True
                break
        state = assemble()
        tokens = size(state)
        if not progressed:
            break

    truncated = False
    if tokens > budget_tokens:
        # Last resort: cut the largest remaining section to fit.
        largest = max(state, key=lambda k: count(state[k]))
        overflow = tokens - budget_tokens
        keep = max(200, count(state[largest]) - overflow - 64)
        state[largest] = _truncate_to_tokens(state[largest], keep, count)
        tokens = size(state)
        truncated = True

    included = features_total
    report_rows: List[Dict[str, Any]] = []
    for s in sections:
        status = "dropped" if s.key in dropped else s.variant_labels[level[s.key]]
        row = {"section": s.key, "rendering": status, "tokens": count(state.get(s.key, "")) if s.key in state else 0}
        if s.feature_counts:
            kept = 0 if s.key in dropped else s.feature_counts[level[s.key]]
            included = kept
            row["features"] = kept
        report_rows.append(row)
    if truncated and "measurements" in state:
        included = sum(1 for line in state["measurements"].splitlines() if line.startswith("- "))
    return PackedState(
        state=state,
        tokens=tokens,
        budget=budget_tokens,
        sections=report_rows,
        features_total=features_total,
        features_included=min(included, features_total),
        truncated=truncated,
    )
