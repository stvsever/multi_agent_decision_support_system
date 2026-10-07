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

Each section has renderings from complete to compact. When the state exceeds the
model's budget the packer walks a fixed ladder, one step at a time, until it fits:
first it shortens what the critic does not measure (group aggregates, tool text,
chunk rows, the study context), then it compacts the measurements by deviation
(|z| at least 0.5, then 1, 1.5 and 2; categorical leaves always stay), and only
then the remaining text. The report records the rendering each section ended at
and how many feature leaves the state still holds, so the critic and the run
record show any coverage loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ...utils.core.record_rendering import (
    measurement_groups,
    measurement_rows,
    render_deviation_profile,
    render_measurements,
    value_representation,
)
from ...utils.toon import json_to_toon

TokenCounter = Callable[[str], int]

# (section, action) in the order the packer applies them. "next" steps the
# section to its next, more compact rendering; "drop" removes it.
PACKING_LADDER: Tuple[Tuple[str, str], ...] = (
    ("deviation_profile", "next"),
    ("phenotype_synthesis", "next"),
    ("chunk_evidence", "next"),
    ("deviation_profile", "drop"),
    ("task_context", "next"),
    ("measurements", "next"),
    ("measurements", "next"),
    ("measurements", "next"),
    ("measurements", "next"),
    ("chunk_evidence", "next"),
    ("phenotype_synthesis", "next"),
    ("clinical_record", "next"),
    ("clinical_record", "next"),
    ("chunk_evidence", "drop"),
    ("phenotype_synthesis", "drop"),
)


@dataclass
class StateSection:
    key: str
    priority: int  # lower is more important (used for report ordering)
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


def truncate_to_tokens(text: str, max_tokens: int, count: TokenCounter) -> str:
    if count(text) <= max_tokens:
        return text
    marker = "\n[truncated to fit the decision model state]"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid] + marker) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + marker


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
                variants=[context, truncate_to_tokens(context, 1500, count)],
                variant_labels=["full", "first 1500 tokens"],
            )
        )

    notes = str(predictor_input.get("non_numerical_data_raw") or executor_output.get("non_numerical_data") or "").strip()
    if notes:
        size = count(notes)
        sections.append(
            StateSection(
                key="clinical_record",
                priority=1,
                variants=[
                    notes,
                    truncate_to_tokens(notes, max(400, size // 2), count),
                    truncate_to_tokens(notes, max(300, size // 4), count),
                ],
                variant_labels=["full", "first half", "first quarter"],
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
        size = count(full)
        sections.append(
            StateSection(
                key="phenotype_synthesis",
                priority=2,
                variants=[
                    full,
                    truncate_to_tokens(full, max(1500, size // 2), count),
                    truncate_to_tokens(full, min(3000, max(750, size // 4)), count),
                ],
                variant_labels=["full", "first half", "first quarter (at most 3000 tokens)"],
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
                variants=[rich, lean, truncate_to_tokens(lean, max(1500, count(lean) // 2), count)],
                variant_labels=["full", "compact rows", "compact rows, first half"],
            )
        )

    multimodal = predictor_input.get("multimodal_unprocessed_raw")
    if multimodal is None:
        multimodal = executor_output.get("multimodal_data") or {}
    rows = measurement_rows(multimodal)
    total_features = len(rows)
    if rows:
        # Thinning by |z| keeps every leaf without a deviation score (native-scale
        # or categorical), so it shrinks only the deviation-scored part. A record
        # with native-scale leaves says so, so that no value is read as a z-score;
        # a fully deviation-scored record keeps its state unchanged.
        representation = value_representation(multimodal)
        value_note = representation["note"] if representation["mode"] in ("native", "mixed") else ""
        variants, labels, counts = [], [], []
        for threshold, label in (
            (0.0, "all leaves"),
            (0.5, "|z| >= 0.5 and categorical"),
            (1.0, "|z| >= 1 and categorical"),
            (1.5, "|z| >= 1.5 and categorical"),
            (2.0, "|z| >= 2 and categorical"),
        ):
            text, kept = render_measurements(rows, min_abs_z=threshold)
            if value_note:
                text = f"Values: {value_note}\n{text}"
            if variants and kept == counts[-1]:
                continue
            variants.append(text)
            labels.append(label)
            counts.append(kept)
        sections.append(
            StateSection(key="measurements", priority=4, variants=variants, variant_labels=labels, feature_counts=counts)
        )

    deviation = (
        executor_output.get("deviation_tree")
        or predictor_input.get("hierarchical_deviation_raw")
        or executor_output.get("hierarchical_deviation")
        or {}
    )
    groups = measurement_groups(multimodal)
    deviation_text = render_deviation_profile(deviation, leaf_groups=groups)
    if deviation_text.strip():
        sections.append(
            StateSection(
                key="deviation_profile",
                priority=5,
                variants=[deviation_text, render_deviation_profile(deviation, max_depth=2, leaf_groups=groups)],
                variant_labels=["all groups", "top two levels"],
            )
        )
    for section in sections:
        _keep_shrinking(section, count)
    return sections, total_features


def _keep_shrinking(section: StateSection, count: TokenCounter) -> None:
    """Drop renderings that are not smaller than the one before, so every ladder step shrinks."""
    keep = [0]
    for i in range(1, len(section.variants)):
        if count(section.variants[i]) < count(section.variants[keep[-1]]):
            keep.append(i)
    section.variants = [section.variants[i] for i in keep]
    section.variant_labels = [section.variant_labels[i] for i in keep]
    if section.feature_counts:
        section.feature_counts = [section.feature_counts[i] for i in keep]


# --------------------------------------------------------------------- packer
def pack_state(
    sections: Sequence[StateSection],
    *,
    budget_tokens: int,
    count: TokenCounter,
    features_total: int,
) -> PackedState:
    by_key = {s.key: s for s in sections}
    level = {s.key: 0 for s in sections}
    dropped: set = set()
    cut: Dict[str, str] = {}  # last-resort truncations

    def text_of(key: str) -> str:
        return cut.get(key, by_key[key].variants[level[key]])

    def assemble() -> Dict[str, str]:
        return {s.key: text_of(s.key) for s in sections if s.key not in dropped and text_of(s.key).strip()}

    def size(state: Dict[str, str]) -> int:
        return count(json.dumps(state, ensure_ascii=False))

    state = assemble()
    tokens = size(state)
    for key, action in PACKING_LADDER:
        if tokens <= budget_tokens:
            break
        section = by_key.get(key)
        if section is None or key in dropped:
            continue
        if action == "drop":
            dropped.add(key)
        elif level[key] < len(section.variants) - 1:
            level[key] += 1
        else:
            continue
        state = assemble()
        tokens = size(state)

    truncated = False
    guard = 0
    while tokens > budget_tokens and state and guard < 12:
        # Last resort: cut the largest remaining section until the state fits.
        guard += 1
        largest = max(state, key=lambda k: count(state[k]))
        overflow = tokens - budget_tokens
        keep = max(50, count(state[largest]) - overflow - 64)
        cut[largest] = truncate_to_tokens(state[largest], keep, count)
        truncated = True
        state = assemble()
        tokens = size(state)

    included = features_total
    report_rows: List[Dict[str, Any]] = []
    for s in sections:
        if s.key in dropped:
            rendering = "dropped"
        else:
            rendering = s.variant_labels[level[s.key]] + (", truncated" if s.key in cut else "")
        row: Dict[str, Any] = {"section": s.key, "rendering": rendering, "tokens": count(state[s.key]) if s.key in state else 0}
        if s.feature_counts:
            if s.key in dropped:
                kept = 0
            elif s.key in cut:
                kept = sum(1 for line in state.get(s.key, "").splitlines() if line.startswith("- "))
            else:
                kept = s.feature_counts[level[s.key]]
            included = kept
            row["features"] = kept
        report_rows.append(row)
    return PackedState(
        state=state,
        tokens=tokens,
        budget=budget_tokens,
        sections=report_rows,
        features_total=features_total,
        features_included=min(included, features_total),
        truncated=truncated,
    )
