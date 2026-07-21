"""
Write the four COMPASS participant files from an arbitrary-depth ontology.

The engine's ``DataLoader`` accepts a "UKB nested" format whose deviation map and
multimodal tree recurse to any depth, so we emit exactly that. This is the whole
point of the ingestion layer: arbitrary pre-processed data is projected onto the
ontology and rendered in the engine's native contract, with no dataset-specific
code inside ``src/full_stack``.

Files produced per participant (each mirrors the ontology hierarchy at full depth):

  data_overview.json              per-node coverage AND token budget at every
                                  level of the hierarchy (not just per domain).
  hierarchical_deviation_map.json signed deviation score at every node down to the
                                  leaves, with ``_stats`` (mean |z|, leaf counts) on
                                  every internal node. This is the aggregate signal
                                  the fusion/anomaly layer reads.
  multimodal_data.json            the same tree with ``_leaves`` holding the actual
                                  values (feature, value label, z). This is the one
                                  place raw values live, so the two files do not
                                  duplicate content.
  non_numerical_data.txt          a compact narrative over the notable features.

Deviation vs. raw: if a normative reference exists (cohort/external), leaves carry
z-scores; in absolute mode (single subject, no reference) leaf scores are null and
the raw value in ``multimodal_data`` carries the signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ontology import is_leaf


def _est_tokens(text: str) -> int:
    return max(1, int(len(str(text)) / 4))


def _leaf_value_text(enc: Dict[str, Any]) -> str:
    """Human-readable value string (the LLM reasons over text, not bare numbers)."""
    if not enc["present"]:
        return "Not measured"
    if enc.get("qualitative") and enc["qualitative"] not in ("Observed", "Missing"):
        return f"{enc['raw_label']} ({enc['qualitative']})"
    return str(enc["raw_label"])


class _NodeResult:
    __slots__ = ("mm", "dev", "cov", "present", "total", "tokens", "absz", "notable")

    def __init__(self):
        self.mm: Any = None
        self.dev: Any = None
        self.cov: Dict[str, Any] = {}
        self.present = 0
        self.total = 0
        self.tokens = 0
        self.absz: List[float] = []
        self.notable: List[str] = []


def _build_node(node: Dict[str, Any], encoded: Dict[str, Dict[str, Any]]) -> _NodeResult:
    r = _NodeResult()

    if is_leaf(node):
        enc = encoded[node["id"]]
        z = enc.get("z_score")
        present = bool(enc["present"])
        value = _leaf_value_text(enc)
        r.total = 1
        r.present = 1 if present else 0
        r.tokens = _est_tokens(f"{node['label']}{value}")
        r.mm = {"feature": node["label"], "value": value, "z_score": z}
        r.dev = {"score": round(float(z), 3) if (present and z is not None) else None}
        if present and z is not None:
            r.absz.append(abs(float(z)))
            if abs(float(z)) >= 1.0:
                r.notable.append(f"{node['label']} is {enc['qualitative'].lower()} (z={z:+.2f})")
        r.cov = {"label": node["label"], "present_leaves": r.present, "total_leaves": 1,
                 "coverage_percentage": 100.0 if present else 0.0, "tokens": r.tokens}
        return r

    mm: Dict[str, Any] = {}
    dev: Dict[str, Any] = {}
    cov_children: Dict[str, Any] = {}
    leaf_records: List[Dict[str, Any]] = []

    for child in node["children"]:
        cr = _build_node(child, encoded)
        r.present += cr.present
        r.total += cr.total
        r.tokens += cr.tokens
        r.absz.extend(cr.absz)
        r.notable.extend(cr.notable)
        dev[child["id"]] = cr.dev
        if is_leaf(child):
            leaf_records.append(cr.mm)
            # Leaf coverage/tokens are folded into this node's totals; the overview
            # reports tokens per hierarchical *group*, not per individual leaf (those
            # values live in multimodal_data.json), which keeps it token-efficient.
        else:
            mm[child["id"]] = cr.mm
            cov_children[child["id"]] = cr.cov

    if leaf_records:
        mm["_leaves"] = leaf_records
    dev["_stats"] = {
        "mean_abs_score": round(float(np.mean(r.absz)), 3) if r.absz else None,
        "n_leaves": r.total,
        "present_leaves": r.present,
    }
    r.mm = mm
    r.dev = dev
    r.cov = {
        "label": node["label"],
        "present_leaves": r.present,
        "total_leaves": r.total,
        "coverage_percentage": round(100.0 * r.present / r.total, 1) if r.total else 0.0,
        "tokens": r.tokens,
    }
    if cov_children:  # only internal children; leaf-only groups stay compact
        r.cov["children"] = cov_children
    return r


def build_participant_payloads(
    participant_id: str,
    ontology: Dict[str, Any],
    encoded: Dict[str, Dict[str, Any]],
    target_note: str,
    reference_mode: str = "cohort",
) -> Dict[str, Any]:
    """Return the four file payloads (three dicts + one text) for a participant."""
    multimodal: Dict[str, Any] = {}
    deviation: Dict[str, Any] = {}
    domain_coverage: Dict[str, Any] = {}
    notable: List[str] = []

    for dom in ontology["domains"]:
        r = _build_node(dom, encoded)
        multimodal[dom["id"]] = r.mm
        deviation[dom["id"]] = r.dev
        domain_coverage[dom["id"]] = {
            "present_leaves": r.present,
            "total_leaves": r.total,
            "coverage_percentage": round(100.0 * r.present / r.total, 1) if r.total else 0.0,
            "missing_count": r.total - r.present,
            "total_tokens": r.tokens,
            "is_available": r.present > 0,
            # Comprehensive per-node breakdown (coverage + tokens at every level).
            "structure": r.cov.get("children", {}),
        }
        notable.extend(r.notable)

    total_tokens = sum(d["total_tokens"] for d in domain_coverage.values())
    available = [d for d, c in domain_coverage.items() if c["is_available"]]
    data_overview = {
        "participant_id": participant_id,
        "reference_mode": reference_mode,
        "domain_coverage": domain_coverage,
        "total_tokens": total_tokens,
        "total_leaves": sum(d["total_leaves"] for d in domain_coverage.values()),
        "present_leaves": sum(d["present_leaves"] for d in domain_coverage.values()),
        "available_domains": available,
        "token_budget": int(total_tokens * 1.1),
    }

    narrative = _build_narrative(participant_id, ontology, encoded, target_note, reference_mode, notable)

    return {
        "data_overview": data_overview,
        "hierarchical_deviation_map": deviation,
        "multimodal_data": multimodal,
        "non_numerical_data": narrative,
    }


def _build_narrative(
    participant_id: str,
    ontology: Dict[str, Any],
    encoded: Dict[str, Dict[str, Any]],
    target_note: str,
    reference_mode: str,
    notable: List[str],
) -> str:
    """Compact textual profile so the non-tabular modality is populated.

    Summarised per domain (not every leaf) to stay token-efficient; the full
    per-feature detail already lives in ``multimodal_data.json``.
    """
    ref_phrase = {
        "cohort": "deviation from the population cohort",
        "external": "deviation from external normative reference",
        "absolute": "absolute pre-processed values, no normative reference",
    }.get(reference_mode, "observed values")

    lines: List[str] = [
        f"Participant ID: {participant_id}",
        f"Source: blinded evaluation record from {ontology.get('dataset', 'dataset')}; "
        "non-cognitive multimodal features only.",
        f"Reference strategy: {reference_mode} ({ref_phrase}).",
        "",
        "TARGET MEASUREMENT REFERENCE:",
        target_note,
        "",
        f"OBSERVED FEATURE PROFILE ({ref_phrase}):",
    ]

    for dom in ontology["domains"]:
        present = [leaf for leaf in _domain_present_leaves(dom, encoded)]
        if not present:
            continue
        # Keep the domain line short: count + a few strongest signals.
        shown = present[:6]
        frag = "; ".join(f"{lbl} = {val}" for lbl, val in shown)
        more = f" (+{len(present) - len(shown)} more)" if len(present) > len(shown) else ""
        lines.append(f"- {dom['label']} [{len(present)} measured]: {frag}{more}")

    lines.append("")
    if notable:
        lines.append("NOTABLE DEVIATIONS: " + "; ".join(notable[:20]) +
                     (f"; (+{len(notable) - 20} more)" if len(notable) > 20 else "") + ".")
    else:
        lines.append("NOTABLE DEVIATIONS: none beyond one standard deviation.")
    lines.append("")
    lines.append(
        "NOTE: No cognitive/IST subscale scores are provided as inputs. The prediction "
        "target must be inferred only from the non-cognitive multimodal evidence provided "
        "in this record."
    )
    return "\n".join(lines)


def _domain_present_leaves(node: Dict[str, Any], encoded: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Present (label, value) pairs under a domain, sorted by |z| descending."""
    out: List[Tuple[str, str, float]] = []

    def walk(n):
        if is_leaf(n):
            enc = encoded[n["id"]]
            if enc["present"]:
                z = enc.get("z_score")
                out.append((n["label"], _leaf_value_text(enc), abs(float(z)) if z is not None else 0.0))
            return
        for c in n["children"]:
            walk(c)

    walk(node)
    out.sort(key=lambda t: t[2], reverse=True)
    return [(lbl, val) for lbl, val, _ in out]


def write_participant(out_dir: Path, payloads: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data_overview.json", "w") as f:
        json.dump(payloads["data_overview"], f, indent=2)
    with open(out_dir / "hierarchical_deviation_map.json", "w") as f:
        json.dump(payloads["hierarchical_deviation_map"], f, indent=2)
    with open(out_dir / "multimodal_data.json", "w") as f:
        json.dump(payloads["multimodal_data"], f, indent=2)
    with open(out_dir / "non_numerical_data.txt", "w") as f:
        f.write(payloads["non_numerical_data"])
