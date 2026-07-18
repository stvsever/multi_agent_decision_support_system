"""
Write the four COMPASS participant files from an ontology + encoded values.

The engine's ``DataLoader`` accepts a "UKB nested" format, so we emit exactly
that shape. This is the whole point of the validation layer: an arbitrary
tabular dataset is projected onto the ontology and rendered in the engine's
native contract, with no dataset-specific code inside ``src/full_stack``.

Files produced per participant:
  data_overview.json            domain coverage summary
  hierarchical_deviation_map.json   ontology tree of aggregated signed z-scores
  multimodal_data.json          ontology tree with per-feature _leaves
  non_numerical_data.txt        short generated narrative over the same features
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _est_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def _leaf_record(feature_node: Dict[str, Any], enc: Dict[str, Any]) -> Dict[str, Any]:
    z = enc["z_score"]
    if enc["present"]:
        value = f"{enc['raw_label']} ({enc['qualitative']})"
    else:
        value = "Not measured"
    return {
        "feature": feature_node["label"],
        "z_score": z,
        "value": value,
        "ref_range": enc["reference"],
    }


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

    for dom in ontology["domains"]:
        dom_id = dom["id"]
        multimodal[dom_id] = {}
        deviation[dom_id] = {}
        present_leaves = 0
        total_leaves = 0
        domain_tokens = 0
        domain_abs_z: List[float] = []

        for sub in dom["subdomains"]:
            sub_id = sub["id"]
            leaves = []
            sub_signed_z: List[float] = []
            for feat in sub["features"]:
                enc = encoded[feat["id"]]
                total_leaves += 1
                rec = _leaf_record(feat, enc)
                leaves.append(rec)
                domain_tokens += _est_tokens(f"{rec['feature']}{rec['value']}{rec['ref_range']}")
                if enc["present"]:
                    present_leaves += 1
                    if enc["z_score"] is not None:
                        sub_signed_z.append(float(enc["z_score"]))
                        domain_abs_z.append(abs(float(enc["z_score"])))
            multimodal[dom_id][sub_id] = {"_leaves": leaves}
            # Subdomain deviation node = mean signed z of its present features.
            if sub_signed_z:
                deviation[dom_id][sub_id] = {"score": round(float(np.mean(sub_signed_z)), 3)}
            else:
                deviation[dom_id][sub_id] = {"score": None}

        # Domain-level stats block consumed by the engine's UKB parser.
        deviation[dom_id]["_stats"] = {
            "mean_abs_score": round(float(np.mean(domain_abs_z)), 3) if domain_abs_z else None,
            "n_leaves": total_leaves,
        }
        coverage_pct = round(100.0 * present_leaves / total_leaves, 1) if total_leaves else 0.0
        domain_coverage[dom_id] = {
            "present_leaves": present_leaves,
            "total_leaves": total_leaves,
            "coverage_percentage": coverage_pct,
            "missing_count": total_leaves - present_leaves,
            "total_tokens": domain_tokens,
            "is_available": present_leaves > 0,
        }

    total_tokens = sum(d["total_tokens"] for d in domain_coverage.values())
    available = [d for d, c in domain_coverage.items() if c["is_available"]]
    data_overview = {
        "participant_id": participant_id,
        "domain_coverage": domain_coverage,
        "total_tokens": total_tokens,
        "available_domains": available,
        "token_budget": int(total_tokens * 1.1),
    }

    narrative = _build_narrative(participant_id, ontology, encoded, target_note, reference_mode)

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
    reference_mode: str = "cohort",
) -> str:
    """Deterministic textual profile so the non-tabular modality is populated."""
    ref_phrase = {
        "cohort": "deviation from the population cohort",
        "external": "deviation from external normative reference",
        "absolute": "absolute pre-processed values, no normative reference",
    }.get(reference_mode, "observed values")
    lines: List[str] = []
    lines.append(f"Participant ID: {participant_id}")
    lines.append(f"Source: {ontology.get('dataset', 'dataset')} tabular phenotype (self-report questionnaires + demographics).")
    lines.append(f"Reference strategy: {reference_mode} ({ref_phrase}).")
    lines.append("")
    lines.append("TARGET MEASUREMENT REFERENCE:")
    lines.append(target_note)
    lines.append("")
    lines.append(f"OBSERVED FEATURE PROFILE ({ref_phrase}):")

    notable: List[str] = []
    for dom in ontology["domains"]:
        dom_present = []
        for sub in dom["subdomains"]:
            for feat in sub["features"]:
                enc = encoded[feat["id"]]
                if not enc["present"]:
                    continue
                dom_present.append(f"{feat['label']} = {enc['raw_label']} ({enc['qualitative']})")
                if enc["z_score"] is not None and abs(enc["z_score"]) >= 1.0:
                    notable.append(f"{feat['label']} is {enc['qualitative'].lower()} (z={enc['z_score']:+.2f})")
        if dom_present:
            lines.append(f"- {dom['label']}: " + "; ".join(dom_present))

    lines.append("")
    if notable:
        lines.append("NOTABLE DEVIATIONS: " + "; ".join(notable) + ".")
    else:
        lines.append("NOTABLE DEVIATIONS: none beyond one standard deviation.")
    lines.append("")
    lines.append(
        "NOTE: No cognitive/IST subscale scores are provided as inputs. The prediction "
        "target must be inferred from demographic, personality, motivational, affective, "
        "identity, and lifestyle features only."
    )
    return "\n".join(lines)


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
