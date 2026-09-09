"""
Ontology view over a participant's linguistified evidence.

Two engine files describe the same taxonomy from different angles:
``hierarchical_deviation_map.json`` carries the signed deviation at every node,
``multimodal_data.json`` carries the underlying feature leaves. This module
merges them into one navigable tree with aggregate statistics at every level,
which is what the explorer renders.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Written by the ontology writer onto every internal node; not a child.
STATS_KEY = "_stats"
#: Feature payloads hang off internal nodes under this key.
LEAVES_KEY = "_leaves"

#: Qualitative bands, matching the thresholds the writer uses for `value` text.
BANDS: Tuple[Tuple[float, str], ...] = (
    (2.0, "very_high"),
    (1.0, "high"),
    (0.5, "high_normal"),
    (-0.5, "normal"),
    (-1.0, "low_normal"),
    (-2.0, "low"),
)


def band_for(z: Optional[float]) -> str:
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return "missing"
    for threshold, name in BANDS:
        if z >= threshold:
            return name
    return "very_low"


def humanise(identifier: str) -> str:
    """Turn a machine id into a readable label without losing information."""
    text = str(identifier or "").strip()
    if not text:
        return ""
    # Generated ids repeat their ancestry, e.g. `Domain__Sub__Leaf`; keep the tail.
    if "__" in text:
        text = text.split("__")[-1]
    text = text.replace("_", " ").strip()
    if not text:
        return str(identifier)
    if text.isupper() or text.islower():
        return text[0].upper() + text[1:]
    return text


def _match_key(value: str) -> str:
    """Collapse an id or a label to a comparable form: `education_years` == `Education (years)`."""
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _is_deviation_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "score" in node and not any(
        isinstance(v, dict) for k, v in node.items() if k != STATS_KEY
    )


def _children_items(node: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    for key, value in node.items():
        if key in (STATS_KEY, LEAVES_KEY):
            continue
        yield key, value


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_features(node: Any) -> List[Dict[str, Any]]:
    """Normalise a `_leaves` array; the pseudo data carries extra optional keys."""
    rows: List[Dict[str, Any]] = []
    for raw in node if isinstance(node, list) else []:
        if not isinstance(raw, dict):
            continue
        z = raw.get("z_score")
        try:
            z_value = float(z) if z is not None else None
        except (TypeError, ValueError):
            z_value = None
        rows.append(
            {
                "feature": str(raw.get("feature") or ""),
                "value": raw.get("value"),
                "z_score": z_value,
                "band": band_for(z_value),
                "ref_range": raw.get("ref_range"),
                "significance": raw.get("significance"),
                "percentile": raw.get("percentile"),
            }
        )
    return rows


def _fold_features_into_leaves(
    children: List[Dict[str, Any]], features: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Fold a node's feature payload into its leaf children.

    A generated ontology records each measurement twice: once as a leaf of the
    deviation map, and once as an entry in the `_leaves` array of that leaf's
    parent. Both files come from the same traversal, so when the counts line up
    and the z-scores agree the pairing is positional. Ids and labels can differ
    completely (`lesion_ratio_p83` against "Lesion overlap: 7Networks_LH_SomMot_1"),
    so name matching is only the fallback.

    Hand authored data describes the two at different granularities and pairs
    nothing, which is why an unmatched feature is kept rather than dropped.
    """
    leaves = [child for child in children if child["kind"] == "leaf"]
    if not leaves or not features:
        return features

    def absorb(leaf: Dict[str, Any], feature: Dict[str, Any]) -> None:
        leaf["value"] = feature.get("value")
        leaf["ref_range"] = feature.get("ref_range")
        leaf["significance"] = feature.get("significance")
        leaf["percentile"] = feature.get("percentile")
        if leaf["score"] is None and feature["z_score"] is not None:
            leaf["score"] = feature["z_score"]
            leaf["band"] = feature["band"]
            leaf["mean_abs_score"] = abs(feature["z_score"])
            leaf["present_leaves"] = 1
            leaf["subtree_signed_mean"] = feature["z_score"]
            leaf["_signed"] = [feature["z_score"]]

    if len(leaves) == len(features) and all(
        leaf["score"] == feature["z_score"] for leaf, feature in zip(leaves, features)
    ):
        for leaf, feature in zip(leaves, features):
            absorb(leaf, feature)
        return []

    by_key = {_match_key(leaf["id"]): leaf for leaf in leaves}
    kept: List[Dict[str, Any]] = []
    for feature in features:
        leaf = by_key.get(_match_key(feature["feature"]))
        if leaf is None:
            kept.append(feature)
        else:
            absorb(leaf, feature)
    return kept


def _multimodal_at(multimodal: Any, path: List[str]) -> Any:
    cursor = multimodal
    for segment in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(segment)
    return cursor


def _build(
    node_id: str,
    node: Any,
    multimodal_root: Any,
    path: List[str],
    depth: int,
) -> Dict[str, Any]:
    node_path = path + [node_id]
    mm_node = _multimodal_at(multimodal_root, node_path)
    features = _collect_features((mm_node or {}).get(LEAVES_KEY)) if isinstance(mm_node, dict) else []

    # The feature payload is often deeper than the deviation map, so children
    # come from the union of both trees. A subtree that exists only in the
    # payload still carries evidence and must stay visible.
    child_ids: List[str] = []
    if isinstance(node, dict) and not _is_deviation_leaf(node):
        child_ids.extend(key for key, _ in _children_items(node))
    if isinstance(mm_node, dict):
        for key, value in mm_node.items():
            if key in (STATS_KEY, LEAVES_KEY) or key in child_ids:
                continue
            if isinstance(value, dict):
                child_ids.append(key)

    if _is_deviation_leaf(node) and not child_ids:
        raw_score = node.get("score")
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None
        return {
            "_signed": [score] if score is not None else [],
            "id": node_id,
            "label": humanise(node_id),
            "path": node_path,
            "depth": depth,
            "kind": "leaf",
            "score": score,
            "subtree_signed_mean": score,
            "value": None,
            "ref_range": None,
            "significance": None,
            "percentile": None,
            "band": band_for(score),
            "mean_abs_score": abs(score) if score is not None else None,
            "leaf_count": 1,
            "present_leaves": 1 if score is not None else 0,
            "feature_count": len(features),
            "features": features,
            "children": [],
        }

    children: List[Dict[str, Any]] = []
    for child_id in child_ids:
        child = node.get(child_id) if isinstance(node, dict) else None
        children.append(_build(child_id, child if child is not None else {}, multimodal_root, node_path, depth + 1))

    features = _fold_features_into_leaves(children, features)

    stats = node.get(STATS_KEY) if isinstance(node, dict) else None
    leaf_count = sum(int(c.get("leaf_count") or 0) for c in children) or len(features)
    present = sum(int(c.get("present_leaves") or 0) for c in children)
    if not children:
        present = sum(1 for f in features if f["z_score"] is not None)

    if isinstance(stats, dict) and stats.get("mean_abs_score") is not None:
        try:
            mean_abs = float(stats["mean_abs_score"])
        except (TypeError, ValueError):
            mean_abs = None
        leaf_count = int(stats.get("n_leaves") or leaf_count)
        present = int(stats.get("present_leaves") or present)
    else:
        magnitudes = [c["mean_abs_score"] for c in children if c.get("mean_abs_score") is not None]
        if not magnitudes:
            magnitudes = [abs(f["z_score"]) for f in features if f["z_score"] is not None]
        mean_abs = round(sum(magnitudes) / len(magnitudes), 3) if magnitudes else None

    # Direction has to roll all the way up: averaging only direct leaf children
    # leaves every branch without a signed score.
    signed: List[float] = [f["z_score"] for f in features if f["z_score"] is not None]
    for child in children:
        signed.extend(child.get("_signed") or [])
    # A node present in the deviation map but expanded by the payload keeps its
    # own score rather than inheriting an average of its children.
    own_score: Any = None
    if isinstance(node, dict) and "score" in node:
        try:
            own_score = float(node["score"]) if node["score"] is not None else None
        except (TypeError, ValueError):
            own_score = None

    subtree_mean = round(sum(signed) / len(signed), 3) if signed else None
    return {
        "_signed": signed,
        "id": node_id,
        "label": humanise(node_id),
        "path": node_path,
        "depth": depth,
        "kind": "domain" if depth == 0 else "node",
        "score": own_score if own_score is not None else subtree_mean,
        "subtree_signed_mean": subtree_mean,
        "band": band_for(mean_abs) if mean_abs is not None else "missing",
        "mean_abs_score": mean_abs,
        "leaf_count": leaf_count,
        "present_leaves": present,
        "feature_count": len(features) + sum(int(c.get("feature_count") or 0) for c in children),
        "features": features,
        "children": children,
    }


def build_ontology(participant_dir: Path) -> Dict[str, Any]:
    """
    Merge the deviation map and the feature payload into one explorable tree.

    Returns the tree plus the summary statistics the explorer shows in its
    header, and an ``extremes`` list used for the "most deviating" ranking.
    """
    deviation = _read_json(participant_dir / "hierarchical_deviation_map.json")
    multimodal = _read_json(participant_dir / "multimodal_data.json")
    overview = _read_json(participant_dir / "data_overview.json")

    # The legacy shape nests everything under an explicit "root" node.
    if "root" in deviation and isinstance(deviation.get("root"), dict):
        deviation = {"root": deviation["root"]}

    domain_ids: List[str] = [k for k in deviation if k != STATS_KEY]
    for key in multimodal:
        if key not in (STATS_KEY, LEAVES_KEY) and key not in domain_ids:
            domain_ids.append(key)
    domains = [_build(domain_id, deviation.get(domain_id) or {}, multimodal, [], 0) for domain_id in domain_ids]
    domains.sort(key=lambda d: (d["mean_abs_score"] is None, -(d["mean_abs_score"] or 0)))

    extremes: List[Dict[str, Any]] = []

    def walk(node: Dict[str, Any]) -> None:
        if node["kind"] == "leaf" and node["score"] is not None:
            extremes.append(
                {
                    "path": node["path"],
                    "label": node["label"],
                    "score": node["score"],
                    "band": node["band"],
                }
            )
        for feature in node["features"]:
            if feature["z_score"] is not None:
                extremes.append(
                    {
                        "path": node["path"] + [feature["feature"]],
                        "label": feature["feature"],
                        "score": feature["z_score"],
                        "band": feature["band"],
                    }
                )
        for child in node["children"]:
            walk(child)

    for domain in domains:
        walk(domain)
    extremes.sort(key=lambda row: -abs(row["score"]))

    for domain in domains:
        _drop_internal(domain)

    total_leaves = sum(int(d.get("leaf_count") or 0) for d in domains)
    present_leaves = sum(int(d.get("present_leaves") or 0) for d in domains)
    max_depth = 0

    def depth_of(node: Dict[str, Any]) -> int:
        if not node["children"]:
            return node["depth"]
        return max(depth_of(child) for child in node["children"])

    for domain in domains:
        max_depth = max(max_depth, depth_of(domain))

    return {
        "participant_id": overview.get("participant_id") or participant_dir.name,
        "domains": domains,
        "extremes": extremes[:200],
        "summary": {
            "domain_count": len(domains),
            "node_count": _count_nodes(domains),
            "leaf_count": total_leaves,
            "present_leaves": present_leaves,
            "feature_count": sum(int(d.get("feature_count") or 0) for d in domains),
            "max_depth": max_depth + 1,
            "coverage": round(present_leaves / total_leaves, 4) if total_leaves else None,
            "total_tokens": overview.get("total_tokens"),
        },
        "domain_coverage": overview.get("domain_coverage") or {},
        "has_deviation_map": bool(deviation),
        "has_multimodal": bool(multimodal),
    }


def _drop_internal(node: Dict[str, Any]) -> None:
    """Remove the signed-sample accumulator used during the roll-up."""
    node.pop("_signed", None)
    for child in node["children"]:
        _drop_internal(child)


def _count_nodes(nodes: List[Dict[str, Any]]) -> int:
    return sum(1 + _count_nodes(node["children"]) for node in nodes)
