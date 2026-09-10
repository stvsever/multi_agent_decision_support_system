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
import re
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


# --- cohort view -------------------------------------------------------------


def _quantile(ordered: List[float], fraction: float) -> Optional[float]:
    """Linear interpolation between order statistics, as numpy would compute it."""
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return round(ordered[low], 4)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 4)


def numeric_summary(values: List[float]) -> Dict[str, Any]:
    """Signed spread of one measurement across a cohort."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None,
                "median": None, "q1": None, "q3": None}
    ordered = sorted(clean)
    mean = sum(ordered) / len(ordered)
    # Sample standard deviation: one participant carries no spread at all, and
    # a population sd would understate it for the small cohorts this serves.
    if len(ordered) > 1:
        variance = sum((v - mean) ** 2 for v in ordered) / (len(ordered) - 1)
        sd = round(math.sqrt(variance), 4)
    else:
        sd = None
    return {
        "n": len(ordered),
        "mean": round(mean, 4),
        "sd": sd,
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "median": _quantile(ordered, 0.5),
        "q1": _quantile(ordered, 0.25),
        "q3": _quantile(ordered, 0.75),
    }


def _sharing(present_in: int, total: int) -> str:
    if total and present_in >= total:
        return "shared"
    if present_in * 2 > total:
        return "common"
    return "partial"


def _aggregate_block(summary: Dict[str, Any], present_in: int, total: int) -> Dict[str, Any]:
    """
    How much of the cohort reached this node, and how it is spread there.

    Both halves belong in one object: a spread means nothing without knowing how
    many participants it describes, and the reader of one always wants the
    other. The same fields stay on the node itself for callers that walk it.
    """
    return {
        "present_in": present_in,
        "participant_count": total,
        "coverage": round(present_in / total, 4) if total else None,
        "membership": _sharing(present_in, total),
        **summary,
    }


def _unique_ids(trees: List[Dict[str, Any]], directories: List[Path]) -> List[str]:
    """
    A stable name per participant.

    Two folders can carry the same participant id, and merging them would
    silently count one cohort member twice.
    """
    identifiers: List[str] = []
    for tree, directory in zip(trees, directories):
        candidate = str(tree["participant_id"])
        if candidate in identifiers:
            candidate = f"{candidate} ({Path(directory).name})"
        while candidate in identifiers:
            candidate = f"{candidate}."
        identifiers.append(candidate)
    return identifiers


def _index_participant(tree: Dict[str, Any]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    index: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    def walk(node: Dict[str, Any]) -> None:
        index[tuple(node["path"])] = node
        for child in node["children"]:
            walk(child)

    for domain in tree["domains"]:
        walk(domain)
    return index


def _merge_features(
    entries: List[Tuple[str, Dict[str, Any]]], total: int
) -> List[Dict[str, Any]]:
    """
    Fold every participant's feature rows for one node into cohort rows.

    Features are matched on their comparable name, because two participants can
    write the same measurement with different punctuation.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for participant_id, node in entries:
        for feature in node.get("features") or []:
            key = _match_key(feature.get("feature") or "")
            bucket = buckets.setdefault(
                key,
                {"feature": feature.get("feature") or "", "participants": [], "z_scores": []},
            )
            bucket["participants"].append(participant_id)
            if feature.get("z_score") is not None:
                bucket["z_scores"].append(float(feature["z_score"]))

    rows: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        aggregate = numeric_summary(bucket["z_scores"])
        mean = aggregate["mean"]
        present_in = len(set(bucket["participants"]))
        rows.append(
            {
                "feature": bucket["feature"],
                "value": None,
                "z_score": mean,
                "band": band_for(mean),
                "ref_range": None,
                "significance": None,
                "percentile": None,
                "present_in": present_in,
                "participant_count": total,
                "coverage": round(present_in / total, 4) if total else None,
                "sharing": _sharing(present_in, total),
                "present_in_ids": sorted(set(bucket["participants"])),
                "aggregate": _aggregate_block(aggregate, present_in, total),
            }
        )
    rows.sort(key=lambda row: (-(row["present_in"]), -abs(row["z_score"] or 0.0), row["feature"]))
    return rows


def aggregate_ontology(participant_dirs: List[Path]) -> Dict[str, Any]:
    """
    One tree over many participants, keeping what only some of them have.

    Nodes are unioned by their full path, so a subtree measured in a single
    participant survives beside the ones the whole cohort shares. Every node
    reports how much of the cohort reached it and how the signed deviation is
    distributed across the participants that did.
    """
    trees = [build_ontology(Path(directory)) for directory in participant_dirs]
    identifiers = _unique_ids(trees, participant_dirs)
    participants = [
        {"participant_id": identifier, "directory": str(directory)}
        for identifier, directory in zip(identifiers, participant_dirs)
    ]
    total = len(trees)
    # A merged tree has to satisfy the same contract as a single participant's,
    # or the explorer has to special-case which of the two it is looking at.
    has_deviation_map = any(bool(tree.get("has_deviation_map")) for tree in trees)
    has_multimodal = any(bool(tree.get("has_multimodal")) for tree in trees)
    if not total:
        return {
            "participant_id": "No participants",
            "participant_ids": [],
            "aggregated": True,
            "participants": [],
            "participant_count": 0,
            "domains": [],
            "extremes": [],
            "summary": {"domain_count": 0, "node_count": 0, "leaf_count": 0, "present_leaves": 0,
                        "feature_count": 0, "max_depth": 0, "coverage": None, "total_tokens": None,
                        "shared_node_count": 0, "partial_node_count": 0},
            "domain_coverage": {},
            "has_deviation_map": False,
            "has_multimodal": False,
        }

    # path -> participant id -> that participant's node at the path.
    by_path: Dict[Tuple[str, ...], Dict[str, Dict[str, Any]]] = {}
    for participant_id, tree in zip(identifiers, trees):
        for path, node in _index_participant(tree).items():
            by_path.setdefault(path, {})[participant_id] = node

    children_of: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = {}
    for path in by_path:
        if len(path) > 1:
            children_of.setdefault(path[:-1], []).append(path)

    def build(path: Tuple[str, ...]) -> Dict[str, Any]:
        entries = by_path[path]
        child_paths = sorted(set(children_of.get(path, [])))
        children = [build(child) for child in child_paths]

        scores = [n["score"] for n in entries.values() if n.get("score") is not None]
        aggregate = numeric_summary(scores)
        magnitudes = [abs(n["mean_abs_score"]) for n in entries.values() if n.get("mean_abs_score") is not None]
        mean_abs = round(sum(magnitudes) / len(magnitudes), 3) if magnitudes else None
        subtree = [n["subtree_signed_mean"] for n in entries.values() if n.get("subtree_signed_mean") is not None]
        subtree_mean = round(sum(subtree) / len(subtree), 3) if subtree else None

        # Two participants can disagree about whether a path is a measurement or
        # a group of measurements. Collapsing them would hide the direct reading
        # under an average of somebody else's children, so the path stays a
        # branch and the direct reading becomes a child of the same name.
        direct = {pid: node for pid, node in entries.items() if not node["children"]}
        if children and direct and path + (path[-1],) not in by_path:
            children.append(_direct_measurement_child(path, direct, total))

        features = _merge_features(list(entries.items()), total)
        present_in = len(entries)
        is_leaf = not children
        return {
            "id": path[-1],
            "label": humanise(path[-1]),
            "path": list(path),
            "depth": len(path) - 1,
            "kind": "domain" if len(path) == 1 else ("leaf" if is_leaf else "node"),
            "score": aggregate["mean"],
            "subtree_signed_mean": subtree_mean,
            "band": band_for(aggregate["mean"]) if is_leaf else (band_for(mean_abs) if mean_abs is not None else "missing"),
            "mean_abs_score": mean_abs,
            "leaf_count": sum(int(c["leaf_count"]) for c in children) or 1,
            "present_leaves": sum(int(c["present_leaves"]) for c in children) or (1 if aggregate["n"] else 0),
            "feature_count": len(features) + sum(int(c["feature_count"]) for c in children),
            "features": features,
            "children": _sorted_children(children),
            "value": None,
            "ref_range": None,
            "significance": None,
            "percentile": None,
            "present_in": present_in,
            "participant_count": total,
            "coverage": round(present_in / total, 4),
            "sharing": _sharing(present_in, total),
            "present_in_ids": sorted(entries),
            "missing_in_ids": sorted({p["participant_id"] for p in participants} - set(entries)),
            "branch_in": sum(1 for node in entries.values() if node["children"]),
            "leaf_in": len(direct),
            "aggregate": _aggregate_block(aggregate, present_in, total),
        }

    domains = [build(path) for path in sorted(p for p in by_path if len(p) == 1)]
    domains.sort(key=lambda d: (d["mean_abs_score"] is None, -(d["mean_abs_score"] or 0)))

    extremes: List[Dict[str, Any]] = []

    def collect(node: Dict[str, Any]) -> None:
        if node["kind"] == "leaf" and node["score"] is not None:
            extremes.append(
                {
                    "path": node["path"],
                    "label": node["label"],
                    "score": node["score"],
                    "band": node["band"],
                    "present_in": node["present_in"],
                    "coverage": node["coverage"],
                    "sharing": node["sharing"],
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
                        "present_in": feature["present_in"],
                        "coverage": feature["coverage"],
                        "sharing": feature["sharing"],
                    }
                )
        for child in node["children"]:
            collect(child)

    for domain in domains:
        collect(domain)
    extremes.sort(key=lambda row: -abs(row["score"]))

    total_leaves = sum(int(d["leaf_count"]) for d in domains)
    present_leaves = sum(int(d["present_leaves"]) for d in domains)

    def depth_of(node: Dict[str, Any]) -> int:
        if not node["children"]:
            return node["depth"]
        return max(depth_of(child) for child in node["children"])

    max_depth = max((depth_of(domain) for domain in domains), default=0)
    shared = sum(1 for node in _flatten(domains) if node["sharing"] == "shared")
    node_count = _count_nodes(domains)
    return {
        # No single participant owns this tree. The identifiers are the real
        # answer; this is the name the header falls back to.
        "participant_id": f"{total} participants" if total != 1 else identifiers[0],
        "participant_ids": list(identifiers),
        "aggregated": True,
        "participants": participants,
        "participant_count": total,
        "domains": domains,
        "extremes": extremes[:200],
        "summary": {
            "domain_count": len(domains),
            "node_count": node_count,
            "leaf_count": total_leaves,
            "present_leaves": present_leaves,
            "feature_count": sum(int(d["feature_count"]) for d in domains),
            "max_depth": max_depth + 1,
            "coverage": round(present_leaves / total_leaves, 4) if total_leaves else None,
            "total_tokens": None,
            "shared_node_count": shared,
            "partial_node_count": node_count - shared,
        },
        "domain_coverage": {},
        "has_deviation_map": has_deviation_map,
        "has_multimodal": has_multimodal,
    }


def _direct_measurement_child(
    path: Tuple[str, ...], direct: Dict[str, Dict[str, Any]], total: int
) -> Dict[str, Any]:
    """The measurement of participants who never expanded this path into a subtree."""
    scores = [node["score"] for node in direct.values() if node.get("score") is not None]
    aggregate = numeric_summary(scores)
    present_in = len(direct)
    return {
        "id": path[-1],
        "label": humanise(path[-1]),
        "path": list(path) + [path[-1]],
        "depth": len(path),
        "kind": "leaf",
        "score": aggregate["mean"],
        "subtree_signed_mean": aggregate["mean"],
        "band": band_for(aggregate["mean"]),
        "mean_abs_score": round(sum(abs(s) for s in scores) / len(scores), 3) if scores else None,
        "leaf_count": 1,
        "present_leaves": 1 if aggregate["n"] else 0,
        "feature_count": 0,
        "features": [],
        "children": [],
        "value": None,
        "ref_range": None,
        "significance": None,
        "percentile": None,
        "present_in": present_in,
        "participant_count": total,
        "coverage": round(present_in / total, 4) if total else None,
        "sharing": _sharing(present_in, total),
        "present_in_ids": sorted(direct),
        "missing_in_ids": [],
        "branch_in": 0,
        "leaf_in": present_in,
        "aggregate": _aggregate_block(aggregate, present_in, total),
        "measured_directly": True,
    }


def _sorted_children(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the cohort shares first, then what deviates most."""
    return sorted(
        children,
        key=lambda node: (-(node["present_in"]), -(node["mean_abs_score"] or 0.0), node["id"]),
    )


def _flatten(nodes: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for node in nodes:
        yield node
        yield from _flatten(node["children"])


# --- one measurement across the cohort ---------------------------------------

#: Word scales that carry an order, so their categories are not shuffled into
#: count order like an unordered label would be.
ORDERED_VOCABULARIES: Tuple[Tuple[str, ...], ...] = (
    ("never", "rarely", "sometimes", "often", "always"),
    ("none", "minimal", "mild", "moderate", "severe", "very severe"),
    ("low", "medium", "high"),
    ("very low", "low", "low normal", "normal", "high normal", "high", "very high"),
)

#: Above this many distinct values a numeric variable is a measurement scale
#: rather than a set of ordered levels.
MAX_ORDINAL_LEVELS = 6

_NUMBER_WITH_UNIT = re.compile(r"^\s*[<>~≈]?\s*([-+]?\d+(?:\.\d+)?)\s*([^\d]{0,16})$")


def _normalise_category(value: Any) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _as_number(value: Any) -> Tuple[Optional[float], str]:
    """A value's numeric reading and its unit, when it has one."""
    if isinstance(value, bool) or value is None:
        return None, ""
    if isinstance(value, (int, float)):
        number = float(value)
        return (None, "") if math.isnan(number) else (number, "")
    match = _NUMBER_WITH_UNIT.match(str(value))
    if not match:
        return None, ""
    try:
        return float(match.group(1)), match.group(2).strip()
    except ValueError:
        return None, ""


def _vocabulary_for(categories: List[str]) -> Optional[Tuple[str, ...]]:
    wanted = {_normalise_category(c) for c in categories}
    for vocabulary in ORDERED_VOCABULARIES:
        if wanted and wanted <= set(vocabulary):
            return vocabulary
    return None


def _histogram(values: List[float]) -> Optional[Dict[str, Any]]:
    """
    Freedman-Diaconis bin width, with a fallback the small cohorts here need.

    The rule divides by the cube root of the sample size, so for a handful of
    participants, or for a sample with no spread in its middle half, it produces
    either one bin or an absurd number of them. Both cases fall back to the
    square-root rule instead.
    """
    if not values:
        return None
    ordered = sorted(values)
    low, high = ordered[0], ordered[-1]
    if high == low:
        return {
            "method": "single_value",
            "bin_width": 0.0,
            "bins": [{"start": round(low, 4), "end": round(low, 4), "count": len(ordered)}],
        }

    q1 = _quantile(ordered, 0.25) or 0.0
    q3 = _quantile(ordered, 0.75) or 0.0
    iqr = q3 - q1
    width = (2 * iqr) / (len(ordered) ** (1 / 3)) if iqr > 0 else 0.0
    method = "freedman_diaconis"
    if len(ordered) < 4 or width <= 0:
        count = max(1, min(10, math.ceil(math.sqrt(len(ordered)))))
        method = "square_root"
    else:
        count = max(1, min(30, math.ceil((high - low) / width)))
    width = (high - low) / count

    bins = []
    for index in range(count):
        start = low + index * width
        end = low + (index + 1) * width
        # The last bin owns its upper edge, or the maximum falls outside every bin.
        in_bin = [v for v in ordered if (start <= v < end) or (index == count - 1 and v == high)]
        bins.append({"start": round(start, 4), "end": round(end, 4), "count": len(in_bin)})
    return {"method": method, "bin_width": round(width, 4), "bins": bins}


def _numeric_block(values: List[float]) -> Dict[str, Any]:
    """The spread of one measurement plus the bins that draw it, in one object."""
    histogram = _histogram(values) or {}
    return {
        **numeric_summary(values),
        "bins": histogram.get("bins") or [],
        "bin_method": histogram.get("method") or "",
        "bin_width": histogram.get("bin_width"),
    }


def _display(value: Any) -> str:
    """
    The reading as the interface prints it.

    A category is highlighted by comparing the focused participant's formatted
    value against the bar's label, so the two have to be formatted the same way
    or the highlight silently never matches.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(round(number, 4))
    return str(value)


def _category_block(
    raw_values: List[Any], vocabulary: Optional[Tuple[str, ...]]
) -> List[Dict[str, Any]]:
    counts: Dict[str, Dict[str, Any]] = {}
    for value in raw_values:
        key = _normalise_category(value)
        bucket = counts.setdefault(key, {"value": value, "label": _display(value), "count": 0})
        bucket["count"] += 1
    total = sum(bucket["count"] for bucket in counts.values()) or 1

    if vocabulary:
        ordered_keys = [key for key in vocabulary if key in counts]
        ordered_keys += sorted(key for key in counts if key not in vocabulary)
    else:
        ordered_keys = sorted(counts, key=lambda key: (-counts[key]["count"], key))

    return [
        {
            "label": counts[key]["label"],
            # The unformatted reading, kept so an ordered numeric scale can be
            # sorted by its own magnitude rather than by its printed form.
            "value": counts[key]["value"],
            "count": counts[key]["count"],
            "proportion": round(counts[key]["count"] / total, 4),
        }
        for key in ordered_keys
    ]


def _infer_type(raw_values: List[Any], numbers: List[float]) -> str:
    if not raw_values:
        return "missing"
    if len(numbers) < len(raw_values):
        return "ordinal" if _vocabulary_for([str(v) for v in raw_values]) else "nominal"
    distinct = {round(n, 10) for n in numbers}
    integral = all(float(n).is_integer() for n in numbers)
    # A handful of repeated whole numbers is a rating scale; whole numbers that
    # never repeat are a count, and anything fractional is a measurement.
    if integral and len(distinct) <= MAX_ORDINAL_LEVELS and len(distinct) < len(numbers):
        return "ordinal"
    return "integer" if integral else "continuous"


def leaf_distribution(participant_dirs: List[Path], path: List[str]) -> Dict[str, Any]:
    """
    Every participant's reading of one node, ready to plot.

    The tree carries both a raw value and a z-score, and which of the two is
    worth plotting depends on the variable, so both are returned along with the
    variable type inferred from what the cohort actually contains.
    """
    key = tuple(str(segment) for segment in path)
    if not key:
        raise ValueError("A node path is required.")

    directories = [Path(directory) for directory in participant_dirs]
    trees = [build_ontology(directory) for directory in directories]
    identifiers = _unique_ids(trees, directories)

    rows: List[Dict[str, Any]] = []
    label = humanise(key[-1])
    for participant_id, directory, tree in zip(identifiers, directories, trees):
        index = _index_participant(tree)
        node = index.get(key)
        value: Any = None
        z_score: Optional[float] = None
        source = ""
        if node is not None:
            source = "node"
            value = node.get("value")
            z_score = node.get("score")
            label = node.get("label") or label
        else:
            parent = index.get(key[:-1])
            wanted = _match_key(key[-1])
            for feature in (parent or {}).get("features") or []:
                if _match_key(feature.get("feature") or "") == wanted:
                    source = "feature"
                    value = feature.get("value")
                    z_score = feature.get("z_score")
                    label = feature.get("feature") or label
                    break
        number, unit = _as_number(value)
        rows.append(
            {
                "participant_id": participant_id,
                "directory": str(directory),
                "value": value,
                "numeric_value": number,
                "unit": unit,
                "z_score": z_score,
                "found": bool(source),
                "source": source,
            }
        )

    present = [row for row in rows if row["found"]]
    raw_values = [row["value"] for row in present if row["value"] is not None]
    z_values = [float(row["z_score"]) for row in present if row["z_score"] is not None]
    units = {row["unit"] for row in present if row["unit"]}

    # A branch node carries a deviation but no reading of its own. Calling that
    # "not measured" is wrong: the deviation is the reading there, so it becomes
    # the value of every row and `reading` records which of the two is on show.
    reading = "value" if raw_values else ("z_score" if z_values else "none")
    if reading == "z_score":
        for row in rows:
            row["value"] = row["z_score"]
            row["numeric_value"] = row["z_score"]
        raw_values = [row["value"] for row in rows if row["value"] is not None]

    numbers = [row["numeric_value"] for row in rows if row["numeric_value"] is not None]
    variable_type = _infer_type(raw_values, numbers) if raw_values else "missing"

    numeric: Optional[Dict[str, Any]] = None
    categories: List[Dict[str, Any]] = []
    if variable_type in ("continuous", "integer"):
        numeric = _numeric_block(numbers)
    elif variable_type == "ordinal" and len(numbers) == len(raw_values):
        # An ordered numeric scale is still worth summarising as numbers.
        numeric = _numeric_block(numbers)
        categories = _category_block(numbers, None)
        categories.sort(key=lambda row: float(row["value"]))
    elif variable_type in ("ordinal", "nominal"):
        categories = _category_block(raw_values, _vocabulary_for([str(v) for v in raw_values]))

    payload: Dict[str, Any] = {
        "path": list(key),
        "label": label,
        "variable_type": variable_type,
        "participant_count": len(rows),
        # One entry per participant in row order, so `values[i]` and `rows[i]`
        # describe the same person and a gap stays visible as a null.
        "values": [row["value"] for row in rows],
        "rows": rows,
        # Which of the two readings the values above are, so a caller can tell a
        # branch node's deviation apart from a measured quantity.
        "reading": reading,
        "plot_field": "numeric" if numeric else ("categories" if categories else ""),
        "found_in": len(present),
        "unit": sorted(units)[0] if len(units) == 1 else "",
        "units": sorted(units),
        "missing_in_ids": [row["participant_id"] for row in rows if not row["found"]],
    }
    # The three summaries are optional in the contract, so an absent one is left
    # out rather than sent as a null the reader has to interpret.
    if numeric is not None:
        payload["numeric"] = numeric
    if categories:
        payload["categories"] = categories
    if z_values:
        payload["z_scores"] = _numeric_block(z_values)
    return payload
