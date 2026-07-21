"""
Project the master ontology onto a data-complexity tier.

A tier is a subset of feature columns (the union of its feature groups). Projecting
keeps only the ontology leaves whose column is in that set and prunes any internal
node left empty, at any depth. Because the master ontology is built once and fixed,
every tier is a clean, consistent sub-ontology of it: individuals within a tier never
differ in structure, only in which leaf values are present. This is what lets the
engine treat each tier as a well-formed hierarchical input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .ontology import is_leaf, count_leaves, _build_column_index


def _prune_node(node: Dict[str, Any], allowed: Set[str]) -> Optional[Dict[str, Any]]:
    """Return a copy of ``node`` keeping only allowed leaves, or None if empty."""
    if is_leaf(node):
        return dict(node) if node["id"] in allowed else None
    kept = [p for p in (_prune_node(c, allowed) for c in node["children"]) if p is not None]
    if not kept:
        return None
    return {**{k: v for k, v in node.items() if k != "children"}, "children": kept}


def project_ontology(ontology: Dict[str, Any], allowed_columns: Set[str]) -> Dict[str, Any]:
    """Return a new ontology containing only leaves in ``allowed_columns``."""
    domains: List[Dict[str, Any]] = []
    for dom in ontology["domains"]:
        pruned = _prune_node(dom, set(allowed_columns))
        if pruned is not None:
            domains.append(pruned)

    n_feats = sum(count_leaves(d) for d in domains)
    return {
        "domains": domains,
        "dataset": ontology.get("dataset"),
        "context": ontology.get("context"),
        "n_features": n_feats,
        "column_index": _build_column_index(domains),
        "construction": ontology.get("construction"),
    }


def tier_columns(tier: Dict[str, Any], groups: Dict[str, List[str]]) -> List[str]:
    """Columns for a tier: the union of its groups' columns, in group order."""
    cols: List[str] = []
    for gid in tier["groups"]:
        cols.extend(groups.get(gid, []))
    return cols
