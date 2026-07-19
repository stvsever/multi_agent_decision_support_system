"""
Project the master ontology onto a data-complexity tier.

A tier is a subset of feature columns (the union of its feature groups). Projecting
keeps only the ontology leaves whose column is in that set and prunes any subdomain
or domain left empty. Because the master ontology is built once and fixed, every tier
is a clean, consistent sub-ontology of it: individuals within a tier never differ in
structure, only in which leaf values are present. This is what lets the engine treat
each tier as a well-formed hierarchical input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def project_ontology(ontology: Dict[str, Any], allowed_columns: Set[str]) -> Dict[str, Any]:
    """Return a new ontology containing only leaves in ``allowed_columns``."""
    domains: List[Dict[str, Any]] = []
    for dom in ontology["domains"]:
        subs = []
        for sub in dom["subdomains"]:
            feats = [f for f in sub["features"] if f["id"] in allowed_columns]
            if feats:
                subs.append({**sub, "features": feats})
        if subs:
            domains.append({**dom, "subdomains": subs})

    n_feats = sum(len(s["features"]) for d in domains for s in d["subdomains"])
    projected = {
        "domains": domains,
        "dataset": ontology.get("dataset"),
        "context": ontology.get("context"),
        "n_features": n_feats,
        "column_index": {
            c: idx for c, idx in (ontology.get("column_index") or {}).items() if c in allowed_columns
        },
        "construction": ontology.get("construction"),
    }
    return projected


def tier_columns(tier: Dict[str, Any], groups: Dict[str, List[str]]) -> List[str]:
    """Columns for a tier: the union of its groups' columns, in group order."""
    cols: List[str] = []
    for gid in tier["groups"]:
        cols.extend(groups.get(gid, []))
    return cols
