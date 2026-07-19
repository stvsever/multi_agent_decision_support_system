"""
LLM-based construction of a non-redundant subclass ontology over dataset features.

Given the deterministic feature manifest, an LLM proposes a strict subclass
hierarchy (Domain -> Subdomain -> Feature) in which every predictor appears
exactly once. The proposal is then *programmatically validated and repaired* so
the final ontology is guaranteed to be complete and non-redundant regardless of
model quality. Two artifacts are emitted:

* ``subclass_structure.json`` - the machine-readable tree plus the
  column -> ontology-path index used by the deviation encoder.
* ``<name>.owl`` - an RDF/XML OWL file (rdfs:subClassOf hierarchy with labels
  and definitions) that loads directly in Protege for visual inspection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .llm import OntologyLLM

_SYSTEM_PROMPT = """You are an ontology engineer building a clean biomedical/psychometric feature ontology.
You organise a flat list of measured features into a strict IS-A (subclass) hierarchy.

Hard rules:
1. Exactly three levels: DOMAIN (broadest) -> SUBDOMAIN -> FEATURE (the leaves).
2. Every input feature must appear as a leaf EXACTLY ONCE. Never drop or duplicate a feature.
3. Grouping must be MECE (mutually exclusive, collectively exhaustive) and non-redundant:
   a feature belongs to one subdomain only, and subdomains to one domain only.
4. Group by shared construct/measurement instrument (e.g. all Big Five scales under one
   personality domain), not by statistical type.
5. Use concise, human-readable domain/subdomain names in UPPER_SNAKE_CASE for domains and
   lower_snake_case for subdomains. Provide a one-sentence definition for each.
6. Do NOT invent features that are not in the input list.

Return ONLY a JSON object, no prose."""

_USER_TEMPLATE = """Dataset: {dataset}
Context: {context}

Organise the following {n} features into the DOMAIN -> SUBDOMAIN -> FEATURE ontology.

Features (column :: label :: description):
{feature_lines}

Return JSON exactly in this shape:
{{
  "domains": [
    {{
      "id": "DOMAIN_NAME",
      "label": "Readable Domain Name",
      "definition": "one sentence",
      "subdomains": [
        {{
          "id": "subdomain_name",
          "label": "Readable Subdomain Name",
          "definition": "one sentence",
          "features": ["column_a", "column_b"]
        }}
      ]
    }}
  ]
}}"""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip()).strip("_")


_LABELING_SYSTEM = """You are an ontology engineer writing clean, interpretable labels for a
feature ontology. You are given a fixed DOMAIN -> SUBDOMAIN -> FEATURE hierarchy (the
structure is decided; do not change it). For every domain and subdomain, write a concise
human-readable label and a one-sentence definition, so a clinician or researcher browsing
the ontology immediately understands the grouping. Return ONLY JSON."""


def build_labeled_ontology(
    features: List[Dict[str, Any]],
    dataset_name: str,
    context: str,
    llm,
) -> Dict[str, Any]:
    """Build a hint-structured, LLM-labeled ontology.

    Each feature dict must carry: id, label, definition, stat_type, units, and the
    structural hints ``domain`` and ``subdomain``. The hierarchy is built
    deterministically from those hints (guaranteeing clean, non-redundant, fully
    covered domains, including keeping brain modalities separate). A single LLM call
    then generates the interpretable parent labels and definitions. This is the
    optimised path for large multi-modal feature sets where free-form grouping by a
    small model is brittle.
    """
    # Deterministic structure from hints (preserves first-seen order).
    domains_order: List[str] = []
    subs_order: Dict[str, List[str]] = {}
    buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for feat in features:
        dom = _slug(feat.get("domain") or "DOMAIN").upper()
        sub = _slug(feat.get("subdomain") or "general").lower()
        if dom not in buckets:
            buckets[dom] = {}
            subs_order[dom] = []
            domains_order.append(dom)
        if sub not in buckets[dom]:
            buckets[dom][sub] = []
            subs_order[dom].append(sub)
        buckets[dom][sub].append(feat)

    labels = _request_node_labels(llm, dataset_name, context, domains_order, subs_order, buckets)

    domains: List[Dict[str, Any]] = []
    for dom in domains_order:
        dom_lbl = labels.get(dom, {})
        subdomains = []
        for sub in subs_order[dom]:
            sub_lbl = dom_lbl.get("subdomains", {}).get(sub, {})
            subdomains.append({
                "id": sub,
                "label": sub_lbl.get("label") or sub.replace("_", " ").title(),
                "definition": sub_lbl.get("definition") or "",
                "features": [
                    {"id": f["id"], "label": f.get("label", f["id"]),
                     "definition": _clean_definition(f.get("label", f["id"]), f.get("definition", "")),
                     "stat_type": f.get("stat_type"), "units": f.get("units")}
                    for f in buckets[dom][sub]
                ],
            })
        domains.append({
            "id": dom,
            "label": dom_lbl.get("label") or dom.replace("_", " ").title(),
            "definition": dom_lbl.get("definition") or "",
            "subdomains": subdomains,
        })

    n_feats = sum(len(s["features"]) for d in domains for s in d["subdomains"])
    ontology = {
        "domains": domains,
        "repair_stats": {"duplicates_removed": 0, "unknown_removed": 0, "unassigned_added": 0},
        "column_index": _build_column_index(domains),
        "dataset": dataset_name,
        "context": context,
        "n_features": n_feats,
        "construction": "hint_structured_llm_labeled",
    }
    return ontology


def _clean_definition(label: str, definition: str) -> str:
    """Drop a leaf definition that merely echoes the label (avoids redundant text)."""
    d = str(definition or "").strip()
    if not d:
        return ""
    lab = str(label or "").strip().lower()
    core = d.lower().rstrip(".")
    # Strip common auto-prefixes then compare to the label.
    for pref in ("freesurfer global measure:", "freesurfer volume of the", "mean functional connectivity"):
        if core.startswith(pref):
            return ""
    if lab and lab in core and (len(core) - len(lab)) < 20:
        return ""
    return d


_AUTO_SYSTEM = """You are an ontology engineer. You are given data-driven feature clusters (features
that behave statistically alike) plus each feature's human label. Organise them into a clean,
non-redundant DOMAIN -> SUBDOMAIN -> FEATURE subclass ontology. Keep every feature exactly once,
group by shared construct/measurement, and give concise interpretable labels and definitions.
Return ONLY JSON."""

_VERIFY_SYSTEM = """You are a senior ontology reviewer. Assess a fixed feature ontology for quality:
mutual exclusivity and collective exhaustiveness (MECE), non-redundancy, and whether each feature
sits in the most sensible subdomain. Be concise and specific. Return ONLY JSON."""


def build_ontology_auto(
    features: List[Dict[str, Any]],
    dataset_name: str,
    context: str,
    llm,
    exploration: Dict[str, Any],
) -> Dict[str, Any]:
    """Hint-free ontology construction driven by automated exploration.

    Statistical feature clusters (from ``explore``) seed the structure; the LLM
    assigns clusters to domains and writes all labels/definitions. Used when a
    dataset provides no domain knowledge; validated and repaired for coverage.
    """
    by_id = {f["id"]: f for f in features}
    clusters = exploration.get("auto_clusters") or {f["id"]: [f["id"]] for f in features}
    cluster_lines = []
    for cid, cols in clusters.items():
        labels = ", ".join(by_id[c]["label"] for c in cols if c in by_id)
        cluster_lines.append(f"{cid}: [{labels}]")
    user = (
        f"Dataset: {dataset_name}\nContext: {context}\n\n"
        "Assign each statistical cluster to a domain and subdomain, and label everything.\n\n"
        + "\n".join(cluster_lines) +
        '\n\nReturn: {"clusters": {"cluster_id": {"domain": "DOMAIN_ID", "domain_label": "..", '
        '"subdomain": "sub_id", "subdomain_label": ".."}}}'
    )
    mapping = {}
    try:
        obj = llm.chat_json(_AUTO_SYSTEM, user, max_tokens=4000)
        mapping = obj.get("clusters", {}) if isinstance(obj, dict) else {}
    except Exception:
        mapping = {}
    # Attach hints from the LLM cluster assignment (fallback: one domain per cluster).
    for cid, cols in clusters.items():
        m = mapping.get(cid, {})
        dom = _slug(m.get("domain") or "DATA").upper()
        sub = _slug(m.get("subdomain") or cid).lower()
        for c in cols:
            if c in by_id:
                by_id[c] = {**by_id[c], "domain": dom, "subdomain": sub}
    return build_labeled_ontology(list(by_id.values()), dataset_name, context, llm)


def assess_ontology(
    ontology: Dict[str, Any],
    exploration: Optional[Dict[str, Any]],
    llm=None,
    verify: bool = True,
) -> Dict[str, Any]:
    """Produce a quality report: coverage, agreement with data clusters, LLM review."""
    domains = ontology["domains"]
    n_dom = len(domains)
    n_sub = sum(len(d["subdomains"]) for d in domains)
    n_feat = sum(len(s["features"]) for d in domains for s in d["subdomains"])
    sizes = [len(s["features"]) for d in domains for s in d["subdomains"]]
    report: Dict[str, Any] = {
        "n_domains": n_dom, "n_subdomains": n_sub, "n_features": n_feat,
        "subdomain_size_min_max_mean": [min(sizes), max(sizes), round(float(np.mean(sizes)), 2)] if sizes else None,
    }
    if exploration:
        report["cluster_agreement"] = _cluster_agreement(ontology, exploration)
        report["redundant_pairs"] = exploration.get("redundant_pairs", [])[:10]
        report["quality_flags"] = exploration.get("quality_flags", {})
    if verify and llm is not None:
        report["llm_review"] = _verify_ontology(llm, ontology)
    return report


def _cluster_agreement(ontology: Dict[str, Any], exploration: Dict[str, Any]) -> Dict[str, Any]:
    """Adjusted Rand Index between the semantic subdomain partition and data clusters."""
    feat_sub = {f["id"]: f"{d['id']}/{s['id']}"
                for d in ontology["domains"] for s in d["subdomains"] for f in s["features"]}
    feat_clu: Dict[str, str] = {}
    for cid, cols in (exploration.get("auto_clusters") or {}).items():
        for c in cols:
            feat_clu[c] = cid
    common = [f for f in feat_sub if f in feat_clu]
    if len(common) < 3:
        return {"adjusted_rand_index": None, "n_features_compared": len(common)}
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score([feat_sub[f] for f in common], [feat_clu[f] for f in common])
        return {"adjusted_rand_index": round(float(ari), 3), "n_features_compared": len(common),
                "interpretation": "1.0 = ontology subdomains match data-driven clusters exactly"}
    except Exception:
        return {"adjusted_rand_index": None, "n_features_compared": len(common)}


def _verify_ontology(llm, ontology: Dict[str, Any]) -> Dict[str, Any]:
    """One compact LLM QA pass over the ontology (structure sent as TOON)."""
    compact = {d["label"]: {s["label"]: [f["label"] for f in s["features"]]
                            for s in d["subdomains"]} for d in ontology["domains"]}
    try:
        from src.full_stack.backend.utils.toon import json_to_toon
        payload = json_to_toon(compact)
    except Exception:
        payload = str(compact)
    user = (
        "Review this DOMAIN -> SUBDOMAIN -> FEATURE ontology (TOON).\n\n" + payload +
        '\n\nReturn: {"mece_ok": true/false, "coherence_1to5": <int>, '
        '"issues": ["..."], "misplaced_features": ["feature -> better subdomain"]}'
    )
    try:
        obj = llm.chat_json(_VERIFY_SYSTEM, user, max_tokens=1500)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _request_node_labels(llm, dataset_name, context, domains_order, subs_order, buckets) -> Dict[str, Any]:
    """One compact LLM call to label every domain/subdomain node.

    The node structure is serialised as TOON (token-oriented object notation) rather
    than JSON to cut prompt tokens, and only representative feature labels are sent
    (not every feature), so the call stays small even for large multi-modal datasets.
    """
    nodes: Dict[str, Any] = {}
    for dom in domains_order:
        nodes[dom] = {sub: [f["label"] for f in buckets[dom][sub][:4]] for sub in subs_order[dom]}
    try:
        from src.full_stack.backend.utils.toon import json_to_toon  # engine's TOON serialiser
        payload = json_to_toon(nodes)
    except Exception:
        payload = "\n".join(
            f"{dom}:\n" + "\n".join(
                f"  {sub}: " + ", ".join(f["label"] for f in buckets[dom][sub][:4])
                for sub in subs_order[dom])
            for dom in domains_order)
    user = (
        f"Dataset: {dataset_name}\nContext: {context}\n\n"
        "The hierarchy below (TOON: DOMAIN -> subdomain -> example feature labels) is fixed. "
        "Write a concise label and one-sentence definition for each domain and each subdomain.\n\n"
        f"{payload}\n\n"
        "Return exactly:\n"
        '{"domains": {"DOMAIN_ID": {"label": "...", "definition": "...", '
        '"subdomains": {"subdomain_id": {"label": "...", "definition": "..."}}}}}'
    )
    try:
        obj = llm.chat_json(_LABELING_SYSTEM, user, max_tokens=4000)
        return obj.get("domains", {}) if isinstance(obj, dict) else {}
    except Exception:
        return {}  # fall back to prettified ids


def build_ontology(
    manifest: Dict[str, Any],
    dataset_name: str,
    context: str,
    llm: OntologyLLM,
) -> Dict[str, Any]:
    """Build and validate the ontology from a feature manifest."""
    predictors = manifest["predictors"]
    by_column = {p["column"]: p for p in predictors}
    all_columns = list(by_column.keys())

    feature_lines = "\n".join(
        f"- {p['column']} :: {p['label']} :: {p['description']}" for p in predictors
    )
    user = _USER_TEMPLATE.format(
        dataset=dataset_name,
        context=context,
        n=len(predictors),
        feature_lines=feature_lines,
    )
    proposal = llm.chat_json(_SYSTEM_PROMPT, user, max_tokens=6000)
    domains = _coerce_domains(proposal)
    ontology = _validate_and_repair(domains, all_columns, by_column)
    ontology["dataset"] = dataset_name
    ontology["context"] = context
    ontology["n_features"] = len(all_columns)
    return ontology


def _coerce_domains(proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    domains = proposal.get("domains")
    if not isinstance(domains, list):
        raise ValueError("Ontology proposal missing 'domains' list")
    return domains


def _validate_and_repair(
    domains: List[Dict[str, Any]],
    all_columns: List[str],
    by_column: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Enforce completeness and non-redundancy on the LLM proposal.

    Any feature the model placed more than once is kept only on its first
    occurrence; any feature the model forgot is appended to an ``UNASSIGNED``
    domain so the ontology always covers every predictor exactly once.
    """
    seen: set[str] = set()
    clean_domains: List[Dict[str, Any]] = []
    stats = {"duplicates_removed": 0, "unknown_removed": 0, "unassigned_added": 0}

    for dom in domains:
        dom_id = _slug(dom.get("id") or dom.get("label") or "DOMAIN").upper()
        clean_subs: List[Dict[str, Any]] = []
        for sub in dom.get("subdomains", []) or []:
            sub_id = _slug(sub.get("id") or sub.get("label") or "subdomain").lower()
            feats: List[str] = []
            for col in sub.get("features", []) or []:
                col = str(col).strip()
                if col not in by_column:
                    stats["unknown_removed"] += 1
                    continue
                if col in seen:
                    stats["duplicates_removed"] += 1
                    continue
                seen.add(col)
                feats.append(col)
            if feats:
                clean_subs.append(
                    {
                        "id": sub_id,
                        "label": str(sub.get("label") or sub_id.replace("_", " ").title()),
                        "definition": str(sub.get("definition") or ""),
                        "features": [_feature_node(by_column[c]) for c in feats],
                    }
                )
        if clean_subs:
            clean_domains.append(
                {
                    "id": dom_id,
                    "label": str(dom.get("label") or dom_id.replace("_", " ").title()),
                    "definition": str(dom.get("definition") or ""),
                    "subdomains": clean_subs,
                }
            )

    missing = [c for c in all_columns if c not in seen]
    if missing:
        stats["unassigned_added"] = len(missing)
        clean_domains.append(
            {
                "id": "UNASSIGNED",
                "label": "Unassigned Features",
                "definition": "Features the ontology model did not place; retained for completeness.",
                "subdomains": [
                    {
                        "id": "unassigned",
                        "label": "Unassigned",
                        "definition": "Auto-assigned to preserve full feature coverage.",
                        "features": [_feature_node(by_column[c]) for c in missing],
                    }
                ],
            }
        )

    return {
        "domains": clean_domains,
        "repair_stats": stats,
        "column_index": _build_column_index(clean_domains),
    }


def _feature_node(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": profile["column"],
        "label": profile["label"],
        "definition": profile["description"],
        "stat_type": profile["stat_type"],
        "units": profile.get("units"),
    }


def _build_column_index(domains: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for dom in domains:
        for sub in dom["subdomains"]:
            for feat in sub["features"]:
                index[feat["id"]] = {
                    "domain": dom["id"],
                    "subdomain": sub["id"],
                    "feature_label": feat["label"],
                }
    return index


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

def write_subclass_json(ontology: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ontology, f, indent=2)


def write_owl(ontology: Dict[str, Any], path: Path, base_iri: Optional[str] = None) -> None:
    """Emit a Protege-loadable OWL file (RDF/XML) using rdflib."""
    from rdflib import Graph, Namespace, Literal, RDF, RDFS, OWL

    dataset = _slug(ontology.get("dataset", "dataset")).lower()
    base_iri = base_iri or f"https://compass.local/ontology/{dataset}#"
    ns = Namespace(base_iri)

    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("", ns)

    ontology_uri = ns[""]
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal(f"COMPASS feature ontology: {ontology.get('dataset')}")))
    g.add((ontology_uri, RDFS.comment, Literal(str(ontology.get("context", "")))))

    # Root class that every domain specialises.
    root = ns.PhenotypeFeature
    g.add((root, RDF.type, OWL.Class))
    g.add((root, RDFS.label, Literal("Phenotype Feature")))
    g.add((root, RDFS.comment, Literal("Root class for all measured participant features.")))

    # Annotation properties for provenance.
    src_col = ns.sourceColumn
    stat_type = ns.statType
    for prop, label in [(src_col, "source column"), (stat_type, "statistical type")]:
        g.add((prop, RDF.type, OWL.AnnotationProperty))
        g.add((prop, RDFS.label, Literal(label)))

    def _cls(local: str):
        return ns[_slug(local)]

    for dom in ontology["domains"]:
        dom_cls = _cls(f"DOMAIN_{dom['id']}")
        g.add((dom_cls, RDF.type, OWL.Class))
        g.add((dom_cls, RDFS.subClassOf, root))
        g.add((dom_cls, RDFS.label, Literal(dom["label"])))
        if dom.get("definition"):
            g.add((dom_cls, RDFS.comment, Literal(dom["definition"])))

        for sub in dom["subdomains"]:
            sub_cls = _cls(f"SUB_{dom['id']}_{sub['id']}")
            g.add((sub_cls, RDF.type, OWL.Class))
            g.add((sub_cls, RDFS.subClassOf, dom_cls))
            g.add((sub_cls, RDFS.label, Literal(sub["label"])))
            if sub.get("definition"):
                g.add((sub_cls, RDFS.comment, Literal(sub["definition"])))

            for feat in sub["features"]:
                feat_cls = _cls(f"FEATURE_{feat['id']}")
                g.add((feat_cls, RDF.type, OWL.Class))
                g.add((feat_cls, RDFS.subClassOf, sub_cls))
                g.add((feat_cls, RDFS.label, Literal(feat["label"])))
                if feat.get("definition"):
                    g.add((feat_cls, RDFS.comment, Literal(feat["definition"])))
                g.add((feat_cls, src_col, Literal(feat["id"])))
                if feat.get("stat_type"):
                    g.add((feat_cls, stat_type, Literal(feat["stat_type"])))

    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="pretty-xml")
