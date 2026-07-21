"""
LLM-driven construction of an arbitrary-depth, non-redundant feature ontology.

The COMPASS engine reasons over a hierarchy of measured features. This module
turns a flat feature manifest into a strict IS-A tree

    Phenotype Feature -> DOMAIN -> ... any number of intermediate levels ... -> FEATURE

in which every predictor appears as a leaf exactly once. Two placement strategies
are supported and can be mixed freely in one dataset:

* deterministic ``path`` hints - a feature spec may carry an explicit
  ``path`` (an ordered list of ``{"id","label","definition"}`` segments from the
  domain down to the leaf's parent). High-resolution modalities (e.g. per-region
  brain morphometry, connectomics) use this to get a clean, deep, reproducible
  structure that does not depend on model quality.
* semantic LLM grouping - features with no ``path`` are grouped by MEANING: the
  model proposes domains from the whole feature set, then organises each domain's
  features into subdomains. This is general to any modality (questionnaires,
  assays, imaging, EEG, lesion masks, ...).

Whatever the strategy, completeness and non-redundancy are enforced in code, so
the final ontology always covers every feature exactly once. The tree is emitted
as ``subclass_structure.json`` (machine-readable, engine-facing) and ``<name>.owl``
(Protege-loadable RDF/XML), and is arbitrary depth throughout - the engine's data
loader and both compressor tools already recurse to any depth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .llm import OntologyLLM


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text).strip()).strip("_")


def _titled(text: str) -> str:
    return _slug(text).replace("_", " ").strip().title()


# =========================================================================== #
# Tree primitives (arbitrary depth)
#
# A node is a dict. A leaf carries a real feature column in ``id`` and has no
# ``children``. An internal node has a ``children`` list. The top-level nodes of
# an ontology are its DOMAINS (kept under the ``domains`` key so the engine, which
# treats top-level keys as domains, sees a familiar shape).
# =========================================================================== #

def is_leaf(node: Dict[str, Any]) -> bool:
    return not node.get("children")


def count_leaves(node: Dict[str, Any]) -> int:
    if is_leaf(node):
        return 1
    return sum(count_leaves(c) for c in node["children"])


def node_depth(node: Dict[str, Any]) -> int:
    """Max number of edges from this node down to its deepest leaf."""
    if is_leaf(node):
        return 0
    return 1 + max(node_depth(c) for c in node["children"])


def iter_leaves(node: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    """Yield ``(ancestor_id_path, leaf_node)`` for every leaf under ``node``."""
    if is_leaf(node):
        yield prefix, node
        return
    for c in node["children"]:
        yield from iter_leaves(c, prefix + (node["id"],))


def ontology_leaves(ontology: Dict[str, Any]) -> Iterable[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    for dom in ontology["domains"]:
        yield from iter_leaves(dom, ())


def _leaf_node(feature: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": feature["id"],
        "label": feature.get("label", feature["id"]),
        "definition": _clean_definition(feature.get("label", feature["id"]), feature.get("definition", "")),
        "stat_type": feature.get("stat_type"),
        "units": feature.get("units"),
    }


def _ensure_child(children: List[Dict[str, Any]], seg: Dict[str, Any]) -> Dict[str, Any]:
    """Find or create an internal child node matching ``seg`` (by id)."""
    sid = _slug(seg["id"])
    for c in children:
        if c["id"] == sid and not is_leaf(c):
            # Backfill a better label/definition if the segment now supplies one.
            if seg.get("label") and (not c.get("label") or c["label"] == _titled(sid)):
                c["label"] = seg["label"]
            if seg.get("definition") and not c.get("definition"):
                c["definition"] = seg["definition"]
            return c
    node = {
        "id": sid,
        "label": seg.get("label") or _titled(sid),
        "definition": seg.get("definition", ""),
        "children": [],
    }
    children.append(node)
    return node


def _insert_feature(domains: List[Dict[str, Any]], path_segs: List[Dict[str, Any]], leaf: Dict[str, Any]) -> None:
    """Insert ``leaf`` under the ``path_segs`` chain (path_segs[0] is the domain)."""
    if not path_segs:
        path_segs = [{"id": "UNASSIGNED", "label": "Unassigned Features"}]
    cursor = _ensure_child(domains, path_segs[0])
    for seg in path_segs[1:]:
        cursor = _ensure_child(cursor["children"], seg)
    cursor["children"].append(leaf)


def _clean_definition(label: str, definition: str) -> str:
    """Drop a leaf definition that merely echoes the label (avoids redundant text)."""
    d = str(definition or "").strip()
    if not d:
        return ""
    lab = str(label or "").strip().lower()
    core = d.lower().rstrip(".")
    for pref in ("freesurfer global measure:", "freesurfer volume of the", "mean functional connectivity"):
        if core.startswith(pref):
            return ""
    if lab and lab in core and (len(core) - len(lab)) < 20:
        return ""
    return d


# =========================================================================== #
# Semantic grouping of un-pathed features (LLM proposes domains + subdomains)
# =========================================================================== #

_DOMAIN_SYSTEM = """You are a biomedical ontology engineer. Given measured features of a dataset
(with labels, descriptions, units and source hints), propose a small set of top-level DOMAINS that
group the features by SEMANTIC meaning and measurement source (e.g. demographics, a questionnaire
construct, a biological assay, brain structure, brain function). Group by what features MEAN, never
by statistical similarity. Aim for 3 to 12 coherent, non-overlapping domains. Return ONLY JSON."""

_ORGANISE_SYSTEM = """You are a biomedical ontology engineer. From the full feature list, select the
features that belong to the given DOMAIN and organise ONLY those into SUBDOMAINS by semantic meaning
(shared construct, sub-system, or anatomy), never by statistical similarity. Each selected feature
appears in exactly one subdomain. Use concise lower_snake_case subdomain ids with readable labels and
one-sentence definitions. Do not invent feature ids. Return ONLY JSON."""

_LABELING_SYSTEM = """You are an ontology engineer writing clean, interpretable labels for a feature
ontology. You are given a fixed hierarchy of internal nodes (the structure is decided; do not change
it). For every node id, write a concise human-readable label and a one-sentence definition, so a
researcher browsing the ontology immediately understands the grouping. Return ONLY JSON."""

_VERIFY_SYSTEM = """You are a senior ontology reviewer assessing a hierarchy over measured dataset
columns. MECE means every supplied column is assigned exactly once to one semantically appropriate
group. Non-redundancy means no column is duplicated; it does NOT require measured columns to be
statistically independent. Aggregate and component measurements may both be valid leaves, directional
questionnaire ratings are distinct measurements, and matrix cells such as unique network pairs are
distinct leaves. Do not propose merging, deleting, or re-encoding source columns. Assess semantic
placement and structural coverage only. Be concise and specific. Return ONLY JSON."""


def _tooned(obj: Any) -> str:
    try:
        from src.full_stack.backend.utils.toon import json_to_toon
        return json_to_toon(obj)
    except Exception:
        return json.dumps(obj)


def _feature_listing(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in features:
        row = {"id": f["id"], "label": f.get("label", f["id"])}
        if f.get("definition"):
            row["desc"] = f["definition"]
        if f.get("units"):
            row["units"] = f["units"]
        if f.get("source"):
            row["source"] = f["source"]
        if f.get("sample") is not None:
            row["sample"] = f["sample"]
        out.append(row)
    return out


def _propose_domains(llm, dataset_name, context, listing, user_guidance) -> List[Dict[str, Any]]:
    guidance = f"\nUser guidance (respect this): {user_guidance.strip()}\n" if user_guidance.strip() else ""
    user = (
        f"Dataset: {dataset_name}\nContext: {context}{guidance}\n"
        f"All {len(listing)} features (TOON):\n{_tooned(listing)}\n\n"
        'Return: {"domains": [{"id": "UPPER_SNAKE_CASE", "label": "..", "definition": ".."}]}'
    )
    try:
        obj = llm.chat_json(_DOMAIN_SYSTEM, user, max_tokens=2500)
        doms = obj.get("domains", []) if isinstance(obj, dict) else []
    except Exception:
        doms = []
    clean = []
    for d in doms:
        did = _slug(d.get("id") or d.get("label") or "").upper()
        if did:
            clean.append({"id": did, "label": d.get("label") or _titled(did),
                          "definition": d.get("definition") or ""})
    return clean or [{"id": "FEATURES", "label": "Features", "definition": ""}]


def _organise_domain(llm, dataset_name, domain, listing, user_guidance) -> Dict[str, Any]:
    guidance = f"\nUser guidance (respect this): {user_guidance.strip()}\n" if user_guidance.strip() else ""
    user = (
        f"Dataset: {dataset_name}\nDOMAIN to populate: {domain['label']} (id {domain['id']}) - "
        f"{domain.get('definition','')}{guidance}\n"
        f"Full feature list (TOON); select only those that belong to this domain:\n{_tooned(listing)}\n\n"
        'Return: {"subdomains": [{"id": "..", "label": "..", "definition": "..", '
        '"features": ["feature_id", ...]}]}'
    )
    try:
        obj = llm.chat_json(_ORGANISE_SYSTEM, user, max_tokens=4000)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _semantic_paths(llm, dataset_name, context, features, user_guidance, max_workers) -> Dict[str, List[Dict[str, Any]]]:
    """Return {feature_id: [domain_seg, subdomain_seg]} by LLM meaning grouping."""
    from concurrent.futures import ThreadPoolExecutor

    by_id = {f["id"]: f for f in features}
    listing = _feature_listing(features)
    domain_specs = _propose_domains(llm, dataset_name, context, listing, user_guidance)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        proposals = list(ex.map(
            lambda d: (d, _organise_domain(llm, dataset_name, d, listing, user_guidance)),
            domain_specs,
        ))

    claimed: Dict[str, List[Dict[str, Any]]] = {}
    for spec, proposal in proposals:
        for sub in proposal.get("subdomains", []) or []:
            sub_seg = {"id": _slug(sub.get("id") or "subdomain").lower(),
                       "label": sub.get("label") or _titled(sub.get("id") or "subdomain"),
                       "definition": sub.get("definition") or ""}
            dom_seg = {"id": spec["id"], "label": spec["label"], "definition": spec.get("definition", "")}
            for cid in sub.get("features") or []:
                if cid in by_id and cid not in claimed:
                    claimed[cid] = [dom_seg, sub_seg]
    # Repair: any feature no domain claimed -> UNASSIGNED (coverage guarantee).
    for f in features:
        claimed.setdefault(f["id"], [
            {"id": "UNASSIGNED", "label": "Unassigned Features",
             "definition": "Features no domain claimed; retained for full coverage."},
            {"id": "unassigned", "label": "Unassigned", "definition": ""},
        ])
    return claimed


def _collect_unlabeled(domains: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Internal nodes whose label/definition still needs writing (id -> node)."""
    todo: Dict[str, Dict[str, Any]] = {}

    def walk(node, key_prefix):
        if is_leaf(node):
            return
        key = f"{key_prefix}/{node['id']}"
        # Only nodes with no definition need labelling. A node that already carries
        # a definition was placed deterministically (path hint) or labelled by the
        # semantic step; never override those (keeps clean names like "Brain").
        if not node.get("definition"):
            todo[key] = node
        for c in node["children"]:
            walk(c, key)

    for d in domains:
        walk(d, "")
    return todo


def _label_internal_nodes(llm, dataset_name, context, domains: List[Dict[str, Any]]) -> None:
    """One compact LLM call to label internal nodes that lack a clean label/definition."""
    todo = _collect_unlabeled(domains)
    if not todo or llm is None:
        return
    skeleton = {}
    for key, node in todo.items():
        examples = []
        for c in node["children"][:5]:
            examples.append(c.get("label", c["id"]))
        skeleton[key] = examples
    user = (
        f"Dataset: {dataset_name}\nContext: {context}\n\n"
        "For each node id below (path shown as parent/child), the example child labels are given. "
        "Write a concise label and one-sentence definition for each node. Keep it factual.\n\n"
        f"{_tooned(skeleton)}\n\n"
        'Return: {"nodes": {"<node_key>": {"label": "..", "definition": ".."}}}'
    )
    try:
        obj = llm.chat_json(_LABELING_SYSTEM, user, max_tokens=4000)
        labels = obj.get("nodes", obj) if isinstance(obj, dict) else {}
    except Exception:
        labels = {}
    for key, node in todo.items():
        lab = labels.get(key) if isinstance(labels, dict) else None
        if isinstance(lab, dict):
            if lab.get("label"):
                node["label"] = lab["label"]
            if lab.get("definition") and not node.get("definition"):
                node["definition"] = lab["definition"]


def build_ontology_tree(
    features: List[Dict[str, Any]],
    dataset_name: str,
    context: str,
    llm=None,
    user_guidance: str = "",
    max_workers: int = 6,
) -> Dict[str, Any]:
    """Build an arbitrary-depth ontology from a mixed feature list.

    Each feature dict carries ``id``, ``label``, ``definition``, ``stat_type``,
    ``units`` and optionally an explicit ``path`` (list of ``{id,label,definition}``
    segments, domain first). Features with a ``path`` are placed deterministically;
    features without one are grouped semantically by the LLM. Every feature ends up
    as a leaf exactly once.
    """
    pathed = [f for f in features if f.get("path")]
    freeform = [f for f in features if not f.get("path")]

    assigned: Dict[str, List[Dict[str, Any]]] = {}
    if freeform:
        if llm is not None:
            assigned = _semantic_paths(llm, dataset_name, context, freeform, user_guidance, max_workers)
        else:
            for f in freeform:
                assigned[f["id"]] = [{"id": "FEATURES", "label": "Features"},
                                     {"id": "general", "label": "General"}]

    domains: List[Dict[str, Any]] = []
    for f in pathed:
        _insert_feature(domains, [dict(s) for s in f["path"]], _leaf_node(f))
    for f in freeform:
        _insert_feature(domains, [dict(s) for s in assigned[f["id"]]], _leaf_node(f))

    _label_internal_nodes(llm, dataset_name, context, domains)

    n_feats = sum(count_leaves(d) for d in domains)
    return {
        "domains": domains,
        "column_index": _build_column_index(domains),
        "dataset": dataset_name,
        "context": context,
        "n_features": n_feats,
        "user_guidance": user_guidance,
        "construction": "hierarchical_path_hints_plus_semantic",
        "repair_stats": {
            "n_pathed": len(pathed),
            "n_semantic": len(freeform),
            "max_depth": max((node_depth(d) for d in domains), default=0),
        },
    }


# Public name preserved for the pipeline (step 02).
def build_semantic_ontology(
    features: List[Dict[str, Any]],
    dataset_name: str,
    context: str,
    llm,
    user_guidance: str = "",
    max_workers: int = 6,
) -> Dict[str, Any]:
    return build_ontology_tree(features, dataset_name, context, llm=llm,
                               user_guidance=user_guidance, max_workers=max_workers)


# =========================================================================== #
# Assessment / QA
# =========================================================================== #

def assess_ontology(
    ontology: Dict[str, Any],
    exploration: Optional[Dict[str, Any]],
    llm=None,
    verify: bool = True,
) -> Dict[str, Any]:
    """Coverage, structure, agreement with data clusters, and an LLM review."""
    domains = ontology["domains"]
    internal_sizes: List[int] = []

    def walk(node):
        if is_leaf(node):
            return
        internal_sizes.append(count_leaves(node))
        for c in node["children"]:
            walk(c)

    for d in domains:
        walk(d)

    n_feat = sum(count_leaves(d) for d in domains)
    depths = [node_depth(d) for d in domains]
    report: Dict[str, Any] = {
        "n_domains": len(domains),
        "n_internal_nodes": len(internal_sizes),
        "n_features": n_feat,
        "max_depth": max(depths) if depths else 0,
        "leaf_paths_example": [
            "|".join(list(p) + [leaf["id"]]) for p, leaf in list(ontology_leaves(ontology))[:6]
        ],
    }
    if exploration:
        report["cluster_agreement"] = _cluster_agreement(ontology, exploration)
        report["redundant_pairs"] = exploration.get("redundant_pairs", [])[:10]
        report["quality_flags"] = exploration.get("quality_flags", {})
    if verify and llm is not None:
        report["llm_review"] = _verify_ontology(llm, ontology)
    return report


def _leaf_group(ontology: Dict[str, Any]) -> Dict[str, str]:
    """Map each leaf id to a 'domain/immediate_parent' partition label."""
    out: Dict[str, str] = {}
    for path, leaf in ontology_leaves(ontology):
        parent = path[-1] if path else "ROOT"
        dom = path[0] if path else "ROOT"
        out[leaf["id"]] = f"{dom}/{parent}"
    return out


def _cluster_agreement(ontology: Dict[str, Any], exploration: Dict[str, Any]) -> Dict[str, Any]:
    feat_group = _leaf_group(ontology)
    feat_clu: Dict[str, str] = {}
    for cid, cols in (exploration.get("auto_clusters") or {}).items():
        for c in cols:
            feat_clu[c] = cid
    common = [f for f in feat_group if f in feat_clu]
    if len(common) < 3:
        return {"adjusted_rand_index": None, "n_features_compared": len(common)}
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score([feat_group[f] for f in common], [feat_clu[f] for f in common])
        return {"adjusted_rand_index": round(float(ari), 3), "n_features_compared": len(common),
                "interpretation": "1.0 = ontology groups match data-driven clusters exactly"}
    except Exception:
        return {"adjusted_rand_index": None, "n_features_compared": len(common)}


def _verify_ontology(llm, ontology: Dict[str, Any]) -> Dict[str, Any]:
    def compact(node):
        if is_leaf(node):
            return node["label"]
        return {node["label"]: [compact(c) for c in node["children"]]}
    tree = {d["label"]: [compact(c) for c in d["children"]] for d in ontology["domains"]}
    try:
        from src.full_stack.backend.utils.toon import json_to_toon
        payload = json_to_toon(tree)
    except Exception:
        payload = json.dumps(tree)
    user = (
        "Review this feature ontology (TOON; nested groups down to feature leaves).\n\n" + payload +
        '\n\nReturn: {"mece_ok": true/false, "coherence_1to5": <int>, '
        '"issues": ["..."], "misplaced_features": ["feature -> better group"]}'
    )
    try:
        obj = llm.chat_json(_VERIFY_SYSTEM, user, max_tokens=1500)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# =========================================================================== #
# Indexing + serialisation
# =========================================================================== #

def _build_column_index(domains: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for dom in domains:
        for path, leaf in iter_leaves(dom, ()):
            full = list(path)
            index[leaf["id"]] = {
                "domain": full[0] if full else dom["id"],
                "subdomain": full[1] if len(full) > 1 else (full[-1] if full else dom["id"]),
                "path": full,
                "feature_label": leaf["label"],
            }
    return index


def hierarchical_names(ontology: Dict[str, Any], sep: str = "|") -> Dict[str, str]:
    """Map each feature id to a flat column name encoding its full ontology path."""
    out: Dict[str, str] = {}
    for path, leaf in ontology_leaves(ontology):
        out[leaf["id"]] = sep.join(list(path) + [leaf["id"]])
    return out


def write_subclass_json(ontology: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ontology, f, indent=2)


def write_owl(ontology: Dict[str, Any], path: Path, base_iri: Optional[str] = None) -> None:
    """Emit a Protege-loadable OWL file (RDF/XML) with the full subclass chain."""
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

    root = ns.PhenotypeFeature
    g.add((root, RDF.type, OWL.Class))
    g.add((root, RDFS.label, Literal("Phenotype Feature")))
    g.add((root, RDFS.comment, Literal("Root class for all measured participant features.")))

    src_col = ns.sourceColumn
    stat_type = ns.statType
    for prop, label in [(src_col, "source column"), (stat_type, "statistical type")]:
        g.add((prop, RDF.type, OWL.AnnotationProperty))
        g.add((prop, RDFS.label, Literal(label)))

    def emit(node: Dict[str, Any], parent_uri, path_ids: Tuple[str, ...]):
        local = "_".join(("FEATURE" if is_leaf(node) else "NODE",) + path_ids + (_slug(node["id"]),))
        cls = ns[_slug(local)]
        g.add((cls, RDF.type, OWL.Class))
        g.add((cls, RDFS.subClassOf, parent_uri))
        g.add((cls, RDFS.label, Literal(node["label"])))
        if node.get("definition"):
            g.add((cls, RDFS.comment, Literal(node["definition"])))
        if is_leaf(node):
            g.add((cls, src_col, Literal(node["id"])))
            if node.get("stat_type"):
                g.add((cls, stat_type, Literal(node["stat_type"])))
        else:
            for c in node["children"]:
                emit(c, cls, path_ids + (_slug(node["id"]),))

    for dom in ontology["domains"]:
        emit(dom, root, ())

    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="pretty-xml")
