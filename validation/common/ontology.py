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
