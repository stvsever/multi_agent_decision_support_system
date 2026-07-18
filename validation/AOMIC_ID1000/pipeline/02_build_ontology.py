#!/usr/bin/env python3
"""
Step 02 - LLM-based ontology construction.

Reads the feature manifest and asks a small OpenRouter model to organise every
predictor into a strict, non-redundant DOMAIN -> SUBDOMAIN -> FEATURE subclass
tree. The proposal is validated and repaired for full coverage, then written as:

  ontology/subclass_structure.json   machine-readable tree + column index
  ontology/aomic_id1000.owl          Protege-loadable OWL (RDF/XML)
"""

import json

import _bootstrap  # noqa: F401

import config
from validation.common import ontology as onto
from validation.common.llm import OntologyLLM


def main() -> None:
    manifest_path = config.ONTOLOGY_DIR / "feature_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[02] Building ontology over {manifest['n_predictors']} features "
          f"with model {config.ONTOLOGY_MODEL} ...")
    llm = OntologyLLM(model=config.ONTOLOGY_MODEL, temperature=0.2)
    ontology = onto.build_ontology(
        manifest=manifest,
        dataset_name=config.DATASET_NAME,
        context=config.ONTOLOGY_CONTEXT,
        llm=llm,
    )

    subclass_path = config.ONTOLOGY_DIR / "subclass_structure.json"
    owl_path = config.ONTOLOGY_DIR / "aomic_id1000.owl"
    onto.write_subclass_json(ontology, subclass_path)
    onto.write_owl(ontology, owl_path)

    n_domains = len(ontology["domains"])
    n_subs = sum(len(d["subdomains"]) for d in ontology["domains"])
    n_feats = sum(len(s["features"]) for d in ontology["domains"] for s in d["subdomains"])
    print(f"[02] Ontology: {n_domains} domains, {n_subs} subdomains, {n_feats} feature leaves")
    print(f"[02] Repair stats: {ontology['repair_stats']}")
    assert n_feats == manifest["n_predictors"], (
        f"coverage mismatch: {n_feats} leaves vs {manifest['n_predictors']} predictors"
    )
    print("[02] Coverage check passed: every predictor is a leaf exactly once.")
    for d in ontology["domains"]:
        subs = ", ".join(f"{s['id']}({len(s['features'])})" for s in d["subdomains"])
        print(f"      {d['id']}: {subs}")
    print(f"[02] Wrote {subclass_path.name} and {owl_path.name}")


if __name__ == "__main__":
    main()
