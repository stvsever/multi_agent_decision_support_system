#!/usr/bin/env python3
"""
Step 02 - build the master multi-modal ontology.

Constructs one non-redundant DOMAIN -> SUBDOMAIN -> FEATURE subclass ontology over
ALL features (tabular self-report + brain morphometry + brain connectome). The
structure comes from per-feature domain/subdomain hints (guaranteeing clean,
non-redundant, fully-covered domains with brain modalities kept separate) and a
small model generates the interpretable parent labels and definitions. Every tier
later reuses a filtered projection of this single ontology. Writes:

  ontology/subclass_structure.json   machine-readable tree + column index
  ontology/aomic_id1000.owl          Protege-loadable OWL (RDF/XML)
"""

import json

import _bootstrap  # noqa: F401

import config
from validation.common import ontology as onto
from validation.common.llm import OntologyLLM


def main() -> None:
    features = config.features_with_hints()
    print(f"[02] Building master ontology over {len(features)} features "
          f"with model {config.ONTOLOGY_MODEL} ...")
    llm = OntologyLLM(model=config.ONTOLOGY_MODEL, temperature=0.2)
    ontology = onto.build_labeled_ontology(
        features=features,
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
    assert n_feats == len(features), f"coverage mismatch: {n_feats} vs {len(features)}"
    print(f"[02] Ontology: {n_domains} domains, {n_subs} subdomains, {n_feats} leaves "
          f"(coverage check passed).")
    for d in ontology["domains"]:
        print(f"      {d['id']} = {d['label']!r}")
        for s in d["subdomains"]:
            print(f"          {s['id']} ({len(s['features'])}) = {s['label']!r}")
    print(f"[02] Wrote {subclass_path.name} and {owl_path.name}")


if __name__ == "__main__":
    main()
