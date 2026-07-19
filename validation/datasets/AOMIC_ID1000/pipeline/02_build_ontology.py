#!/usr/bin/env python3
"""
Step 02 - automated data exploration + master ontology construction.

First runs a sophisticated, dataset-agnostic exploration (type inference,
distributions, rank-correlation structure, hierarchical feature clustering,
redundancy and quality flags, target associations). Then builds one
non-redundant DOMAIN -> SUBDOMAIN -> FEATURE ontology over all features, and
assesses its quality: statistical agreement between the semantic subdomains and
the data-driven clusters (adjusted Rand index) plus a compact LLM review.

Writes:
  ontology/exploration_report.json   automated understanding of the data
  ontology/subclass_structure.json   machine-readable ontology + column index
  ontology/aomic_id1000.owl          Protege-loadable OWL (RDF/XML)
  ontology/ontology_report.json      coverage, cluster agreement, LLM review
"""

import json

import _bootstrap  # noqa: F401

import config
from validation.common import explore as expl, ontology as onto
from validation.common.llm import OntologyLLM


def main() -> None:
    df = config.load_merged_frame()
    specs = config.all_feature_specs()

    print(f"[02] Automated exploration of {len(specs)} features over {len(df)} participants ...")
    exploration = expl.explore(df, specs, target=config.TARGET["column"])
    with open(config.ONTOLOGY_DIR / "exploration_report.json", "w") as f:
        json.dump(exploration, f, indent=2)
    print(f"[02] Types: {exploration['type_counts']} | data-driven clusters: "
          f"{exploration['n_auto_clusters']} | redundant pairs: {len(exploration['redundant_pairs'])}")
    flags = {k: len(v) for k, v in exploration["quality_flags"].items() if v}
    if flags:
        print(f"[02] Quality flags: {flags}")

    print(f"[02] Building master ontology with model {config.ONTOLOGY_MODEL} ...")
    llm = OntologyLLM(model=config.ONTOLOGY_MODEL, temperature=0.2)
    features = config.features_with_hints()
    ontology = onto.build_labeled_ontology(features, config.DATASET_NAME, config.ONTOLOGY_CONTEXT, llm)

    onto.write_subclass_json(ontology, config.ONTOLOGY_DIR / "subclass_structure.json")
    onto.write_owl(ontology, config.ONTOLOGY_DIR / "aomic_id1000.owl")

    report = onto.assess_ontology(ontology, exploration, llm=llm, verify=True)
    with open(config.ONTOLOGY_DIR / "ontology_report.json", "w") as f:
        json.dump(report, f, indent=2)

    n_feats = report["n_features"]
    assert n_feats == len(features), f"coverage mismatch: {n_feats} vs {len(features)}"
    ari = (report.get("cluster_agreement") or {}).get("adjusted_rand_index")
    rev = report.get("llm_review") or {}
    print(f"[02] Ontology: {report['n_domains']} domains, {report['n_subdomains']} subdomains, "
          f"{n_feats} leaves (coverage passed).")
    for d in ontology["domains"]:
        print(f"      {d['id']} = {d['label']!r} ({len(d['subdomains'])} subdomains)")
    print(f"[02] Ontology-vs-data cluster agreement (ARI): {ari}")
    print(f"[02] LLM review: MECE={rev.get('mece_ok')} coherence={rev.get('coherence_1to5')}/5 "
          f"issues={len(rev.get('issues', []))}")
    print(f"[02] Wrote exploration_report.json, subclass_structure.json, aomic_id1000.owl, ontology_report.json")


if __name__ == "__main__":
    main()
