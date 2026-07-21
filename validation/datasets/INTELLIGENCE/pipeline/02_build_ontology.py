#!/usr/bin/env python3
"""
Step 02 - automated exploration + semantic, LLM-driven ontology construction.

First runs a dataset-agnostic exploration (types, distributions, correlation
structure, feature clustering, redundancy, target associations) purely to
document the data and to quality-check the ontology later (it never drives the
structure). Then builds one non-redundant DOMAIN -> SUBDOMAIN -> FEATURE ontology
by MEANING: the LLM is given every feature's label, description, units, source and
sample values, proposes the domains itself, and organises each domain into
subdomains. Optional free-text user guidance is injected into every prompt (the
backend hook for a future UI). Finally, a single flattened benchmark CSV encodes
the whole hierarchy in the column names for downstream ML comparison.

Writes:
  ontology/exploration_report.json    automated understanding of the data
  ontology/subclass_structure.json    machine-readable ontology + column index
  ontology/aomic_id1000.owl           Protege-loadable OWL (RDF/XML)
  ontology/ontology_report.json       coverage, cluster agreement, LLM review
  ontology/ontology_features.csv       all participants x all features, hierarchy-encoded
"""

import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

import config
from validation.common import explore as expl, ontology as onto
from validation.common.llm import OntologyLLM


def _write_benchmark_csv(df, ontology, path):
    """One CSV: rows=participants, cols=DOMAIN|subdomain|feature (+ target)."""
    names = onto.hierarchical_names(ontology)
    cols = [c for c in names if c in df.columns]
    out = df[["participant_id"] + cols].rename(columns=names)
    tcol = config.TARGET["column"]
    out["TARGET|" + tcol] = pd.to_numeric(df[tcol], errors="coerce")
    out.to_csv(path, index=False)
    return len(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guidance", default=config.ONTOLOGY_USER_GUIDANCE,
                    help="free-text guidance injected into ontology-building prompts")
    args = ap.parse_args()

    df = config.load_merged_frame()
    evaluation_ids = set(config.select_subset_ids(df))
    reference_df = df[~df["participant_id"].isin(evaluation_ids)].copy()
    specs = config.all_feature_specs()

    print(f"[02] Automated exploration of {len(specs)} features over "
          f"{len(reference_df)} reference participants ...")
    exploration = expl.explore(reference_df, specs, target=config.TARGET["column"])
    with open(config.ONTOLOGY_DIR / "exploration_report.json", "w") as f:
        json.dump(exploration, f, indent=2)
    print(f"[02] Types: {exploration['type_counts']} | data-driven clusters (QA only): "
          f"{exploration['n_auto_clusters']} | redundant pairs: {len(exploration['redundant_pairs'])}")

    print(f"[02] Semantic LLM ontology construction with {config.ONTOLOGY_MODEL} ...")
    if args.guidance.strip():
        print(f"[02] User guidance: {args.guidance.strip()!r}")
    llm = OntologyLLM(model=config.ONTOLOGY_MODEL, temperature=0.2)
    features = config.features_for_ontology(reference_df)
    ontology = onto.build_semantic_ontology(
        features, config.DATASET_NAME, config.ONTOLOGY_CONTEXT, llm, user_guidance=args.guidance,
    )

    onto.write_subclass_json(ontology, config.ONTOLOGY_DIR / "subclass_structure.json")
    onto.write_owl(ontology, config.ONTOLOGY_DIR / "aomic_id1000.owl")
    n_cols = _write_benchmark_csv(df, ontology, config.ONTOLOGY_DIR / "ontology_features.csv")

    from validation.common import viewer
    viewer.write_viewer(ontology, config.ONTOLOGY_DIR / "ontology_viewer.html", title=config.DATASET_LABEL)

    report = onto.assess_ontology(ontology, exploration, llm=llm, verify=True)
    with open(config.ONTOLOGY_DIR / "ontology_report.json", "w") as f:
        json.dump(report, f, indent=2)

    n_feats = report["n_features"]
    assert n_feats == len(features), f"coverage mismatch: {n_feats} vs {len(features)}"
    ari = (report.get("cluster_agreement") or {}).get("adjusted_rand_index")
    rev = report.get("llm_review") or {}
    print(f"[02] Ontology (construction={ontology['construction']}): {report['n_domains']} domains, "
          f"{report['n_internal_nodes']} internal nodes, max depth {report['max_depth']}, "
          f"{n_feats} leaves; repair={ontology['repair_stats']}")
    for d in ontology["domains"]:
        from validation.common.ontology import count_leaves
        print(f"      {d['id']} = {d['label']!r} ({len(d['children'])} children, {count_leaves(d)} leaves)")
    print(f"[02] Ontology-vs-data cluster agreement (ARI, QA): {ari} | "
          f"LLM review MECE={rev.get('mece_ok')} coherence={rev.get('coherence_1to5')}/5")
    print(f"[02] Benchmark CSV: {n_cols} feature columns + target -> ontology_features.csv")


if __name__ == "__main__":
    main()
