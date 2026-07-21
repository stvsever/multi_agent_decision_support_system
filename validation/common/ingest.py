"""
Generic, modality-agnostic single-subject ingestion engine (with CLI).

This is the reusable "automated data ingestion" that turns pre-processed data for
N subjects into the engine's native per-subject input format. It is deliberately
modality-agnostic: any modality (FreeSurfer morphometry, functional connectome,
lesion-mask statistics, EEG band powers, questionnaires, ...) is first reduced by a
small adapter to a per-subject feature table plus feature specs, and this engine
then does the rest, identically for every modality:

  1. auto-detect the N subjects (one per row of the merged feature matrix),
  2. build (or load) the arbitrary-depth ontology - deterministic ``path`` hints
     where a modality provides them, LLM semantic grouping otherwise,
  3. choose a reference strategy (cohort z-scores / external norms / raw absolute),
  4. auto-detect any per-subject free-text note and fold it into the text modality,
  5. write one clean ``loaded/subject_001/ ... /subject_00N/`` folder per subject,
     each containing the four COMPASS files, plus an ingest manifest.

The CLI requires explicit arguments so ingestion is bounded and reproducible (it
never guesses inputs). The important arguments select the *mode* of ingestion:
``--reference-mode`` (how deviation is computed), ``--ontology`` vs ``--build-ontology``
(structure source), ``--text-dir`` (free-text modality), and ``--limit`` (cost).

Example
-------
    python -m validation.common.ingest \
        --features morphometry.csv --features connectome.csv \
        --specs brain_specs.json --id-column participant_id \
        --text-dir notes/ --reference-mode auto \
        --out ./ingested --dataset MYSTUDY --context "..." --build-ontology
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import compass_writer, deviation
from . import ontology as onto
from . import tiers
from .explore import infer_stat_type


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def load_feature_frame(feature_files: List[Path], id_column: str) -> pd.DataFrame:
    """Load and merge one or more subject-by-feature tables on ``id_column``."""
    frame: Optional[pd.DataFrame] = None
    for path in feature_files:
        sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
        part = pd.read_csv(path, sep=sep, na_values=["n/a", "N/A", ""])
        if id_column not in part.columns:
            raise SystemExit(f"[ingest] '{id_column}' not found in {path}")
        frame = part if frame is None else frame.merge(part, on=id_column, how="outer")
    if frame is None:
        raise SystemExit("[ingest] no feature files provided")
    return frame


def auto_specs(df: pd.DataFrame, id_column: str) -> Dict[str, Dict[str, Any]]:
    """Infer minimal specs (label + stat_type) for every non-id column."""
    specs: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        if col == id_column:
            continue
        specs[col] = {"label": col.replace("_", " ").strip().title(),
                      "stat_type": infer_stat_type(df[col])}
    return specs


def features_for_ontology(specs: Dict[str, Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Ontology-ready feature descriptors; carries ``path`` hints when present."""
    out = []
    for col, spec in specs.items():
        sample = None
        if col in df.columns:
            sample = [str(v) for v in df[col].dropna().unique()[:4]]
        feat = {"id": col, "label": spec.get("label", col),
                "definition": spec.get("description", ""),
                "stat_type": spec.get("stat_type", "numeric"),
                "units": spec.get("units"), "source": spec.get("source", ""),
                "sample": sample}
        if spec.get("path"):
            feat["path"] = spec["path"]
        out.append(feat)
    return out


def detect_free_text(text_dir: Optional[Path], source_id: str) -> str:
    """Auto-detect a per-subject free-text note (``<source_id>.txt`` or a folder)."""
    if not text_dir:
        return ""
    candidates = [text_dir / f"{source_id}.txt", text_dir / source_id / "notes.txt"]
    folder = text_dir / source_id
    if folder.is_dir():
        candidates += sorted(folder.glob("*.txt"))
    for cand in candidates:
        if cand.is_file():
            return cand.read_text().strip()
    return ""


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #

def ingest_subjects(
    df: pd.DataFrame,
    specs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    id_column: str = "participant_id",
    dataset: str = "DATASET",
    context: str = "",
    guidance: str = "",
    reference_mode: str = "auto",
    external_norms: Optional[Dict[str, Dict[str, float]]] = None,
    ontology: Optional[Dict[str, Any]] = None,
    build_ontology: bool = False,
    llm=None,
    text_dir: Optional[Path] = None,
    target_note: str = "",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Ingest every subject (row) into ``out_dir/loaded/subject_NNN/``.

    Returns the ingest manifest (also written to ``loaded/ingest_manifest.json``).
    """
    feature_cols = [c for c in specs if c in df.columns]
    df = df.dropna(axis=0, how="all", subset=feature_cols).reset_index(drop=True)

    # 1. Ontology: prebuilt, freshly built (LLM/deterministic), or path-hint only.
    if ontology is None:
        feats = features_for_ontology(specs, df)
        ontology = onto.build_ontology_tree(
            feats, dataset, context, llm=(llm if build_ontology else None),
            user_guidance=guidance,
        )
    # Project onto the features we actually have, so a prebuilt (superset) ontology
    # ingests cleanly and empty branches are pruned at any depth.
    ontology = tiers.project_ontology(ontology, set(feature_cols))
    if ontology["n_features"] == 0:
        raise SystemExit("[ingest] no ontology leaves match the provided feature columns")

    # 2. Reference strategy (cohort z / external norms / raw absolute), auto-resolved.
    n_ref = int(df[feature_cols].notna().sum().min()) if feature_cols else 0
    mode = deviation.resolve_reference_mode(reference_mode, n_ref, bool(external_norms))
    ref_specs = {c: specs[c] for c in feature_cols}
    ref = deviation.ReferenceModel(ref_specs, mode=mode)
    ref.fit(df, external_norms=external_norms)

    # 3. Auto-detect subjects (rows) -> subject_001 ... subject_00N.
    loaded_root = out_dir / "loaded"
    loaded_root.mkdir(parents=True, exist_ok=True)
    ids = df[id_column].astype(str).tolist()
    if limit:
        ids = ids[:limit]

    manifest_subjects = []
    width = max(3, len(str(len(ids))))
    for index, source_id in enumerate(ids, 1):
        subject_key = f"subject_{index:0{width}d}"
        row = df[df[id_column].astype(str) == source_id].iloc[0]
        encoded = ref.encode_participant(row)
        payloads = compass_writer.build_participant_payloads(
            participant_id=subject_key, ontology=ontology, encoded=encoded,
            target_note=target_note, reference_mode=mode,
        )
        note = detect_free_text(text_dir, source_id)
        if note:
            payloads["non_numerical_data"] += (
                "\n\nPROVIDED SUBJECT NOTES (free text):\n" + note
            )
        compass_writer.write_participant(loaded_root / subject_key, payloads)
        cov = payloads["data_overview"]
        manifest_subjects.append({
            "subject": subject_key,
            "source_id": source_id,
            "present_leaves": cov["present_leaves"],
            "total_leaves": cov["total_leaves"],
            "total_tokens": cov["total_tokens"],
            "has_free_text": bool(note),
        })

    manifest = {
        "dataset": dataset,
        "n_subjects": len(manifest_subjects),
        "reference_mode": mode,
        "id_column": id_column,
        "n_features": ontology["n_features"],
        "n_domains": len(ontology["domains"]),
        "ontology_construction": ontology.get("construction"),
        "subjects": manifest_subjects,
    }
    with open(loaded_root / "ingest_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    # Persist the ontology used, next to the loaded subjects, for reproducibility.
    onto.write_subclass_json(ontology, loaded_root / "ontology.json")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generic modality-agnostic ingestion: pre-processed features -> "
                    "loaded/subject_NNN/ COMPASS inputs.")
    ap.add_argument("--features", action="append", required=True, type=Path,
                    help="subject-by-feature CSV/TSV (repeatable; merged on --id-column)")
    ap.add_argument("--out", required=True, type=Path, help="output dir (writes <out>/loaded/)")
    ap.add_argument("--id-column", default="participant_id")
    ap.add_argument("--specs", type=Path, help="feature specs JSON (label/stat_type/units/path); "
                                               "auto-profiled if omitted")
    ap.add_argument("--ontology", type=Path, help="prebuilt subclass_structure.json (skips building)")
    ap.add_argument("--build-ontology", action="store_true",
                    help="build the ontology with the LLM (semantic grouping of un-pathed features)")
    ap.add_argument("--text-dir", type=Path, help="dir of per-subject free-text notes (<source_id>.txt)")
    ap.add_argument("--reference-mode", default="auto",
                    choices=["auto", "cohort", "external", "absolute"],
                    help="deviation strategy: cohort z-scores / external norms / raw absolute")
    ap.add_argument("--external-norms", type=Path, help="{col: {mean, std}} JSON for --reference-mode external")
    ap.add_argument("--dataset", default="DATASET")
    ap.add_argument("--context", default="")
    ap.add_argument("--guidance", default="", help="free-text guidance injected into ontology prompts")
    ap.add_argument("--target-note", default="", help="target description folded into each narrative")
    ap.add_argument("--model", default="google/gemini-3.1-flash-lite", help="ontology LLM (OpenRouter)")
    ap.add_argument("--limit", type=int, help="ingest only the first N subjects (cost control)")
    args = ap.parse_args()

    df = load_feature_frame(args.features, args.id_column)
    specs = json.loads(args.specs.read_text()) if args.specs else auto_specs(df, args.id_column)
    ontology = json.loads(args.ontology.read_text()) if args.ontology else None
    external_norms = json.loads(args.external_norms.read_text()) if args.external_norms else None

    llm = None
    if args.build_ontology and ontology is None:
        from .llm import OntologyLLM
        llm = OntologyLLM(model=args.model)

    manifest = ingest_subjects(
        df=df, specs=specs, out_dir=args.out, id_column=args.id_column,
        dataset=args.dataset, context=args.context, guidance=args.guidance,
        reference_mode=args.reference_mode, external_norms=external_norms,
        ontology=ontology, build_ontology=args.build_ontology, llm=llm,
        text_dir=args.text_dir, target_note=args.target_note, limit=args.limit,
    )
    print(f"[ingest] {manifest['n_subjects']} subjects -> {args.out / 'loaded'} "
          f"| {manifest['n_features']} features, {manifest['n_domains']} domains "
          f"| reference={manifest['reference_mode']}")
    with_text = sum(s["has_free_text"] for s in manifest["subjects"])
    print(f"[ingest] free-text notes detected for {with_text}/{manifest['n_subjects']} subjects; "
          f"ontology + manifest written under loaded/")


if __name__ == "__main__":
    main()
