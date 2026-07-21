#!/usr/bin/env python3
"""
Step 11 - extract functional connectome features (movie-watching fMRI).

For each subject in the reference cohort: download fMRIPrep BOLD + confounds,
parcellate with Schaefer-100/Yeo-7, denoise, and reduce to 28 network-level FC
features. Raw BOLD is deleted after each subject to bound disk use. Writes:

  brain/connectome/connectome_features.csv    participant_id + 28 FC features
  brain/connectome/connectome_specs.json      feature label/description/type/units
  brain/connectome/network_fc/<sub>.npy       7x7 network FC matrix (for notebooks)

The final extraction contains 100 modality-complete evaluation participants plus
at least 20 disjoint participants for reference-only normalization.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config
from validation.common import connectome as conn


def _extract_one(pid: str, keep_bold: bool):
    """Process one participant in an isolated process."""
    paths = {}
    try:
        atlas = conn.load_atlas(n_rois=100, yeo_networks=7, resolution_mm=2)
        networks = conn.atlas_networks(atlas)
        cache = config.BRAIN_CACHE_DIR / "connectome"
        paths = conn.download_func(pid, config.ACCESSION, cache)
        if not paths.get("bold") or not paths.get("confounds"):
            return pid, None, "missing BOLD or confounds"
        feats, net_mat, _parcel_fc = conn.extract_subject_fc(paths, atlas, networks)
        fc_dir = config.CONNECTOME_DIR / "network_fc"
        fc_dir.mkdir(parents=True, exist_ok=True)
        np.save(fc_dir / f"{pid}.npy", net_mat)
        return pid, feats, None
    except Exception as exc:  # noqa: BLE001
        return pid, None, f"{type(exc).__name__}: {exc}"
    finally:
        if not keep_bold:
            for key in ("bold", "mask"):
                path = paths.get(key)
                try:
                    if path and path.exists():
                        path.unlink()
                except Exception:
                    pass


def _run_parallel(participant_ids, workers, keep_bold, label):
    updates = {}
    failures = {}
    if not participant_ids:
        return updates, failures
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_extract_one, pid, keep_bold): pid for pid in participant_ids}
        for index, future in enumerate(as_completed(futures), 1):
            pid, feats, error = future.result()
            if feats is None:
                failures[pid] = error
                print(f"[11] {label} ({index}/{len(participant_ids)}) {pid}: FAILED {error}")
            else:
                updates[pid] = feats
                print(f"[11] {label} ({index}/{len(participant_ids)}) {pid}: {len(feats)} FC features")
    return updates, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-size", type=int, default=config.BRAIN_CONN_REF_SIZE)
    ap.add_argument("--workers", type=int, default=3,
                    help="bounded process parallelism; fMRI extraction is memory-heavy")
    ap.add_argument("--keep-bold", action="store_true", help="do not delete raw BOLD after use")
    args = ap.parse_args()

    df = pd.read_csv(config.PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])
    subset = config.select_subset_ids(df)
    ref_ids = config.brain_reference_ids(df, args.ref_size, include=subset)
    print(f"[11] Connectome target: {len(ref_ids)} subjects "
          f"(incl. {len(subset)} evaluation subjects), workers={args.workers}")

    # Warm the atlas cache before child processes start, avoiding download races.
    atlas = conn.load_atlas(n_rois=100, yeo_networks=7, resolution_mm=2)
    del atlas
    fc_dir = config.CONNECTOME_DIR / "network_fc"
    fc_dir.mkdir(parents=True, exist_ok=True)

    specs = conn.feature_specs()
    cols = list(specs)
    table_path = config.CONNECTOME_DIR / "connectome_features.csv"
    if table_path.exists():
        existing = pd.read_csv(table_path).set_index("participant_id")
        existing = existing.reindex(columns=cols)
    else:
        existing = pd.DataFrame(columns=cols)

    complete_existing = {
        pid for pid, row in existing.iterrows()
        if row.notna().all() and (fc_dir / f"{pid}.npy").exists()
    }
    to_run = [pid for pid in ref_ids if pid not in complete_existing]
    print(f"[11] Reusing {len(ref_ids) - len(to_run)} complete target rows; "
          f"extracting {len(to_run)}")

    updates, failures = _run_parallel(to_run, args.workers, args.keep_bold, "initial")

    for pid, feats in updates.items():
        existing.loc[pid, cols] = [feats.get(c, np.nan) for c in cols]
    table = existing.dropna(how="all").sort_index()[cols]

    # Backfill missing evaluation scans using the same seeded, target-blind
    # candidate order. Modality availability is the only replacement criterion.
    candidate_order = config.evaluation_candidate_ids(df)
    initial_evaluation = set(subset)
    complete = set(table.dropna(subset=cols).index.astype(str))
    final_evaluation = [pid for pid in subset if pid in complete]
    attempted = set(ref_ids)

    def use_complete_backups():
        for pid in candidate_order[config.SUBSET_SIZE:]:
            if len(final_evaluation) >= config.SUBSET_SIZE:
                break
            if pid in complete and pid not in final_evaluation:
                final_evaluation.append(pid)

    use_complete_backups()
    backup_cursor = 0
    backups = [pid for pid in candidate_order[config.SUBSET_SIZE:] if pid not in attempted]
    while len(final_evaluation) < config.SUBSET_SIZE and backup_cursor < len(backups):
        need = config.SUBSET_SIZE - len(final_evaluation)
        batch_n = max(args.workers, min(len(backups) - backup_cursor, need * 2))
        batch = backups[backup_cursor:backup_cursor + batch_n]
        backup_cursor += batch_n
        attempted.update(batch)
        more_updates, more_failures = _run_parallel(
            batch, args.workers, args.keep_bold, "backfill"
        )
        failures.update(more_failures)
        for pid, feats in more_updates.items():
            table.loc[pid, cols] = [feats.get(c, np.nan) for c in cols]
        complete = set(table.dropna(subset=cols).index.astype(str))
        use_complete_backups()

    final_evaluation = sorted(final_evaluation[:config.SUBSET_SIZE])
    complete = set(table.dropna(subset=cols).index.astype(str))
    disjoint_reference_n = len(complete - set(final_evaluation))
    while disjoint_reference_n < 20 and backup_cursor < len(backups):
        need = 20 - disjoint_reference_n
        batch_n = max(args.workers, min(len(backups) - backup_cursor, need * 2))
        batch = backups[backup_cursor:backup_cursor + batch_n]
        backup_cursor += batch_n
        more_updates, more_failures = _run_parallel(
            batch, args.workers, args.keep_bold, "reference-backfill"
        )
        failures.update(more_failures)
        for pid, feats in more_updates.items():
            table.loc[pid, cols] = [feats.get(c, np.nan) for c in cols]
        complete = set(table.dropna(subset=cols).index.astype(str))
        disjoint_reference_n = len(complete - set(final_evaluation))

    config.EVALUATION_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EVALUATION_IDS_PATH, "w") as handle:
        json.dump({
            "participant_ids": final_evaluation,
            "n": len(final_evaluation),
            "seed": config.RANDOM_SEED,
            "selection": (
                "seeded target-blind tabular-eligible draw; missing fMRI backfilled "
                "in seeded target-blind order based only on modality availability"
            ),
            "initial_missing_imaging": sorted(initial_evaluation - complete),
        }, handle, indent=2)

    config.CONNECTOME_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path)
    kept_specs = {c: specs[c] for c in cols}
    with open(config.CONNECTOME_DIR / "connectome_specs.json", "w") as f:
        json.dump(kept_specs, f, indent=2)

    complete = table.dropna(subset=cols).index.astype(str).tolist()
    missing_evaluation = sorted(set(final_evaluation) - set(complete))
    disjoint_reference_n = len(set(complete) - set(final_evaluation))
    print(f"[11] Extracted {len(cols)} connectome features for {len(complete)} complete subjects.")
    print(f"[11] Complete evaluation={len(final_evaluation) - len(missing_evaluation)}/{config.SUBSET_SIZE}; "
          f"disjoint references={disjoint_reference_n}")
    print(f"[11] Wrote brain/connectome/connectome_features.csv and connectome_specs.json")
    if len(final_evaluation) != config.SUBSET_SIZE or missing_evaluation or disjoint_reference_n < 20:
        raise SystemExit(
            "[11] Incomplete leakage-safe cohort: "
            f"evaluation_n={len(final_evaluation)}, missing evaluation={missing_evaluation}, "
            f"disjoint references={disjoint_reference_n}. "
            "Rerun extraction to retry failures."
        )


if __name__ == "__main__":
    main()
