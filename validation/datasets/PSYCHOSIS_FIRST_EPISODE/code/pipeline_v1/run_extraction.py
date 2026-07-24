"""Batch driver: extracts eeg_fep_rest_v1 features for every subject in ds003944 + ds003947.

Parallelized across subjects (confirmed with user - pure engineering change,
no effect on the science). Each worker process limits its own internal
thread count via the pool initializer so 8 concurrent processes don't
oversubscribe the machine's cores fighting each other for BLAS/OMP threads.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_WORKERS = 8
SCHEMA_VERSION = "eeg_fep_rest_v1"

# This file is code/pipeline_v1/run_extraction.py, so the dataset root
# (PSYCHOSIS_FIRST_EPISODE/) is three parents up, not two.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS = {"ds003944": BASE_DIR / "data" / "ds003944", "ds003947": BASE_DIR / "data" / "ds003947"}
OUTPUT_DIR = BASE_DIR / "data" / "processed" / SCHEMA_VERSION


def _worker_init():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _iter_participants():
    for dataset, dataset_path in DATASETS.items():
        participants = pd.read_csv(dataset_path / "participants.tsv", sep="\t")
        for participant_id in participants["participant_id"]:
            yield dataset, dataset_path, participant_id


def _write_run_manifest(feature_manifest_hash: str) -> None:
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=BASE_DIR).decode().strip())
    except Exception:
        git_commit, git_dirty = None, None

    versions = {}
    for pkg in ["mne", "numpy", "scipy", "pandas", "specparam", "antropy", "mne_connectivity", "networkx", "pyprep", "autoreject"]:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "datasets": ["ds003944@1.0.1", "ds003947@1.0.1"],
        "n_workers": N_WORKERS,
        "package_versions": versions,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "feature_manifest_hash": feature_manifest_hash,
        "performance_mitigations": {
            "sample_entropy_epoch_cap": 40,
            "sample_entropy_epoch_cap_seed": 97,
            "parallel_workers": N_WORKERS,
        },
        "group_f_status": "dropped_from_v1",
        "n_features": 804,
        "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from pipeline_v1.schema import write_feature_manifest

    feature_manifest_hash = write_feature_manifest(OUTPUT_DIR / "feature_manifest.json")
    _write_run_manifest(feature_manifest_hash)

    features_csv = OUTPUT_DIR / "features_eeg_804.csv"
    qc_tsv = OUTPUT_DIR / "qc_subjects.tsv"
    exclusions_tsv = OUTPUT_DIR / "exclusions.tsv"
    log_path = OUTPUT_DIR / "extraction_log.txt"

    subjects = list(_iter_participants())
    n_total = len(subjects)
    n_ok = n_excluded = n_failed = 0
    start = time.time()

    features_header_written = features_csv.exists()
    qc_header_written = qc_tsv.exists()
    exclusions_header_written = exclusions_tsv.exists()

    from pipeline_v1.extract_subject import extract_subject

    with open(log_path, "a") as log, ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as pool:
        futures = {
            pool.submit(extract_subject, dataset, dataset_path, participant_id): (dataset, participant_id)
            for dataset, dataset_path, participant_id in subjects
        }

        for i, future in enumerate(as_completed(futures), 1):
            dataset, participant_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                msg = f"[CRASH] {dataset}/{participant_id}: {exc}"
                print(msg)
                log.write(msg + "\n")
                n_failed += 1
                continue

            status = result["status"]
            if status in ("ok", "excluded"):
                # Every non-failed participant becomes a features row (excluded ones
                # are all-blank, per user's "all 143 as rows" choice). Rows are
                # written in completion order; sort by participant_id downstream if
                # a canonical order is needed.
                if status == "ok":
                    n_ok += 1
                else:
                    n_excluded += 1
                row_df = pd.DataFrame([result["row"]])
                row_df.to_csv(features_csv, mode="a", index=False, header=not features_header_written)
                features_header_written = True
            if status == "excluded":
                exc_df = pd.DataFrame([{"dataset": dataset, "participant_id": participant_id, "reason": result["error"]}])
                exc_df.to_csv(exclusions_tsv, mode="a", index=False, sep="\t", header=not exclusions_header_written)
                exclusions_header_written = True
            elif status not in ("ok", "excluded"):
                n_failed += 1
                msg = f"[FAIL] {dataset}/{participant_id}: {result['error']}"
                log.write(msg + "\n")

            if result["qc_row"] is not None:
                qc_row = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in result["qc_row"].items()}
                qc_df = pd.DataFrame([qc_row])
                qc_df.to_csv(qc_tsv, mode="a", index=False, sep="\t", header=not qc_header_written)
                qc_header_written = True

            elapsed = time.time() - start
            print(
                f"[{i}/{n_total}] {dataset}/{participant_id}: {status} "
                f"(ok={n_ok} excluded={n_excluded} failed={n_failed}, {elapsed / 60:.1f} min elapsed)"
            )

    elapsed_min = (time.time() - start) / 60
    print(f"\nDone in {elapsed_min:.1f} min. ok={n_ok} excluded={n_excluded} failed={n_failed} / {n_total}")

    if features_csv.exists():
        df = pd.read_csv(features_csv)
        df.to_parquet(OUTPUT_DIR / "features_eeg_804.parquet", index=False)
        print(f"Wrote {features_csv} and .parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()
