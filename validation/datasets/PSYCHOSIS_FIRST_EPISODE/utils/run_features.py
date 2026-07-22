#!/usr/bin/env python3
"""CLI: build group microstate templates, extract 836 features per subject,
assemble results/eeg_features.csv and its QC companions.

    python -m utils.run_features                 # all subjects
    python -m utils.run_features --limit 5       # first 5 (smoke test)
    python -m utils.run_features --overwrite     # recompute checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config as C  # noqa: E402
from utils import extract as X  # noqa: E402
from utils import features as F  # noqa: E402


def _logger() -> logging.Logger:
    C.PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(C.PROCESSED_ROOT / "run_features.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    mne.set_log_level("ERROR")
    return logger


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rebuild-templates", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = C.load_config()
    logger = _logger()
    groups, names = F.build_schema()
    records = X.load_records(cfg)

    logger.info("building / loading group microstate templates")
    modk = X.build_group_microstate_templates(records, cfg, logger=logger,
                                              overwrite=args.rebuild_templates)

    subset = records[: args.limit] if args.limit else records
    logger.info("extracting features for %d recordings", len(subset))
    rows, qc_rows = [], []
    t0 = time.time()
    n = len(subset)
    for i, record in enumerate(subset, 1):
        row, qc = X.extract_subject(record, modk, groups, names, cfg,
                                    overwrite=args.overwrite, logger=logger)
        rows.append(row)
        qc_rows.append(qc)
        if i % 10 == 0 or i == n:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            complete = sum(q.get("status") == "complete" for q in qc_rows)
            logger.info("[%d/%d] complete=%d elapsed=%.0fs eta=%.0fs",
                        i, n, complete, elapsed, eta)

    C.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=["recording_id", *names])
    table.to_csv(C.RESULTS_ROOT / "eeg_features.csv", index=False, na_rep="")
    try:
        table.to_parquet(C.RESULTS_ROOT / "eeg_features.parquet", index=False)
    except Exception as exc:
        logger.warning("parquet not written: %s", exc)

    # Feature manifest and group QC tables.
    manifest = {
        "schema_version": cfg["schema_version"],
        "feature_count": len(names),
        "group_counts": {k: len(v) for k, v in groups.items()},
        "ordered_feature_names": names,
        "ordered_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
    }
    (C.PROCESSED_ROOT / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    qc_frame = pd.DataFrame([
        {"recording_id": q["recording_id"], "dataset_id": q["dataset_id"],
         "participant_id": q["participant_id"], "status": q["status"],
         "feature_count_finite": q.get("feature_count_finite", 0),
         "feature_count_missing": q.get("feature_count_missing", len(names)),
         **{f"group__{k}": v for k, v in q.get("group_status", {}).items()}}
        for q in qc_rows])
    qc_frame.to_csv(C.RESULTS_ROOT / "feature_group_availability.csv", index=False)

    values = table[names].apply(pd.to_numeric, errors="coerce")
    finite = int(np.isfinite(values.to_numpy()).sum())
    summary = {
        "recording_count": int(len(table)),
        "feature_count": len(names),
        "finite_cells": finite,
        "missing_cells": int(values.size - finite),
        "all_nan_rows": int((values.notna().sum(axis=1) == 0).sum()),
        "complete_subjects": sum(q.get("status") == "complete" for q in qc_rows),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (C.RESULTS_ROOT / "feature_extraction_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("features complete: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
