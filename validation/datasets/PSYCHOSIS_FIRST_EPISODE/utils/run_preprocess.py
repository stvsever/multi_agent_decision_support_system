#!/usr/bin/env python3
"""CLI: harmonized preprocessing for every recording.

By default it reuses the audited ICA-cleaned native checkpoints (fast) and only
re-derives the harmonized 49-channel, average-referenced output. Pass
``--from-raw`` to run the full load / filter / ICA cleaning from the source
binaries instead. Progress and an ETA are printed for the batch.

    python -m utils.run_preprocess               # reuse ICA, harmonize all
    python -m utils.run_preprocess --from-raw    # full recompute from raw
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import mne  # noqa: E402
import pandas as pd  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config as C  # noqa: E402
from utils import io as eeg_io  # noqa: E402
from utils import preprocess as P  # noqa: E402

REUSE_ROOT = C.ROOT / "data" / "processed" / "eeg_fep_rest_v1" / "per_subject"


def _logger() -> logging.Logger:
    C.PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(C.PROCESSED_ROOT / "run_preprocess.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        logger.addHandler(h)
    mne.set_log_level("ERROR")
    return logger


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-raw", action="store_true", help="Full recompute (incl. ICA)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = C.load_config()
    logger = _logger()
    records = eeg_io.discover_records()
    if args.limit:
        records = records[: args.limit]
    reuse_root = None if args.from_raw else REUSE_ROOT
    logger.info("preprocessing %d recordings (%s)", len(records),
                "from raw" if args.from_raw else "reuse ICA + harmonize")

    summaries = []
    t0 = time.time()
    n = len(records)
    for i, record in enumerate(records, 1):
        summaries.append(P.run_subject(record, cfg, overwrite=args.overwrite or not args.from_raw,
                                       reuse_root=reuse_root, logger=logger))
        if i % 20 == 0 or i == n:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            done = sum(s.get("status") == "processed" for s in summaries)
            elig = sum(bool(s.get("feature_eligible")) for s in summaries)
            logger.info("[%d/%d] processed=%d eligible=%d elapsed=%.0fs eta=%.0fs",
                        i, n, done, elig, elapsed, eta)

    frame = pd.DataFrame(summaries)
    keep = [c for c in ["dataset_id", "participant_id", "recording_id", "status",
                        "feature_eligible", "harmonized_channel_count",
                        "interpolated_channel_count", "bad_channel_fraction",
                        "epoch_count_retained", "epoch_count_total", "bad_epoch_fraction",
                        "usable_duration_s", "exclusion_reasons", "runtime_s"]
            if c in frame.columns]
    frame[keep].to_csv(C.PROCESSED_ROOT / "preprocessing_summary.csv", index=False)
    processed = sum(s.get("status") == "processed" for s in summaries)
    logger.info("preprocessing complete: %d/%d processed, %d eligible in %.0fs",
                processed, n, sum(bool(s.get("feature_eligible")) for s in summaries),
                time.time() - t0)
    return 0 if processed == n else 2


if __name__ == "__main__":
    raise SystemExit(main())
