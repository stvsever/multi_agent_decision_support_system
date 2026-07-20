#!/usr/bin/env python3
"""
Run the full tiered AOMIC ID1000 validation.

  python run_tiers.py                 # offline: brain extraction + ontology + tier inputs
  python run_tiers.py --live          # also run the engine on every tier + evaluate
  python run_tiers.py --skip-brain    # reuse already-extracted brain features

Steps:
  10 extract FreeSurfer morphometry     (network I/O)
  11 extract functional connectome      (fMRI download + nilearn)
  01 explore merged feature structure
  02 build master multi-modal ontology  (one LLM call)
  03 project ontology onto every tier -> COMPASS inputs
  04 run the engine per tier            (--live)
  05 evaluate per tier + aggregate      (--live)
"""

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent


def _run(script: str, *extra: str) -> None:
    print(f"\n{'='*72}\n  {script} {' '.join(extra)}\n{'='*72}")
    r = subprocess.run([sys.executable, str(PIPELINE_DIR / script), *extra], cwd=PIPELINE_DIR)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed ({r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--skip-brain", action="store_true")
    ap.add_argument("--workers", type=int, default=12,
                    help="bounded process workers for live tier-participant jobs")
    args = ap.parse_args()

    if not args.skip_brain:
        _run("10_extract_freesurfer.py")
        _run("11_extract_connectome.py")
    for s in ("01_explore_structure.py", "02_build_ontology.py", "03_build_compass_inputs.py"):
        _run(s)
    if args.live:
        _run("04_run_compass.py", "--workers", str(args.workers))
        _run("05_evaluate.py")
    print("\n[run_tiers] Done.")


if __name__ == "__main__":
    main()
