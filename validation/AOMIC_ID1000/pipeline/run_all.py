#!/usr/bin/env python3
"""
Run the full AOMIC ID1000 validation pipeline sequentially.

  python run_all.py            # steps 01-03 (offline: explore, ontology, inputs)
  python run_all.py --live     # also run 04 (live COMPASS) and 05 (evaluate)

Steps 02 and 04 make OpenRouter calls; everything else is offline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
STEPS_OFFLINE = ["01_explore_structure.py", "02_build_ontology.py", "03_build_compass_inputs.py"]
STEPS_LIVE = ["04_run_compass.py", "05_evaluate.py"]


def _run(script: str) -> None:
    print(f"\n{'='*70}\n  {script}\n{'='*70}")
    result = subprocess.run([sys.executable, str(PIPELINE_DIR / script)], cwd=PIPELINE_DIR)
    if result.returncode != 0:
        raise SystemExit(f"Step {script} failed with exit code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also run the live COMPASS engine + evaluation")
    args = ap.parse_args()

    for script in STEPS_OFFLINE:
        _run(script)
    if args.live:
        for script in STEPS_LIVE:
            _run(script)
    print("\n[run_all] Done.")


if __name__ == "__main__":
    main()
