"""Path bootstrap shared by the numbered pipeline scripts."""

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PIPELINE_DIR.parent                 # validation/AOMIC_ID1000
REPO_ROOT = PIPELINE_DIR.parents[2]                # repository root

for p in (str(REPO_ROOT), str(PIPELINE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
