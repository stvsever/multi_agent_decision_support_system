"""Path bootstrap shared by the numbered pipeline scripts.

Locates the repository root robustly (by walking up to the folder that contains
``src/full_stack``) so the dataset folder can live at any depth under validation/.
"""

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PIPELINE_DIR.parent                      # datasets/<DATASET>


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src" / "full_stack").is_dir():
            return parent
    return start.parents[3]  # sensible fallback


REPO_ROOT = _find_repo_root(PIPELINE_DIR)

for p in (str(REPO_ROOT), str(PIPELINE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
