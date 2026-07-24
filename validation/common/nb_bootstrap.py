"""Make a single validation smoke-test cell runnable from a fresh kernel.

If a smoke-test cell is pressed before the setup cells have been executed in this
kernel session, or after the notebook's helper API changed while the kernel stayed
alive, `ensure_notebook_setup` runs every code cell ABOVE the first smoke-test
cell into the caller's namespace. It stops at the first engine cell, so it never
triggers a paid run. A version marker—not merely the presence of
`run_smoketest`—prevents stale in-memory helpers from being mistaken for current
ones.
"""

from __future__ import annotations

from pathlib import Path

# Everything above this cell id is setup; this id and below are the engine cells.
_FIRST_ENGINE_CELL = "c0438afa"
_NB_NAME = "validation_with_openneuro_datasets.ipynb"
SETUP_VERSION = "2026-07-24-live-verbose-v8"


def _repo_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "full_stack").is_dir():
            return base
    raise RuntimeError(
        "Could not locate the repo root (the folder containing src/full_stack) from the "
        "current working directory. Open the notebook from inside the compass_engine repo."
    )


def ensure_notebook_setup(g: dict) -> None:
    """Execute setup cells unless `g` already holds this exact helper version."""
    if g.get("VALIDATION_HELPERS_VERSION") == SETUP_VERSION:
        return
    import nbformat

    nb_path = _repo_root() / "validation" / _NB_NAME
    if not nb_path.exists():
        raise FileNotFoundError(f"Could not find the validation notebook at {nb_path}")
    doc = nbformat.read(nb_path, as_version=4)
    for cell in doc.cells:
        if cell.get("id") == _FIRST_ENGINE_CELL:
            break
        if cell.get("cell_type") == "code":
            src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
            exec(compile(src, f"<setup {cell.get('id')}>", "exec"), g)
