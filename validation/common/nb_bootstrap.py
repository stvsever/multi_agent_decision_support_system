"""Make a single validation smoke-test cell runnable from a fresh kernel.

If a smoke-test cell is pressed before the setup cells have been executed in this
kernel session (a fresh kernel shows only saved outputs, not live definitions),
`ensure_notebook_setup` runs every code cell ABOVE the first smoke-test cell
(environment, provider panel, dataset registry, cohorts, run helpers, and the
smoke-test helpers) into the caller's namespace. It stops at the first engine
cell, so it never triggers a paid run. During a normal top-to-bottom run it is a
no-op, because `run_smoketest` is already defined.
"""

from __future__ import annotations

from pathlib import Path

# Everything above this cell id is setup; this id and below are the engine cells.
_FIRST_ENGINE_CELL = "c0438afa"
_NB_NAME = "validation_with_openneuro_datasets.ipynb"


def _repo_root() -> Path:
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "full_stack").is_dir():
            return base
    raise RuntimeError(
        "Could not locate the repo root (the folder containing src/full_stack) from the "
        "current working directory. Open the notebook from inside the compass_engine repo."
    )


def ensure_notebook_setup(g: dict) -> None:
    """Execute the notebook's setup cells into globals `g` if setup is not loaded yet."""
    if "run_smoketest" in g:
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
