"""
The evaluate() section of validation_guide.ipynb runs offline on the bundled
examples, from the folder Jupyter starts the kernel in (the notebook's own).
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIBRARY = REPO / "src" / "full_stack" / "backend" / "utils" / "validation" / "with_annotated_dataset"
NOTEBOOK = LIBRARY / "validation_guide.ipynb"
TAG = "evaluate-example"


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _tagged(cell_type):
    return [c for c in _cells() if c["cell_type"] == cell_type and TAG in (c.get("metadata", {}).get("tags") or [])]


def test_notebook_is_valid_and_has_the_evaluate_section():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "\u2014" not in text
    headings = ["".join(c["source"]).splitlines()[0] for c in _tagged("markdown")]
    assert headings[0].startswith("## 9. The Evaluation Library")
    # The legacy workflows stay in the guide next to the new section.
    assert "run_validation_metrics.py" in text and "detailed_analysis.py" in text
    assert all(not c.get("outputs") for c in _cells() if c["cell_type"] == "code")


def test_evaluate_section_runs_offline_on_bundled_examples(monkeypatch):
    code_cells = _tagged("code")
    assert len(code_cells) >= 5
    monkeypatch.chdir(NOTEBOOK.parent)
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"__name__": "__validation_guide__"}
    try:
        for cell in code_cells:
            exec(compile("".join(cell["source"]), f"<{cell.get('id', 'cell')}>", "exec"), namespace)
        result = namespace["result"]
        assert result.summary["status_counts"] == {"ok": 7, "missing_prediction": 1}
        metrics = result.metrics
        score = metrics[(metrics["output"] == "total_score") & (metrics["metric"] == "mae")].iloc[0]
        assert (score["n"], score["n_total"]) == (3, 4)
        assert any(Path(p).name.startswith("scatter_") for p in result.figures)
        multiclass = namespace["multiclass"]
        assert not multiclass.comparison.empty
        assert namespace["cli"].returncode == 0
        assert (namespace["OUT"] / "cli_example" / "metrics_long.csv").exists()
    finally:
        out = namespace.get("OUT")
        if out is not None:
            shutil.rmtree(out, ignore_errors=True)
