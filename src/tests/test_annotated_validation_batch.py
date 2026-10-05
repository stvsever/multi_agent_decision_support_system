"""batch_run.py post-hoc validation runs the legacy scripts and the evaluation library."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

BATCH_RUN = Path(__file__).resolve().parents[1] / "full_stack" / "backend" / "utils" / "batch_run.py"


def _load_batch_run():
    spec = importlib.util.spec_from_file_location("compass_batch_run_under_test", BATCH_RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_runs(root: Path, n: int = 12):
    rng = np.random.default_rng(0)
    spec = {"root": {"node_id": "root", "display_name": "Target", "mode": "binary_classification",
                     "class_labels": ["TARGET", "CONTROL"]}}
    targets = {}
    for i in range(n):
        pid = f"{i + 1:02d}"
        case = i % 2 == 0
        targets[pid] = {"label": "CASE" if case else "CONTROL", "disorder": "GROUP_A" if i < n // 2 else "GROUP_B"}
        p = float(np.clip((0.75 if case else 0.25) + rng.normal(0, 0.15), 0.01, 0.99))
        node = {"node_id": "root", "mode": "binary_classification", "children": [],
                "classification": {"predicted_label": "TARGET" if p > 0.5 else "CONTROL",
                                   "probabilities": {"TARGET": p, "CONTROL": 1 - p}}}
        folder = root / f"participant_ID{pid}"
        folder.mkdir(parents=True)
        report = {"participant_id": pid, "report_id": f"RPT_{pid}",
                  "prediction": {"prediction_task_spec": spec, "root_prediction": node, "probability": p},
                  "evaluation": {"verdict": "SATISFACTORY", "checklist_passed": 4, "checklist_total": 4},
                  "execution": {"iterations": 1, "tokens_used": 100}}
        (folder / f"report_{pid}.json").write_text(json.dumps(report))
    return targets


def test_posthoc_validation_runs_legacy_and_evaluation(tmp_path):
    batch_run = _load_batch_run()
    runs = tmp_path / "participant_runs"
    targets = _write_runs(runs)
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps(targets))
    out = tmp_path / "analysis"
    args = Namespace(
        run_validation=True,
        validation_results_dir=str(runs),
        validation_output_dir=str(out),
        validation_targets_file=str(targets_path),
        validation_annotations_json="",
        validation_disorder_groups="",
        validation_engine="both",
        validation_group_by="disorder",
        prediction_type="binary",
        task_spec_file="",
    )
    batch_run.run_posthoc_validation(args)
    assert (out / "binary_confusion_matrix" / "binary_metrics_integrated.json").exists()
    metrics = out / "evaluation" / "metrics_long.csv"
    assert metrics.exists()
    import pandas as pd

    table = pd.read_csv(metrics)
    assert set(table["disorder"]) == {"GROUP_A", "GROUP_B"}
    assert "auroc" in set(table["metric"])
    assert (out / "evaluation" / "figures").is_dir()
