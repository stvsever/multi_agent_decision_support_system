"""Every standard figure of the annotated evaluation library renders on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LIB = Path(__file__).resolve().parents[1] / "full_stack" / "backend" / "utils" / "validation" / "with_annotated_dataset"
sys.path.insert(0, str(LIB))

from core import figures as F  # noqa: E402
from core.stats import compare_configurations, compute_metrics  # noqa: E402
from core.tidy import build_tidy  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    rows = []
    for tier in ("T1", "T2", "T3"):
        for pred in ("llm", "decision"):
            for i in range(30):
                t = rng.normal(10, 3)
                noise = {"T1": 3.0, "T2": 2.0, "T3": 1.0}[tier] * (1.0 if pred == "llm" else 1.4)
                rows.append({"participant_id": f"p{i}", "tier": tier, "predictor": pred, "output": "score",
                             "true_value": t, "pred_value": t + rng.normal(0, noise)})
                case = i % 2 == 0
                p = float(np.clip((0.7 if case else 0.3) + rng.normal(0, 0.2), 0, 1))
                rows.append({"participant_id": f"p{i}", "tier": tier, "predictor": pred, "output": "dx",
                             "kind": "binary", "true_label": "case" if case else "control",
                             "pred_label": "case" if p > 0.5 else "control", "p_positive": p})
    tidy = build_tidy(pd.DataFrame(rows), positive_labels={"dx": "case"})
    tables = compute_metrics(tidy, ["tier", "predictor"], n_boot=100)
    comp = compare_configurations(tidy, "predictor", "llm", "decision", ["tier"], n_boot=100)
    return tidy, tables, comp


def _save(fig, tmp_path, name):
    paths = F.save_figure(fig, tmp_path / name, formats=("png", "pdf"))
    assert len(paths) == 2 and all(Path(p).stat().st_size > 1000 for p in paths)


def _node_of(tidy, output):
    return tidy.loc[tidy["output"] == output, "node_id"].iloc[0]


def test_scatter_and_residuals(frames, tmp_path):
    tidy, tables, _ = frames
    node = _node_of(tidy, "score")
    _save(F.scatter_grid(tidy, node, "score", facet_cols=["tier", "predictor"], metrics=tables.metrics), tmp_path, "scatter")
    _save(F.residual_plot(tidy, node, "score", color_col="tier"), tmp_path, "residuals")
    with pytest.raises(ValueError):
        F.scatter_grid(tidy, "no_such_node", "score")
    with F.figure_style():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        F.scatter_true_vs_pred(tidy[(tidy["output"] == "score") & (tidy["tier"] == "T1")], ax=ax, title="one panel")
        _save(fig, tmp_path, "single")


def test_forest_by_tier_confusion_roc(frames, tmp_path):
    tidy, tables, _ = frames
    _save(F.forest_plot(tables.metrics, "pearson_r", label_cols=["tier", "predictor", "output"], color_col="predictor"), tmp_path, "forest")
    _save(F.metric_by_tier(tables.metrics[tables.metrics["output"] == "score"], "mae", tier_col="tier",
                           config_cols=["predictor"], tier_order=["T1", "T2", "T3"]), tmp_path, "by_tier")
    conf = tables.confusion[(tables.confusion["tier"] == "T1") & (tables.confusion["predictor"] == "llm")]
    _save(F.confusion_matrix_plot(conf, title="dx"), tmp_path, "confusion")
    _save(F.roc_plot(tidy, "dx", color_cols=["tier"]), tmp_path, "roc")
    _save(F.calibration_plot(tidy, "dx", color_cols=["predictor"]), tmp_path, "calibration")


def test_paired_plots(frames, tmp_path):
    tidy, _, comp = frames
    _save(F.paired_comparison_plot(comp, metric="mae", label_cols=["tier", "output"]), tmp_path, "paired")
    _save(F.paired_errors_plot(tidy, "predictor", "llm", "decision", _node_of(tidy, "score"), "score"), tmp_path, "paired_errors")


def test_roc_points_and_palette():
    fpr, tpr = F.roc_curve_points(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.3, 0.1]))
    assert fpr[0] == 0 and tpr[-1] == 1 and fpr[-1] == 1
    colors = F.color_for(["b", "a", "c"])
    assert colors["a"] == F.PALETTE[0] and colors["b"] == F.PALETTE[1]
    assert F.metric_label("macro_pearson_r") == "Macro Pearson r"
