"""Synthetic tests of the annotated evaluation library (tidy table, metrics, CLI).

Every task kind (binary, multiclass, multi-label, hierarchical mixed trees,
univariate and multivariate regression) and every input format (COMPASS
result folders, JSON and JSONL records, tidy and wide CSV, Python objects,
annotation JSON templates, CSV with a column mapping) is exercised on small
synthetic data. No network, no LLM.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

LIB = Path(__file__).resolve().parents[1] / "full_stack" / "backend" / "utils" / "validation" / "with_annotated_dataset"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(LIB))

from core.evaluate import evaluate, main as cli_main  # noqa: E402
from core.io_utils import extract_generalized_prediction  # noqa: E402
from core.stats import compare_configurations, compute_metrics, test_retest  # noqa: E402
from core.tidy import TIDY_COLUMNS, build_tidy, load_annotations, load_predictions, read_tidy, tidy_to_csv  # noqa: E402

N_BOOT = 200


def _metric(table, metric, **filters):
    sub = table[table["metric"] == metric]
    for key, value in filters.items():
        sub = sub[sub[key].astype(str) == str(value)]
    assert len(sub) == 1, (metric, filters, len(sub))
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def _regression_records(n=40, seed=0, outputs=("score",), failed=(), tiers=("T1",), predictors=("llm",)):
    rng = np.random.default_rng(seed)
    truth = {f"sub-{i:02d}": {o: float(rng.normal(100, 15)) for o in outputs} for i in range(n)}
    records = []
    for tier in tiers:
        for pred in predictors:
            noise = 5.0 if pred == predictors[0] else 12.0
            for pid, gt in truth.items():
                rec = {"subject": pid, "tier": tier, "predictor": pred, "outputs": list(outputs), "ground_truth": gt}
                if pid in failed:
                    rec.update(ok=False, prediction=None)
                else:
                    rec.update(ok=True, prediction={"label": None, "probs": None,
                                                    "regression": {o: v + rng.normal(0, noise) for o, v in gt.items()},
                                                    "classifications": {}})
                records.append(rec)
    return records, truth


def _node(node_id, mode, *, label=None, probs=None, values=None, children=()):
    node = {"node_id": node_id, "mode": mode, "children": list(children)}
    if label is not None or probs is not None:
        node["classification"] = {"predicted_label": label, "probabilities": probs or {}}
    if values is not None:
        node["regression"] = {"values": values}
    return node


def _write_compass_folder(root: Path, pid: str, report_pred: dict, spec: dict, perf_extra=None):
    folder = root / f"participant_{pid}"
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "report_id": f"RPT_{pid}",
        "participant_id": pid,
        "prediction": {"prediction_type": "binary", "prediction_task_spec": spec, "root_prediction": report_pred},
        "evaluation": {"verdict": "SATISFACTORY", "checklist_passed": 5, "checklist_total": 5},
        "execution": {"iterations": 1},
        "clinical_summary": "synthetic",
    }
    (folder / f"report_{pid}.json").write_text(json.dumps(report))
    perf = {"participant_id": pid, "predictor": {"kind": "llm", "model": "test/model"},
            "routing": {"selected_route": "direct"}, "token_usage": {"total_tokens": 1000}}
    perf.update(perf_extra or {})
    (folder / f"performance_report_{pid}.json").write_text(json.dumps(perf))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_univariate_regression_from_jsonl_records_matches_scipy(tmp_path):
    records, truth = _regression_records(n=40, failed=("sub-03", "sub-07"))
    path = tmp_path / "jobs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    result = evaluate(str(path), n_boot=N_BOOT)
    tidy = result.tidy
    assert set(TIDY_COLUMNS).issubset(tidy.columns)
    assert (tidy["status"] == "missing_prediction").sum() == 2
    row = _metric(result.metrics, "pearson_r", output="score")
    ok = tidy[tidy["status"] == "ok"]
    expected = sps.pearsonr(ok["true_value"], ok["pred_value"])[0]
    assert row["value"] == pytest.approx(expected)
    assert row["n"] == 38 and row["n_missing_prediction"] == 2 and row["coverage"] == pytest.approx(38 / 40)
    assert row["ci_low"] < row["value"] < row["ci_high"]
    rho = _metric(result.metrics, "spearman_rho", output="score")["value"]
    assert rho == pytest.approx(sps.spearmanr(ok["true_value"], ok["pred_value"])[0])
    mae = _metric(result.metrics, "mae", output="score")["value"]
    assert mae == pytest.approx(np.mean(np.abs(ok["pred_value"] - ok["true_value"])))
    slope = _metric(result.metrics, "ols_slope", output="score")["value"]
    assert slope == pytest.approx(sps.linregress(ok["true_value"], ok["pred_value"]).slope)
    nmae = _metric(result.metrics, "nmae", output="score")["value"]
    assert nmae == pytest.approx(mae / np.std(ok["true_value"], ddof=0))


def test_regression_is_not_recalibrated():
    rows = []
    for i in range(30):
        t = float(i)
        rows.append({"participant_id": str(i), "output": "y", "pred_value": t + 50.0, "true_value": t})
    result = evaluate(pd.DataFrame(rows), n_boot=0, figures=False)
    assert _metric(result.metrics, "pearson_r", output="y")["value"] == pytest.approx(1.0)
    assert _metric(result.metrics, "r2", output="y")["value"] < -20  # shifted predictions are penalized
    assert _metric(result.metrics, "bias", output="y")["value"] == pytest.approx(50.0)
    assert _metric(result.metrics, "lin_ccc", output="y")["value"] < 0.1


def test_multivariate_regression_grouped_with_macro_rows():
    records, _ = _regression_records(n=30, outputs=("a", "b", "c"), tiers=("T1", "T2"), predictors=("llm", "decision"))
    result = evaluate(records, group_by=["tier", "predictor"], n_boot=N_BOOT, figures=False)
    m = result.metrics
    assert set(m["tier"]) == {"T1", "T2"} and set(m["predictor"]) == {"llm", "decision"}
    assert set(m.loc[m["kind"] == "regression", "output"]) == {"a", "b", "c", "__macro__"}
    macro = _metric(m, "macro_pearson_r", tier="T1", predictor="llm")
    per = [
        _metric(m, "pearson_r", tier="T1", predictor="llm", output=o)["value"] for o in ("a", "b", "c")
    ]
    assert macro["value"] == pytest.approx(np.mean(per))
    wide = result.metrics_wide
    assert {"pearson_r", "pearson_r_ci_low", "pearson_r_ci_high", "mae"}.issubset(wide.columns)
    assert len(wide) == 2 * 2 * 4  # tier x predictor x (3 outputs + macro)


def test_tidy_csv_long_round_trip(tmp_path):
    records, _ = _regression_records(n=25, outputs=("a", "b"))
    tidy = build_tidy(records)
    path = tmp_path / "tidy.csv"
    tidy_to_csv(tidy, path)
    again = read_tidy(path)
    assert len(again) == len(tidy)
    r1 = evaluate(tidy, n_boot=N_BOOT, figures=False).metrics
    r2 = evaluate(str(path), n_boot=N_BOOT, figures=False).metrics
    pd.testing.assert_series_equal(r1["value"].reset_index(drop=True), r2["value"].reset_index(drop=True))


def test_wide_csv_predictions_with_csv_annotation_mapping(tmp_path):
    rng = np.random.default_rng(3)
    ids = [f"P{i:03d}" for i in range(30)]
    truth = rng.normal(50, 10, size=30)
    pd.DataFrame({"participant_id": ids, "tier": "T1", "score": truth + rng.normal(0, 3, 30)}).to_csv(tmp_path / "pred.csv", index=False)
    pd.DataFrame({"pid": ids, "TOTAL_SCORE": truth, "site": ["A"] * 15 + ["B"] * 15}).to_csv(tmp_path / "ann.csv", index=False)
    # The two tables use different id columns: load each explicitly.
    preds = load_predictions(str(tmp_path / "pred.csv"), prediction_columns={"score": "score"})
    ann = load_annotations(str(tmp_path / "ann.csv"), id_column="pid", columns={"score": "TOTAL_SCORE"})
    result = evaluate(preds, ann, group_by=["site"], n_boot=N_BOOT, figures=False)
    assert set(result.metrics["site"]) == {"A", "B"}
    r = _metric(result.metrics, "pearson_r", site="A", output="score")
    assert r["n"] == 15 and r["value"] > 0.8


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _binary_folders(root: Path, n=40, seed=1):
    rng = np.random.default_rng(seed)
    spec = {"root": {"node_id": "root", "display_name": "Depression", "mode": "binary_classification",
                     "class_labels": ["MAJOR_DEPRESSIVE_DISORDER", "CONTROL"]}}
    targets = {}
    for i in range(n):
        pid = f"{i + 1:02d}"
        case = i % 2 == 0
        targets[pid] = {"label": "CASE" if case else "CONTROL", "disorder": "MDD"}
        if i == 5:
            (root / f"participant_{pid}").mkdir(parents=True)  # failed run: folder without a report
            continue
        p = float(np.clip((0.7 if case else 0.3) + rng.normal(0, 0.2), 0.01, 0.99))
        label = "MAJOR_DEPRESSIVE_DISORDER" if p > 0.5 else "CONTROL"
        node = _node("root", "binary_classification", label=label,
                     probs={"MAJOR_DEPRESSIVE_DISORDER": p, "CONTROL": 1 - p})
        _write_compass_folder(root, pid, node, spec)
    return targets


def test_binary_from_compass_folders_with_case_control_targets(tmp_path):
    targets = _binary_folders(tmp_path / "runs")
    ann_path = tmp_path / "targets.json"
    ann_path.write_text(json.dumps(targets))
    result = evaluate(str(tmp_path / "runs"), str(ann_path), n_boot=N_BOOT, figures=False)
    tidy = result.tidy
    assert set(tidy["true_label"].dropna()) == {"MAJOR_DEPRESSIVE_DISORDER", "CONTROL"}
    assert (tidy["positive_label"] == "MAJOR_DEPRESSIVE_DISORDER").all()
    assert (tidy["status"] == "missing_prediction").sum() == 1
    assert {"predictor_kind", "route", "critic_verdict"}.issubset(tidy.columns)
    assert "clinical_summary" not in tidy.columns
    ok = tidy[tidy["status"] == "ok"]
    y = (ok["true_label"] == "MAJOR_DEPRESSIVE_DISORDER").astype(float).to_numpy()
    p = ok["p_positive"].to_numpy(float)
    auc_manual = np.mean([(a > b) + 0.5 * (a == b) for a in p[y == 1] for b in p[y == 0]])
    assert _metric(result.metrics, "auroc")["value"] == pytest.approx(auc_manual)
    assert _metric(result.metrics, "brier")["value"] == pytest.approx(np.mean((p - y) ** 2))
    acc = _metric(result.metrics, "accuracy")
    miss = _metric(result.metrics, "accuracy_missing_as_wrong")
    assert miss["value"] == pytest.approx(acc["value"] * acc["n"] / (acc["n"] + 1))
    sens = _metric(result.metrics, "sensitivity")["value"]
    spec = _metric(result.metrics, "specificity")["value"]
    assert _metric(result.metrics, "balanced_accuracy")["value"] == pytest.approx((sens + spec) / 2)
    conf = result.confusion
    assert "(missing)" in set(conf["pred_label"])
    assert conf["count"].sum() == len(tidy)


def test_multiclass_from_json_records_with_template_annotations(tmp_path):
    rng = np.random.default_rng(5)
    labels = ["anxious", "depressed", "healthy"]
    records, ann = [], {"annotations": []}
    for i in range(45):
        pid = f"S{i}"
        true = labels[i % 3]
        pred = true if rng.random() < 0.7 else labels[(i + 1) % 3]
        probs = {lab: (0.6 if lab == pred else 0.2) for lab in labels}
        records.append({"participant_id": pid, "prediction": {"label": pred.upper(), "probs": probs}})
        ann["annotations"].append({"eid": pid, "label": true, "group": "G1" if i < 20 else "G2"})
    (tmp_path / "pred.json").write_text(json.dumps({"records": records}))
    (tmp_path / "ann.json").write_text(json.dumps(ann))
    spec = {"root": {"node_id": "root", "mode": "multiclass_classification", "class_labels": labels}}
    result = evaluate(str(tmp_path / "pred.json"), str(tmp_path / "ann.json"), task_spec=spec,
                      group_by=["group"], n_boot=N_BOOT, figures=False)
    tidy = result.tidy
    assert set(tidy["pred_label"].dropna()) <= set(labels)  # case-insensitive canonicalization
    assert set(tidy["kind"]) == {"multiclass"}
    pc = result.per_class
    assert set(pc["label"]) == set(labels) and {"precision", "recall_ci_low", "f1_ci_high"}.issubset(pc.columns)
    g1 = tidy[(tidy["group"] == "G1") & (tidy["status"] == "ok")]
    acc = (g1["true_label"] == g1["pred_label"]).mean()
    assert _metric(result.metrics, "accuracy", group="G1")["value"] == pytest.approx(acc)
    assert np.isfinite(_metric(result.metrics, "auroc", group="G1")["value"])


def test_multilabel_from_label_sets_and_probabilities():
    rng = np.random.default_rng(7)
    universe = ["adhd", "anxiety", "asd"]
    records, ann = [], {}
    for i in range(40):
        pid = f"c{i}"
        true = [lab for lab in universe if rng.random() < 0.4]
        probs = {lab: float(np.clip((0.75 if lab in true else 0.25) + rng.normal(0, 0.15), 0, 1)) for lab in universe}
        records.append({"participant_id": pid, "prediction": {"multilabel": {"comorbidity": probs}}})
        ann[pid] = {"multilabel": {"comorbidity": true}}
    result = evaluate(records, ann, n_boot=N_BOOT, figures=False)
    tidy = result.tidy
    assert set(tidy["kind"]) == {"multilabel"}
    assert set(tidy["output"]) == set(universe)
    assert len(tidy) == 40 * 3 and (tidy["status"] == "ok").all()
    ml = result.multilabel
    overall = ml[ml["output"] == "__all__"]
    assert {"macro_f1", "micro_f1", "hamming_loss", "subset_accuracy", "macro_auroc"}.issubset(set(overall["metric"]))
    y = tidy["true_value"].to_numpy()
    yhat = tidy["pred_value"].to_numpy()
    tp, fp, fn = ((y == 1) & (yhat == 1)).sum(), ((y == 0) & (yhat == 1)).sum(), ((y == 1) & (yhat == 0)).sum()
    assert _metric(overall, "micro_f1")["value"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert _metric(overall, "hamming_loss")["value"] == pytest.approx((y != yhat).mean())
    per_label = ml[(ml["output"] == "adhd") & (ml["metric"] == "f1")]
    assert len(per_label) == 1


def test_multilabel_groups_of_binary_nodes():
    spec = {"root": {"node_id": "root", "mode": "multiclass_classification", "class_labels": ["a", "b", "c"],
                     "children": [
                         {"node_id": "dx_adhd", "mode": "binary_classification", "class_labels": ["present", "absent"]},
                         {"node_id": "dx_anx", "mode": "binary_classification", "class_labels": ["present", "absent"]},
                     ]}}
    rng = np.random.default_rng(11)
    records, ann = [], {}
    for i in range(30):
        pid = f"m{i}"
        t1, t2 = rng.random() < 0.5, rng.random() < 0.5
        root = _node("root", "multiclass_classification", label="a", children=[
            _node("dx_adhd", "binary_classification", label="present" if (t1 ^ (i % 5 == 0)) else "absent"),
            _node("dx_anx", "binary_classification", label="present" if t2 else "absent"),
        ])
        records.append({"participant_id": pid, "prediction": {"root_prediction": root}})
        ann[pid] = {"nodes": {"root": {"label": "a"}, "dx_adhd": {"label": "present" if t1 else "absent"},
                              "dx_anx": {"label": "present" if t2 else "absent"}}}
    result = evaluate(records, ann, task_spec=spec, multilabel_groups={"comorbidity": ["dx_adhd", "dx_anx"]},
                      n_boot=N_BOOT, figures=False)
    ml = result.multilabel
    assert set(ml["node_id"]) == {"comorbidity"}
    assert _metric(ml[ml["output"] == "dx_anx"], "f1")["value"] == pytest.approx(1.0)
    assert _metric(ml[ml["output"] == "__all__"], "macro_f1")["value"] < 1.0
    assert (result.tidy.loc[result.tidy["node_id"] == "dx_adhd", "positive_label"] == "present").all()


# ---------------------------------------------------------------------------
# Hierarchical mixed trees
# ---------------------------------------------------------------------------


def _hier_spec():
    return {
        "schema_version": "1.0",
        "root": {
            "node_id": "root", "display_name": "Profile", "mode": "binary_classification",
            "class_labels": ["PATIENT", "CONTROL"],
            "children": [
                {"node_id": "subtype", "display_name": "Subtype", "mode": "multiclass_classification",
                 "class_labels": ["x", "y", "z"],
                 "children": [{"node_id": "severity", "display_name": "Severity", "mode": "univariate_regression",
                               "regression_outputs": ["total"]}]},
                {"node_id": "cognition", "display_name": "Cognition", "mode": "multivariate_regression",
                 "regression_outputs": ["memory", "speed"]},
            ],
        },
    }


def _hier_records(n=36, seed=13, drop_child_for=()):
    rng = np.random.default_rng(seed)
    records, ann = [], {}
    for i in range(n):
        pid = f"h{i:02d}"
        patient = i % 2 == 0
        sub = "xyz"[i % 3]
        total, mem, spd = rng.normal(20, 5), rng.normal(0, 1), rng.normal(0, 1)
        ann[pid] = {"nodes": {
            "root": {"mode": "binary_classification", "label": "PATIENT" if patient else "CONTROL"},
            "subtype": {"mode": "multiclass_classification", "label": sub},
            "severity": {"mode": "univariate_regression", "values": {"total": total}},
            "cognition": {"mode": "multivariate_regression", "values": {"memory": mem, "speed": spd}},
        }}
        root_ok = rng.random() < 0.8
        pred_root = ("PATIENT" if patient else "CONTROL") if root_ok else ("CONTROL" if patient else "PATIENT")
        sev = _node("severity", "univariate_regression", values={"total": total + rng.normal(0, 2)})
        subtype = _node("subtype", "multiclass_classification", label=sub if rng.random() < 0.7 else "x",
                        probs={"x": 0.4, "y": 0.3, "z": 0.3}, children=[sev])
        cog = _node("cognition", "multivariate_regression", values={"memory": mem + rng.normal(0, .5), "speed": spd + rng.normal(0, .5)})
        children = [subtype] if pid in drop_child_for else [subtype, cog]
        root = _node("root", "binary_classification", label=pred_root,
                     probs={"PATIENT": 0.8 if pred_root == "PATIENT" else 0.2, "CONTROL": 0.2 if pred_root == "PATIENT" else 0.8},
                     children=children)
        records.append({"participant_id": pid, "tier": "T1" if i < n // 2 else "T2",
                        "prediction": {"root_prediction": root, "prediction_task_spec": _hier_spec()}})
    return records, ann


def test_hierarchical_mixed_tree_from_nested_root_prediction():
    records, ann = _hier_records(drop_child_for=("h01", "h02"))
    result = evaluate(records, ann, group_by=["tier"], n_boot=N_BOOT, figures=False)  # spec read from the records
    tidy = result.tidy
    kinds = tidy.drop_duplicates(["node_id", "output"]).set_index("output")["kind"].to_dict()
    assert kinds == {"root": "binary", "subtype": "multiclass", "total": "regression", "memory": "regression", "speed": "regression"}
    assert tidy.loc[tidy["node_id"] == "severity", "depth"].iloc[0] == 2
    assert tidy.loc[tidy["node_id"] == "severity", "parent_node_id"].iloc[0] == "subtype"
    missing = tidy[tidy["status"] == "missing_prediction"]
    assert set(missing["output"]) == {"memory", "speed"} and len(missing) == 4
    hier = result.hierarchy
    assert {"node_coverage", "exact_match", "path_accuracy", "child_accuracy_given_parent_correct"}.issubset(set(hier["metric"]))
    paths = set(hier.loc[hier["metric"] == "path_accuracy", "path"])
    assert "root > subtype > severity" in paths and "root > cognition" in paths
    t1 = tidy[(tidy["tier"] == "T1")]
    cls = t1[t1["kind"].isin(["binary", "multiclass"])].pivot_table(index="instance_id", columns="node_id", values="correct")
    exact = (cls == 1).all(axis=1).mean()
    assert _metric(hier, "exact_match", tier="T1")["value"] == pytest.approx(exact)
    cov = _metric(hier, "node_coverage", tier="T1")["value"]
    assert cov < 1.0


def test_hierarchical_spec_as_pydantic_object():
    try:
        from src.full_stack.backend.data.models.prediction_task import PredictionTaskSpec
    except Exception:  # engine package not importable in this environment
        pytest.skip("engine PredictionTaskSpec not importable")
    spec = PredictionTaskSpec(**_hier_spec())
    records, ann = _hier_records()
    result = evaluate(records, ann, task_spec=spec, n_boot=0, figures=False)
    assert set(result.tidy["node_id"]) == {"root", "subtype", "severity", "cognition"}


def test_legacy_extractor_walks_nested_children(tmp_path):
    records, _ = _hier_records(n=3)
    root = records[0]["prediction"]["root_prediction"]
    folder = tmp_path / "participant_h00"
    _write_compass_folder(tmp_path, "h00", root, _hier_spec())
    pred = extract_generalized_prediction(folder)
    assert set(pred["nodes"]) == {"root", "subtype", "severity", "cognition"}


# ---------------------------------------------------------------------------
# Comparison, test-retest, missing participants, determinism
# ---------------------------------------------------------------------------


def test_paired_comparison_detects_better_configuration():
    records, _ = _regression_records(n=60, predictors=("good", "noisy"), seed=21)
    tidy = build_tidy(records)
    comp = compare_configurations(tidy, "predictor", "noisy", "good", n_boot=500)
    mae = _metric(comp, "mae")
    assert mae["diff"] < 0 and mae["diff_ci_high"] < 0 and mae["better"] == "good"
    assert mae["n_matched"] == 60 and mae["p_boot"] < 0.05
    r = _metric(comp, "pearson_r")
    assert r["diff"] > 0 and r["better"] == "good"
    result = evaluate(records, compare="predictor:noisy,good", n_boot=N_BOOT, figures=False)
    assert len(result.comparison) > 0


def test_test_retest_icc_and_kappa():
    rng = np.random.default_rng(4)
    rows = []
    for i in range(30):
        base = rng.normal(0, 1)
        lab = "A" if i % 2 else "B"
        for rep in (1, 2, 3):
            rows.append({"participant_id": str(i), "repeat": rep, "output": "y", "pred_value": base + rng.normal(0, 0.05),
                         "true_value": base})
            rows.append({"participant_id": str(i), "repeat": rep, "output": "dx", "kind": "binary",
                         "pred_label": lab, "true_label": lab})
    tidy = build_tidy(pd.DataFrame(rows))
    rt = test_retest(tidy, "repeat", n_boot=N_BOOT)
    assert _metric(rt, "icc_2_1", output="y")["value"] > 0.95
    assert _metric(rt, "fleiss_kappa", output="dx")["value"] == pytest.approx(1.0)
    assert _metric(rt, "percent_agreement", output="dx")["value"] == pytest.approx(1.0)
    two = test_retest(tidy[tidy["repeat"].astype(int) <= 2], "repeat", n_boot=0)
    assert _metric(two, "cohen_kappa", output="dx")["value"] == pytest.approx(1.0)
    # evaluate() runs test-retest automatically when a 'repeat' column exists.
    assert not evaluate(tidy, n_boot=0, figures=False).test_retest.empty


def test_expected_participants_from_annotations_counts_absent_runs():
    records, truth = _regression_records(n=10, tiers=("T1", "T2"))
    records = [r for r in records if not (r["tier"] == "T2" and r["subject"] in {"sub-00", "sub-01", "sub-02"})]
    ann = {pid: {"regression": gt} for pid, gt in truth.items()}
    observed = evaluate(records, ann, group_by=["tier"], n_boot=0, figures=False)
    expected = evaluate(records, ann, group_by=["tier"], n_boot=0, figures=False, expected_participants="annotations")
    assert _metric(observed.metrics, "mae", tier="T2")["n_missing_prediction"] == 0
    row = _metric(expected.metrics, "mae", tier="T2")
    assert row["n_missing_prediction"] == 3 and row["n"] == 7


def test_bootstrap_is_seeded_and_order_independent():
    records, _ = _regression_records(n=30, outputs=("a", "b"), tiers=("T1", "T2"))
    tidy = build_tidy(records)
    m1 = compute_metrics(tidy, ["tier"], n_boot=300, seed=7).metrics
    m2 = compute_metrics(tidy.iloc[::-1].reset_index(drop=True), ["tier"], n_boot=300, seed=7).metrics
    key = ["tier", "output", "metric"]
    a = m1.sort_values(key).reset_index(drop=True)
    b = m2.sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(a["value"], b["value"])
    np.testing.assert_allclose(a["ci_low"], b["ci_low"], equal_nan=True)
    m3 = compute_metrics(tidy, ["tier"], n_boot=300, seed=8).metrics.sort_values(key).reset_index(drop=True)
    assert not np.allclose(a["ci_low"].fillna(0), m3["ci_low"].fillna(0))


def test_lenient_participant_id_matching(tmp_path):
    records, truth = _regression_records(n=8)
    for rec in records:
        rec["subject"] = "participant_ID" + rec["subject"].split("-")[1]
    ann = {f"{int(pid.split('-')[1]):03d}": {"regression": gt} for pid, gt in truth.items()}
    result = evaluate(records, ann, n_boot=0, figures=False)
    assert (result.tidy["status"] == "ok").all()


def test_python_objects_and_predictionset_concatenation():
    rec_a, truth = _regression_records(n=12, predictors=("llm",))
    rec_b, _ = _regression_records(n=12, predictors=("llm",), seed=9)
    a = load_predictions(rec_a, meta={"provider": "p1"})
    b = load_predictions(rec_b, meta={"provider": "p2"})
    result = evaluate([a, b], group_by=["provider"], n_boot=0, figures=False)
    assert set(result.metrics["provider"]) == {"p1", "p2"}


# ---------------------------------------------------------------------------
# Outputs and CLI
# ---------------------------------------------------------------------------


def test_evaluate_writes_tables_and_figures(tmp_path):
    records, ann = _hier_records()
    out = tmp_path / "eval"
    result = evaluate(records, ann, group_by=["tier"], out_dir=out, n_boot=100)
    for name in ("tidy.csv", "metrics_long.csv", "metrics_long.json", "metrics_wide.csv", "per_class.csv",
                 "confusion.csv", "hierarchy.csv", "summary.json"):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["group_by"] == ["tier"] and summary["status_counts"]["ok"] > 0
    names = {Path(p).name for p in result.figures}
    assert any(n.startswith("scatter_") for n in names)
    assert any(n.startswith("confusion_") for n in names)
    assert any(n.startswith("roc_") for n in names)
    assert any(n.startswith("forest_") for n in names)
    assert any(n.startswith("by_tier_") for n in names)
    assert all(Path(p).stat().st_size > 1000 for p in result.figures)
    json.loads((out / "metrics_long.json").read_text())


def test_cli_main_and_standalone_script(tmp_path):
    records, _ = _regression_records(n=20, tiers=("T1", "T2"), predictors=("llm", "decision"))
    path = tmp_path / "jobs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    out = tmp_path / "cli_out"
    code = cli_main(["evaluate", "--predictions", str(path), "--group-by", "tier,predictor", "--out", str(out),
                     "--n-boot", "50", "--compare", "predictor:llm,decision", "--no-figures"])
    assert code == 0
    comp = pd.read_csv(out / "comparison.csv")
    assert set(comp["tier"]) == {"T1", "T2"}
    out2 = tmp_path / "script_out"
    proc = subprocess.run(
        [sys.executable, str(LIB / "run_evaluation.py"), "evaluate", "--predictions", str(path), "--out", str(out2),
         "--n-boot", "20", "--no-figures"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out2 / "metrics_wide.csv").exists()


def test_cli_reports_bad_group_column(tmp_path, capsys):
    records, _ = _regression_records(n=5)
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(records))
    code = cli_main(["evaluate", "--predictions", str(path), "--group-by", "nope", "--out", str(tmp_path / "o")])
    assert code == 1
    assert "group_by columns not found" in capsys.readouterr().err


def test_embedded_truth_is_per_record_when_ids_repeat_across_groups():
    # The same participant id in two datasets with different truths: each record keeps its own truth.
    records = []
    rng = np.random.default_rng(2)
    for dataset in ("D1", "D2"):
        for i in range(20):
            t = float(rng.normal(0, 1))
            records.append({"participant_id": f"s{i}", "dataset": dataset, "ground_truth": {"y": t},
                            "prediction": {"regression": {"y": t + 0.01}}})
    records.append({"participant_id": "s0", "dataset": "D3", "ok": False})  # no truth anywhere consistent
    result = evaluate(records, group_by=["dataset"], n_boot=0, figures=False)
    for dataset in ("D1", "D2"):
        assert _metric(result.metrics, "mae", dataset=dataset)["value"] == pytest.approx(0.01)
    d3 = result.tidy[result.tidy["dataset"] == "D3"]
    assert d3["status"].tolist() == ["missing_both"] or d3.empty or d3["true_value"].isna().all()


def test_node_aware_classification_records_with_labels_truth():
    # Job records with classifications per node and labels_truth per node (no tree).
    rng = np.random.default_rng(8)
    records = []
    for i in range(40):
        t_root = "case" if i % 2 else "control"
        t_sub = "A" if i % 3 else "B"
        p = float(np.clip((0.8 if t_root == "case" else 0.2) + rng.normal(0, 0.1), 0, 1))
        records.append({
            "subject": f"e{i}", "tier": "T1", "ok": True,
            "labels_truth": {"dx": t_root, "subtype": t_sub},
            "prediction": {"label": None, "probs": None, "regression": {},
                           "classifications": {"dx": {"label": "case" if p > 0.5 else "control",
                                                      "probs": {"case": p, "control": 1 - p}},
                                               "subtype": {"label": t_sub, "probabilities": {"A": 0.5, "B": 0.5}}}},
        })
    result = evaluate(records, n_boot=N_BOOT, figures=False)
    m = result.metrics
    assert _metric(m, "accuracy", node_id="subtype")["value"] == pytest.approx(1.0)
    assert _metric(m, "auroc", node_id="dx")["value"] > 0.95
    assert 0 <= _metric(m, "ece", node_id="dx")["value"] < 0.3
    assert (result.tidy.loc[result.tidy["node_id"] == "dx", "positive_label"] == "case").all()


def test_cli_multiple_labelled_sources(tmp_path):
    a, _ = _regression_records(n=15, seed=1)
    b, _ = _regression_records(n=15, seed=1, predictors=("other",))
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in a))
    (tmp_path / "b.jsonl").write_text("\n".join(json.dumps(r) for r in b))
    out = tmp_path / "o"
    code = cli_main(["evaluate", "--predictions", f"run_a={tmp_path / 'a.jsonl'}", f"run_b={tmp_path / 'b.jsonl'}",
                     "--source-col", "run", "--group-by", "run", "--compare", "run", "--out", str(out),
                     "--n-boot", "30", "--no-figures"])
    assert code == 0
    table = pd.read_csv(out / "metrics_long.csv")
    assert set(table["run"]) == {"run_a", "run_b"}
    assert (out / "comparison.csv").exists()


def test_nested_compass_folders_with_path_groups(tmp_path):
    spec = {"root": {"node_id": "root", "mode": "univariate_regression", "regression_outputs": ["score"]}}
    rng = np.random.default_rng(6)
    ann = {}
    for i in range(12):
        pid = f"{i:02d}"
        truth = float(rng.normal(10, 2))
        ann[pid] = {"regression": {"score": truth}}
        for tier in ("T1", "T2"):
            for predictor in ("llm", "decision"):
                node = _node("root", "univariate_regression", values={"score": truth + rng.normal(0, 1)})
                _write_compass_folder(tmp_path / "runs" / tier / predictor, pid, node, spec)
    result = evaluate(str(tmp_path / "runs"), ann, path_groups=["tier", "predictor"], group_by=["tier", "predictor"],
                      n_boot=0, figures=False)
    assert len(result.metrics_wide) == 4
    assert (result.metrics_wide["n"] == 12).all()


def test_numeric_class_codes_in_csv_annotations(tmp_path):
    # Truth coded 0/1 in a table, predictions given as labels "1"/"0" with P(1).
    rng = np.random.default_rng(12)
    ids = [f"r{i}" for i in range(30)]
    truth = (np.arange(30) % 2).astype(int)
    p = np.clip(0.3 + 0.4 * truth + rng.normal(0, 0.15, 30), 0, 1)
    pd.DataFrame({"participant_id": ids, "dx": truth, "age": rng.integers(20, 60, 30)}).to_csv(tmp_path / "ann.csv", index=False)
    records = [{"participant_id": pid, "prediction": {"classifications": {"dx": {"label": str(int(pp > 0.5)), "probs": {"1": pp, "0": 1 - pp}}}}}
               for pid, pp in zip(ids, p)]
    result = evaluate(records, str(tmp_path / "ann.csv"), annotation_columns={"dx": "dx"},
                      n_boot=0, figures=False)
    tidy = result.tidy
    assert set(tidy["kind"]) == {"binary"}
    assert set(tidy["true_label"]) == {"0", "1"}
    assert (tidy["positive_label"] == "1").all()
    assert _metric(result.metrics, "auroc", node_id="dx")["value"] > 0.8
