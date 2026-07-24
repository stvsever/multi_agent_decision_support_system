import math

from validation.common.batch_report import performance_tables, tidy_results


def _reg_row(subject, predicted, truth):
    return {
        "ok": True,
        "dataset": "NUMERACY",
        "task": "approximate_numeracy",
        "subject": subject,
        "tier": "T1",
        "model": "provider/model",
        "outputs": ["approximate_numeracy"],
        "ground_truth": {"approximate_numeracy": truth},
        "prediction": {
            "label": None,
            "probs": None,
            "regression": {"approximate_numeracy": predicted},
        },
        "attempts": 1,
        "total_seconds": 10,
    }


def _dx_row(subject, truth, predicted, p_case):
    return {
        "ok": True,
        "dataset": "PSYCHOSIS",
        "task": "psychosis",
        "subject": subject,
        "tier": "T2",
        "model": "provider/model",
        "outputs": [],
        "ground_truth": {},
        "diagnosis": truth,
        "prediction": {
            "label": predicted,
            "probs": {
                "Control": 1 - p_case,
                "First-Episode Psychosis": p_case,
            },
            "regression": {},
        },
        "attempts": 2,
        "total_seconds": 20,
    }


def test_regression_metrics_are_reported_per_output():
    frame = tidy_results(
        [
            _reg_row("s1", 1, 1),
            _reg_row("s2", 2, 3),
            _reg_row("s3", 4, 3),
        ]
    )

    metrics = performance_tables(frame)["regression_by_output"].loc[
        "approximate_numeracy"
    ]

    assert metrics["n"] == 3
    assert metrics["MAE"] == 0.667
    assert metrics["RMSE"] == 0.816
    assert metrics["bias"] == 0
    assert math.isfinite(metrics["pearson_r"])
    assert math.isfinite(metrics["spearman_rho"])


def test_cross_output_summaries_use_macro_normalized_mae():
    rows = [
        _reg_row("s1", 1, 0),
        _reg_row("s2", 3, 2),
        _reg_row("s3", 5, 4),
    ]
    for row, predicted, truth in zip(rows, (100, 120, 140), (90, 110, 130)):
        row["outputs"].append("large_scale")
        row["ground_truth"]["large_scale"] = truth
        row["prediction"]["regression"]["large_scale"] = predicted

    tables = performance_tables(tidy_results(rows))

    assert "regression_by_provider" not in tables
    summary = tables["regression_macro_nmae_by_provider"].loc["model"]
    assert summary["n"] == 6
    assert summary["n_outputs"] == 2
    assert math.isfinite(summary["macro_NMAE"])


def test_binary_metrics_include_balance_and_probability_ranking():
    rows = [
        _dx_row("s1", "Control", "Control", 0.1),
        _dx_row("s2", "First-Episode Psychosis", "Control", 0.4),
        _dx_row("s3", "Control", "Control", 0.2),
        _dx_row("s4", "First-Episode Psychosis", "First-Episode Psychosis", 0.8),
    ]
    rows[0]["prediction"]["probs"] = {}
    frame = tidy_results(rows)

    metrics = performance_tables(frame)["diagnosis_overall"].iloc[0]

    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 1.0
    assert metrics["AUROC"] == 1.0


def test_psychosis_ground_truth_alias_matches_prediction_case_label():
    frame = tidy_results(
        [
            _dx_row(
                "s1",
                "Psychosis",
                "First-Episode Psychosis",
                0.9,
            ),
            _dx_row("s2", "Control", "Control", 0.1),
        ]
    )

    metrics = performance_tables(frame)["diagnosis_overall"].iloc[0]

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["AUROC"] == 1.0
