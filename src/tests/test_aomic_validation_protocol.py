import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from validation.common.evaluation import evaluate_regression
from validation.common.freesurfer import _parse_measures


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "validation" / "datasets" / "AOMIC_ID1000" / "pipeline" / "config.py"


def _load_config():
    spec = importlib.util.spec_from_file_location("aomic_test_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_subset_selection_is_invariant_to_target_values():
    config = _load_config()
    n = 160
    frame = pd.DataFrame({
        "participant_id": [f"sub-{i:04d}" for i in range(n)],
        "IST_intelligence_total": np.arange(n, dtype=float),
        "BMI": np.full(n, 22.0),
    })
    for column in config.CORE_COMPLETE:
        frame[column] = 1.0
    selected = config.select_subset_ids(frame, use_lock=False)
    changed = frame.copy()
    changed["IST_intelligence_total"] = np.arange(n, dtype=float)[::-1] * 1000
    assert config.select_subset_ids(changed, use_lock=False) == selected
    assert len(selected) == config.SUBSET_SIZE


def test_target_and_subscales_are_not_predictors():
    config = _load_config()
    predictors = set(config.all_feature_specs())
    assert not predictors.intersection(config.EXCLUDED_COLUMNS)


def test_tie_correct_spearman_and_rank_rows():
    actual = [1, 2, 3, 4, 5, 6]
    predicted = [10, 10, 20, 20, 30, 30]
    metrics, ranks = evaluate_regression(
        [f"eval-{i}" for i in range(6)], actual, predicted,
        reference_mean=3.5, reference_sd=1.5, n_boot=100,
    )
    expected = float(spearmanr(actual, predicted).statistic)
    assert metrics["spearman_rho"] == round(expected, 4)
    assert len(ranks) == 6
    assert ranks[0]["predicted_rank"] == ranks[1]["predicted_rank"]
    assert metrics["rank_mae_positions"] >= 0


def test_freesurfer_measure_parser_keeps_long_and_short_etiv_aliases():
    text = (
        "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total "
        "Intracranial Volume, 1543210.0, mm^3\n"
    )
    parsed = _parse_measures(text)
    assert parsed["EstimatedTotalIntraCranialVol"] == 1543210.0
    assert parsed["eTIV"] == 1543210.0
