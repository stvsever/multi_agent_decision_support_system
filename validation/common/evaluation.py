"""Leakage-aware regression and rank-recovery metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def _rounded(value, digits=4):
    value = float(value)
    return None if not np.isfinite(value) else round(value, digits)


def _pairwise_order_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of actual pair orderings recovered, with predicted ties worth 0.5."""
    scores = []
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            actual_delta = actual[i] - actual[j]
            if actual_delta == 0:
                continue
            predicted_delta = predicted[i] - predicted[j]
            if predicted_delta == 0:
                scores.append(0.5)
            else:
                scores.append(float(np.sign(actual_delta) == np.sign(predicted_delta)))
    return float(np.mean(scores)) if scores else float("nan")


def _top_bottom_overlap(actual: np.ndarray, predicted: np.ndarray, fraction=0.25):
    k = max(1, int(np.ceil(len(actual) * fraction)))
    actual_order = np.argsort(-actual, kind="stable")
    predicted_order = np.argsort(-predicted, kind="stable")
    top = len(set(actual_order[:k]).intersection(predicted_order[:k])) / k
    bottom = len(set(actual_order[-k:]).intersection(predicted_order[-k:])) / k
    return float(top), float(bottom), k


def evaluate_regression(
    participant_ids: Iterable[str],
    ground_truth: Iterable[float],
    predicted: Iterable[float],
    reference_mean: float,
    reference_sd: float,
    n_boot: int = 2000,
    seed: int = 42,
):
    """Return point, scale, and rank metrics plus per-person rank rows.

    IQ-equivalent values are a transparent linear transformation using the
    disjoint reference split. They are not official normed IST IQ scores.
    """
    ids = list(participant_ids)
    actual = np.asarray(list(ground_truth), dtype=float)
    pred = np.asarray(list(predicted), dtype=float)
    if len(actual) < 2 or len(actual) != len(pred) or len(ids) != len(actual):
        raise ValueError("evaluation requires at least two equally sized paired arrays")
    if not np.isfinite(reference_sd) or reference_sd <= 0:
        raise ValueError("reference_sd must be positive")

    errors = pred - actual
    abs_errors = np.abs(errors)
    mae = float(abs_errors.mean())
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rho = float(spearmanr(actual, pred).statistic)
    tau = float(kendalltau(actual, pred, variant="b").statistic)
    pearson = (
        float(np.corrcoef(actual, pred)[0, 1])
        if np.std(actual) > 1e-12 and np.std(pred) > 1e-12
        else float("nan")
    )

    # Rank 1 is the highest intelligence score. Average ranks handle ties correctly.
    actual_rank = rankdata(-actual, method="average")
    predicted_rank = rankdata(-pred, method="average")
    absolute_rank_error = np.abs(predicted_rank - actual_rank)
    rank_mae = float(absolute_rank_error.mean())
    rank_mae_percentile = 100.0 * rank_mae / max(1, len(actual) - 1)
    top_overlap, bottom_overlap, quartile_k = _top_bottom_overlap(actual, pred)

    baseline_errors = np.abs(reference_mean - actual)
    baseline_mae = float(baseline_errors.mean())
    improvement = 100.0 * (baseline_mae - mae) / baseline_mae if baseline_mae else float("nan")

    rng = np.random.default_rng(seed)
    boot_mae = []
    boot_rho = []
    for _ in range(n_boot):
        index = rng.integers(0, len(actual), size=len(actual))
        boot_mae.append(float(np.mean(np.abs(pred[index] - actual[index]))))
        value = float(spearmanr(actual[index], pred[index]).statistic)
        if np.isfinite(value):
            boot_rho.append(value)
    rho_values = np.asarray(boot_rho, dtype=float)
    mae_values = np.asarray(boot_mae, dtype=float)

    iq_factor = 15.0 / reference_sd
    metrics = {
        "n": len(actual),
        "mae_native_ist": _rounded(mae, 3),
        "median_absolute_error_native_ist": _rounded(np.median(abs_errors), 3),
        "rmse_native_ist": _rounded(rmse, 3),
        "mae_native_ist_ci95": [
            _rounded(np.percentile(mae_values, 2.5), 3),
            _rounded(np.percentile(mae_values, 97.5), 3),
        ],
        "mae_in_reference_sd": _rounded(mae / reference_sd, 4),
        "mae_iq15_equivalent": _rounded(mae * iq_factor, 3),
        "rmse_iq15_equivalent": _rounded(rmse * iq_factor, 3),
        "iq_equivalent_caveat": "Reference-standardized interpretive transform, not an official IST norm conversion",
        "reference_mean_baseline_mae_native_ist": _rounded(baseline_mae, 3),
        "mae_improvement_vs_reference_mean_percent": _rounded(improvement, 2),
        "r2": _rounded(r2, 4),
        "pearson_r": _rounded(pearson, 4),
        "spearman_rho": _rounded(rho, 4),
        "spearman_bootstrap_ci95": (
            [_rounded(np.percentile(rho_values, 2.5), 4),
             _rounded(np.percentile(rho_values, 97.5), 4)]
            if rho_values.size else [None, None]
        ),
        "spearman_bootstrap_mean": _rounded(np.mean(rho_values), 4) if rho_values.size else None,
        "spearman_bootstrap_sd": _rounded(np.std(rho_values), 4) if rho_values.size else None,
        "kendall_tau_b": _rounded(tau, 4),
        "pairwise_order_accuracy": _rounded(_pairwise_order_accuracy(actual, pred), 4),
        "rank_mae_positions": _rounded(rank_mae, 3),
        "rank_mae_percentile_points": _rounded(rank_mae_percentile, 3),
        "top_quartile_overlap": _rounded(top_overlap, 4),
        "bottom_quartile_overlap": _rounded(bottom_overlap, 4),
        "quartile_n": quartile_k,
        # Compact backward-compatible aliases used by older visualizations.
        "mae": _rounded(mae, 3),
        "rmse": _rounded(rmse, 3),
        "normalised_mae": _rounded(mae / reference_sd, 4),
    }

    rank_rows = []
    for index, participant_id in enumerate(ids):
        rank_rows.append({
            "participant_id": participant_id,
            "ground_truth_native_ist": _rounded(actual[index], 4),
            "predicted_native_ist": _rounded(pred[index], 4),
            "ground_truth_iq_equivalent": _rounded(100 + (actual[index] - reference_mean) * iq_factor, 4),
            "predicted_iq_equivalent": _rounded(100 + (pred[index] - reference_mean) * iq_factor, 4),
            "actual_rank": _rounded(actual_rank[index], 3),
            "predicted_rank": _rounded(predicted_rank[index], 3),
            "absolute_rank_error": _rounded(absolute_rank_error[index], 3),
        })
    return metrics, rank_rows
