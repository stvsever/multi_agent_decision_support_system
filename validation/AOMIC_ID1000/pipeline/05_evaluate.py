#!/usr/bin/env python3
"""
Step 05 - evaluate subset predictions against ground truth.

Reports point-error metrics (MAE, RMSE, normalised MAE) and, because the subset
is small and rank recovery is often the quantity of interest for a deep-phenotype
engine, rank-agreement metrics: Pearson r, Spearman rho, and a bootstrap
rank-stability estimate (mean/sd of Spearman rho over resamples plus a
leave-one-out minimum). Writes ``results/metrics.json`` and prints a summary.
"""

import json

import _bootstrap  # noqa: F401
import numpy as np

import config


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-9 or np.std(rb) < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap_rank_stability(gt, pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(gt)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            continue
        rho = _spearman(gt[idx], pred[idx])
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.array(rhos) if rhos else np.array([float("nan")])
    # Leave-one-out Spearman (sensitivity of the ranking to any single subject).
    loo = []
    for i in range(n):
        mask = np.arange(n) != i
        loo.append(_spearman(gt[mask], pred[mask]))
    loo = [x for x in loo if not np.isnan(x)]
    return {
        "bootstrap_spearman_mean": round(float(np.nanmean(rhos)), 4),
        "bootstrap_spearman_sd": round(float(np.nanstd(rhos)), 4),
        "bootstrap_spearman_p05": round(float(np.nanpercentile(rhos, 5)), 4),
        "bootstrap_spearman_p95": round(float(np.nanpercentile(rhos, 95)), 4),
        "leave_one_out_spearman_min": round(float(np.min(loo)), 4) if loo else None,
        "leave_one_out_spearman_mean": round(float(np.mean(loo)), 4) if loo else None,
    }


def main() -> None:
    with open(config.RESULTS_DIR / "predictions.json") as f:
        preds = json.load(f)
    rows = [r for r in preds["predictions"] if r.get("predicted") is not None]
    if not rows:
        print("[05] No numeric predictions to evaluate.")
        return

    gt = np.array([r["ground_truth"] for r in rows], dtype=float)
    pr = np.array([r["predicted"] for r in rows], dtype=float)
    err = pr - gt
    target_sd = float(preds["target"].get("std", 0)) or float(np.std(gt)) or 1.0

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")

    metrics = {
        "dataset": preds["dataset"],
        "model": preds.get("model"),
        "n_evaluated": len(rows),
        "target": preds["target"],
        "point_error": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "normalised_mae_vs_target_sd": round(mae / target_sd, 3),
            "r2": round(r2, 4),
            "mean_bias": round(float(np.mean(err)), 3),
        },
        "rank_agreement": {
            "pearson_r": round(_pearson(gt, pr), 4),
            "spearman_rho": round(_spearman(gt, pr), 4),
        },
        "rank_stability": _bootstrap_rank_stability(gt, pr),
        "note": (
            "Predicting total intelligence from non-cognitive features (personality, "
            "demographics, SES, identity, lifestyle) is intrinsically hard; modest "
            "point accuracy is expected. This run validates the ingestion, ontology, "
            "and end-to-end multi-agent regression flow on real open data with a tiny "
            "model, not maximal predictive performance. Metrics on N="
            f"{len(rows)} are indicative only."
        ),
    }
    with open(config.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[05] N={len(rows)}  MAE={mae:.1f}  RMSE={rmse:.1f}  "
          f"nMAE={mae/target_sd:.2f}  R2={r2:.3f}")
    print(f"[05] Pearson r={metrics['rank_agreement']['pearson_r']}  "
          f"Spearman rho={metrics['rank_agreement']['spearman_rho']}")
    print(f"[05] Rank stability (bootstrap Spearman): "
          f"{metrics['rank_stability']['bootstrap_spearman_mean']} "
          f"+/- {metrics['rank_stability']['bootstrap_spearman_sd']}")
    print("[05] Wrote results/metrics.json")


if __name__ == "__main__":
    main()
