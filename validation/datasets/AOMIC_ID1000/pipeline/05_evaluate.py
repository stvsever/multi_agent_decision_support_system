#!/usr/bin/env python3
"""
Step 05 - evaluate predictions per tier and aggregate across tiers.

For each tier with predictions, computes point-error metrics (MAE, RMSE, normalised
MAE, R2) and rank metrics (Pearson, Spearman) plus a bootstrap rank-stability
estimate. Writes ``results/<tier_id>/metrics.json`` and a combined
``results/tiers_summary.json`` showing how performance changes as modalities are
added.
"""

import json

import _bootstrap  # noqa: F401
import numpy as np

import config


def _spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) < 1e-9 or np.std(rb) < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a, b):
    if len(a) < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank_stability(gt, pr, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(gt)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2:
            continue
        r = _spearman(gt[idx], pr[idx])
        if not np.isnan(r):
            rhos.append(r)
    rhos = np.array(rhos) if rhos else np.array([np.nan])
    return round(float(np.nanmean(rhos)), 3), round(float(np.nanstd(rhos)), 3)


def _eval_rows(rows, target_sd):
    rows = [r for r in rows if r.get("predicted") is not None]
    if len(rows) < 2:
        return None
    gt = np.array([r["ground_truth"] for r in rows], float)
    pr = np.array([r["predicted"] for r in rows], float)
    err = pr - gt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    rho_mean, rho_sd = _rank_stability(gt, pr)
    mae = float(np.mean(np.abs(err)))
    return {
        "n": len(rows),
        "mae": round(mae, 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "normalised_mae": round(mae / target_sd, 3),
        "r2": round(1.0 - ss_res / ss_tot, 3) if ss_tot > 1e-9 else None,
        "pearson_r": round(_pearson(gt, pr), 3),
        "spearman_rho": round(_spearman(gt, pr), 3),
        "rank_stability_mean": rho_mean,
        "rank_stability_sd": rho_sd,
    }


def main() -> None:
    with open(config.INPUTS_DIR / "tiers.json") as f:
        tiers_meta = {t["id"]: t for t in json.load(f)["tiers"]}
    target_sd = float(config.TARGET.get("std_hint", 40.4)) if isinstance(config.TARGET, dict) else 40.4

    summary = []
    for tier in config.TIERS:
        tid = tier["id"]
        pred_path = config.RESULTS_DIR / tid / "predictions.json"
        if not pred_path.exists():
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        # target sd from the manifest if available
        try:
            with open(config.ONTOLOGY_DIR / "feature_manifest.json") as mf:
                target_sd = float(json.load(mf)["target"]["std"])
        except Exception:
            pass
        metrics = _eval_rows(preds["predictions"], target_sd)
        if metrics is None:
            continue
        metrics.update({"tier": tid, "label": tier["label"],
                        "n_features": tiers_meta.get(tid, {}).get("n_features"),
                        "groups": tier["groups"]})
        with open(config.RESULTS_DIR / tid / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        summary.append(metrics)

    summary_path = config.RESULTS_DIR / "tiers_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"dataset": config.DATASET_LABEL, "target": config.TARGET,
                   "note": ("Subset validation across data-complexity tiers. Small N, so "
                            "rank metrics (Spearman, stability) are the informative signal."),
                   "tiers": summary}, f, indent=2)

    print(f"\n[05] {'tier':22} {'n_feat':>6} {'MAE':>7} {'nMAE':>6} {'Pears':>6} {'Spear':>6} {'stab':>6}")
    for m in summary:
        print(f"[05] {m['tier']:22} {str(m.get('n_features')):>6} {m['mae']:>7} "
              f"{m['normalised_mae']:>6} {m['pearson_r']:>6} {m['spearman_rho']:>6} "
              f"{m['rank_stability_mean']:>6}")
    print(f"[05] Wrote {summary_path.name}")


if __name__ == "__main__":
    main()
