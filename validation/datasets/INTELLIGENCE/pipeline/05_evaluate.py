#!/usr/bin/env python3
"""Step 05: evaluate every tier on one common successful participant cohort."""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import pandas as pd

import config
from validation.common.evaluation import evaluate_regression


def _valid_rows(payload):
    return {
        row["participant_id"]: row
        for row in payload.get("predictions", [])
        if row.get("predicted") is not None
    }


def main() -> None:
    subset_payload = json.loads((config.RESULTS_DIR / "subset.json").read_text())
    requested = subset_payload["participants"]
    requested_order = {row["participant_id"]: index for index, row in enumerate(requested)}
    calibration = subset_payload["target_calibration"]
    reference_mean = float(calibration["native_mean"])
    reference_sd = float(calibration["native_sd"])
    tiers_meta = {
        tier["id"]: tier
        for tier in json.loads((config.INPUTS_DIR / "tiers.json").read_text())["tiers"]
    }

    prediction_payloads = {}
    valid_by_tier = {}
    for tier in config.TIERS:
        path = config.RESULTS_DIR / tier["id"] / "predictions.json"
        if path.exists():
            payload = json.loads(path.read_text())
            prediction_payloads[tier["id"]] = payload
            valid_by_tier[tier["id"]] = _valid_rows(payload)
    if not valid_by_tier:
        raise SystemExit("[05] No prediction files found")

    common_ids = set.intersection(*(set(rows) for rows in valid_by_tier.values()))
    common_ids = sorted(common_ids, key=requested_order.get)
    if len(common_ids) < 2:
        raise SystemExit("[05] Fewer than two participants succeeded in every tier")

    summary = []
    for tier in config.TIERS:
        tier_id = tier["id"]
        if tier_id not in valid_by_tier:
            continue
        valid = valid_by_tier[tier_id]
        common_rows = [valid[participant_id] for participant_id in common_ids]
        metrics, rank_rows = evaluate_regression(
            participant_ids=common_ids,
            ground_truth=[row["ground_truth"] for row in common_rows],
            predicted=[row["predicted"] for row in common_rows],
            reference_mean=reference_mean,
            reference_sd=reference_sd,
        )

        available_ids = sorted(valid, key=requested_order.get)
        available_rows = [valid[participant_id] for participant_id in available_ids]
        available_metrics, _ = evaluate_regression(
            participant_ids=available_ids,
            ground_truth=[row["ground_truth"] for row in available_rows],
            predicted=[row["predicted"] for row in available_rows],
            reference_mean=reference_mean,
            reference_sd=reference_sd,
        )
        metrics.update({
            "tier": tier_id,
            "label": tier["label"],
            "n_features": tiers_meta.get(tier_id, {}).get("n_features"),
            "groups": tier["groups"],
            "evaluation_cohort": "common success intersection across all evaluated tiers",
            "n_attempted": len(requested),
            "n_valid": len(valid),
            "n_failed": len(requested) - len(valid),
            "success_rate": round(len(valid) / len(requested), 4),
            "available_case_metrics": available_metrics,
        })
        tier_root = config.RESULTS_DIR / tier_id
        (tier_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame(rank_rows).to_csv(tier_root / "rank_comparison.csv", index=False)
        summary.append(metrics)

    summary_payload = {
        "dataset": config.DATASET_LABEL,
        "target": config.TARGET,
        "target_scale": calibration,
        "evaluation": {
            "requested_n": len(requested),
            "common_success_n": len(common_ids),
            "common_success_participant_ids": common_ids,
            "comparison_policy": "Every headline tier metric uses the same common successful participants",
            "rank_definition": "Rank 1 is highest; ties receive average ranks",
            "rank_uncertainty": "Paired participant bootstrap, 2000 resamples, percentile 95% CI",
        },
        "tiers": summary,
    }
    (config.RESULTS_DIR / "tiers_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(f"[05] Headline comparison cohort: N={len(common_ids)} common successes "
          f"from {len(requested)} requested")
    print(f"[05] {'tier':22} {'valid':>7} {'MAE IST':>9} {'IQeq':>7} "
          f"{'R2':>7} {'Spear':>7} {'rankMAE':>8}")
    for metrics in summary:
        print(
            f"[05] {metrics['tier']:22} "
            f"{metrics['n_valid']:>3}/{metrics['n_attempted']:<3} "
            f"{metrics['mae_native_ist']:>9.3f} "
            f"{metrics['mae_iq15_equivalent']:>7.3f} "
            f"{str(metrics['r2']):>7} "
            f"{str(metrics['spearman_rho']):>7} "
            f"{metrics['rank_mae_positions']:>8.3f}"
        )
    print("[05] Wrote metrics.json, rank_comparison.csv, and tiers_summary.json")


if __name__ == "__main__":
    main()
