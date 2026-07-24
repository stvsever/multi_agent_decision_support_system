"""Batch performance and cross-provider analysis for the OpenNeuro validation.

Reusable, dataset-agnostic tools that turn a dataset's cached full-batch result rows
(as produced by the validation notebook: dicts with subject, tier, model, outputs,
prediction, ground_truth, diagnosis, ok) into:

  - a tidy long frame (one row per predicted output),
  - per-output, per-tier, and per-provider performance tables (regression MAE,
    RMSE, bias, Pearson r, and Spearman rho; macro normalized MAE for summaries
    that span differently scaled outputs; diagnosis accuracy, balanced accuracy,
    sensitivity, specificity, and AUROC),
  - a multiplot of the nested balanced design (tier x provider subject counts), drawn
    BEFORE any run, and
  - a per-dataset performance multiplot (tier vs metric, provider vs metric, and a
    predicted-vs-truth panel), drawn AFTER a batch, with tables and figure saved.

The provider axis is the supplementary cross-provider check: because the design balances
providers within every tier, a provider effect here is read at matched data complexity.

No interpretation is produced here; these functions only quantify and draw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PANEL_COLORS = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#b07aa1", "#76b7b2", "#edc948"]


def _short(model: Optional[str]) -> str:
    return "" if model is None else str(model).split("/")[-1]


def _case_label(labels: List[str]) -> Optional[str]:
    """Pick the non-control label from a binary probability mapping."""
    labels = [str(label) for label in labels]
    for label in labels:
        if "psychosis" in label.strip().casefold():
            return label
    for label in labels:
        normalized = label.strip().casefold()
        if normalized not in {"control", "healthy", "negative", "0"}:
            return label
    return None


def _is_case(value: Any, case_label: Optional[str]) -> Optional[bool]:
    """Map semantically equivalent binary labels onto the same case class."""
    if value is None or case_label is None:
        return None
    normalized = str(value).strip().casefold()
    normalized_case = str(case_label).strip().casefold()
    if "psychosis" in normalized_case:
        return "psychosis" in normalized
    return normalized == normalized_case


def tidy_results(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Long frame: one row per predicted output. kind is 'reg' or 'dx' (diagnosis label)."""
    recs: List[Dict[str, Any]] = []
    diagnosis_labels: List[str] = []
    for r in rows or []:
        pred = r.get("prediction") or {}
        diagnosis_labels.extend(str(label) for label in (pred.get("probs") or {}))
        if pred.get("label") is not None:
            diagnosis_labels.append(str(pred["label"]))
        if r.get("diagnosis") is not None:
            diagnosis_labels.append(str(r["diagnosis"]))
    global_case_label = _case_label(diagnosis_labels)

    for r in rows or []:
        if not r.get("ok"):
            continue
        subj, tier, model = r.get("subject"), r.get("tier"), r.get("model")
        common = dict(
            dataset=r.get("dataset"),
            task=r.get("task"),
            subject=subj,
            tier=tier,
            model=_short(model),
            attempts=int(r.get("attempts", 1) or 1),
            seconds=float(r.get("total_seconds", r.get("seconds", np.nan))),
        )
        pred = r.get("prediction") or {}
        if pred.get("label") is not None:
            true = r.get("diagnosis")
            probs = pred.get("probs") or {}
            case_label = global_case_label or _case_label(
                list(probs) + [pred["label"], true]
            )
            p_case = probs.get(case_label, np.nan)
            true_case_value = _is_case(true, case_label)
            pred_case_value = _is_case(pred["label"], case_label)
            true_case = (
                np.nan if true_case_value is None else float(true_case_value)
            )
            pred_case = (
                np.nan if pred_case_value is None else float(pred_case_value)
            )
            recs.append(
                dict(
                    **common,
                    kind="dx",
                    output="diagnosis",
                    predicted=pred["label"],
                    truth=true,
                    error=np.nan,
                    correct=(
                        np.nan
                        if true_case_value is None or pred_case_value is None
                        else float(pred_case_value == true_case_value)
                    ),
                    case_label=case_label,
                    p_case=float(p_case) if p_case is not None else np.nan,
                    true_case=true_case,
                    pred_case=pred_case,
                )
            )
        gt = r.get("ground_truth") or {}
        for o in r.get("outputs", []):
            p = (pred.get("regression") or {}).get(o)
            t = gt.get(o)
            if p is None:
                continue
            recs.append(
                dict(
                    **common,
                    kind="reg",
                    output=o,
                    predicted=float(p),
                    truth=(np.nan if t is None else float(t)),
                    error=(np.nan if t is None else float(p) - float(t)),
                    correct=np.nan,
                    case_label=None,
                    p_case=np.nan,
                    true_case=np.nan,
                    pred_case=np.nan,
                )
            )
    return pd.DataFrame(recs)


def _reg_metrics(g: pd.DataFrame) -> pd.Series:
    e = g["error"].dropna()
    sub = g.dropna(subset=["predicted", "truth"])
    has_variance = (
        len(sub) >= 3
        and sub["predicted"].nunique() > 1
        and sub["truth"].nunique() > 1
    )
    pearson = (
        float(sub["predicted"].corr(sub["truth"]))
        if has_variance
        else np.nan
    )
    spearman = (
        float(sub["predicted"].corr(sub["truth"], method="spearman"))
        if has_variance
        else np.nan
    )
    return pd.Series(
        {
            "n": int(len(sub)),
            "MAE": float(e.abs().mean()) if len(e) else np.nan,
            "RMSE": float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan,
            "bias": float(e.mean()) if len(e) else np.nan,
            "pearson_r": pearson,
            "spearman_rho": spearman,
        }
    )


def _macro_nmae(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Macro-average output-wise MAE after scaling by each output's truth SD."""
    valid = frame.dropna(subset=["error", "truth"]).copy()
    truth_sd = valid.groupby("output")["truth"].transform(
        lambda values: values.std(ddof=0)
    )
    valid["normalized_absolute_error"] = (
        valid["error"].abs() / truth_sd.where(truth_sd > 0)
    )
    output_cells = (
        valid.groupby([by, "output"], sort=True)["normalized_absolute_error"]
        .agg(["count", "mean"])
        .reset_index()
    )
    if output_cells.empty:
        return pd.DataFrame(columns=["n", "n_outputs", "macro_NMAE"])
    return (
        output_cells.groupby(by, sort=True)
        .agg(
            n=("count", "sum"),
            n_outputs=("output", "nunique"),
            macro_NMAE=("mean", "mean"),
        )
        .round(3)
    )


def _binary_auroc(truth: pd.Series, scores: pd.Series) -> float:
    paired = pd.DataFrame({"truth": truth, "score": scores}).dropna()
    positives = paired.loc[paired.truth == 1, "score"].to_numpy()
    negatives = paired.loc[paired.truth == 0, "score"].to_numpy()
    if not len(positives) or not len(negatives):
        return np.nan
    wins = sum((positive > negative) + 0.5 * (positive == negative)
               for positive in positives for negative in negatives)
    return float(wins / (len(positives) * len(negatives)))


def _dx_metrics(g: pd.DataFrame) -> pd.Series:
    valid = g.dropna(subset=["true_case", "pred_case"])
    c = valid["correct"].dropna()
    sensitivity = (
        float((valid.loc[valid.true_case == 1, "pred_case"] == 1).mean())
        if (valid.true_case == 1).any()
        else np.nan
    )
    specificity = (
        float((valid.loc[valid.true_case == 0, "pred_case"] == 0).mean())
        if (valid.true_case == 0).any()
        else np.nan
    )
    balanced = (
        float(np.nanmean([sensitivity, specificity]))
        if np.isfinite(sensitivity) or np.isfinite(specificity)
        else np.nan
    )
    return pd.Series(
        {
            "n": int(len(valid)),
            "accuracy": float(c.mean()) if len(c) else np.nan,
            "balanced_accuracy": balanced,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "AUROC": _binary_auroc(valid["true_case"], valid["p_case"]),
        }
    )


def _group_metrics(
    frame: pd.DataFrame, by: Any, metric: Any
) -> pd.DataFrame:
    """Apply a metric without depending on pandas' changing group-column behavior."""
    grouped = frame.groupby(by, sort=True)
    try:
        return grouped.apply(metric, include_groups=False).round(3)
    except TypeError:  # pandas < 2.2
        return grouped.apply(metric).round(3)


def performance_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per-output/tier/provider regression tables and binary diagnosis tables."""
    out: Dict[str, pd.DataFrame] = {}
    reg, dx = df[df.kind == "reg"], df[df.kind == "dx"]
    if not reg.empty:
        out["regression_by_output"] = _group_metrics(reg, "output", _reg_metrics)
        out["regression_by_output_tier"] = _group_metrics(
            reg, ["output", "tier"], _reg_metrics
        )
        out["regression_by_output_provider"] = _group_metrics(
            reg, ["output", "model"], _reg_metrics
        )
        out["regression_macro_nmae_by_tier"] = _macro_nmae(reg, "tier")
        out["regression_macro_nmae_by_provider"] = _macro_nmae(reg, "model")
    if not dx.empty:
        out["diagnosis_overall"] = _dx_metrics(dx).to_frame().T.round(3)
        out["diagnosis_by_tier"] = _group_metrics(dx, "tier", _dx_metrics)
        out["diagnosis_by_provider"] = _group_metrics(dx, "model", _dx_metrics)
    return out


def plot_design(designs: Dict[str, List[Dict[str, Any]]], panel_ids: List[str], show: bool = True):
    """Multiplot (one panel per dataset): tier x provider heatmap of subject counts.

    Visualizes the nested balanced design before any run: rows are data-complexity tiers,
    columns are the five providers, each cell is how many subjects sit in that tier/provider
    pair. Even shading and equal row/column margins show the crossing is balanced.
    """
    cols = [_short(m) for m in panel_ids]
    n = len(designs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (dkey, design) in zip(axes, designs.items()):
        tiers = sorted({a["tier"] for a in design})
        mat = pd.DataFrame(0, index=tiers, columns=cols)
        for a in design:
            mat.loc[a["tier"], _short(a["model"])] += 1
        im = ax.imshow(mat.values, cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(tiers))); ax.set_yticklabels(tiers, fontsize=7)
        for i in range(len(tiers)):
            for j in range(len(cols)):
                ax.text(j, i, int(mat.values[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if mat.values[i, j] > mat.values.max() * 0.6 else "black")
        ax.set_title(f"{dkey}\n(n={len(design)} subjects)", fontsize=9)
    fig.suptitle("Nested balanced design: tier x provider subject counts (per dataset)", fontsize=11)
    if show:
        plt.show()
    return fig


def plot_performance(dkey: str, df: pd.DataFrame, show: bool = True):
    """Per-dataset performance multiplot: tier vs metric, provider vs metric, predicted vs truth.

    The metric is diagnosis accuracy when a classification label is present,
    otherwise macro normalized MAE. The third panel always shows predicted vs
    truth for the regression outputs (colored by tier), with the identity line.
    """
    reg, dx = df[df.kind == "reg"], df[df.kind == "dx"]
    has_dx = not dx.empty
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

    def _bar(a, table, col, ylabel, title):
        if table is None or table.empty:
            a.text(0.5, 0.5, "no data", ha="center", va="center"); a.set_title(title); return
        a.bar(range(len(table)), table[col].values, color="#4e79a7", alpha=0.85)
        a.set_xticks(range(len(table))); a.set_xticklabels(table.index, rotation=45, ha="right", fontsize=7)
        a.set_ylabel(ylabel); a.set_title(title)
        for i, v in enumerate(table[col].values):
            if np.isfinite(v):
                a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    tables = performance_tables(df)
    if has_dx:
        _bar(ax[0], tables.get("diagnosis_by_tier"), "accuracy", "accuracy",
             f"{dkey}: diagnosis accuracy by tier")
        _bar(ax[1], tables.get("diagnosis_by_provider"), "accuracy", "accuracy",
             "diagnosis accuracy by provider")
    else:
        _bar(
            ax[0],
            tables.get("regression_macro_nmae_by_tier"),
            "macro_NMAE",
            "macro normalized MAE",
            f"{dkey}: normalized regression error by tier",
        )
        _bar(
            ax[1],
            tables.get("regression_macro_nmae_by_provider"),
            "macro_NMAE",
            "macro normalized MAE",
            "normalized regression error by provider",
        )

    if not reg.empty:
        sub = reg.dropna(subset=["predicted", "truth"])
        tiers = sorted(sub["tier"].unique())
        for k, t in enumerate(tiers):
            s = sub[sub.tier == t]
            ax[2].scatter(s["truth"], s["predicted"], s=28, alpha=0.8,
                          color=_PANEL_COLORS[k % len(_PANEL_COLORS)], label=t)
        if len(sub):
            lo = float(np.nanmin(sub[["truth", "predicted"]].to_numpy()))
            hi = float(np.nanmax(sub[["truth", "predicted"]].to_numpy()))
            ax[2].plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax[2].set_xlabel("true"); ax[2].set_ylabel("predicted")
        ax[2].set_title("predicted vs true (regression outputs)")
        ax[2].legend(fontsize=6, frameon=False)
    else:
        ax[2].axis("off")
    if show:
        plt.show()
    return fig


def report(dkey: str, rows: List[Dict[str, Any]], out_dir: Optional[Path] = None, show: bool = True) -> Optional[Dict[str, pd.DataFrame]]:
    """Quantify + visualize + save one dataset's full-batch results. Graceful if empty."""
    df = tidy_results(rows)
    ok = sum(1 for r in (rows or []) if r.get("ok"))
    if df.empty:
        print(f"[{dkey}] no full-batch results yet. Set RUN_FULL_BATCH=True and run this dataset's "
              f"full-batch cell above first; then re-run this analysis cell.")
        return None
    print(f"=== {dkey}: full-batch performance over {ok} ok runs ===")
    tables = performance_tables(df)
    for name, tab in tables.items():
        print(f"\n[{name}]"); print(tab.to_string())
    plot_performance(dkey, df, show=show)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{dkey}_results_tidy.csv", index=False)
        for name, tab in tables.items():
            tab.to_csv(out_dir / f"{dkey}_{name}.csv")
        fig = plot_performance(dkey, df, show=False)
        fig.savefig(out_dir / f"{dkey}_performance.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved tidy results, per-tier/provider tables, and performance figure to {out_dir}")
    return tables
