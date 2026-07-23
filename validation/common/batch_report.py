"""Batch performance and cross-provider analysis for the OpenNeuro validation.

Reusable, dataset-agnostic tools that turn a dataset's cached full-batch result rows
(as produced by the validation notebook: dicts with subject, tier, model, outputs,
prediction, ground_truth, diagnosis, ok) into:

  - a tidy long frame (one row per predicted output),
  - per-tier and per-provider performance tables (regression MAE and Pearson r,
    and diagnosis accuracy where a classification label is present),
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


def tidy_results(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Long frame: one row per predicted output. kind is 'reg' or 'dx' (diagnosis label)."""
    recs: List[Dict[str, Any]] = []
    for r in rows or []:
        if not r.get("ok"):
            continue
        subj, tier, model = r.get("subject"), r.get("tier"), r.get("model")
        pred = r.get("prediction") or {}
        if pred.get("label") is not None:
            true = r.get("diagnosis")
            recs.append(dict(subject=subj, tier=tier, model=_short(model), kind="dx",
                             output="diagnosis", predicted=pred["label"], truth=true,
                             error=np.nan, correct=(np.nan if true is None else float(pred["label"] == true))))
        gt = r.get("ground_truth") or {}
        for o in r.get("outputs", []):
            p = (pred.get("regression") or {}).get(o)
            t = gt.get(o)
            if p is None:
                continue
            recs.append(dict(subject=subj, tier=tier, model=_short(model), kind="reg", output=o,
                             predicted=float(p), truth=(np.nan if t is None else float(t)),
                             error=(np.nan if t is None else float(p) - float(t)), correct=np.nan))
    return pd.DataFrame(recs)


def _reg_metrics(g: pd.DataFrame) -> pd.Series:
    e = g["error"].dropna()
    sub = g.dropna(subset=["predicted", "truth"])
    r = float(sub["predicted"].corr(sub["truth"])) if len(sub) >= 3 else np.nan
    return pd.Series({"n": int(len(g)), "MAE": float(e.abs().mean()) if len(e) else np.nan, "pearson_r": r})


def _dx_metrics(g: pd.DataFrame) -> pd.Series:
    c = g["correct"].dropna()
    return pd.Series({"n": int(len(g)), "accuracy": float(c.mean()) if len(c) else np.nan})


def performance_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per-tier and per-provider tables for regression and (if present) diagnosis."""
    out: Dict[str, pd.DataFrame] = {}
    reg, dx = df[df.kind == "reg"], df[df.kind == "dx"]
    if not reg.empty:
        out["regression_by_tier"] = reg.groupby("tier", sort=True).apply(_reg_metrics).round(3)
        out["regression_by_provider"] = reg.groupby("model", sort=True).apply(_reg_metrics).round(3)
    if not dx.empty:
        out["diagnosis_by_tier"] = dx.groupby("tier", sort=True).apply(_dx_metrics).round(3)
        out["diagnosis_by_provider"] = dx.groupby("model", sort=True).apply(_dx_metrics).round(3)
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

    The metric is diagnosis accuracy when a classification label is present, otherwise
    regression MAE. The third panel always shows predicted vs truth for the regression
    outputs (colored by tier), with the identity line.
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
        _bar(ax[0], tables.get("regression_by_tier"), "MAE", "MAE (native units)",
             f"{dkey}: regression MAE by tier")
        _bar(ax[1], tables.get("regression_by_provider"), "MAE", "MAE (native units)",
             "regression MAE by provider")

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
