"""Rich performance and run-procedure analysis for the OpenNeuro full validation.

This module reads ONLY the already-saved canonical artifacts of the completed
seeded validation run (``predictions.json`` and ``summary.json`` under a results
directory). It never launches the engine and never contacts a provider. Its job
is to turn the 453 accepted job records into publication-style figures and rich
metric tables so the notebook can present, per dataset and per target phenotype:

  - association metrics that survive differing native scales: Pearson r (with a
    Fisher 95% interval), Spearman rho, and Kendall tau,
  - calibration metrics: the coefficient of determination (R^2, 1 - SS_res/SS_tot,
    which can be negative when raw predictions are biased or mis-scaled), the OLS
    slope, plus MAE, RMSE, normalized RMSE and bias for completeness,
  - predicted-vs-truth panels per output with the identity and OLS-fit lines,
  - Pearson-r heatmaps over output x tier and output x provider,
  - binary-diagnosis quality (accuracy, balanced accuracy, sensitivity,
    specificity, AUROC, F1, MCC and the confusion matrix) for the psychosis task,
  - run-procedure descriptives (execution time, whole-pipeline attempts, and the
    self-correction/recovery loop that made every job structurally valid), and
  - the supplementary question of whether run cost (time, attempts) predicts
    accuracy, answered by correlating per-subject effort with per-subject error.

The provider axis is a supplementary cross-provider check: because the seeded
design balances the five providers within every tier, a provider difference is
read at matched data complexity, not confounded with it. Nothing here is a
within-subject ablation, so the module quantifies and draws; it does not claim
causal modality or provider effects.

No tool-call trace was serialized for this run, so run-procedure analysis uses
the fields that were persisted: ``attempts``, typed ``attempt_errors``,
``total_seconds`` and the per-provider execution table. That limit is stated
wherever it matters rather than filled with a proxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import kendalltau, pearsonr, spearmanr

from . import batch_report as BR

# --- stable presentation order and colors for the five-provider panel ----------
PROVIDER_ORDER = [
    "deepseek/deepseek-v4-flash",
    "poolside/laguna-xs-2.1",
    "nex-agi/nex-n2-mini",
    "qwen/qwen3.5-9b",
    "openai/gpt-oss-120b",
]
PROVIDER_COLORS = {
    "deepseek-v4-flash": "#4e79a7",
    "laguna-xs-2.1": "#59a14f",
    "nex-n2-mini": "#e15759",
    "qwen3.5-9b": "#f28e2b",
    "gpt-oss-120b": "#b07aa1",
}
TIER_COLORS = ["#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#b07aa1", "#76b7b2"]
_POS = "#2a7fff"   # calibrated / good direction
_NEG = "#e15759"   # miscalibrated / bad direction

DATASET_TITLE = {
    "INTELLIGENCE": "INTELLIGENCE  (AOMIC ID1000, ds003097)",
    "PSYCHOSIS": "PSYCHOSIS  (first-episode, ds003944 + ds003947)",
    "NUMERACY": "NUMERACY  (numeracy after stroke, ds006533)",
}

# Compact, human-readable labels for the hierarchical phenotype outputs.
OUTPUT_LABELS = {
    "IST_intelligence_total": "Intelligence total",
    "IST_fluid": "Fluid",
    "IST_memory": "Memory",
    "IST_crystallised": "Crystallised",
    "approximate_numeracy": "Approximate numeracy",
    "precise_numeracy": "Precise numeracy",
}


def _short(model: Optional[str]) -> str:
    return "" if model is None else str(model).split("/")[-1]


def pretty_output(name: str) -> str:
    if name in OUTPUT_LABELS:
        return OUTPUT_LABELS[name]
    stub = str(name).split("__")[-1]
    for prefix in ("BPRS_", "SAPS_", "SANS_"):
        if stub.startswith(prefix):
            scale = prefix.rstrip("_")
            return f"{scale} {stub[len(prefix):].replace('_', ' ')}"
    return stub.replace("_", " ")


def pretty_tier(name: str) -> str:
    stub = str(name)
    if "_" in stub and stub[0] == "T" and stub[1].isdigit():
        head, _, rest = stub.partition("_")
        return f"{head} {rest.replace('_', ' ')}"
    return stub.replace("_", " ")


def provider_color(short_model: str) -> str:
    return PROVIDER_COLORS.get(short_model, "#8c8c8c")


def ordered_providers(present: Sequence[str]) -> List[str]:
    """Providers in panel order, restricted to those present in the data."""
    short_present = list(dict.fromkeys(present))
    ordered = [_short(m) for m in PROVIDER_ORDER if _short(m) in short_present]
    ordered += [m for m in short_present if m not in ordered]
    return ordered


# ---------------------------------------------------------------------------
# Loading and framing
# ---------------------------------------------------------------------------
def load_canonical(results_dir) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (prediction rows, summary dict) from a saved results directory."""
    results_dir = Path(results_dir)
    preds = json.loads((results_dir / "predictions.json").read_text())
    rows = preds["predictions"] if isinstance(preds, dict) else preds
    summary_path = results_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return rows, summary


def job_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per prediction job, with execution and self-correction fields."""

    def _err_types(row: Dict[str, Any]) -> List[str]:
        return [_classify_error(a.get("error")) for a in (row.get("attempt_errors") or [])]

    recs = []
    for r in rows or []:
        errors = _err_types(r)
        recs.append(
            dict(
                dataset=r.get("dataset"),
                task=r.get("task"),
                tier=r.get("tier"),
                subject=r.get("subject"),
                model=_short(r.get("model")),
                ok=bool(r.get("ok")),
                attempts=int(r.get("attempts", 1) or 1),
                seconds=float(r.get("total_seconds", r.get("seconds", np.nan))),
                n_prior_failures=len(r.get("attempt_errors") or []),
                recovered=len(r.get("attempt_errors") or []) > 0,
                error_types=errors,
            )
        )
    return pd.DataFrame(recs)


def long_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per predicted output (delegates to the shared tidy builder)."""
    return BR.tidy_results(rows)


def _classify_error(text: Optional[str]) -> str:
    text = str(text or "")
    if "Missing regression output" in text or "omitted a required" in text:
        return "predictor: missing output"
    if "Orchestrator plan" in text or "no steps" in text:
        return "orchestrator: empty plan"
    if "off-scale" in text or "outside" in text:
        return "validator: off-scale value"
    if "non-finite" in text or "missing/non-finite" in text:
        return "validator: non-finite value"
    if "timeout" in text.lower() or "timed out" in text.lower():
        return "watchdog: timeout"
    if "invalid classification" in text:
        return "validator: invalid label"
    return "other"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _pearson_ci(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    n = len(x)
    if n < 4:
        return (np.nan, np.nan, np.nan)
    r = float(pearsonr(x, y)[0])
    if not np.isfinite(r) or abs(r) >= 1:
        return (r, np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return (r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))


def rich_regression_metrics(g: pd.DataFrame) -> pd.Series:
    """Association + calibration + error metrics for one output/tier/provider cell."""
    sub = g.dropna(subset=["predicted", "truth"])
    x = sub["truth"].to_numpy(dtype=float)
    y = sub["predicted"].to_numpy(dtype=float)
    e = (y - x)
    n = len(sub)
    has_var = n >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1
    r, r_lo, r_hi = _pearson_ci(x, y) if has_var else (np.nan, np.nan, np.nan)
    rho = float(spearmanr(x, y)[0]) if has_var else np.nan
    tau = float(kendalltau(x, y)[0]) if has_var else np.nan
    ss_tot = float(np.sum((x - x.mean()) ** 2)) if n else np.nan
    r2_cod = 1.0 - float(np.sum(e ** 2)) / ss_tot if ss_tot and ss_tot > 0 else np.nan
    slope = np.polyfit(x, y, 1)[0] if has_var else np.nan
    sd_truth = float(np.std(x, ddof=0)) if n else np.nan
    rmse = float(np.sqrt(np.mean(e ** 2))) if n else np.nan
    return pd.Series(
        {
            "n": int(n),
            "pearson_r": r,
            "r_lo": r_lo,
            "r_hi": r_hi,
            "spearman_rho": rho,
            "kendall_tau": tau,
            "R2": r2_cod,
            "r_squared": r ** 2 if np.isfinite(r) else np.nan,
            "slope": float(slope) if np.isfinite(slope) else np.nan,
            "MAE": float(np.mean(np.abs(e))) if n else np.nan,
            "RMSE": rmse,
            "nRMSE": rmse / sd_truth if sd_truth and sd_truth > 0 else np.nan,
            "bias": float(np.mean(e)) if n else np.nan,
        }
    )


def regression_by(long: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    reg = long[long.kind == "reg"].dropna(subset=["predicted", "truth"])
    if reg.empty:
        return pd.DataFrame()
    try:
        out = reg.groupby(list(by), sort=True).apply(rich_regression_metrics, include_groups=False)
    except TypeError:  # pandas < 2.2
        out = reg.groupby(list(by), sort=True).apply(rich_regression_metrics)
    return out.round(3)


def diagnosis_metrics(dx: pd.DataFrame) -> Dict[str, Any]:
    """Binary-diagnosis quality plus the 2x2 confusion counts."""
    valid = dx.dropna(subset=["true_case", "pred_case"])
    t = valid["true_case"].to_numpy()
    p = valid["pred_case"].to_numpy()
    tp = float(np.sum((t == 1) & (p == 1)))
    tn = float(np.sum((t == 0) & (p == 0)))
    fp = float(np.sum((t == 0) & (p == 1)))
    fn = float(np.sum((t == 1) & (p == 0)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    acc = (tp + tn) / len(valid) if len(valid) else np.nan
    bal = np.nanmean([sens, spec]) if (np.isfinite(sens) or np.isfinite(spec)) else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and (prec + sens) else np.nan
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else np.nan
    auroc = BR._binary_auroc(valid["true_case"], valid["p_case"])
    return dict(
        n=int(len(valid)), accuracy=acc, balanced_accuracy=bal, sensitivity=sens,
        specificity=spec, precision=prec, f1=f1, mcc=mcc, auroc=auroc,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def diagnosis_by(dx: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in dx.groupby(by, sort=True):
        m = diagnosis_metrics(g)
        m[by] = key
        rows.append(m)
    return pd.DataFrame(rows).set_index(by) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def _annotate_bars(ax, values, fmt="{:.2f}", fontsize=8, dy=0.0):
    for i, v in enumerate(values):
        if v is not None and np.isfinite(v):
            ax.text(i, v + dy, fmt.format(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=fontsize)


def _heatmap(ax, mat: pd.DataFrame, title: str, vmin=-1, vmax=1, cmap="RdBu_r",
             fmt="{:.2f}", show_ylabels=True):
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(list(mat.index) if show_ylabels else [], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.55 else "black")
    ax.set_title(title, fontsize=9)
    return im


# ---------------------------------------------------------------------------
# Per-dataset performance composite
# ---------------------------------------------------------------------------
def fig_dataset_performance(
    dkey: str, long: pd.DataFrame, out_dir: Optional[Path] = None, show: bool = True
) -> plt.Figure:
    """One rich composite per dataset: predicted-vs-truth panels, r heatmaps, macro bars."""
    reg = long[long.kind == "reg"].dropna(subset=["predicted", "truth"]).copy()
    outputs = [o for o in _output_order(dkey, reg)]
    k = len(outputs)
    tiers = sorted(reg["tier"].unique())
    providers = ordered_providers(reg["model"].unique())

    scat_cols = min(k, 4)
    scat_rows = int(np.ceil(k / scat_cols))
    fig = plt.figure(figsize=(4.7 * scat_cols, 3.5 * scat_rows + 6.6))
    gs = GridSpec(
        scat_rows + 2, max(scat_cols, 2), figure=fig,
        height_ratios=[3.4] * scat_rows + [4.0, 3.0], hspace=0.62, wspace=0.32,
    )

    tier_color = {t: TIER_COLORS[i % len(TIER_COLORS)] for i, t in enumerate(tiers)}
    for idx, out in enumerate(outputs):
        ax = fig.add_subplot(gs[idx // scat_cols, idx % scat_cols])
        s = reg[reg.output == out]
        for t in tiers:
            st = s[s.tier == t]
            ax.scatter(st["truth"], st["predicted"], s=26, alpha=0.78,
                       color=tier_color[t], edgecolor="none", label=pretty_tier(t))
        xv = s["truth"].to_numpy(float)
        yv = s["predicted"].to_numpy(float)
        lo = float(np.nanmin([xv.min(), yv.min()]))
        hi = float(np.nanmax([xv.max(), yv.max()]))
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.6)
        if np.unique(xv).size > 1:
            b, a = np.polyfit(xv, yv, 1)
            xs = np.array([lo - pad, hi + pad])
            ax.plot(xs, b * xs + a, color="#d1495b", lw=1.6, alpha=0.9)
        m = rich_regression_metrics(s)
        ax.set_title(pretty_output(out), fontsize=9.5)
        ax.set_xlabel("true", fontsize=8)
        ax.set_ylabel("predicted", fontsize=8)
        ax.tick_params(labelsize=7)
        txt = (f"r = {m['pearson_r']:.2f}  [{m['r_lo']:.2f}, {m['r_hi']:.2f}]\n"
               f"rho = {m['spearman_rho']:.2f}   tau = {m['kendall_tau']:.2f}\n"
               f"R2 = {m['R2']:.2f}   slope = {m['slope']:.2f}\n"
               f"n = {int(m['n'])}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    # single shared tier legend, framed, in the first scatter's upper-right corner
    # (the metrics annotation box sits in the upper-left, so they never collide)
    first = fig.axes[0]
    first.legend(fontsize=6.6, frameon=True, framealpha=0.9, edgecolor="#cccccc",
                 loc="upper right", title="tier", title_fontsize=6.8)

    # r heatmaps: output x tier and output x provider (the provider heatmap reuses
    # the tier heatmap's output rows, so its y-labels are suppressed to avoid overlap)
    r_tier = _pivot_metric(reg, "tier", "pearson_r", outputs, tiers)
    r_prov = _pivot_metric(reg, "model", "pearson_r", outputs, providers)
    ax_ht = fig.add_subplot(gs[scat_rows, 0])
    _heatmap(ax_ht, r_tier.rename(index=pretty_output, columns=pretty_tier),
             "Pearson r by output x tier")
    ax_hp = fig.add_subplot(gs[scat_rows, 1] if max(scat_cols, 2) > 1 else gs[scat_rows, 0])
    im = _heatmap(ax_hp, r_prov, "Pearson r by output x provider", show_ylabels=False)
    fig.colorbar(im, ax=ax_hp, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    # macro association by tier and provider (mean over outputs of r and rho)
    ax_bt = fig.add_subplot(gs[scat_rows + 1, 0])
    _macro_assoc_bars(ax_bt, reg, "tier", tiers, pretty_tier, "macro association by tier")
    ax_bp = fig.add_subplot(gs[scat_rows + 1, 1] if max(scat_cols, 2) > 1 else gs[scat_rows + 1, 0])
    _macro_assoc_bars(ax_bp, reg, "model", providers, lambda s: s, "macro association by provider")

    fig.suptitle(f"{DATASET_TITLE.get(dkey, dkey)} - predictive performance across phenotype outputs, "
                 f"tiers and providers", fontsize=12.5, y=0.997)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dkey.lower()}_performance.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _macro_assoc_bars(ax, reg, by, order, labeler, title):
    tab = regression_by(reg, [by, "output"])
    if tab.empty:
        ax.axis("off"); ax.set_title(title, fontsize=9); return
    macro = tab.groupby(level=0)[["pearson_r", "spearman_rho"]].mean().reindex(order)
    xs = np.arange(len(macro))
    ax.bar(xs - 0.19, macro["pearson_r"].values, width=0.36, color="#4e79a7", label="mean r")
    ax.bar(xs + 0.19, macro["spearman_rho"].values, width=0.36, color="#f28e2b", label="mean rho")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([labeler(o) for o in macro.index], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean over outputs", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.8, frameon=False, ncol=2)
    ax.tick_params(labelsize=7)


def _pivot_metric(reg, by, metric, outputs, order) -> pd.DataFrame:
    tab = regression_by(reg, ["output", by])
    if tab.empty:
        return pd.DataFrame(index=outputs, columns=order, dtype=float)
    mat = tab[metric].unstack(by)
    return mat.reindex(index=outputs, columns=order)


def _output_order(dkey: str, reg: pd.DataFrame) -> List[str]:
    present = list(dict.fromkeys(reg["output"].tolist()))
    preferred = [
        "IST_intelligence_total", "IST_fluid", "IST_memory", "IST_crystallised",
        "approximate_numeracy", "precise_numeracy",
    ]
    ordered = [o for o in preferred if o in present]
    ordered += [o for o in present if o not in ordered]
    return ordered


# ---------------------------------------------------------------------------
# Diagnosis composite (psychosis)
# ---------------------------------------------------------------------------
def fig_diagnosis(dkey: str, long: pd.DataFrame, out_dir: Optional[Path] = None,
                  show: bool = True) -> Optional[plt.Figure]:
    dx = long[long.kind == "dx"].dropna(subset=["true_case", "pred_case"]).copy()
    if dx.empty:
        return None
    overall = diagnosis_metrics(dx)
    providers = ordered_providers(dx["model"].unique())
    tiers = sorted(dx["tier"].unique())

    fig = plt.figure(figsize=(16, 4.4))
    gs = GridSpec(1, 4, figure=fig, wspace=0.42, width_ratios=[1.0, 1.25, 1.35, 1.35])

    # confusion matrix
    ax0 = fig.add_subplot(gs[0, 0])
    cm = np.array([[overall["tn"], overall["fp"]], [overall["fn"], overall["tp"]]])
    ax0.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax0.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax0.set_xticks([0, 1]); ax0.set_xticklabels(["pred control", "pred case"], fontsize=8)
    ax0.set_yticks([0, 1]); ax0.set_yticklabels(["true control", "true case"], fontsize=8)
    ax0.set_title(f"Confusion (n={overall['n']})", fontsize=9)

    # headline metrics
    ax1 = fig.add_subplot(gs[0, 1])
    names = ["accuracy", "balanced\naccuracy", "sensitivity", "specificity", "AUROC", "MCC", "F1"]
    vals = [overall["accuracy"], overall["balanced_accuracy"], overall["sensitivity"],
            overall["specificity"], overall["auroc"], overall["mcc"], overall["f1"]]
    colors = [_POS if (np.isfinite(v) and v >= 0.5) else _NEG for v in vals]
    ax1.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax1.axhline(0.5, color="#888", ls=":", lw=0.9)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax1.set_ylim(min(0, np.nanmin(vals)) - 0.05, 1.0)
    _annotate_bars(ax1, vals, fmt="{:.2f}", fontsize=7)
    ax1.set_title("Diagnosis quality (overall)", fontsize=9)

    # balanced accuracy + AUROC by tier
    bt = diagnosis_by(dx, "tier").reindex(tiers)
    ax2 = fig.add_subplot(gs[0, 2])
    xs = np.arange(len(bt))
    ax2.bar(xs - 0.19, bt["balanced_accuracy"].values, width=0.36, color="#4e79a7", label="balanced acc")
    ax2.bar(xs + 0.19, bt["auroc"].values, width=0.36, color="#59a14f", label="AUROC")
    ax2.axhline(0.5, color="#888", ls=":", lw=0.9)
    ax2.set_xticks(xs); ax2.set_xticklabels([pretty_tier(t) for t in bt.index], rotation=35, ha="right", fontsize=7)
    ax2.set_ylim(0, 1); ax2.legend(fontsize=7, frameon=False); ax2.set_title("Diagnosis by tier", fontsize=9)

    # balanced accuracy + AUROC by provider
    bp = diagnosis_by(dx, "model").reindex(providers)
    ax3 = fig.add_subplot(gs[0, 3])
    xs = np.arange(len(bp))
    ax3.bar(xs - 0.19, bp["balanced_accuracy"].values, width=0.36, color="#4e79a7", label="balanced acc")
    ax3.bar(xs + 0.19, bp["auroc"].values, width=0.36, color="#59a14f", label="AUROC")
    ax3.axhline(0.5, color="#888", ls=":", lw=0.9)
    ax3.set_xticks(xs); ax3.set_xticklabels(bp.index, rotation=35, ha="right", fontsize=7)
    ax3.set_ylim(0, 1); ax3.legend(fontsize=7, frameon=False); ax3.set_title("Diagnosis by provider", fontsize=9)

    fig.suptitle(f"{DATASET_TITLE.get(dkey, dkey)} - binary diagnosis discrimination", fontsize=12, y=1.03)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dkey.lower()}_diagnosis.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Run procedure and self-correction (critic-actor / recovery loop)
# ---------------------------------------------------------------------------
def execution_table(rows) -> pd.DataFrame:
    jobs = job_frame(rows)
    grp = jobs.groupby(["dataset", "model"])
    tab = grp.agg(
        n_jobs=("subject", "size"),
        median_seconds=("seconds", "median"),
        p95_seconds=("seconds", lambda s: float(np.nanpercentile(s, 95))),
        recovered_jobs=("recovered", "sum"),
        recorded_attempts=("attempts", "sum"),
    ).round(2)
    return tab


def fig_run_procedure(rows, summary: Dict[str, Any], out_dir: Optional[Path] = None,
                      show: bool = True) -> plt.Figure:
    """Descriptive multiplot of how the run executed and how the recovery loop behaved."""
    jobs = job_frame(rows)
    datasets = [d for d in ("INTELLIGENCE", "PSYCHOSIS", "NUMERACY") if d in set(jobs.dataset)]
    providers = ordered_providers(jobs["model"].unique())

    fig = plt.figure(figsize=(16, 9.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.3)

    # (1) median execution seconds by provider, grouped by dataset
    ax = fig.add_subplot(gs[0, 0])
    width = 0.8 / max(1, len(datasets))
    for di, d in enumerate(datasets):
        med = [jobs[(jobs.dataset == d) & (jobs.model == p)]["seconds"].median() for p in providers]
        ax.bar(np.arange(len(providers)) + di * width, med, width=width, label=d.title(),
               color=TIER_COLORS[di % len(TIER_COLORS)])
    ax.set_xticks(np.arange(len(providers)) + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(providers, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("median seconds / job"); ax.set_title("Execution time by provider", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # (2) execution-time distribution (all jobs) by provider
    ax = fig.add_subplot(gs[0, 1])
    data = [jobs[jobs.model == p]["seconds"].dropna().to_numpy() for p in providers]
    bp = ax.boxplot(data, vert=True, showfliers=False, patch_artist=True, widths=0.6)
    for patch, p in zip(bp["boxes"], providers):
        patch.set_facecolor(provider_color(p)); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks(range(1, len(providers) + 1)); ax.set_xticklabels(providers, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("seconds / job (all datasets)"); ax.set_title("Execution-time spread by provider", fontsize=10)

    # (3) whole-pipeline attempts distribution
    ax = fig.add_subplot(gs[0, 2])
    counts = jobs["attempts"].value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values, color="#4e79a7", alpha=0.85)
    for a, c in zip(counts.index.astype(int), counts.values):
        ax.text(a, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(counts.index.astype(int))
    ax.set_xlabel("recorded attempts"); ax.set_ylabel("jobs")
    n_rec = int(jobs["recovered"].sum())
    ax.set_title(f"Attempts per job ({n_rec}/{len(jobs)} used the recovery loop)", fontsize=10)

    # (4) recovered jobs by provider (stacked by dataset)
    ax = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(providers))
    for di, d in enumerate(datasets):
        vals = [int(jobs[(jobs.dataset == d) & (jobs.model == p)]["recovered"].sum()) for p in providers]
        ax.bar(providers, vals, bottom=bottom, label=d.title(), color=TIER_COLORS[di % len(TIER_COLORS)])
        bottom += np.array(vals)
    ax.set_ylabel("jobs that entered recovery"); ax.set_title("Self-correction load by provider", fontsize=10)
    ax.set_xticklabels(providers, rotation=40, ha="right", fontsize=7); ax.legend(fontsize=7, frameon=False)

    # (5) recovered-error taxonomy
    ax = fig.add_subplot(gs[1, 1])
    tax: Dict[str, int] = {}
    for types in jobs["error_types"]:
        for t in types:
            tax[t] = tax.get(t, 0) + 1
    tax = dict(sorted(tax.items(), key=lambda kv: kv[1], reverse=True))
    if tax:
        ax.barh(list(tax.keys())[::-1], list(tax.values())[::-1], color="#e15759", alpha=0.85)
        for i, v in enumerate(list(tax.values())[::-1]):
            ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_xlabel("recovered attempts"); ax.set_title("What the recovery loop caught", fontsize=10)
    ax.tick_params(labelsize=7)

    # (6) validity funnel: attempted -> valid, with the one discarded off-scale value
    ax = fig.add_subplot(gs[1, 2])
    total_attempts = int(jobs["attempts"].sum())
    n_jobs = len(jobs)
    n_first_ok = int((jobs["attempts"] == 1).sum())
    discarded = len(summary.get("discarded_invalid_predictions", []) or [])
    labels = ["total\nattempts", "valid on\nfirst try", "recovered\nby loop", "final valid\njobs"]
    vals = [total_attempts, n_first_ok, n_jobs - n_first_ok, n_jobs]
    ax.bar(range(4), vals, color=["#9c9c9c", "#59a14f", "#f28e2b", "#4e79a7"], alpha=0.88)
    _annotate_bars(ax, vals, fmt="{:.0f}", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"Validity funnel ({n_jobs}/{n_jobs} valid; {discarded} off-scale discarded+rerun)", fontsize=9.5)

    fig.suptitle("Run procedure and self-correction (critic-actor recovery) loop", fontsize=13, y=0.98)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "run_procedure.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _subject_effort_error(rows) -> pd.DataFrame:
    """Per (dataset, subject, task) effort (seconds, attempts) vs normalized abs error / correctness."""
    long = long_frame(rows)
    reg = long[long.kind == "reg"].dropna(subset=["error", "truth"]).copy()
    sd = reg.groupby(["dataset", "output"])["truth"].transform(lambda v: v.std(ddof=0))
    reg["abs_z_error"] = reg["error"].abs() / sd.where(sd > 0)
    reg_g = reg.groupby(["dataset", "subject", "task"]).agg(
        abs_z_error=("abs_z_error", "mean"), seconds=("seconds", "first"),
        attempts=("attempts", "first"), model=("model", "first"), tier=("tier", "first"),
    ).reset_index()
    dx = long[long.kind == "dx"].dropna(subset=["correct"]).copy()
    dx_g = dx.groupby(["dataset", "subject", "task"]).agg(
        correct=("correct", "first"), seconds=("seconds", "first"),
        attempts=("attempts", "first"), model=("model", "first"), tier=("tier", "first"),
    ).reset_index()
    return reg_g, dx_g


def fig_procedure_vs_performance(rows, out_dir: Optional[Path] = None, show: bool = True) -> plt.Figure:
    """Supplementary: does run cost (time / attempts) predict accuracy?"""
    reg_g, dx_g = _subject_effort_error(rows)
    datasets = [d for d in ("INTELLIGENCE", "PSYCHOSIS", "NUMERACY") if d in set(reg_g.dataset)]

    fig = plt.figure(figsize=(16, 8.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32)

    # top row: seconds vs mean |z error| per subject, one panel per dataset
    for di, d in enumerate(datasets):
        ax = fig.add_subplot(gs[0, di])
        s = reg_g[reg_g.dataset == d].dropna(subset=["seconds", "abs_z_error"])
        for p in ordered_providers(s["model"].unique()):
            sp = s[s.model == p]
            ax.scatter(sp["seconds"], sp["abs_z_error"], s=24, alpha=0.75,
                       color=provider_color(p), label=p, edgecolor="none")
        if len(s) >= 4 and s["seconds"].nunique() > 1:
            r, pval = pearsonr(s["seconds"], s["abs_z_error"])
            b, a = np.polyfit(s["seconds"], s["abs_z_error"], 1)
            xs = np.array([s["seconds"].min(), s["seconds"].max()])
            ax.plot(xs, b * xs + a, color="#d1495b", lw=1.5)
            ax.text(0.03, 0.97, f"r = {r:.2f}\np = {pval:.2f}\nn = {len(s)}", transform=ax.transAxes,
                    va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))
        ax.set_title(f"{d.title()}: effort vs error", fontsize=9.5)
        ax.set_xlabel("seconds / job"); ax.set_ylabel("mean |z error| per subject", fontsize=8)
        if di == 0:
            ax.legend(fontsize=6.2, frameon=False, loc="upper right")

    # bottom row: error (or accuracy) by attempts group, per dataset
    for di, d in enumerate(datasets):
        ax = fig.add_subplot(gs[1, di])
        s = reg_g[reg_g.dataset == d].dropna(subset=["abs_z_error"]).copy()
        s["attempt_grp"] = np.where(s["attempts"] > 1, "recovered\n(>=2 attempts)", "first try")
        groups = ["first try", "recovered\n(>=2 attempts)"]
        data = [s[s.attempt_grp == g]["abs_z_error"].to_numpy() for g in groups]
        ns = [len(x) for x in data]
        if any(ns):
            bp = ax.boxplot([x for x in data if len(x)], labels=[g for g, x in zip(groups, data) if len(x)],
                            showfliers=False, patch_artist=True, widths=0.55)
            for patch in bp["boxes"]:
                patch.set_facecolor("#4e79a7"); patch.set_alpha(0.6)
            means = [np.nanmean(x) if len(x) else np.nan for x in data if len(x)]
            for i, mv in enumerate(means):
                ax.scatter(i + 1, mv, color="#d1495b", zorder=5, s=30)
        ax.set_title(f"{d.title()}: error by attempt count", fontsize=9.5)
        ax.set_ylabel("mean |z error| per subject", fontsize=8)
        ax.tick_params(labelsize=7.5)

    fig.suptitle("Supplementary: does run cost (execution time, retries) predict accuracy?", fontsize=12.5, y=0.98)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "procedure_vs_performance.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Cross-provider composite (one figure across all datasets)
# ---------------------------------------------------------------------------
def fig_cross_provider(rows, out_dir: Optional[Path] = None, show: bool = True) -> plt.Figure:
    """Cross-provider view at matched data complexity: macro association + timing + recovery."""
    long = long_frame(rows)
    jobs = job_frame(rows)
    datasets = [d for d in ("INTELLIGENCE", "PSYCHOSIS", "NUMERACY") if d in set(jobs.dataset)]
    providers = ordered_providers(jobs["model"].unique())

    fig = plt.figure(figsize=(16, 8.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.32)

    # top: macro Pearson r by provider, one panel per dataset (regression outputs)
    for di, d in enumerate(datasets):
        ax = fig.add_subplot(gs[0, di])
        reg = long[(long.kind == "reg") & (long.dataset == d)]
        tab = regression_by(reg, ["model", "output"])
        if not tab.empty:
            macro = tab.groupby(level=0)[["pearson_r", "spearman_rho"]].mean().reindex(providers)
            xs = np.arange(len(macro))
            ax.bar(xs - 0.19, macro["pearson_r"].values, width=0.36, color="#4e79a7", label="mean r")
            ax.bar(xs + 0.19, macro["spearman_rho"].values, width=0.36, color="#f28e2b", label="mean rho")
            ax.axhline(0, color="#888", lw=0.8)
            ax.set_xticks(xs); ax.set_xticklabels(macro.index, rotation=40, ha="right", fontsize=7)
            ax.legend(fontsize=6.6, frameon=False, ncol=2)
        ax.set_title(f"{d.title()}: macro association by provider", fontsize=9.5)
        ax.set_ylabel("mean over outputs", fontsize=8)

    # bottom-left: diagnosis balanced accuracy + AUROC by provider (psychosis)
    ax = fig.add_subplot(gs[1, 0])
    dx = long[long.kind == "dx"].dropna(subset=["true_case", "pred_case"])
    if not dx.empty:
        bp = diagnosis_by(dx, "model").reindex(providers)
        xs = np.arange(len(bp))
        ax.bar(xs - 0.19, bp["balanced_accuracy"].values, width=0.36, color="#59a14f", label="balanced acc")
        ax.bar(xs + 0.19, bp["auroc"].values, width=0.36, color="#4e79a7", label="AUROC")
        ax.axhline(0.5, color="#888", ls=":", lw=0.9)
        ax.set_xticks(xs); ax.set_xticklabels(bp.index, rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 1); ax.legend(fontsize=6.8, frameon=False)
    ax.set_title("Psychosis diagnosis by provider", fontsize=9.5)

    # bottom-middle: median seconds by provider (all datasets)
    ax = fig.add_subplot(gs[1, 1])
    med = [jobs[jobs.model == p]["seconds"].median() for p in providers]
    ax.bar(providers, med, color=[provider_color(p) for p in providers], alpha=0.85)
    _annotate_bars(ax, med, fmt="{:.0f}", fontsize=8)
    ax.set_xticklabels(providers, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("median seconds / job"); ax.set_title("Median execution time by provider", fontsize=9.5)

    # bottom-right: recovery rate by provider
    ax = fig.add_subplot(gs[1, 2])
    rate = [100.0 * jobs[jobs.model == p]["recovered"].mean() for p in providers]
    ax.bar(providers, rate, color=[provider_color(p) for p in providers], alpha=0.85)
    _annotate_bars(ax, rate, fmt="{:.0f}%", fontsize=8)
    ax.set_xticklabels(providers, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("percent of jobs recovered"); ax.set_title("Recovery-loop rate by provider", fontsize=9.5)

    fig.suptitle("Cross-provider comparison at matched data complexity (providers balanced within every tier)",
                 fontsize=12.5, y=0.98)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "cross_provider.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Table helpers for inline display
# ---------------------------------------------------------------------------
def overall_regression_table(long: pd.DataFrame) -> pd.DataFrame:
    reg = long[long.kind == "reg"]
    tab = regression_by(reg, ["dataset", "output"])
    if tab.empty:
        return tab
    keep = ["n", "pearson_r", "r_lo", "r_hi", "spearman_rho", "kendall_tau", "R2", "slope", "MAE", "RMSE", "nRMSE", "bias"]
    return tab[keep]


def macro_association(long: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean-over-outputs association and calibration, one row per tier or provider.

    Correlation-based metrics (r, rho, tau) and the coefficient of determination
    are scale-free, so averaging them across differently scaled phenotype outputs
    is well defined; that is what makes this a fair macro summary.
    """
    reg = long[long.kind == "reg"]
    tab = regression_by(reg, [by, "output"])
    if tab.empty:
        return tab
    macro = tab.groupby(level=0)[["pearson_r", "spearman_rho", "kendall_tau", "R2"]].mean()
    macro["n_outputs"] = tab.groupby(level=0).size()
    return macro.round(3)
