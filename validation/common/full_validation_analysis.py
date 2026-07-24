"""Rich performance and run-procedure analysis for the OpenNeuro full validation.

This module reads ONLY the already-saved canonical artifacts of the completed
seeded validation run (``predictions.json`` and ``summary.json`` under a results
directory). It never launches the engine and never contacts a provider. Its job
is to turn the 453 accepted job records into rich metric tables and figures so
the notebook can present, per dataset and per target phenotype:

  - association metrics that survive differing native scales: Pearson r (with a
    Fisher 95% interval), Spearman rho, and Kendall tau,
  - calibration metrics: the coefficient of determination (R^2, 1 - SS_res/SS_tot,
    which can be negative when raw predictions are biased or mis-scaled), the OLS
    slope, plus MAE, RMSE, normalized RMSE and bias for completeness,
  - predicted-vs-truth panels per output with the identity line, a pooled OLS
    fit, and a per-tier linear (Pearson) fit plus a per-tier monotonic
    (Spearman-flavored, isotonic) fit, so a tier's ranking signal and its
    linear signal can be told apart even when they diverge,
  - Pearson-r heatmaps over output x tier and output x provider,
  - binary-diagnosis quality (accuracy, balanced accuracy, sensitivity,
    specificity, AUROC, F1, MCC and the confusion matrix) for the psychosis task,
  - run-procedure descriptives (execution time, whole-pipeline attempts, and the
    self-correction/recovery loop that made every job structurally valid), and
  - the supplementary question of whether run cost (time, attempts) predicts
    accuracy, answered by correlating per-subject effort with per-subject error.

Design choices worth stating up front:

  - Figures never draw their own main title. A baked-in title duplicates the
    notebook markdown that already introduces each figure, and it survives
    stale in a saved PNG if the surrounding prose changes. Every subplot still
    carries its own descriptive title.
  - Bar charts over tiers or providers color each bar with that tier's or
    provider's own, figure-to-figure-consistent color (the same color used for
    that tier/provider everywhere else), instead of one flat color for every
    bar with only the metric distinguished by hue.
  - The ten psychosis symptom outputs are noisy, low-n (79 subjects with ground
    truth) and inconsistent in sign, so their tier/provider heatmaps and macro
    bars use absolute Pearson r (how much association exists at all) rather
    than signed r (which direction). Sign is never discarded outright: the
    headline per-output table keeps signed r with its 95% interval, and the
    forest plot (section 9.4) marks each output's sign with a +/- prefix.
  - Multiplot complexity is scaled to how much there is to show: intelligence
    (4 outputs) and numeracy (2 outputs) get a full predicted-vs-truth grid;
    psychosis (10 outputs, most individually non-significant) gets a compact
    forest plot of all ten plus full scatter detail only for the outputs whose
    95% interval excludes zero, instead of ten mostly-uninformative panels.
    Every dataset's per-dataset composite is split into exactly two uniformly
    gridded figures (``predictions`` and ``association``) rather than one
    figure with a ragged row/column count, which the earlier single-composite
    design produced whenever a dataset had more scatter panels than the two
    columns used by its heatmaps and bars.
  - In every predicted-vs-truth panel, a tier's linear fit is solid and its
    monotonic (isotonic) fit is dashed, both drawn in that tier's own color and
    restricted to that tier's own x-range; the pooled OLS fit (all tiers
    combined) is a thicker solid red line, and the identity reference is a
    thin grey dotted line, chosen specifically so it is never confused with a
    per-tier dashed line.

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import kendalltau, mannwhitneyu, pearsonr, spearmanr

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
FAMILY_COLORS = {"BPRS": "#b07aa1", "SAPS": "#e15759", "SANS": "#4e79a7", "other": "#888888"}
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


def _instrument_family(output: str) -> str:
    stub = str(output).lower()
    if "bprs" in stub:
        return "BPRS"
    if "saps" in stub:
        return "SAPS"
    if "sans" in stub:
        return "SANS"
    return "other"


def provider_color(short_model: str) -> str:
    return PROVIDER_COLORS.get(short_model, "#8c8c8c")


def ordered_providers(present: Sequence[str]) -> List[str]:
    """Providers in panel order, restricted to those present in the data."""
    short_present = list(dict.fromkeys(present))
    ordered = [_short(m) for m in PROVIDER_ORDER if _short(m) in short_present]
    ordered += [m for m in short_present if m not in ordered]
    return ordered


def _tier_color_map(tiers: Sequence[str]) -> Dict[str, str]:
    """Position-based tier color, stable across every figure for a given dataset."""
    return {t: TIER_COLORS[i % len(TIER_COLORS)] for i, t in enumerate(sorted(tiers))}


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


def _macro_r_by(reg: pd.DataFrame, by: str, order: Sequence[str], use_abs: bool = False) -> np.ndarray:
    """Mean Pearson r (or |r|) over outputs, one value per tier/provider, in `order`."""
    tab = regression_by(reg, [by, "output"])
    if tab.empty:
        return np.full(len(order), np.nan)
    col = tab["pearson_r"].abs() if use_abs else tab["pearson_r"]
    macro = col.groupby(level=0).mean().reindex(order)
    return macro.to_numpy()


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
    mid = vmin + 0.55 * (vmax - vmin)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                dark = (v > mid) if vmin >= 0 else (abs(v) > 0.55)
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "black")
    ax.set_title(title, fontsize=9)
    return im


def _category_bar_single(ax, categories: Sequence[str], values: np.ndarray,
                         color_of: Callable[[str], str], labeler: Callable[[str], str],
                         title: str, ylabel: str, use_abs: bool = False):
    """One bar per tier/provider, colored with that category's own established color."""
    colors = [color_of(c) for c in categories]
    xs = np.arange(len(categories))
    ax.bar(xs, values, color=colors, alpha=0.88)
    _annotate_bars(ax, values, fmt="{:.2f}")
    if use_abs:
        ax.set_ylim(0, 1)
    else:
        ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([labeler(c) for c in categories], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _isotonic_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Monotone least-squares fit via pool-adjacent-violators (direction from Spearman's sign).

    A dependency-free (pure numpy/scipy) isotonic regression: it fits the best
    non-decreasing (or non-increasing) step function through the points, so it
    reflects a rank-consistent trend without assuming linearity the way an OLS
    fit does. Used to draw each tier's Spearman-flavored line, visually distinct
    from that tier's linear (Pearson) fit line.
    """
    order = np.argsort(x)
    xs = x[order]
    ys = y[order].astype(float)
    rho = spearmanr(xs, ys)[0] if len(xs) >= 2 else 0.0
    direction = -1.0 if (np.isfinite(rho) and rho < 0) else 1.0
    levels = [[float(v) * direction, 1] for v in ys]
    i = 0
    while i < len(levels) - 1:
        if levels[i][0] > levels[i + 1][0] + 1e-12:
            wa, wb = levels[i][1], levels[i + 1][1]
            merged = (levels[i][0] * wa + levels[i + 1][0] * wb) / (wa + wb)
            levels[i:i + 2] = [[merged, wa + wb]]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted: List[float] = []
    for val, w in levels:
        fitted.extend([val] * int(w))
    y_fit = np.array(fitted) * direction
    return xs, y_fit


def _scatter_with_fit(ax, s: pd.DataFrame, tiers: Sequence[str],
                      tier_color_of: Callable[[str], str], title: str) -> pd.Series:
    """Predicted-vs-truth scatter (colored by tier) with identity, pooled-OLS, and
    per-tier trend lines, plus a pooled stats box.

    Each tier with enough points (n >= 3, more than one unique x) gets two lines
    in its own color: a solid linear (Pearson/OLS) fit and a dashed monotonic
    (Spearman-flavored, isotonic) fit, restricted to that tier's own x-range so
    neither line extrapolates beyond its data. The thick red line is the pooled
    OLS fit across all tiers combined; the grey dotted line is the identity
    reference (perfect prediction), kept dotted specifically so it is never
    confused with a per-tier dashed line.
    """
    for t in tiers:
        st = s[s.tier == t]
        if st.empty:
            continue
        color = tier_color_of(t)
        ax.scatter(st["truth"], st["predicted"], s=26, alpha=0.78,
                   color=color, edgecolor="none", label=pretty_tier(t))
        xt = st["truth"].to_numpy(float)
        yt = st["predicted"].to_numpy(float)
        if len(xt) >= 3 and np.unique(xt).size > 1:
            b, a = np.polyfit(xt, yt, 1)
            xs_lin = np.array([xt.min(), xt.max()])
            ax.plot(xs_lin, b * xs_lin + a, color=color, lw=1.3, alpha=0.85,
                    linestyle="-", zorder=2)
            xs_iso, y_iso = _isotonic_fit(xt, yt)
            ax.plot(xs_iso, y_iso, color=color, lw=1.3, alpha=0.85,
                    linestyle="--", zorder=2)
    xv = s["truth"].to_numpy(float)
    yv = s["predicted"].to_numpy(float)
    lo = float(np.nanmin([xv.min(), yv.min()]))
    hi = float(np.nanmax([xv.max(), yv.max()]))
    pad = 0.05 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#555555", lw=0.8,
            alpha=0.6, linestyle=":")
    if np.unique(xv).size > 1:
        b, a = np.polyfit(xv, yv, 1)
        xs = np.array([lo - pad, hi + pad])
        ax.plot(xs, b * xs + a, color="#d1495b", lw=1.8, alpha=0.9, linestyle="-", zorder=3)
    m = rich_regression_metrics(s)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("true", fontsize=8)
    ax.set_ylabel("predicted", fontsize=8)
    ax.tick_params(labelsize=7)
    txt = (f"r = {m['pearson_r']:.2f}  [{m['r_lo']:.2f}, {m['r_hi']:.2f}]\n"
           f"rho = {m['spearman_rho']:.2f}   tau = {m['kendall_tau']:.2f}\n"
           f"R2 = {m['R2']:.2f}   slope = {m['slope']:.2f}\n"
           f"n = {int(m['n'])}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    return m


def _forest_plot(ax, tab: pd.DataFrame, order: Sequence[str], title: str):
    """Horizontal bars of |Pearson r| per output, colored by clinical-instrument family.

    Bar length is absolute magnitude because many of these per-output correlations
    are noisy and inconsistent in sign at this sample size; a +/- prefix on each
    label and a solid outline (95% CI on the SIGNED r excludes zero) keep direction
    and statistical reliability visible without cluttering the geometry with sign.
    """
    tab = tab.reindex(order)
    abs_r = tab["pearson_r"].abs()
    plot_order = abs_r.sort_values(ascending=True).index
    y = np.arange(len(plot_order))
    vals = abs_r.reindex(plot_order).to_numpy()
    families = [_instrument_family(o) for o in plot_order]
    colors = [FAMILY_COLORS[f] for f in families]
    signif = [(tab.loc[o, "r_lo"] > 0) or (tab.loc[o, "r_hi"] < 0) for o in plot_order]
    edgecolors = ["black" if sig else "none" for sig in signif]
    linewidths = [1.3 if sig else 0.0 for sig in signif]
    ax.barh(y, vals, color=colors, edgecolor=edgecolors, linewidth=linewidths, alpha=0.9)
    for i, o in enumerate(plot_order):
        r = tab.loc[o, "pearson_r"]
        sign = "+" if r >= 0 else "-"
        star = " *" if signif[i] else ""
        ax.text(vals[i] + 0.015, i, f"{sign}{vals[i]:.2f}{star}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_output(o) for o in plot_order], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("|Pearson r|   (sign shown as +/-,  * = 95% CI excludes 0)", fontsize=7.5)
    ax.set_title(title, fontsize=9.5)
    present = [f for f in FAMILY_COLORS if f in set(families)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in present]
    ax.legend(handles, present, fontsize=6.8, frameon=False, loc="lower right", title="instrument",
              title_fontsize=7)


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
# Per-dataset performance composite: two uniformly-gridded figures per dataset
# ---------------------------------------------------------------------------
def fig_dataset_performance(
    dkey: str, long: pd.DataFrame, out_dir: Optional[Path] = None, show: bool = True
) -> Tuple[plt.Figure, plt.Figure]:
    """Predictive-performance figures for one dataset: (predictions, association).

    Each dataset gets exactly two figures, both with a uniform row x column grid
    (no row with more or fewer populated columns than another, which the earlier
    single-composite design produced whenever a dataset had more scatter panels
    than the two columns used by its heatmaps and bars):

      - ``predictions``: few outputs (intelligence: 4, numeracy: 2) get a full
        predicted-vs-truth grid at 2 columns (2x2 for 4 outputs, 1x2 for 2).
        Many outputs (psychosis: 10 symptom scores, mostly individually
        non-significant) get a compact forest plot of all of them in one row,
        plus full scatter detail only for the outputs whose 95% interval
        excludes zero, so panel count tracks how much there is to show rather
        than the raw output count.
      - ``association``: always the same clean 2x2 grid (heatmap by tier,
        heatmap by provider, macro bar by tier, macro bar by provider). Uses
        absolute Pearson r for psychosis (noisy, inconsistent sign at n=79)
        and signed r elsewhere (few, individually reliable outputs).
    """
    reg = long[long.kind == "reg"].dropna(subset=["predicted", "truth"]).copy()
    outputs = _output_order(dkey, reg)
    tiers = sorted(reg["tier"].unique())
    providers = ordered_providers(reg["model"].unique())
    use_abs = len(outputs) > 4

    if use_abs:
        fig_pred = _predictions_forest(dkey, reg, outputs, tiers, out_dir, show)
    else:
        fig_pred = _predictions_grid(dkey, reg, outputs, tiers, out_dir, show)
    fig_assoc = _association_grid(dkey, reg, outputs, tiers, providers, use_abs, out_dir, show)
    return fig_pred, fig_assoc


def _predictions_grid(dkey, reg, outputs, tiers, out_dir, show) -> plt.Figure:
    """Predicted-vs-truth panels for few (<=4) outputs, at a fixed 2-column grid."""
    k = len(outputs)
    cols = 2 if k > 1 else 1
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.6 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    tier_color = _tier_color_map(tiers)
    for idx, out in enumerate(outputs):
        s = reg[reg.output == out]
        _scatter_with_fit(axes[idx], s, tiers, lambda t: tier_color[t], pretty_output(out))
    for ax in axes[len(outputs):]:
        ax.axis("off")
    axes[0].legend(fontsize=6.8, frameon=True, framealpha=0.9, edgecolor="#cccccc",
                   loc="upper right", title="tier", title_fontsize=7.2)

    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dkey.lower()}_predictions.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _predictions_forest(dkey, reg, outputs, tiers, out_dir, show) -> plt.Figure:
    """Forest plot of all outputs (one row) plus scatter detail for significant ones."""
    tab = regression_by(reg, ["output"]).reindex(outputs)
    sig = tab[(tab.r_lo > 0) | (tab.r_hi < 0)].copy()
    sig["abs_r"] = sig["pearson_r"].abs()
    examples = sig.sort_values("abs_r", ascending=False).index.tolist()[:2]
    n_ex = len(examples)

    ncols = 1 + n_ex
    width_ratios = [1.6] + [1.0] * n_ex if n_ex else [1.0]
    fig, axes = plt.subplots(
        1, ncols, figsize=(6.8 + 4.8 * n_ex, 5.6),
        gridspec_kw=dict(width_ratios=width_ratios), constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    _forest_plot(axes[0], tab, outputs, f"Association with truth, all {len(outputs)} outputs")

    tier_color = _tier_color_map(tiers)
    for j, out in enumerate(examples):
        ax = axes[1 + j]
        s = reg[reg.output == out]
        _scatter_with_fit(ax, s, tiers, lambda t: tier_color[t], pretty_output(out))
        if j == 0:
            ax.legend(fontsize=6.4, frameon=True, framealpha=0.9, edgecolor="#ccc",
                      loc="lower right", title="tier", title_fontsize=6.6)

    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dkey.lower()}_predictions.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _association_grid(dkey, reg, outputs, tiers, providers, use_abs, out_dir, show) -> plt.Figure:
    """Always a 2x2 grid: heatmap by tier, heatmap by provider, macro bar by each."""
    r_tier = _pivot_metric(reg, "tier", "pearson_r", outputs, tiers)
    r_prov = _pivot_metric(reg, "model", "pearson_r", outputs, providers)
    if use_abs:
        r_tier, r_prov = r_tier.abs(), r_prov.abs()
    heat_kwargs = dict(vmin=0, vmax=1, cmap="Reds") if use_abs else dict(vmin=-1, vmax=1, cmap="RdBu_r")
    label = "|Pearson r|" if use_abs else "Pearson r"
    tier_color = _tier_color_map(tiers)

    fig, ax = plt.subplots(2, 2, figsize=(11.8, 9.0), constrained_layout=True)
    _heatmap(ax[0, 0], r_tier.rename(index=pretty_output, columns=pretty_tier),
             f"{label} by output x tier", **heat_kwargs)
    im = _heatmap(ax[0, 1], r_prov, f"{label} by output x provider", show_ylabels=False, **heat_kwargs)
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    _category_bar_single(ax[1, 0], tiers, _macro_r_by(reg, "tier", tiers, use_abs=use_abs),
                          lambda t: tier_color[t], pretty_tier,
                          f"macro {label} by tier", f"mean {label} over outputs", use_abs=use_abs)
    _category_bar_single(ax[1, 1], providers, _macro_r_by(reg, "model", providers, use_abs=use_abs),
                          provider_color, lambda p: p,
                          f"macro {label} by provider", f"mean {label} over outputs", use_abs=use_abs)

    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{dkey.lower()}_association.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


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

    fig = plt.figure(figsize=(16, 4.6))
    gs = GridSpec(1, 4, figure=fig, wspace=0.42, width_ratios=[1.0, 1.25, 1.35, 1.35])

    # confusion matrix
    ax0 = fig.add_subplot(gs[0, 0])
    cm = np.array([[overall["tn"], overall["fp"]], [overall["fn"], overall["tp"]]])
    ax0.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax0.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax0.set_xticks([0, 1]); ax0.set_xticklabels(["pred control", "pred case"], fontsize=8, rotation=20, ha="right")
    ax0.set_yticks([0, 1]); ax0.set_yticklabels(["true control", "true case"], fontsize=8)
    ax0.set_title(f"Confusion (n={overall['n']})", fontsize=9)

    # headline metrics
    ax1 = fig.add_subplot(gs[0, 1])
    names = ["accuracy", "balanced accuracy", "sensitivity", "specificity", "AUROC", "MCC", "F1"]
    vals = [overall["accuracy"], overall["balanced_accuracy"], overall["sensitivity"],
            overall["specificity"], overall["auroc"], overall["mcc"], overall["f1"]]
    colors = [_POS if (np.isfinite(v) and v >= 0.5) else _NEG for v in vals]
    ax1.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax1.axhline(0.5, color="#888", ls=":", lw=0.9)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=7.5, rotation=35, ha="right")
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

    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "run_procedure.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _subject_effort_error(rows) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def procedure_vs_performance_stats(rows) -> pd.DataFrame:
    """Per-dataset effort/accuracy correlations plus the pooled first-try-vs-recovered test.

    Returned as a table so the notebook can print the exact numbers the figure
    only shows visually (Pearson r and Spearman rho of seconds vs error, and the
    Mann-Whitney U p-value comparing first-try to recovered-job error).
    """
    reg_g, _ = _subject_effort_error(rows)
    rows_out = []
    for d, s in reg_g.groupby("dataset"):
        s = s.dropna(subset=["seconds", "abs_z_error"])
        if len(s) >= 4 and s["seconds"].nunique() > 1:
            r, p_r = pearsonr(s["seconds"], s["abs_z_error"])
            rho, p_rho = spearmanr(s["seconds"], s["abs_z_error"])
        else:
            r = p_r = rho = p_rho = np.nan
        first = s.loc[s["attempts"] == 1, "abs_z_error"]
        recovered = s.loc[s["attempts"] > 1, "abs_z_error"]
        if len(first) >= 3 and len(recovered) >= 3:
            u_p = float(mannwhitneyu(first, recovered, alternative="two-sided").pvalue)
        else:
            u_p = np.nan
        rows_out.append(dict(
            dataset=d, n=len(s), n_first_try=len(first), n_recovered=len(recovered),
            pearson_r_seconds_vs_error=round(r, 3) if np.isfinite(r) else np.nan,
            p_pearson=round(p_r, 3) if np.isfinite(p_r) else np.nan,
            spearman_rho_seconds_vs_error=round(rho, 3) if np.isfinite(rho) else np.nan,
            p_spearman=round(p_rho, 3) if np.isfinite(p_rho) else np.nan,
            mean_error_first_try=round(float(first.mean()), 3) if len(first) else np.nan,
            mean_error_recovered=round(float(recovered.mean()), 3) if len(recovered) else np.nan,
            mannwhitney_p=round(u_p, 3) if np.isfinite(u_p) else np.nan,
        ))
    return pd.DataFrame(rows_out).set_index("dataset")


def fig_procedure_vs_performance(rows, out_dir: Optional[Path] = None, show: bool = True) -> plt.Figure:
    """Supplementary: does run cost (time / attempts) predict accuracy?

    Top row: one effort-vs-error scatter per dataset (own provider colors), with
    Pearson r/p and Spearman rho/p annotated. Bottom: a single panel pooling all
    three datasets' normalized error (comparable because each is already scaled
    by its own output's population SD) to compare first-try vs recovered jobs,
    with a Mann-Whitney U test, instead of three near-duplicate per-dataset
    boxplots -- one pooled comparison already answers the question.
    """
    reg_g, _ = _subject_effort_error(rows)
    datasets = [d for d in ("INTELLIGENCE", "PSYCHOSIS", "NUMERACY") if d in set(reg_g.dataset)]

    fig = plt.figure(figsize=(16, 8.0))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32, height_ratios=[1.0, 0.85])

    for di, d in enumerate(datasets):
        ax = fig.add_subplot(gs[0, di])
        s = reg_g[reg_g.dataset == d].dropna(subset=["seconds", "abs_z_error"])
        for p in ordered_providers(s["model"].unique()):
            sp = s[s.model == p]
            ax.scatter(sp["seconds"], sp["abs_z_error"], s=24, alpha=0.75,
                       color=provider_color(p), label=p, edgecolor="none")
        if len(s) >= 4 and s["seconds"].nunique() > 1:
            r, p_r = pearsonr(s["seconds"], s["abs_z_error"])
            rho, p_rho = spearmanr(s["seconds"], s["abs_z_error"])
            b, a = np.polyfit(s["seconds"], s["abs_z_error"], 1)
            xs = np.array([s["seconds"].min(), s["seconds"].max()])
            ax.plot(xs, b * xs + a, color="#d1495b", lw=1.5)
            ax.text(0.03, 0.97, f"r = {r:.2f} (p={p_r:.2f})\nrho = {rho:.2f} (p={p_rho:.2f})\nn = {len(s)}",
                    transform=ax.transAxes, va="top", fontsize=7.6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))
        ax.set_title(f"{d.title()}: effort vs error", fontsize=9.5)
        ax.set_xlabel("seconds / job"); ax.set_ylabel("mean |z error| per subject", fontsize=8)
        if di == 0:
            ax.legend(fontsize=6.2, frameon=False, loc="upper right")

    # single pooled panel: first try vs recovered, all datasets combined
    ax = fig.add_subplot(gs[1, :])
    pooled = reg_g.dropna(subset=["abs_z_error"]).copy()
    pooled["group"] = np.where(pooled["attempts"] > 1, "recovered (>=2 attempts)", "first try")
    groups = ["first try", "recovered (>=2 attempts)"]
    data = [pooled.loc[pooled.group == g, "abs_z_error"].to_numpy() for g in groups]
    bp = ax.boxplot(data, labels=groups, showfliers=False, patch_artist=True, widths=0.5,
                    vert=False)
    for patch, c in zip(bp["boxes"], ["#9c9c9c", "#f28e2b"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for i, d in enumerate(data, start=1):
        if len(d):
            ax.scatter(np.mean(d), i, color="#d1495b", zorder=5, s=40, label="mean" if i == 1 else None)
    if all(len(d) >= 3 for d in data):
        u_p = float(mannwhitneyu(data[0], data[1], alternative="two-sided").pvalue)
        ax.text(0.98, 0.06, f"Mann-Whitney U, p = {u_p:.2f}  (n={len(data[0])} vs n={len(data[1])})",
                transform=ax.transAxes, ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))
    ax.set_xlabel("mean |z error| per subject (pooled across all three datasets)", fontsize=8.5)
    ax.set_title("Does needing the recovery loop change accuracy? (pooled across datasets)", fontsize=10)

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
    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.32)

    # top: macro association by provider, one panel per dataset (own provider colors;
    # psychosis uses |r| since its per-output correlations are noisy in sign)
    for di, d in enumerate(datasets):
        ax = fig.add_subplot(gs[0, di])
        reg = long[(long.kind == "reg") & (long.dataset == d)]
        use_abs = d == "PSYCHOSIS"
        vals = _macro_r_by(reg, "model", providers, use_abs=use_abs)
        _category_bar_single(ax, providers, vals, provider_color, lambda p: p,
                             f"{d.title()}: macro association by provider",
                             "mean |Pearson r|" if use_abs else "mean Pearson r", use_abs=use_abs)

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
    """Full signed per-output metrics (direction is scientifically meaningful here,
    e.g. a negative bias means the model underestimates truth), pooled over tiers
    and providers. Heatmaps and macro bars elsewhere may switch to |r| for
    high-output, noisy-sign settings (psychosis); this table never does."""
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
    is well defined. Both the signed mean and the mean absolute magnitude are
    returned; use the absolute columns where sign is noisy across many outputs
    (as in psychosis) and the signed columns where direction matters and there
    are few, individually reliable outputs (intelligence, numeracy).
    """
    reg = long[long.kind == "reg"]
    tab = regression_by(reg, [by, "output"])
    if tab.empty:
        return tab
    g = tab.groupby(level=0)
    macro = pd.DataFrame({
        "pearson_r": g["pearson_r"].mean(),
        "abs_pearson_r": tab["pearson_r"].abs().groupby(level=0).mean(),
        "spearman_rho": g["spearman_rho"].mean(),
        "abs_spearman_rho": tab["spearman_rho"].abs().groupby(level=0).mean(),
        "kendall_tau": g["kendall_tau"].mean(),
        "R2": g["R2"].mean(),
        "r_squared": g["r_squared"].mean(),
        "n_outputs": g.size(),
    })
    return macro.round(3)
