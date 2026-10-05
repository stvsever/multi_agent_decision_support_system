"""Publication figures for annotated validation (matplotlib, Agg backend).

One consistent visual system: a fixed-order categorical palette validated for
color-vision deficiency (configurations keep their color across figures), a
single-hue blue ramp for magnitude (density, confusion proportions), text in
neutral ink, light recessive grid. Every function takes tidy or metric tables
(see ``tidy.py`` and ``stats.py``), draws on a given ``ax`` or a new figure,
and ``save_figure`` writes PNG (and optionally PDF or SVG).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots in fixed order (identity follows the entity, never its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OVERFLOW_COLOR = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#ffffff"
BLUE_RAMP = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("compass_blue", BLUE_RAMP)

METRIC_LABELS = {
    "pearson_r": "Pearson r",
    "spearman_rho": "Spearman rho",
    "r2": "R²",
    "mae": "MAE",
    "rmse": "RMSE",
    "nmae": "Normalized MAE",
    "bias": "Bias (pred - true)",
    "lin_ccc": "Lin CCC",
    "ols_slope": "OLS slope",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "auroc": "AUROC",
    "brier": "Brier score",
    "kappa": "Cohen kappa",
    "f1": "F1",
    "micro_f1": "Micro F1",
    "subset_accuracy": "Subset accuracy",
    "icc_2_1": "ICC(2,1)",
}
# Reference lines that mean "no skill" for a metric.
METRIC_REFERENCE = {"pearson_r": 0.0, "spearman_rho": 0.0, "r2": 0.0, "lin_ccc": 0.0, "auroc": 0.5,
                    "balanced_accuracy": None, "kappa": 0.0, "bias": 0.0, "ols_slope": 1.0}


@contextmanager
def figure_style():
    """rcParams for every figure of this module."""
    params = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "font.family": "sans-serif",
        "lines.linewidth": 2.0,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
    with plt.rc_context(params):
        yield


_COLOR_ORDER: Dict[str, int] = {}


def register_colors(values: Sequence[Any]) -> None:
    """Fix the palette slot of each entity for all later figures (call once
    with every configuration of an evaluation, so a subset never repaints)."""
    _COLOR_ORDER.clear()
    for i, v in enumerate(dict.fromkeys(sorted((str(x) for x in values)))):
        _COLOR_ORDER[v] = i


def color_for(values: Sequence[Any]) -> Dict[Any, str]:
    """Stable color per entity: registered slot if any, else slot i in sorted
    order; entities past the eighth slot are gray."""
    uniq = list(dict.fromkeys(sorted(values, key=lambda v: str(v))))
    out: Dict[Any, str] = {}
    if _COLOR_ORDER and all(str(v) in _COLOR_ORDER for v in uniq):
        for v in uniq:
            i = _COLOR_ORDER[str(v)]
            out[v] = PALETTE[i] if i < len(PALETTE) else OVERFLOW_COLOR
        return out
    return {v: (PALETTE[i] if i < len(PALETTE) else OVERFLOW_COLOR) for i, v in enumerate(uniq)}


def metric_label(metric: str) -> str:
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    if metric.startswith("macro_") and metric[len("macro_"):] in METRIC_LABELS:
        return "Macro " + METRIC_LABELS[metric[len("macro_"):]]
    return metric.replace("_", " ")


def save_figure(fig: Any, path: Union[str, Path], formats: Sequence[str] = ("png",)) -> List[str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(target)
        out.append(str(target))
    plt.close(fig)
    return out


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"


def _fmt_ci(row: Mapping[str, Any], digits: int = 2) -> str:
    value = _fmt(row.get("value"), digits)
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return value
    return f"{value} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def _density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 10:
        return np.full(len(x), 0.5)
    try:
        from scipy.stats import gaussian_kde

        xy = np.vstack([x, y])
        if np.linalg.matrix_rank(np.cov(xy)) < 2:
            return np.full(len(x), 0.5)
        dens = gaussian_kde(xy)(xy)
        return (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
    except Exception:
        return np.full(len(x), 0.5)


def scatter_true_vs_pred(
    frame: pd.DataFrame,
    *,
    ax: Any = None,
    title: Optional[str] = None,
    xlabel: str = "True value",
    ylabel: str = "Predicted value",
    stats: Optional[Mapping[str, Mapping[str, Any]]] = None,
    show_box: bool = True,
    limits: Optional[Tuple[float, float]] = None,
) -> Any:
    """Density-colored scatter of predicted vs true with identity line, OLS
    fit with its 95% confidence band, and a stats box.

    ``frame``: tidy rows of one output (rows with ``status == "ok"`` are drawn).
    ``stats``: optional ``{metric: {"value", "ci_low", "ci_high"}}`` for the box
    (computed from the points without CIs when absent). ``limits`` fixes
    both axes (used by ``scatter_grid`` so facets share one scale).
    """
    from scipy import stats as sps

    data = frame[frame["status"] == "ok"] if "status" in frame.columns else frame
    x = data["true_value"].to_numpy(float)
    y = data["pred_value"].to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))
    if len(x) == 0:
        ax.text(0.5, 0.5, "no scored pairs", ha="center", va="center", color=INK_2, transform=ax.transAxes)
        ax.set_title(title or "")
        return ax
    dens = _density(x, y)
    order = np.argsort(dens)
    ax.scatter(x[order], y[order], c=dens[order], cmap=SEQUENTIAL_CMAP, vmin=-0.25, vmax=1.0, s=22,
               edgecolors=SURFACE, linewidths=0.4, zorder=3)
    if limits is not None:
        lo, hi = float(limits[0]), float(limits[1])
    else:
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=INK_3, zorder=2, label="Identity")
    n = len(x)
    if n >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(lo, hi, 100)
        fit = intercept + slope * grid
        resid = y - (intercept + slope * x)
        s = math.sqrt(float(np.sum(resid ** 2)) / max(1, n - 2))
        sxx = float(np.sum((x - x.mean()) ** 2))
        tcrit = float(sps.t.ppf(0.975, max(1, n - 2)))
        half = tcrit * s * np.sqrt(1.0 / n + (grid - x.mean()) ** 2 / sxx)
        ax.fill_between(grid, fit - half, fit + half, color=PALETTE[1], alpha=0.18, lw=0, zorder=1)
        ax.plot(grid, fit, color=PALETTE[1], lw=2, zorder=4, label="OLS fit (95% CI)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if show_box:
        if not stats:
            r = np.corrcoef(x, y)[0, 1] if n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")
            mae = float(np.mean(np.abs(y - x)))
            stats = {"pearson_r": {"value": r}, "mae": {"value": mae}}
        lines = [f"n = {n}"]
        for metric, name in (("pearson_r", "r"), ("spearman_rho", "ρ"), ("r2", "R²"), ("mae", "MAE"), ("lin_ccc", "CCC")):
            if metric in stats:
                lines.append(f"{name} = {_fmt_ci(stats[metric])}")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID, alpha=0.92), zorder=5)
    ax.legend(loc="lower right", fontsize=8)
    return ax


def scatter_grid(
    tidy: pd.DataFrame,
    node_id: str,
    output: str,
    *,
    facet_cols: Sequence[str] = (),
    metrics: Optional[pd.DataFrame] = None,
    max_facets: int = 12,
    title: Optional[str] = None,
) -> Any:
    """One density scatter per facet (combination of ``facet_cols``) for one output."""
    data = tidy[(tidy["node_id"] == node_id) & (tidy["output"] == output)]
    facet_cols = [c for c in facet_cols if c in data.columns]
    n_all = 1
    if facet_cols:
        all_keys = list(data.groupby(facet_cols, dropna=False, sort=True).groups.keys())
        n_all = len(all_keys)
        keys = all_keys[:max_facets]
    else:
        keys = [None]
    if data.empty:
        raise ValueError(f"no rows for node '{node_id}' and output '{output}'")
    n = len(keys)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    ok = data[data["status"] == "ok"] if "status" in data.columns else data
    vals = np.concatenate([ok["true_value"].to_numpy(float), ok["pred_value"].to_numpy(float)])
    vals = vals[np.isfinite(vals)]
    limits = None
    if vals.size:
        span = float(vals.max() - vals.min()) or 1.0
        limits = (float(vals.min()) - 0.05 * span, float(vals.max()) + 0.05 * span)
    with figure_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.2 * nrows), squeeze=False)
        for ax, key in zip(axes.ravel(), keys):
            sub = data
            label = None
            stats = None
            if key is not None:
                key = key if isinstance(key, tuple) else (key,)
                mask = np.ones(len(data), dtype=bool)
                for c, v in zip(facet_cols, key):
                    mask &= (data[c].astype(str) == str(v)).to_numpy()
                sub = data[mask]
                label = " | ".join(str(v) for v in key)
            if metrics is not None and not metrics.empty:
                m = metrics[(metrics["node_id"] == node_id) & (metrics["output"] == output)]
                if key is not None:
                    for c, v in zip(facet_cols, key):
                        if c in m.columns:
                            m = m[m[c].astype(str) == str(v)]
                stats = {r["metric"]: r for r in m.to_dict("records")}
            scatter_true_vs_pred(sub, ax=ax, title=label, stats=stats or None, limits=limits)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        heading = title or f"{output}: predicted vs true"
        if n_all > len(keys):
            heading += f" (first {len(keys)} of {n_all} groups)"
        fig.suptitle(heading, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
        fig.tight_layout()
    return fig


def residual_plot(
    tidy: pd.DataFrame,
    node_id: str,
    output: str,
    *,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
) -> Any:
    """Residual (pred - true) vs true, and a Bland-Altman panel (difference vs
    mean with bias and 95% limits of agreement). ``color_col`` colors groups."""
    data = tidy[(tidy["node_id"] == node_id) & (tidy["output"] == output) & (tidy["status"] == "ok")]
    with figure_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
        groups = [(None, data)]
        colors = {None: PALETTE[0]}
        if color_col and color_col in data.columns and data[color_col].nunique() > 1:
            colors = color_for(data[color_col].astype(str).unique())
            groups = [(k, g) for k, g in data.groupby(data[color_col].astype(str), sort=True)]
        for key, g in groups:
            t = g["true_value"].to_numpy(float)
            p = g["pred_value"].to_numpy(float)
            label = None if key is None else str(key)
            ax1.scatter(t, p - t, s=16, color=colors[key], alpha=0.75, edgecolors=SURFACE, linewidths=0.3, label=label)
            ax2.scatter((t + p) / 2, p - t, s=16, color=colors[key], alpha=0.75, edgecolors=SURFACE, linewidths=0.3)
        t = data["true_value"].to_numpy(float)
        p = data["pred_value"].to_numpy(float)
        d = p - t
        ax1.axhline(0, color=INK_3, lw=1.2, ls="--")
        ax1.set_xlabel("True value")
        ax1.set_ylabel("Residual (predicted - true)")
        ax1.set_title("Residuals vs true", loc="left")
        if len(d) >= 3:
            bias = float(np.mean(d))
            sd = float(np.std(d, ddof=1))
            for yv, ls in ((bias, "-"), (bias + 1.96 * sd, ":"), (bias - 1.96 * sd, ":")):
                ax2.axhline(yv, color=INK_2, lw=1.2, ls=ls)
            ax2.text(0.99, 0.02, f"bias {_fmt(bias)}, LoA [{_fmt(bias - 1.96 * sd)}, {_fmt(bias + 1.96 * sd)}]",
                     transform=ax2.transAxes, ha="right", va="bottom", fontsize=8, color=INK_2)
        ax2.axhline(0, color=INK_3, lw=1.0, ls="--")
        ax2.set_xlabel("Mean of true and predicted")
        ax2.set_ylabel("Predicted - true")
        ax2.set_title("Bland-Altman agreement", loc="left")
        if len(groups) > 1:
            ax1.legend(fontsize=8, loc="best")
        fig.suptitle(title or f"{output}: residual diagnostics", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metric summaries
# ---------------------------------------------------------------------------


def forest_plot(
    metrics: pd.DataFrame,
    metric: str,
    *,
    label_cols: Sequence[str] = ("output",),
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    reference: Optional[float] = "auto",  # type: ignore[assignment]
    ax: Any = None,
) -> Any:
    """Point estimate and 95% CI of one metric per row (group x output)."""
    data = metrics[metrics["metric"] == metric].copy()
    label_cols = [c for c in label_cols if c in data.columns]
    data["_label"] = data[label_cols].astype(str).agg(" | ".join, axis=1) if label_cols else metric
    data = data.iloc[::-1].reset_index(drop=True)
    height = max(2.2, 0.32 * len(data) + 1.1)
    with figure_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.2, height))
        else:
            fig = ax.figure
        colors = color_for(data[color_col].astype(str).unique()) if color_col and color_col in data.columns else {}
        for i, row in data.iterrows():
            c = colors.get(str(row[color_col]), PALETTE[0]) if colors else PALETTE[0]
            v, lo, hi = row["value"], row.get("ci_low"), row.get("ci_high")
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
            if np.isfinite(v):
                ax.plot([v], [i], marker="o", ms=7, color=c, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ax.text(1.01, i, _fmt_ci(row), transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data["_label"], fontsize=8)
        ax.set_ylim(-0.7, len(data) - 0.3)
        ref = METRIC_REFERENCE.get(metric) if reference == "auto" else reference
        if ref is not None:
            ax.axvline(ref, color=INK_3, lw=1.0, ls="--", zorder=0)
        ax.set_xlabel(metric_label(metric))
        ax.grid(axis="y", visible=False)
        if colors and len(colors) > 1:
            handles = [plt.Line2D([0], [0], marker="o", color=c, lw=2, label=k) for k, c in colors.items()]
            ax.legend(handles=handles, title=color_col, loc="upper left", bbox_to_anchor=(1.0, -0.06), fontsize=8)
        ax.set_title(title or f"{metric_label(metric)} with 95% CI", loc="left")
        fig.tight_layout()
    return fig


def metric_by_tier(
    metrics: pd.DataFrame,
    metric: str,
    *,
    tier_col: str = "tier",
    config_cols: Sequence[str] = (),
    tier_order: Optional[Sequence[Any]] = None,
    title: Optional[str] = None,
    ax: Any = None,
) -> Any:
    """Metric (with 95% CI) across tiers, one line per configuration."""
    data = metrics[metrics["metric"] == metric].copy()
    config_cols = [c for c in config_cols if c in data.columns and c != tier_col]
    data["_config"] = data[config_cols].astype(str).agg(" | ".join, axis=1) if config_cols else "all"
    tiers = list(tier_order) if tier_order else sorted(data[tier_col].dropna().astype(str).unique())
    pos = {t: i for i, t in enumerate(tiers)}
    configs = sorted(data["_config"].unique())
    colors = color_for(configs)
    width = 0.5
    with figure_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(tiers) + 2.5), 4.2))
        else:
            fig = ax.figure
        for j, cfg in enumerate(configs):
            sub = data[data["_config"] == cfg].copy()
            sub["_x"] = sub[tier_col].astype(str).map(pos)
            sub = sub.dropna(subset=["_x"]).sort_values("_x")
            off = (j - (len(configs) - 1) / 2) * (width / max(1, len(configs))) if len(configs) > 1 else 0.0
            xs = sub["_x"].to_numpy(float) + off
            c = colors[cfg]
            ax.plot(xs, sub["value"], color=c, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=cfg, zorder=3)
            lo, hi = sub["ci_low"].to_numpy(float), sub["ci_high"].to_numpy(float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            ax.vlines(xs[ok], lo[ok], hi[ok], color=c, lw=2, alpha=0.8, zorder=2)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers, rotation=20 if max((len(t) for t in tiers), default=0) > 8 else 0, ha="right" if max((len(t) for t in tiers), default=0) > 8 else "center")
        ref = METRIC_REFERENCE.get(metric)
        if ref is not None:
            ax.axhline(ref, color=INK_3, lw=1.0, ls="--", zorder=0)
        ax.set_xlabel(tier_col)
        ax.set_ylabel(metric_label(metric))
        if len(configs) > 1:
            ax.legend(title=" | ".join(config_cols) if config_cols else None, loc="best", fontsize=8)
        ax.set_title(title or f"{metric_label(metric)} by {tier_col} (95% CI)", loc="left")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def confusion_matrix_plot(confusion: pd.DataFrame, *, title: Optional[str] = None, normalize: bool = True, ax: Any = None) -> Any:
    """Heatmap of one confusion matrix (rows: true, columns: predicted).

    ``confusion``: rows of the ``confusion`` table for ONE node and ONE group
    (columns ``true_label``, ``pred_label``, ``count``). Cells show the row
    proportion and the count; a ``(missing)`` column holds missing predictions.
    """
    true_labels = list(dict.fromkeys(confusion["true_label"]))
    pred_labels = list(dict.fromkeys([*true_labels, *confusion["pred_label"]]))
    mat = np.zeros((len(true_labels), len(pred_labels)))
    for r in confusion.itertuples(index=False):
        mat[true_labels.index(r.true_label), pred_labels.index(r.pred_label)] += r.count
    rows = mat.sum(1, keepdims=True)
    prop = np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)
    with figure_style():
        if ax is None:
            size = max(3.6, 0.75 * len(pred_labels) + 2.0)
            fig, ax = plt.subplots(figsize=(size + 0.8, size))
        else:
            fig = ax.figure
        im = ax.imshow(prop if normalize else mat, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1 if normalize else None)
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = prop[i, j]
                color = SURFACE if (val > 0.55 if normalize else mat[i, j] > 0.55 * mat.max()) else INK
                ax.text(j, i, f"{val:.2f}\n({int(mat[i, j])})" if normalize else f"{int(mat[i, j])}",
                        ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(true_labels)))
        ax.set_yticklabels(true_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Row proportion" if normalize else "Count")
        cb.outline.set_visible(False)
        ax.set_title(title or "Confusion matrix", loc="left")
        fig.tight_layout()
    return fig


def roc_curve_points(y: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) of a binary score (ties handled as one threshold)."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    distinct = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.r_[0.0, tps / P] if P > 0 else np.r_[0.0, np.zeros(len(tps))]
    fpr = np.r_[0.0, fps / N] if N > 0 else np.r_[0.0, np.zeros(len(fps))]
    return fpr, tpr


def roc_plot(
    tidy: pd.DataFrame,
    node_id: str,
    *,
    color_cols: Sequence[str] = (),
    metrics: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    ax: Any = None,
) -> Any:
    """ROC curves of one classification node. Binary: one curve per
    configuration (combination of ``color_cols``, at most 8). Multiclass:
    one-vs-rest curve per class (pooled over configurations)."""
    data = tidy[(tidy["node_id"] == node_id) & (tidy["status"] == "ok")]
    kind = str(data["kind"].iloc[0]) if len(data) else "binary"
    color_cols = [c for c in color_cols if c in data.columns]
    with figure_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.8, 4.6))
        else:
            fig = ax.figure
        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=INK_3, label="Chance")
        if kind == "binary":
            data = data[np.isfinite(data["p_positive"].to_numpy(float))]
            keys = data[color_cols].astype(str).agg(" | ".join, axis=1) if color_cols else pd.Series("all", index=data.index)
            colors = color_for(keys.unique())
            for key in sorted(keys.unique())[:8]:
                sub = data[keys == key]
                y = (sub["true_label"].astype(str) == sub["positive_label"].astype(str)).to_numpy(float)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                fpr, tpr = roc_curve_points(y, sub["p_positive"].to_numpy(float))
                from .stats import weighted_auroc

                auc = float(weighted_auroc(y, sub["p_positive"].to_numpy(float), np.ones((1, len(y))))[0])
                ax.step(fpr, tpr, where="post", color=colors[key], lw=2, label=f"{key} (AUROC {auc:.2f})")
        else:
            probs = data["probs"]
            have = np.array([isinstance(p, dict) and bool(p) for p in probs], dtype=bool)
            sub = data[have]
            labels = sorted({k for p in sub["probs"] for k in p})
            colors = color_for(labels)
            for lab in labels[:8]:
                y = (sub["true_label"].astype(str) == lab).to_numpy(float)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                s = np.array([p.get(lab, 0.0) for p in sub["probs"]])
                fpr, tpr = roc_curve_points(y, s)
                from .stats import weighted_auroc

                auc = float(weighted_auroc(y, s, np.ones((1, len(y))))[0])
                ax.step(fpr, tpr, where="post", color=colors[lab], lw=2, label=f"{lab} vs rest ({auc:.2f})")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title or f"ROC: {node_id}", loc="left")
        fig.tight_layout()
    return fig


def calibration_plot(
    tidy: pd.DataFrame,
    node_id: str,
    *,
    color_cols: Sequence[str] = (),
    n_bins: int = 10,
    title: Optional[str] = None,
    ax: Any = None,
) -> Any:
    """Reliability diagram of P(positive) for one binary node: observed
    positive rate per equal-width probability bin (marker size = count), one
    line per configuration (combination of ``color_cols``, at most 8)."""
    data = tidy[(tidy["node_id"] == node_id) & (tidy["status"] == "ok")]
    data = data[np.isfinite(data["p_positive"].to_numpy(float))]
    color_cols = [c for c in color_cols if c in data.columns]
    keys = data[color_cols].astype(str).agg(" | ".join, axis=1) if color_cols else pd.Series("all", index=data.index)
    colors = color_for(keys.unique())
    edges = np.linspace(0, 1, n_bins + 1)
    with figure_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.8, 4.6))
        else:
            fig = ax.figure
        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=INK_3, label="Perfect calibration")
        for key in sorted(keys.unique())[:8]:
            sub = data[keys == key]
            p = sub["p_positive"].to_numpy(float)
            y = (sub["true_label"].astype(str) == sub["positive_label"].astype(str)).to_numpy(float)
            idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
            xs, ys, ns = [], [], []
            for b in range(n_bins):
                m = idx == b
                if m.any():
                    xs.append(p[m].mean())
                    ys.append(y[m].mean())
                    ns.append(int(m.sum()))
            if not xs:
                continue
            ns_arr = np.asarray(ns, dtype=float)
            ax.plot(xs, ys, color=colors[key], lw=2, label=str(key) if key != "all" else "Observed")
            ax.scatter(xs, ys, s=20 + 80 * ns_arr / ns_arr.max(), color=colors[key], edgecolors=SURFACE, linewidths=1.0, zorder=3)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted probability of the positive class")
        ax.set_ylabel("Observed positive rate")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(title or f"Calibration: {node_id}", loc="left")
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def paired_comparison_plot(comparison: pd.DataFrame, *, metric: Optional[str] = None,
                           label_cols: Sequence[str] = ("output",), title: Optional[str] = None) -> Any:
    """Forest of paired differences (config b minus config a) with 95% CIs.

    Left panel: both configurations' metric values with CIs; right panel: the
    paired difference, a zero line and the bootstrap p-value.
    """
    data = comparison if metric is None else comparison[comparison["metric"] == metric]
    data = data.copy()
    label_cols = [c for c in label_cols if c in data.columns]
    if metric is None:
        label_cols = [*label_cols, "metric"]
    data["_label"] = data[label_cols].astype(str).agg(" | ".join, axis=1)
    data = data.iloc[::-1].reset_index(drop=True)
    a = str(data["config_a"].iloc[0]) if len(data) else "a"
    b = str(data["config_b"].iloc[0]) if len(data) else "b"
    colors = color_for([a, b])
    height = max(2.4, 0.36 * len(data) + 1.3)
    with figure_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, height), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
        for i, row in data.iterrows():
            for key, off, cfg in (("value_a", -0.14, a), ("value_b", 0.14, b)):
                v, lo, hi = row[key], row.get(f"{key}_ci_low"), row.get(f"{key}_ci_high")
                if np.isfinite(lo) and np.isfinite(hi):
                    ax1.plot([lo, hi], [i + off, i + off], color=colors[cfg], lw=2)
                ax1.plot([v], [i + off], marker="o", ms=6, color=colors[cfg], markeredgecolor=SURFACE)
            d, lo, hi = row["diff"], row["diff_ci_low"], row["diff_ci_high"]
            if np.isfinite(lo) and np.isfinite(hi):
                ax2.plot([lo, hi], [i, i], color=INK_2, lw=2)
            ax2.plot([d], [i], marker="D", ms=6, color=INK, markeredgecolor=SURFACE)
            p = row.get("p_boot")
            ax2.text(1.01, i, f"p = {_fmt(p, 3)}", transform=ax2.get_yaxis_transform(), va="center", fontsize=8, color=INK_2)
        ax1.set_yticks(range(len(data)))
        ax1.set_yticklabels(data["_label"], fontsize=8)
        ax2.axvline(0, color=INK_3, ls="--", lw=1.0)
        ax1.set_xlabel(metric_label(metric) if metric else "Metric value")
        ax2.set_xlabel(f"Difference ({b} - {a})")
        ax1.grid(axis="y", visible=False)
        ax2.grid(axis="y", visible=False)
        handles = [plt.Line2D([0], [0], marker="o", color=c, lw=2, label=k) for k, c in colors.items()]
        ax1.legend(handles=handles, loc="best", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)
        fig.suptitle(title or "Paired comparison on matched participants (95% CI)", x=0.01, ha="left",
                     fontsize=12, fontweight="bold", color=INK)
        fig.tight_layout()
    return fig


def paired_errors_plot(tidy: pd.DataFrame, config_col: str, a: Any, b: Any, node_id: str, output: str,
                       *, title: Optional[str] = None) -> Any:
    """Per participant absolute error under ``a`` (x) vs ``b`` (y); points
    below the identity line are participants where ``b`` is closer to truth."""
    data = tidy[(tidy["node_id"] == node_id) & (tidy["output"] == output) & (tidy["status"] == "ok")]
    fa = data[data[config_col].astype(str) == str(a)].groupby("participant_id")["error"].apply(lambda e: np.mean(np.abs(e)))
    fb = data[data[config_col].astype(str) == str(b)].groupby("participant_id")["error"].apply(lambda e: np.mean(np.abs(e)))
    both = pd.concat([fa.rename("a"), fb.rename("b")], axis=1).dropna()
    with figure_style():
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        if len(both):
            hi = float(max(both["a"].max(), both["b"].max())) * 1.05 or 1.0
            ax.plot([0, hi], [0, hi], ls="--", lw=1.2, color=INK_3)
            better_b = both["b"] < both["a"]
            colors = color_for([str(a), str(b)])
            ax.scatter(both.loc[better_b, "a"], both.loc[better_b, "b"], s=20, color=colors[str(b)], edgecolors=SURFACE,
                       linewidths=0.4, label=f"{b} closer ({int(better_b.sum())})")
            ax.scatter(both.loc[~better_b, "a"], both.loc[~better_b, "b"], s=20, color=colors[str(a)], edgecolors=SURFACE,
                       linewidths=0.4, label=f"{a} closer or tie ({int((~better_b).sum())})")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel(f"Mean |error| per participant, {a}")
        ax.set_ylabel(f"Mean |error| per participant, {b}")
        ax.set_title(title or f"{output}: paired absolute errors", loc="left")
        fig.tight_layout()
    return fig
