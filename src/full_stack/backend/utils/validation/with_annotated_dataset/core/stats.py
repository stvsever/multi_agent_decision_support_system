"""Metrics with seeded participant-cluster bootstrap confidence intervals.

Every metric is computed from bootstrap *weights*: a replicate is a vector of
multinomial counts over participants (the resampling unit), broadcast to that
participant's rows. The point estimate uses unit weights, so point estimate
and intervals come from the same code path, repeated rows of one participant
(for example repeats that are not a grouping column) stay together, and two
configurations compared on matched participants share the same replicates
(paired bootstrap). Intervals are percentile intervals; seeds are derived
from the global seed and the cell key, so results do not depend on the order
in which cells are processed.

Predictions are never recalibrated (zero-shot evaluation): R2 is
``1 - SS_res / SS_tot`` of the raw predictions and can be negative.
"""

from __future__ import annotations

import json
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _nanmean(values: Any, axis: int) -> np.ndarray:
    """``np.nanmean`` without its "Mean of empty slice" warning: an all-NaN
    column (a bootstrap resample or a small group without a defined value) is
    simply NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260101
DEFAULT_CI = 0.95

REGRESSION_METRICS = [
    "pearson_r", "spearman_rho", "r2", "mae", "rmse", "nmae", "bias",
    "ols_slope", "ols_intercept", "lin_ccc",
]
CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "kappa", "accuracy_missing_as_wrong"]
BINARY_METRICS = ["sensitivity", "specificity", "ppv", "npv", "f1", "mcc"]
PROBABILITY_METRICS = ["auroc", "brier", "ece"]
MULTILABEL_METRICS = [
    "macro_f1", "micro_f1", "micro_precision", "micro_recall", "macro_balanced_accuracy",
    "macro_auroc", "hamming_loss", "subset_accuracy",
]
COMPARE_DEFAULTS = {
    "regression": ["mae", "rmse", "nmae", "pearson_r", "spearman_rho", "r2", "lin_ccc"],
    "binary": ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "brier"],
    "multiclass": ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "brier"],
}
LOWER_IS_BETTER = {"mae", "rmse", "nmae", "brier", "hamming_loss"}

_BLOCK_ELEMENTS = 4_000_000


# ---------------------------------------------------------------------------
# Bootstrap plumbing
# ---------------------------------------------------------------------------


def cell_rng(seed: int, key: Any) -> np.random.Generator:
    """Deterministic generator for one cell (independent of processing order)."""
    digest = zlib.crc32(json.dumps(key, default=str, sort_keys=True).encode("utf-8"))
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, digest])


def cluster_weights(n_clusters: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(1 + n_boot) x n_clusters weights: row 0 is the point estimate (all ones)."""
    if n_clusters <= 0:
        return np.ones((1, 0))
    ones = np.ones((1, n_clusters))
    if n_boot <= 0:
        return ones
    draws = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot).astype(float)
    return np.vstack([ones, draws])


def summarize(values: np.ndarray, ci: float = DEFAULT_CI) -> Tuple[float, float, float]:
    """(point, ci_low, ci_high) from a (1 + B,) replicate vector."""
    values = np.asarray(values, dtype=float)
    point = float(values[0]) if values.size else float("nan")
    boot = values[1:]
    boot = boot[np.isfinite(boot)]
    if not np.isfinite(point) or boot.size < 20:
        return point, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)


def _blocks(n_rep: int, n_cols: int) -> Iterable[slice]:
    size = max(1, int(_BLOCK_ELEMENTS // max(1, n_cols)))
    for start in range(0, n_rep, size):
        yield slice(start, min(n_rep, start + size))


def _div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=den != 0)
    return out


# ---------------------------------------------------------------------------
# Weighted primitives (W: replicates x rows)
# ---------------------------------------------------------------------------


def _wmean(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _div(W @ x, W.sum(1))


def _wcorr(W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    sw = W.sum(1, keepdims=True)
    mx = _div((W * X).sum(1, keepdims=True), sw)
    my = _div((W * Y).sum(1, keepdims=True), sw)
    dx = X - mx
    dy = Y - my
    cov = (W * dx * dy).sum(1)
    vx = (W * dx * dx).sum(1)
    vy = (W * dy * dy).sum(1)
    den = np.sqrt(vx * vy)
    ok = (vx > 1e-12 * np.maximum(1.0, np.abs(mx[:, 0]) ** 2)) & (vy > 1e-12 * np.maximum(1.0, np.abs(my[:, 0]) ** 2))
    return np.where(ok, _div(cov, den), np.nan)


def _weighted_midranks(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Ranks each replicate's resampled sample would give (ties get the mid rank)."""
    uniq, inv = np.unique(x, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
    Wu = np.add.reduceat(W[:, order], starts, axis=1)
    before = np.cumsum(Wu, axis=1) - Wu
    mid = before + (Wu + 1.0) / 2.0
    return mid[:, inv]


def weighted_auroc(y: np.ndarray, score: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC per replicate (ties count one half)."""
    y = np.asarray(y, dtype=float)
    uniq, inv = np.unique(score, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
    P = np.add.reduceat((W * y)[:, order], starts, axis=1)
    N = np.add.reduceat((W * (1.0 - y))[:, order], starts, axis=1)
    below = np.cumsum(N, axis=1) - N
    num = (P * (below + 0.5 * N)).sum(1)
    return _div(num, P.sum(1) * N.sum(1))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def regression_stats(t: np.ndarray, p: np.ndarray, W: np.ndarray) -> Dict[str, np.ndarray]:
    """All regression metrics per replicate. ``t`` true, ``p`` predicted."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    out: Dict[str, List[np.ndarray]] = {m: [] for m in REGRESSION_METRICS}
    ct, cp = (float(np.mean(t)), float(np.mean(p))) if t.size else (0.0, 0.0)
    tc, pc = t - ct, p - cp
    e = p - t
    for block in _blocks(W.shape[0], t.size):
        Wb = W[block]
        sw = Wb.sum(1)
        mt = _div(Wb @ tc, sw)
        mp = _div(Wb @ pc, sw)
        vt = _div(Wb @ (tc * tc), sw) - mt ** 2
        vp = _div(Wb @ (pc * pc), sw) - mp ** 2
        cov = _div(Wb @ (tc * pc), sw) - mt * mp
        vt = np.where(vt > 1e-12 * max(1.0, ct * ct), vt, 0.0)
        vp = np.where(vp > 1e-12 * max(1.0, cp * cp), vp, 0.0)
        mse = _div(Wb @ (e * e), sw)
        mae = _div(Wb @ np.abs(e), sw)
        out["bias"].append(_div(Wb @ e, sw))
        out["mae"].append(mae)
        out["rmse"].append(np.sqrt(mse))
        out["pearson_r"].append(np.where((vt > 0) & (vp > 0), _div(cov, np.sqrt(vt * vp)), np.nan))
        out["r2"].append(np.where(vt > 0, 1.0 - _div(mse, vt), np.nan))
        out["nmae"].append(np.where(vt > 0, _div(mae, np.sqrt(vt)), np.nan))
        slope = np.where(vt > 0, _div(cov, vt), np.nan)
        out["ols_slope"].append(slope)
        out["ols_intercept"].append((mp + cp) - slope * (mt + ct))
        den = vt + vp + (mt + ct - mp - cp) ** 2
        out["lin_ccc"].append(np.where(den > 0, _div(2.0 * cov, den), np.nan))
        rt = _weighted_midranks(t, Wb)
        rp = _weighted_midranks(p, Wb)
        out["spearman_rho"].append(_wcorr(Wb, rt, rp))
    result = {m: np.concatenate(v) if v else np.full(W.shape[0], np.nan) for m, v in out.items()}
    n_eff = (W > 0).sum(1)
    for m in ("pearson_r", "spearman_rho", "lin_ccc", "ols_slope", "ols_intercept", "r2", "nmae"):
        result[m] = np.where(n_eff >= 3, result[m], np.nan)
    return result


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def confusion_counts(true_idx: np.ndarray, pred_idx: np.ndarray, K: int, W: np.ndarray) -> np.ndarray:
    """Weighted confusion matrices, shape (replicates, K, K) [true, pred]."""
    pair = np.asarray(true_idx) * K + np.asarray(pred_idx)
    M = np.zeros((pair.size, K * K))
    M[np.arange(pair.size), pair] = 1.0
    return (W @ M).reshape(W.shape[0], K, K)


def classification_stats(
    true_idx: np.ndarray,
    pred_idx: np.ndarray,
    K: int,
    W: np.ndarray,
    positive_idx: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    C = confusion_counts(true_idx, pred_idx, K, W)
    row = C.sum(2)
    col = C.sum(1)
    diag = np.diagonal(C, axis1=1, axis2=2)
    n = row.sum(1)
    recall = _div(diag, row)
    precision = np.where(col > 0, _div(diag, col), np.where(row > 0, 0.0, np.nan))
    f1 = np.where(np.isfinite(precision) & np.isfinite(recall) & (precision + recall > 0),
                  _div(2 * precision * recall, precision + recall), np.where((row > 0) | (col > 0), 0.0, np.nan))
    with np.errstate(invalid="ignore"):
        out: Dict[str, np.ndarray] = {
            "accuracy": _div(diag.sum(1), n),
            "balanced_accuracy": _nanmean(np.where(row > 0, recall, np.nan), axis=1) if K else np.full(len(n), np.nan),
            "macro_f1": _nanmean(f1, axis=1) if K else np.full(len(n), np.nan),
        }
        po = out["accuracy"]
        pe = _div((row * col).sum(1), n * n)
        out["kappa"] = np.where(np.abs(1 - pe) > 1e-12, _div(po - pe, 1 - pe), np.nan)
    out["_precision"] = precision
    out["_recall"] = recall
    out["_f1"] = f1
    out["_support"] = row
    out["_predicted"] = col
    out["_confusion"] = C
    if positive_idx is not None and K == 2:
        neg = 1 - positive_idx
        tp = C[:, positive_idx, positive_idx]
        fn = C[:, positive_idx, neg]
        fp = C[:, neg, positive_idx]
        tn = C[:, neg, neg]
        out["sensitivity"] = _div(tp, tp + fn)
        out["specificity"] = _div(tn, tn + fp)
        out["ppv"] = _div(tp, tp + fp)
        out["npv"] = _div(tn, tn + fn)
        out["f1"] = _div(2 * tp, 2 * tp + fp + fn)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out["mcc"] = np.where(den > 0, _div(tp * tn - fp * fn, den), np.nan)
    return out


def weighted_ece(y: np.ndarray, p: np.ndarray, W: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Expected calibration error of P(positive): count-weighted mean over 10
    equal-width bins of |observed rate - mean probability|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    total = np.zeros(W.shape[0])
    for b in range(n_bins):
        mask = (idx == b).astype(float)
        if not mask.any():
            continue
        total += np.abs(W @ (mask * y) - W @ (mask * p))
    return _div(total, W.sum(1))


def probability_stats(Y: np.ndarray, P: np.ndarray, W: np.ndarray, binary_positive: Optional[int] = None) -> Dict[str, np.ndarray]:
    """AUROC and Brier from a one-hot truth matrix ``Y`` and probability matrix ``P``.

    Binary: AUROC of P(positive), Brier = mean (p - y)^2. Multiclass: macro
    one-vs-rest AUROC over classes with both outcomes present, Brier = mean of
    the summed squared error over classes (range 0 to 2).
    """
    if binary_positive is not None:
        y = Y[:, binary_positive]
        p = P[:, binary_positive]
        return {"auroc": weighted_auroc(y, p, W), "brier": _wmean(W, (p - y) ** 2), "ece": weighted_ece(y, p, W)}
    aucs = []
    for k in range(Y.shape[1]):
        if 0 < Y[:, k].sum() < Y.shape[0]:
            aucs.append(weighted_auroc(Y[:, k], P[:, k], W))
    with np.errstate(invalid="ignore"):
        auroc = _nanmean(np.vstack(aucs), axis=0) if aucs else np.full(W.shape[0], np.nan)
    return {"auroc": auroc, "brier": _wmean(W, ((P - Y) ** 2).sum(1))}


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def _cluster_codes(frame: pd.DataFrame, cluster: str) -> Tuple[np.ndarray, List[str]]:
    keys = frame[cluster].astype(str).to_numpy()
    uniq, codes = np.unique(keys, return_inverse=True)
    return codes, list(uniq)


def _labels_for(frame: pd.DataFrame, structure_labels: Sequence[str]) -> List[str]:
    labels = [str(x) for x in structure_labels]
    for col in ("true_label", "pred_label"):
        for lab in frame[col].dropna().astype(str):
            if lab not in labels:
                labels.append(lab)
    return labels


def _group_iter(tidy: pd.DataFrame, group_by: Sequence[str]) -> Iterable[Tuple[Dict[str, Any], pd.DataFrame]]:
    group_by = [g for g in group_by if g in tidy.columns]
    if not group_by:
        yield {}, tidy
        return
    for key, frame in tidy.groupby(group_by, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        yield {g: k for g, k in zip(group_by, key)}, frame


def _counts(frame: pd.DataFrame) -> Dict[str, Any]:
    status = frame["status"]
    n = int((status == "ok").sum())
    miss_pred = int((status == "missing_prediction").sum())
    return {
        "n": n,
        "n_total": int(len(frame)),
        "n_missing_prediction": miss_pred,
        "n_missing_truth": int((status == "missing_truth").sum()),
        "coverage": (n / (n + miss_pred)) if (n + miss_pred) else float("nan"),
    }


def _row(group: Mapping[str, Any], node_id: str, output: str, kind: str, metric: str,
         reps: np.ndarray, counts: Mapping[str, Any], ci: float) -> Dict[str, Any]:
    point, lo, hi = summarize(reps, ci)
    return {**group, "node_id": node_id, "output": output, "kind": kind, "metric": metric,
            "value": point, "ci_low": lo, "ci_high": hi, **counts}


def _structure_labels(tidy: pd.DataFrame, node_id: str, structure: Any) -> List[str]:
    if structure is not None and node_id in getattr(structure, "nodes", {}):
        return list(structure.nodes[node_id].class_labels)
    return []


# ---------------------------------------------------------------------------
# Metric tables
# ---------------------------------------------------------------------------


@dataclass
class MetricTables:
    """Output of ``compute_metrics``. All tables are tidy (long) DataFrames."""

    metrics: pd.DataFrame
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    multilabel: pd.DataFrame
    hierarchy: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)

    def wide(self) -> pd.DataFrame:
        return metrics_wide(self.metrics)


def metrics_wide(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per group x node x output; ``<metric>``, ``<metric>_ci_low``, ``<metric>_ci_high``."""
    if metrics.empty:
        return metrics.copy()
    count_cols = ["n", "n_total", "n_missing_prediction", "n_missing_truth", "coverage"]
    id_cols = [c for c in metrics.columns if c not in {"metric", "value", "ci_low", "ci_high", *count_cols}]
    base = metrics.drop_duplicates(id_cols)[id_cols + [c for c in count_cols if c in metrics.columns]]
    pieces = []
    indexed = metrics.drop_duplicates([*id_cols, "metric"]).set_index([*id_cols, "metric"])
    for suffix, col in (("", "value"), ("_ci_low", "ci_low"), ("_ci_high", "ci_high")):
        piv = indexed[col].unstack("metric")
        piv.columns = [f"{m}{suffix}" for m in piv.columns]
        pieces.append(piv)
    wide = pd.concat(pieces, axis=1).reset_index()
    order = []
    for m in dict.fromkeys(metrics["metric"]):
        order.extend([m, f"{m}_ci_low", f"{m}_ci_high"])
    wide = base.merge(wide, on=id_cols, how="left")
    return wide[[*id_cols, *[c for c in count_cols if c in wide.columns], *[c for c in order if c in wide.columns]]]


def _regression_cells(group, frame, node_id, n_boot, seed, ci, cluster, rows_out):
    outputs = list(dict.fromkeys(frame["output"]))
    scored = frame[frame["status"] == "ok"]
    codes_all, uniq = _cluster_codes(scored, cluster) if len(scored) else (np.zeros(0, int), [])
    rng = cell_rng(seed, [group, node_id, "regression"])
    Wp = cluster_weights(len(uniq), n_boot, rng)
    per_output: Dict[str, Dict[str, np.ndarray]] = {}
    for output in outputs:
        sub_all = frame[frame["output"] == output]
        counts = _counts(sub_all)
        mask = (scored["output"] == output).to_numpy()
        sub = scored[mask]
        if len(sub) == 0:
            for m in REGRESSION_METRICS:
                rows_out.append(_row(group, node_id, output, "regression", m, np.array([np.nan]), counts, ci))
            continue
        W = Wp[:, codes_all[mask]]
        stats = regression_stats(sub["true_value"].to_numpy(float), sub["pred_value"].to_numpy(float), W)
        per_output[output] = stats
        for m in REGRESSION_METRICS:
            rows_out.append(_row(group, node_id, output, "regression", m, stats[m], counts, ci))
    if len(per_output) > 1:
        counts = _counts(frame)
        for m in ("pearson_r", "spearman_rho", "r2", "nmae", "lin_ccc"):
            stack = np.vstack([per_output[o][m] for o in per_output])
            with np.errstate(invalid="ignore"):
                reps = _nanmean(stack, axis=0) if np.isfinite(stack).any() else np.full(stack.shape[1], np.nan)
            rows_out.append(_row(group, node_id, "__macro__", "regression", f"macro_{m}", reps, counts, ci))


def _classification_cell(group, frame, node_id, kind, labels_struct, positive_label, n_boot, seed, ci, cluster,
                         rows_out, per_class_out, confusion_out):
    counts = _counts(frame)
    truth_rows = frame[frame["status"].isin(["ok", "missing_prediction"])]
    scored = truth_rows[truth_rows["status"] == "ok"]
    labels = _labels_for(truth_rows, labels_struct)
    K = len(labels)
    metrics_list = list(CLASSIFICATION_METRICS) + (BINARY_METRICS if kind == "binary" else []) + [
        m for m in PROBABILITY_METRICS if kind == "binary" or m != "ece"]
    if len(truth_rows) == 0 or K == 0:
        for m in metrics_list:
            rows_out.append(_row(group, node_id, node_id, kind, m, np.array([np.nan]), counts, ci))
        return
    codes_all, uniq = _cluster_codes(truth_rows, cluster)
    rng = cell_rng(seed, [group, node_id, kind])
    Wp = cluster_weights(len(uniq), n_boot, rng)
    ok_mask = (truth_rows["status"] == "ok").to_numpy()
    W_all = Wp[:, codes_all]
    W = W_all[:, ok_mask]
    index = {lab: i for i, lab in enumerate(labels)}
    true_idx = scored["true_label"].astype(str).map(index).to_numpy(int)
    pred_idx = scored["pred_label"].astype(str).map(index).to_numpy(int)
    pos_idx = index.get(positive_label) if (kind == "binary" and positive_label is not None and K == 2) else None
    stats = classification_stats(true_idx, pred_idx, K, W, positive_idx=pos_idx)
    correct_all = np.zeros(len(truth_rows))
    correct_all[ok_mask] = (true_idx == pred_idx).astype(float)
    stats["accuracy_missing_as_wrong"] = _wmean(W_all, correct_all)
    # Probabilities.
    prob_rows = [i for i, p in enumerate(scored["probs"]) if isinstance(p, dict) and p]
    if prob_rows and K >= 2:
        P = np.zeros((len(prob_rows), K))
        for r, i in enumerate(prob_rows):
            for lab, v in scored["probs"].iloc[i].items():
                if lab in index:
                    P[r, index[lab]] = v
        Y = np.zeros_like(P)
        Y[np.arange(len(prob_rows)), true_idx[prob_rows]] = 1.0
        stats.update(probability_stats(Y, P, W[:, prob_rows], binary_positive=pos_idx))
    for m in metrics_list:
        rows_out.append(_row(group, node_id, node_id, kind, m, stats.get(m, np.array([np.nan])), counts, ci))
    for k, lab in enumerate(labels):
        rec = {**group, "node_id": node_id, "label": lab,
               "support": int(stats["_support"][0, k]), "n_predicted": int(stats["_predicted"][0, k])}
        for name in ("precision", "recall", "f1"):
            point, lo, hi = summarize(stats[f"_{name}"][:, k], ci)
            rec.update({name: point, f"{name}_ci_low": lo, f"{name}_ci_high": hi})
        per_class_out.append(rec)
    C = stats["_confusion"][0]
    for i, tl in enumerate(labels):
        for j, pl in enumerate(labels):
            confusion_out.append({**group, "node_id": node_id, "true_label": tl, "pred_label": pl, "count": int(C[i, j])})
    n_missing = int((~ok_mask).sum())
    if n_missing:
        for tl in labels:
            cnt = int(((truth_rows["status"] == "missing_prediction") & (truth_rows["true_label"].astype(str) == tl)).sum())
            confusion_out.append({**group, "node_id": node_id, "true_label": tl, "pred_label": "(missing)", "count": cnt})


def _multilabel_cell(group, frame, ml_group, n_boot, seed, ci, cluster, out):
    """Multi-label metrics over 0/1 rows (columns: participant, instance, output=label, true/pred/p)."""
    truth_rows = frame[pd.notna(frame["true_value"])]
    labels = list(dict.fromkeys(truth_rows["output"].astype(str)))
    counts = _counts(frame)
    if truth_rows.empty:
        return
    codes, uniq = _cluster_codes(truth_rows, cluster)
    Wp = cluster_weights(len(uniq), n_boot, cell_rng(seed, [group, ml_group, "multilabel"]))
    W_all = Wp[:, codes]
    ok = (truth_rows["status"] == "ok").to_numpy()
    y = truth_rows["true_value"].to_numpy(float)
    yhat = np.where(ok, truth_rows["pred_value"].to_numpy(float), np.nan)
    p = truth_rows["p_positive"].to_numpy(float) if "p_positive" in truth_rows else np.full(len(y), np.nan)
    lab = truth_rows["output"].astype(str).to_numpy()
    reps: Dict[str, np.ndarray] = {}
    tp_sum = fp_sum = fn_sum = 0.0
    f1s, baccs, aucs = [], [], []
    per_label_rows = []
    for name in labels:
        m = (lab == name) & ok
        if not m.any():
            continue
        W = W_all[:, m]
        yt, yp = y[m], yhat[m]
        tp = W @ ((yt == 1) & (yp == 1)).astype(float)
        fp = W @ ((yt == 0) & (yp == 1)).astype(float)
        fn = W @ ((yt == 1) & (yp == 0)).astype(float)
        tn = W @ ((yt == 0) & (yp == 0)).astype(float)
        tp_sum, fp_sum, fn_sum = tp_sum + tp, fp_sum + fp, fn_sum + fn
        f1 = np.where(2 * tp + fp + fn > 0, _div(2 * tp, 2 * tp + fp + fn), np.nan)
        sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
        with np.errstate(invalid="ignore"):
            bacc = _nanmean(np.vstack([sens, spec]), axis=0)
        f1s.append(f1)
        baccs.append(bacc)
        label_metrics = {"f1": f1, "sensitivity": sens, "specificity": spec, "balanced_accuracy": bacc,
                         "prevalence": _wmean(W, (yt == 1).astype(float))}
        pm = np.isfinite(p[m])
        if pm.sum() >= 2 and 0 < yt[pm].sum() < pm.sum():
            auc = weighted_auroc(yt[pm], p[m][pm], W[:, pm])
            aucs.append(auc)
            label_metrics["auroc"] = auc
            label_metrics["brier"] = _wmean(W[:, pm], (p[m][pm] - yt[pm]) ** 2)
        lab_counts = _counts(frame[frame["output"].astype(str) == name])
        for metric, values in label_metrics.items():
            per_label_rows.append(_row(group, ml_group, name, "multilabel", metric, values, lab_counts, ci))
    with np.errstate(invalid="ignore"):
        if f1s:
            reps["macro_f1"] = _nanmean(np.vstack(f1s), axis=0)
            reps["macro_balanced_accuracy"] = _nanmean(np.vstack(baccs), axis=0)
        if aucs:
            reps["macro_auroc"] = _nanmean(np.vstack(aucs), axis=0)
    reps["micro_precision"] = _div(tp_sum, tp_sum + fp_sum)
    reps["micro_recall"] = _div(tp_sum, tp_sum + fn_sum)
    reps["micro_f1"] = _div(2 * tp_sum, 2 * tp_sum + fp_sum + fn_sum)
    reps["hamming_loss"] = _wmean(W_all[:, ok], (y[ok] != yhat[ok]).astype(float)) if ok.any() else np.full(W_all.shape[0], np.nan)
    # Subset accuracy per instance (an instance with a missing label counts as not exact).
    inst = truth_rows.assign(_ok=ok, _hit=(y == yhat) & ok)
    by_inst = inst.groupby("instance_id").agg(_all=("_hit", "all"), _pid=(cluster, "first"))
    pid_index = {pid: i for i, pid in enumerate(uniq)}
    inst_codes = by_inst["_pid"].astype(str).map(pid_index).to_numpy(int)
    reps["subset_accuracy"] = _wmean(Wp[:, inst_codes], by_inst["_all"].to_numpy(float))
    for metric in MULTILABEL_METRICS:
        if metric in reps:
            out.append(_row(group, ml_group, "__all__", "multilabel", metric, reps[metric], counts, ci))
    out.extend(per_label_rows)


def _multilabel_frame(tidy: pd.DataFrame, groups: Mapping[str, Sequence[str]], structure: Any) -> pd.DataFrame:
    """0/1 rows for multi-label groups made of binary nodes (label = node id)."""
    frames = []
    for name, nodes in groups.items():
        sub = tidy[tidy["node_id"].isin(list(nodes)) & (tidy["kind"] == "binary")].copy()
        if sub.empty:
            continue
        pos = sub["positive_label"].astype(str)
        sub["true_value"] = np.where(sub["true_label"].notna(), (sub["true_label"].astype(str) == pos).astype(float), np.nan)
        sub["pred_value"] = np.where(sub["pred_label"].notna(), (sub["pred_label"].astype(str) == pos).astype(float), np.nan)
        sub["output"] = sub["node_id"]
        sub["node_id"] = name
        sub["kind"] = "multilabel"
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else tidy.iloc[0:0]


def _hierarchy_cell(group, frame, structure, n_boot, seed, ci, cluster, out):
    """Per instance hierarchy summaries: node coverage, exact match, path accuracy,
    and accuracy of a child node conditional on its parent being right or wrong."""
    cls_nodes = [n for n in structure.classification_nodes()]
    expected = frame[frame["status"].isin(["ok", "missing_prediction"])]
    if expected.empty:
        return
    inst = expected.groupby("instance_id").agg(_pid=(cluster, "first"))
    pids = inst["_pid"].astype(str)
    uniq = sorted(pids.unique())
    pid_index = {p: i for i, p in enumerate(uniq)}
    codes = pids.map(pid_index).to_numpy(int)
    Wp = cluster_weights(len(uniq), n_boot, cell_rng(seed, [group, "hierarchy"]))
    W = Wp[:, codes]
    cov = expected.assign(_p=(expected["status"] == "ok").astype(float)).groupby("instance_id")["_p"].mean().reindex(inst.index)
    base = {"n_instances": int(len(inst))}

    def _emit(metric, values, label="__tree__", valid=None):
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values) if valid is None else valid
        if not valid.any():
            return
        reps = _wmean(W[:, valid], values[valid])
        point, lo, hi = summarize(reps, ci)
        out.append({**group, "metric": metric, "path": label, "value": point, "ci_low": lo, "ci_high": hi,
                    "n": int(valid.sum()), **base})

    _emit("node_coverage", cov.to_numpy(float))
    if not cls_nodes:
        return
    cls = expected[expected["node_id"].isin(cls_nodes)]
    corr = cls.assign(_c=np.where(cls["status"] == "ok", cls["correct"], 0.0)).pivot_table(
        index="instance_id", columns="node_id", values="_c", aggfunc="first").reindex(inst.index)
    if corr.shape[1] >= 1:
        all_present = corr.notna().all(axis=1).to_numpy()
        exact = np.where(all_present, (corr.fillna(0).to_numpy() == 1).all(axis=1).astype(float), np.nan)
        _emit("exact_match", exact)
    for path in structure.paths():
        cpath = [n for n in path if n in corr.columns]
        if len(cpath) < 1 or len(path) < 2:
            continue
        sub = corr[cpath]
        present = sub.notna().all(axis=1).to_numpy()
        vals = np.where(present, (sub.fillna(0).to_numpy() == 1).all(axis=1).astype(float), np.nan)
        _emit("path_accuracy", vals, label=" > ".join(path))
    for node_id in cls_nodes:
        anc = [a for a in structure.ancestors(node_id) if a in corr.columns]
        if not anc or node_id not in corr.columns:
            continue
        parent = anc[0]
        child = corr[node_id].to_numpy(float)
        par = corr[parent].to_numpy(float)
        _emit("child_accuracy_given_parent_correct", np.where(par == 1, child, np.nan), label=f"{parent} > {node_id}")
        _emit("child_accuracy_given_parent_wrong", np.where(par == 0, child, np.nan), label=f"{parent} > {node_id}")


def compute_metrics(
    tidy: pd.DataFrame,
    group_by: Sequence[str] = (),
    *,
    structure: Any = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    cluster: str = "participant_id",
    multilabel_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> MetricTables:
    """Metrics per group x node x output with bootstrap CIs.

    ``tidy`` needs the tidy columns ``participant_id``, ``node_id``,
    ``output``, ``kind``, ``status`` and, per kind, ``true_value`` /
    ``pred_value`` (regression, multilabel) or ``true_label`` / ``pred_label``
    (binary, multiclass; optional ``probs`` dicts and ``positive_label``).
    ``structure`` (a ``TaskStructure``) adds the declared class labels and the
    hierarchy summaries; ``multilabel_groups`` maps a group name to binary
    node ids that form one multi-label task.
    """
    rows: List[Dict[str, Any]] = []
    per_class: List[Dict[str, Any]] = []
    confusion: List[Dict[str, Any]] = []
    ml_rows: List[Dict[str, Any]] = []
    hier_rows: List[Dict[str, Any]] = []
    group_by = [g for g in group_by if g in tidy.columns]
    if "instance_id" not in tidy.columns:
        tidy = tidy.assign(instance_id=pd.factorize(tidy[[cluster, *group_by]].astype(str).agg("|".join, axis=1))[0])
    for group, gframe in _group_iter(tidy, group_by):
        for node_id, nframe in gframe.groupby("node_id", sort=False):
            kind = str(nframe["kind"].iloc[0])
            if kind == "regression":
                _regression_cells(group, nframe, node_id, n_boot, seed, ci, cluster, rows)
            elif kind in {"binary", "multiclass"}:
                pos = nframe["positive_label"].dropna()
                _classification_cell(group, nframe, node_id, kind, _structure_labels(tidy, node_id, structure),
                                     pos.iloc[0] if len(pos) else None, n_boot, seed, ci, cluster, rows, per_class, confusion)
            elif kind == "multilabel":
                _multilabel_cell(group, nframe, node_id, n_boot, seed, ci, cluster, ml_rows)
        if multilabel_groups:
            mf = _multilabel_frame(gframe, multilabel_groups, structure)
            for name, mframe in mf.groupby("node_id", sort=False):
                _multilabel_cell(group, mframe, name, n_boot, seed, ci, cluster, ml_rows)
        if structure is not None and getattr(structure, "source", "") == "spec" and len(getattr(structure, "nodes", {})) > 1:
            _hierarchy_cell(group, gframe, structure, n_boot, seed, ci, cluster, hier_rows)
    return MetricTables(
        metrics=pd.DataFrame(rows),
        per_class=pd.DataFrame(per_class),
        confusion=pd.DataFrame(confusion),
        multilabel=pd.DataFrame(ml_rows),
        hierarchy=pd.DataFrame(hier_rows),
        settings={"n_boot": n_boot, "seed": seed, "ci": ci, "cluster": cluster, "group_by": list(group_by),
                  "interval": "percentile bootstrap, participants resampled"},
    )


# ---------------------------------------------------------------------------
# Paired comparison of two configurations
# ---------------------------------------------------------------------------


def compare_configurations(
    tidy: pd.DataFrame,
    config_col: str,
    a: Any,
    b: Any,
    group_by: Sequence[str] = (),
    *,
    metrics: Optional[Mapping[str, Sequence[str]]] = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    cluster: str = "participant_id",
) -> pd.DataFrame:
    """Paired comparison of configuration ``b`` against ``a`` on matched participants.

    For every group x node x output, only participants scored under both
    configurations enter; both metrics use the same bootstrap replicates, so
    ``diff = b - a`` has a paired percentile interval. ``p_boot`` is the
    two-sided bootstrap p-value (twice the smaller tail share around zero).
    ``better`` names the configuration with the better point estimate given
    the metric direction (lower is better for errors and Brier).
    """
    metrics = {**COMPARE_DEFAULTS, **dict(metrics or {})}
    group_by = [g for g in group_by if g in tidy.columns and g != config_col]
    out: List[Dict[str, Any]] = []
    sub = tidy[tidy[config_col].astype(str).isin([str(a), str(b)]) & (tidy["status"] == "ok")]
    for group, gframe in _group_iter(sub, group_by):
        for (node_id, output), cframe in gframe.groupby(["node_id", "output"], sort=False):
            kind = str(cframe["kind"].iloc[0])
            fa = cframe[cframe[config_col].astype(str) == str(a)]
            fb = cframe[cframe[config_col].astype(str) == str(b)]
            matched = sorted(set(fa[cluster].astype(str)) & set(fb[cluster].astype(str)))
            if len(matched) < 2:
                continue
            fa = fa[fa[cluster].astype(str).isin(matched)]
            fb = fb[fb[cluster].astype(str).isin(matched)]
            index = {p: i for i, p in enumerate(matched)}
            Wp = cluster_weights(len(matched), n_boot, cell_rng(seed, [group, node_id, output, "compare", str(a), str(b)]))
            Wa = Wp[:, fa[cluster].astype(str).map(index).to_numpy(int)]
            Wb = Wp[:, fb[cluster].astype(str).map(index).to_numpy(int)]
            if kind == "regression":
                sa = regression_stats(fa["true_value"].to_numpy(float), fa["pred_value"].to_numpy(float), Wa)
                sb = regression_stats(fb["true_value"].to_numpy(float), fb["pred_value"].to_numpy(float), Wb)
            elif kind in {"binary", "multiclass"}:
                labels = _labels_for(cframe, [])
                pos = cframe["positive_label"].dropna()
                positive = pos.iloc[0] if len(pos) else None
                sa = _cls_all(fa, labels, Wa, kind, positive)
                sb = _cls_all(fb, labels, Wb, kind, positive)
            else:
                continue
            for metric in metrics.get(kind, []):
                if metric not in sa or metric not in sb:
                    continue
                va, vb = np.asarray(sa[metric], float), np.asarray(sb[metric], float)
                diff = vb - va
                point, lo, hi = summarize(diff, ci)
                boot = diff[1:][np.isfinite(diff[1:])]
                p_boot = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))) if boot.size >= 20 else float("nan")
                pa, pa_lo, pa_hi = summarize(va, ci)
                pb, pb_lo, pb_hi = summarize(vb, ci)
                lower = metric in LOWER_IS_BETTER
                better = None
                if np.isfinite(pa) and np.isfinite(pb) and pa != pb:
                    better = str(b) if ((pb < pa) if lower else (pb > pa)) else str(a)
                out.append({**group, "node_id": node_id, "output": output, "kind": kind, "metric": metric,
                            "config_col": config_col, "config_a": str(a), "config_b": str(b), "n_matched": len(matched),
                            "value_a": pa, "value_a_ci_low": pa_lo, "value_a_ci_high": pa_hi,
                            "value_b": pb, "value_b_ci_low": pb_lo, "value_b_ci_high": pb_hi,
                            "diff": point, "diff_ci_low": lo, "diff_ci_high": hi, "p_boot": p_boot,
                            "better": better, "lower_is_better": lower})
    return pd.DataFrame(out)


def _cls_all(frame: pd.DataFrame, labels: List[str], W: np.ndarray, kind: str, positive: Optional[str]) -> Dict[str, np.ndarray]:
    index = {lab: i for i, lab in enumerate(labels)}
    ti = frame["true_label"].astype(str).map(index).to_numpy(int)
    pi = frame["pred_label"].astype(str).map(index).to_numpy(int)
    pos_idx = index.get(positive) if (kind == "binary" and positive is not None and len(labels) == 2) else None
    stats = classification_stats(ti, pi, len(labels), W, positive_idx=pos_idx)
    prob_rows = [i for i, p in enumerate(frame["probs"]) if isinstance(p, dict) and p] if "probs" in frame else []
    if prob_rows and len(prob_rows) == len(frame):
        P = np.zeros((len(prob_rows), len(labels)))
        for r, i in enumerate(prob_rows):
            for lab, v in frame["probs"].iloc[i].items():
                if lab in index:
                    P[r, index[lab]] = v
        Y = np.zeros_like(P)
        Y[np.arange(len(prob_rows)), ti[prob_rows]] = 1.0
        stats.update(probability_stats(Y, P, W[:, prob_rows], binary_positive=pos_idx))
    return stats


# ---------------------------------------------------------------------------
# Test-retest agreement across repeats
# ---------------------------------------------------------------------------


def _icc_reps(Y: np.ndarray, Wp: np.ndarray) -> Dict[str, np.ndarray]:
    """ICC(2,1) absolute agreement and ICC(3,1) consistency (Shrout and Fleiss),
    two-way ANOVA with participants weighted by bootstrap counts."""
    P, k = Y.shape
    N = Wp.sum(1)
    row_mean = Y.mean(1)
    grand = _div(Wp @ row_mean, N)
    col_mean = _div(Wp @ Y, N[:, None])
    ssr = k * (Wp * (row_mean[None, :] - grand[:, None]) ** 2).sum(1)
    ssc = N * ((col_mean - grand[:, None]) ** 2).sum(1)
    sst = (Wp[:, :, None] * (Y[None, :, :] - grand[:, None, None]) ** 2).sum((1, 2))
    sse = sst - ssr - ssc
    msr = _div(ssr, N - 1)
    msc = _div(ssc, np.full_like(N, k - 1.0))
    mse = _div(sse, (N - 1) * (k - 1))
    icc21 = _div(msr - mse, msr + (k - 1) * mse + _div(k * (msc - mse), N))
    icc31 = _div(msr - mse, msr + (k - 1) * mse)
    return {"icc_2_1": icc21, "icc_3_1": icc31, "within_sd": np.sqrt(np.maximum(mse, 0))}


def _kappa_reps(L: np.ndarray, Wp: np.ndarray) -> Dict[str, np.ndarray]:
    """Cohen's kappa (2 repeats) or Fleiss' kappa (3 or more) and percent agreement."""
    P, k = L.shape
    labels = sorted(set(L.ravel().tolist()))
    index = {lab: i for i, lab in enumerate(labels)}
    codes = np.vectorize(index.get)(L)
    K = len(labels)
    agree = (codes == codes[:, :1]).all(1).astype(float)
    out = {"percent_agreement": _div(Wp @ agree, Wp.sum(1))}
    if k == 2:
        stats = classification_stats(codes[:, 0], codes[:, 1], K, Wp)
        out["cohen_kappa"] = stats["kappa"]
    else:
        counts = np.zeros((P, K))
        for j in range(k):
            counts[np.arange(P), codes[:, j]] += 1
        Pi = (counts * (counts - 1)).sum(1) / (k * (k - 1))
        N = Wp.sum(1)
        pbar = _div(Wp @ Pi, N)
        pj = _div(Wp @ counts, (N * k)[:, None])
        pe = (pj ** 2).sum(1)
        out["fleiss_kappa"] = np.where(np.abs(1 - pe) > 1e-12, _div(pbar - pe, 1 - pe), np.nan)
    return out


def test_retest(
    tidy: pd.DataFrame,
    repeat_col: str = "repeat",
    group_by: Sequence[str] = (),
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    cluster: str = "participant_id",
) -> pd.DataFrame:
    """Agreement of predictions across repeats of the same configuration.

    Uses predictions only (truth not needed). Participants with a prediction
    in every repeat enter. Regression: ICC(2,1), ICC(3,1) and the within
    participant SD; labels: Cohen's kappa (2 repeats) or Fleiss' kappa (3 or
    more) and percent agreement. Bootstrap CIs resample participants.
    """
    group_by = [g for g in group_by if g in tidy.columns and g != repeat_col]
    out: List[Dict[str, Any]] = []
    has_pred = np.where(tidy["kind"].isin(["binary", "multiclass"]), tidy["pred_label"].notna(), tidy["pred_value"].notna())
    data = tidy[has_pred]
    for group, gframe in _group_iter(data, group_by):
        for (node_id, output), cframe in gframe.groupby(["node_id", "output"], sort=False):
            kind = str(cframe["kind"].iloc[0])
            col = "pred_value" if kind in {"regression", "multilabel"} else "pred_label"
            wide = cframe.pivot_table(index=cluster, columns=repeat_col, values=col, aggfunc="first")
            wide = wide.dropna()
            if wide.shape[1] < 2 or wide.shape[0] < 3:
                continue
            Wp = cluster_weights(wide.shape[0], n_boot, cell_rng(seed, [group, node_id, output, "retest"]))
            if kind in {"regression"}:
                reps = _icc_reps(wide.to_numpy(float), Wp)
            else:
                reps = _kappa_reps(wide.astype(str).to_numpy(), Wp)
            for metric, values in reps.items():
                point, lo, hi = summarize(values, ci)
                out.append({**group, "node_id": node_id, "output": output, "kind": kind, "metric": metric,
                            "value": point, "ci_low": lo, "ci_high": hi, "n_participants": int(wide.shape[0]),
                            "n_repeats": int(wide.shape[1])})
    return pd.DataFrame(out)


test_retest.__test__ = False
