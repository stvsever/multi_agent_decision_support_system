"""
Sophisticated, dataset-agnostic automated data exploration.

Given any tabular frame (and optional feature specs), this produces a structured
"understanding" of the data that both documents it and grounds the ontology
builder in statistics rather than names alone:

* per-feature profiling with automatic type inference (numeric / binary /
  ordinal / nominal) when specs are not supplied,
* robust distribution stats (skew, kurtosis, quantiles, IQR-outlier fraction),
* missingness profile,
* rank (Spearman) correlation structure over numeric features,
* hierarchical clustering of features into data-driven groups,
* near-duplicate / redundancy detection,
* target associations (if a target is given),
* dataset quality flags (constant, near-constant, highly-missing, redundant).

The output is a JSON-serialisable report. Nothing here is dataset-specific.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- typing
def infer_stat_type(series: pd.Series) -> str:
    """Infer numeric / binary / ordinal / nominal from a raw column."""
    s = series.dropna()
    if s.empty:
        return "nominal"
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_frac = numeric.notna().mean()
    nun = int(s.nunique())
    if numeric_frac >= 0.9:
        if nun <= 2:
            return "binary"
        if nun <= 10 and numeric.dropna().apply(float.is_integer).all():
            return "ordinal"
        return "numeric"
    return "binary" if nun == 2 else "nominal"


def _as_numeric(series: pd.Series, stat_type: str, spec: Dict[str, Any]) -> pd.Series:
    invalid = [float(v) for v in (spec or {}).get("invalid_values", [])]
    if stat_type == "numeric":
        num = pd.to_numeric(series, errors="coerce")
        return num.mask(num.isin(invalid)) if invalid else num
    text = series.astype(str).str.lower().mask(lambda t: t == "n/a")
    if stat_type == "ordinal":
        order = [str(x).lower() for x in (spec or {}).get("ordinal_order", [])]
        mapping = {c: i for i, c in enumerate(order)} if order else {
            c: i for i, c in enumerate(sorted(text.dropna().unique()))}
    else:
        mapping = {c: i for i, c in enumerate(sorted(text.dropna().unique()))}
    return text.map(mapping)


# ------------------------------------------------------------------ profiling
def profile_feature(series: pd.Series, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = spec or {}
    stat_type = str(spec.get("stat_type") or infer_stat_type(series))
    num = _as_numeric(series, stat_type, spec).dropna().astype(float)
    n = len(series)
    present = int(num.shape[0]) if stat_type != "nominal" else int(series.notna().sum())
    prof: Dict[str, Any] = {
        "label": spec.get("label", series.name),
        "stat_type": stat_type,
        "units": spec.get("units"),
        "n_present": present,
        "coverage_pct": round(100.0 * present / n, 1) if n else 0.0,
        "n_unique": int(series.dropna().nunique()),
    }
    if stat_type in ("numeric", "ordinal", "binary") and len(num) > 1:
        q = num.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
        iqr = q[3] - q[1]
        lo, hi = q[1] - 1.5 * iqr, q[3] + 1.5 * iqr
        prof.update({
            "mean": round(float(num.mean()), 4), "std": round(float(num.std(ddof=0)), 4),
            "min": round(float(num.min()), 4), "max": round(float(num.max()), 4),
            "skew": round(float(num.skew()), 3) if len(num) > 2 else None,
            "kurtosis": round(float(num.kurtosis()), 3) if len(num) > 3 else None,
            "quantiles_5_25_50_75_95": q,
            "outlier_fraction": round(float(((num < lo) | (num > hi)).mean()), 3),
        })
    else:
        vc = series.dropna().astype(str).value_counts(normalize=True).head(8)
        prof["categories"] = {k: round(float(v), 3) for k, v in vc.items()}
    return prof


# ---------------------------------------------------------------- correlations
def correlation_analysis(
    df: pd.DataFrame, specs: Dict[str, Dict[str, Any]]
) -> Tuple[pd.DataFrame, List[str]]:
    """Spearman correlation over encoded numeric-like features."""
    cols, encoded = [], {}
    for c, spec in specs.items():
        st = str(spec.get("stat_type") or infer_stat_type(df[c]))
        if st == "nominal":
            continue
        v = _as_numeric(df[c], st, spec)
        if v.notna().sum() > 10 and v.dropna().nunique() > 1:
            encoded[c] = v
            cols.append(c)
    if len(cols) < 2:
        return pd.DataFrame(), cols
    enc = pd.DataFrame(encoded)
    return enc.corr(method="spearman"), cols


def cluster_features(corr: pd.DataFrame, distance_threshold: float = 0.7) -> Dict[str, List[str]]:
    """Hierarchically cluster features on 1 - |Spearman r| (Ward linkage)."""
    if corr.empty or corr.shape[0] < 2:
        return {}
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    # Pairs with no overlapping observations yield NaN correlation -> treat as
    # maximally dissimilar (distance 1.0) so clustering stays well-defined.
    dist = 1.0 - corr.abs().values
    dist = np.nan_to_num(dist, nan=1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")
    clusters: Dict[str, List[str]] = {}
    for col, lab in zip(corr.columns, labels):
        clusters.setdefault(f"cluster_{int(lab)}", []).append(col)
    return clusters


def redundant_pairs(corr: pd.DataFrame, threshold: float = 0.9) -> List[Dict[str, Any]]:
    if corr.empty:
        return []
    out = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                out.append({"a": cols[i], "b": cols[j], "spearman_r": round(float(r), 3)})
    return sorted(out, key=lambda d: -abs(d["spearman_r"]))


def target_associations(
    df: pd.DataFrame, target: str, specs: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    t = pd.to_numeric(df[target], errors="coerce")
    rows = []
    for c, spec in specs.items():
        st = str(spec.get("stat_type") or infer_stat_type(df[c]))
        if st == "nominal":
            continue
        v = _as_numeric(df[c], st, spec)
        m = v.notna() & t.notna()
        if m.sum() > 20 and v[m].nunique() > 1:
            rho = float(pd.Series(v[m].values).corr(pd.Series(t[m].values), method="spearman"))
            rows.append({"feature": c, "label": spec.get("label", c), "spearman_r": round(rho, 3)})
    return sorted(rows, key=lambda d: -abs(d["spearman_r"]))


# --------------------------------------------------------------------- report
def explore(
    df: pd.DataFrame,
    feature_specs: Dict[str, Dict[str, Any]],
    target: Optional[str] = None,
    cluster_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Full automated exploration report for a set of predictor columns."""
    profiles = {c: profile_feature(df[c], feature_specs.get(c, {})) for c in feature_specs}
    corr, corr_cols = correlation_analysis(df, feature_specs)
    clusters = cluster_features(corr, cluster_threshold)
    redun = redundant_pairs(corr)

    flags = {
        "constant": [c for c, p in profiles.items() if p["n_unique"] <= 1],
        "near_constant": [c for c, p in profiles.items()
                          if p.get("std") is not None and p["n_unique"] > 1 and p["std"] < 1e-9],
        "high_missing": [c for c, p in profiles.items() if p["coverage_pct"] < 50.0],
        "highly_skewed": [c for c, p in profiles.items()
                          if p.get("skew") is not None and abs(p["skew"]) > 2.0],
    }
    report: Dict[str, Any] = {
        "n_participants": int(len(df)),
        "n_features": len(feature_specs),
        "type_counts": pd.Series([p["stat_type"] for p in profiles.values()]).value_counts().to_dict(),
        "profiles": profiles,
        "correlation_features": corr_cols,
        "auto_clusters": clusters,
        "n_auto_clusters": len(clusters),
        "redundant_pairs": redun,
        "quality_flags": flags,
    }
    if target and target in df.columns:
        report["target"] = target
        report["target_associations"] = target_associations(df, target, feature_specs)
    return report
