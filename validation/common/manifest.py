"""
Deterministic (non-LLM) exploration of a tabular dataset's feature structure.

This is the "hardcoded exploration" half of ingestion: before any LLM sees the
data, we profile every candidate predictor column (statistical type, cardinality,
missingness, distribution) and attach human-readable labels/descriptions. The
resulting manifest is the single source of truth that both the ontology builder
and the deviation encoder consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureProfile:
    column: str
    label: str
    description: str
    stat_type: str            # numeric | binary | ordinal | nominal
    units: Optional[str]
    n_present: int
    n_missing: int
    coverage_pct: float
    n_unique: int
    categories: List[str] = field(default_factory=list)
    ordinal_order: List[str] = field(default_factory=list)
    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_numeric(series: pd.Series, invalid_values: List[float]) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    if invalid_values:
        num = num.mask(num.isin(invalid_values))
    return num


def profile_features(
    df: pd.DataFrame,
    feature_specs: Dict[str, Dict[str, Any]],
) -> List[FeatureProfile]:
    """Profile each predictor column described by ``feature_specs``.

    ``feature_specs[column]`` may contain: ``label``, ``description``, ``units``,
    ``stat_type`` (numeric|binary|ordinal|nominal), ``ordinal_order`` (for ordinal),
    and ``invalid_values`` (numeric sentinels to treat as missing, e.g. BMI == 0).
    """
    n_rows = len(df)
    profiles: List[FeatureProfile] = []

    for column, spec in feature_specs.items():
        if column not in df.columns:
            raise KeyError(f"Column '{column}' declared in config is absent from the data")
        raw = df[column]
        stat_type = str(spec.get("stat_type", "numeric")).lower()
        invalid_values = [float(v) for v in spec.get("invalid_values", [])]

        if stat_type == "numeric":
            values = _clean_numeric(raw, invalid_values)
            present_mask = values.notna()
        elif stat_type in ("binary", "ordinal", "nominal"):
            present_mask = raw.notna() & (raw.astype(str).str.lower() != "n/a")
            values = raw.where(present_mask)
        else:
            raise ValueError(f"Unknown stat_type '{stat_type}' for column '{column}'")

        n_present = int(present_mask.sum())
        n_missing = int(n_rows - n_present)
        coverage = round(100.0 * n_present / n_rows, 1) if n_rows else 0.0

        profile = FeatureProfile(
            column=column,
            label=str(spec.get("label", column)),
            description=str(spec.get("description", "")),
            stat_type=stat_type,
            units=spec.get("units"),
            n_present=n_present,
            n_missing=n_missing,
            coverage_pct=coverage,
            n_unique=int(values.dropna().nunique()),
        )

        if stat_type == "numeric":
            clean = values.dropna().astype(float)
            if len(clean):
                profile.mean = round(float(clean.mean()), 4)
                profile.std = round(float(clean.std(ddof=0)), 4)
                profile.minimum = round(float(clean.min()), 4)
                profile.maximum = round(float(clean.max()), 4)
        else:
            cats = sorted(values.dropna().astype(str).unique().tolist())
            profile.categories = cats
            if stat_type == "ordinal":
                order = [str(x) for x in spec.get("ordinal_order", [])]
                # Keep only categories that actually occur, preserving declared order.
                profile.ordinal_order = [c for c in order if c in cats] or cats

        profiles.append(profile)

    return profiles


def build_manifest(
    df: pd.DataFrame,
    dataset_name: str,
    target: Dict[str, Any],
    feature_specs: Dict[str, Dict[str, Any]],
    excluded: Dict[str, str],
) -> Dict[str, Any]:
    """Assemble the full feature manifest for a dataset."""
    profiles = profile_features(df, feature_specs)
    target_series = _clean_numeric(df[target["column"]], [])
    return {
        "dataset": dataset_name,
        "n_participants": int(len(df)),
        "target": {
            "column": target["column"],
            "label": target["label"],
            "description": target["description"],
            "units": target.get("units"),
            "n_present": int(target_series.notna().sum()),
            "mean": round(float(target_series.mean()), 4),
            "std": round(float(target_series.std(ddof=0)), 4),
            "minimum": round(float(target_series.min()), 4),
            "maximum": round(float(target_series.max()), 4),
        },
        "n_predictors": len(profiles),
        "predictors": [p.to_dict() for p in profiles],
        "excluded_columns": excluded,
    }
