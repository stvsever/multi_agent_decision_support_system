"""
Shape-changing transforms for skewed continuous variables.

Kept separate from z-scoring (a linear rescale that cannot fix skew) and from
the feature-extraction pipeline itself (validation/common/lesion.py, the
NUMERACY_STROKE pipeline scripts) - these are a modeling-time choice, applied
after loading data/processed/*.csv, not baked into the shared processed files.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def rank_int(x) -> np.ndarray:
    """Rank-based inverse normal transform.

    Distribution-free: always produces an (approximately) Gaussian result
    regardless of the input's original shape, at the cost of discarding the
    original spacing between values (only rank order is used). Recommended
    default for ceiling/floor-effect psychometric scores (e.g. aphasia
    severity, composite accuracy scores) where no single parametric transform
    fits well.

    Fit/apply notes: recompute on the training split only, then rank the
    validation/test values against those same training ranks before applying
    this formula, to avoid leaking test-set distribution shape into training.
    """
    x = np.asarray(x, dtype=float)
    ranks = stats.rankdata(x)
    return stats.norm.ppf((ranks - 0.5) / len(ranks))


def flag_outliers_zscore(x, threshold: float = 3.0) -> np.ndarray:
    """Boolean mask, True where |z-score| exceeds threshold (default: 3 SD).

    Operates on raw values (not already-transformed ones). As with the
    transforms above, compute this on the training split only, then apply the
    same training mean/std/threshold to flag (and drop, or keep and just note)
    outliers in validation/test - don't recompute the threshold per split.
    """
    x = np.asarray(x, dtype=float)
    z = (x - np.nanmean(x)) / np.nanstd(x)
    return np.abs(z) > threshold


def log_transform(x) -> np.ndarray:
    """Plain natural log. No parameters to fit - safe to apply to train and test alike.

    Requires strictly positive input (true of lesion_volume: a proportion > 0
    for every subject in this dataset). Raises if any value is <= 0 rather than
    silently producing NaN/-inf.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("log_transform requires all values > 0; got a non-positive value.")
    return np.log(x)


def yeojohnson_transform(x, lmbda: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Yeo-Johnson power transform. Handles negative values (unlike Box-Cox).

    Returns (transformed, lmbda). Pass ``lmbda=None`` to *fit* on this data
    (typically your training split) - the fitted lambda comes back as the
    second return value. Pass that same lambda back in on subsequent calls
    (typically for validation/test) to *apply* the already-fitted transform
    without refitting - refitting per split would leak that split's own
    distribution shape into the transform.
    """
    x = np.asarray(x, dtype=float)
    if lmbda is None:
        transformed, fitted_lmbda = stats.yeojohnson(x)
        return transformed, float(fitted_lmbda)
    return stats.yeojohnson(x, lmbda=lmbda), float(lmbda)


# Per-variable recommended transform for this dataset's DVs/covariates, chosen
# in validation/datasets/NUMERACY_STROKE/utils/inspection.ipynb by comparing
# skewness across candidates on the full cohort:
#   - aphasia_quotient:     rank-INT            (raw skew -3.67 -> -0.12)
#   - approximate_numeracy: Yeo-Johnson         (raw skew -1.20 -> -0.04)
#   - precise_numeracy:     Yeo-Johnson         (raw skew -2.95 -> -0.45; the
#     best-scoring option was reflect+Box-Cox at 0.39, but 47/104 subjects
#     (45%) tie exactly at this variable's ceiling value - that's a point-mass,
#     not ordinary skew, and no monotonic transform fixes it. Box-Cox's fitted
#     lambda there (-2.27) is extreme and likely unstable out-of-sample, so
#     Yeo-Johnson is the more reliable choice despite the slightly worse skew.)
#   - lesion_volume:        log                 (raw skew 2.78 -> -0.05)
RECOMMENDED_TRANSFORM = {
    "aphasia_quotient": "rank_int",
    "approximate_numeracy": "yeojohnson",
    "precise_numeracy": "yeojohnson",
    "lesion_volume": "log",
}


def apply_recommended_transform(column: str, x, fitted_params: Optional[dict] = None):
    """Apply this dataset's recommended transform for a given column by name.

    Returns (transformed, fitted_params). Pass ``fitted_params=None`` when
    fitting (your training split); pass back the returned ``fitted_params``
    on subsequent calls (validation/test) to reuse the fitted lambda instead
    of refitting. rank_int and log have no parameters, so fitted_params is
    always {} for those - the argument only matters for yeojohnson.
    """
    kind = RECOMMENDED_TRANSFORM.get(column)
    if kind is None:
        raise KeyError(
            f"No recommended transform registered for column '{column}'. "
            f"Known columns: {sorted(RECOMMENDED_TRANSFORM)}"
        )
    if kind == "rank_int":
        return rank_int(x), {}
    if kind == "log":
        return log_transform(x), {}
    if kind == "yeojohnson":
        lmbda = (fitted_params or {}).get("lmbda")
        transformed, fitted_lmbda = yeojohnson_transform(x, lmbda=lmbda)
        return transformed, {"lmbda": fitted_lmbda}
    raise AssertionError(f"unreachable: unknown transform kind '{kind}'")
