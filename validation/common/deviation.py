"""
Turn raw feature values into COMPASS-style feature evidence.

Not every dataset comes with a normative reference. This module therefore
supports several *reference strategies* and can auto-select between them:

* ``cohort``   - standardise against this batch's own mean/sd (population-referenced).
                 Requires enough participants for a stable sd.
* ``external`` - standardise against externally supplied normative stats
                 ({column: {mean, std}}), e.g. published norms or a healthy cohort.
* ``absolute`` - no normative reference is available (single-subject inference or
                 a batch with no reference group). Raw pre-processed values are
                 passed through as-is. An optional, one-shot LLM range estimate can
                 assign qualitative High/Normal/Low labels from general knowledge.

Whatever the mode, the human-readable value label is always preserved next to the
(possibly null) z-score, because the multi-agent engine reasons over the text
labels, not bare numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

VALID_MODES = ("cohort", "external", "absolute")
MIN_COHORT_N = 20  # below this a batch sd is too unstable to trust as a reference


def resolve_reference_mode(
    requested: str,
    n_participants: int,
    has_external_norms: bool,
) -> str:
    """Decide the reference strategy, honouring an explicit request when valid.

    ``requested`` may be one of VALID_MODES or ``"auto"``. Auto-selection:
    external norms win if present; otherwise a sufficiently large batch is used as
    its own reference; otherwise we fall back to absolute (raw pass-through).
    """
    requested = str(requested or "auto").lower()
    if requested in VALID_MODES:
        if requested == "external" and not has_external_norms:
            raise ValueError("reference_mode='external' requires external_norms")
        if requested == "cohort" and n_participants < MIN_COHORT_N:
            # Respect the request but the caller should be aware; not fatal.
            pass
        return requested
    if requested != "auto":
        raise ValueError(f"Unknown reference_mode '{requested}'")

    if has_external_norms:
        return "external"
    if n_participants >= MIN_COHORT_N:
        return "cohort"
    return "absolute"


def _qualitative_from_z(z: Optional[float]) -> str:
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "Missing"
    if z >= 2.0:
        return "Very High"
    if z >= 1.0:
        return "High"
    if z >= 0.5:
        return "High-Normal"
    if z > -0.5:
        return "Normal"
    if z > -1.0:
        return "Low-Normal"
    if z > -2.0:
        return "Low"
    return "Very Low"


class ReferenceModel:
    """Cohort/external/absolute reference used to encode participants consistently."""

    def __init__(
        self,
        feature_specs: Dict[str, Dict[str, Any]],
        mode: str = "cohort",
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode}")
        self.feature_specs = feature_specs
        self.mode = mode
        self.numeric_stats: Dict[str, Dict[str, float]] = {}
        self.encoding: Dict[str, Dict[str, float]] = {}
        self.summary: Dict[str, str] = {}
        self.llm_ranges: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        df: pd.DataFrame,
        external_norms: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> "ReferenceModel":
        """Fit reference statistics according to ``self.mode``.

        ``df`` is always used to learn categorical encodings and value ranges even
        in absolute mode (so labels are consistent); it is only used for z-score
        statistics in cohort mode.
        """
        for column, spec in self.feature_specs.items():
            stat_type = str(spec.get("stat_type", "numeric")).lower()
            invalid = [float(v) for v in spec.get("invalid_values", [])]
            encoded = self._encode_column(df[column], stat_type, spec, invalid, learn=True)
            clean = encoded.dropna().astype(float)

            if self.mode == "external" and external_norms and column in external_norms:
                mean = float(external_norms[column].get("mean", 0.0))
                std = float(external_norms[column].get("std", 1.0)) or 1.0
                self.summary[column] = f"normative mean {mean:.1f}, sd {std:.1f} (external reference)"
            elif self.mode == "cohort":
                mean = float(clean.mean()) if len(clean) else 0.0
                std = float(clean.std(ddof=0)) if len(clean) else 0.0
                std = std if std > 1e-9 else 1.0
                if stat_type == "numeric":
                    self.summary[column] = f"cohort mean {mean:.1f}, sd {std:.1f} (population-referenced)"
                else:
                    self.summary[column] = self._categorical_summary(df[column])
            else:
                # absolute (or external without a norm for this column): no z baseline.
                mean, std = None, None
                rng = self.llm_ranges.get(column)
                if rng:
                    self.summary[column] = (
                        f"estimated typical range {rng.get('mean', 0):.1f} +/- {rng.get('std', 0):.1f} (LLM prior)"
                    )
                elif stat_type == "numeric":
                    self.summary[column] = "no normative reference (absolute value)"
                else:
                    self.summary[column] = self._categorical_summary(df[column])

            self.numeric_stats[column] = {"mean": mean, "std": std}
        return self

    def set_llm_ranges(self, ranges: Dict[str, Dict[str, float]]) -> None:
        """Attach LLM-estimated {column: {mean, std}} priors for absolute mode."""
        self.llm_ranges = ranges or {}

    def _categorical_summary(self, series: pd.Series) -> str:
        counts = series.astype(str).str.lower()
        counts = counts[counts != "n/a"].value_counts(normalize=True)
        top = ", ".join(f"{k}:{v*100:.0f}%" for k, v in counts.head(4).items())
        return f"cohort distribution {top}" if top else "categorical"

    def _encode_column(self, series, stat_type, spec, invalid, learn=False):
        if stat_type == "numeric":
            num = pd.to_numeric(series, errors="coerce")
            if invalid:
                num = num.mask(num.isin(invalid))
            return num
        text = series.astype(str).str.lower()
        text = text.mask(text == "n/a")
        if learn:
            if stat_type == "ordinal":
                order = [str(x).lower() for x in spec.get("ordinal_order", [])]
                mapping = {cat: float(i) for i, cat in enumerate(order)}
            else:
                cats = sorted([c for c in text.dropna().unique().tolist()])
                mapping = {cat: float(i) for i, cat in enumerate(cats)}
            self.encoding[series.name] = mapping
        return text.map(self.encoding.get(series.name, {}))

    # -------------------------------------------------------------- encode
    def encode_participant(self, row: pd.Series) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for column, spec in self.feature_specs.items():
            stat_type = str(spec.get("stat_type", "numeric")).lower()
            invalid = [float(v) for v in spec.get("invalid_values", [])]
            raw = row.get(column)
            raw_label, numeric = self._participant_value(raw, stat_type, spec, invalid, column)
            present = numeric is not None and not (isinstance(numeric, float) and np.isnan(numeric))

            z: Optional[float] = None
            stats = self.numeric_stats.get(column, {})
            base_mean, base_std = stats.get("mean"), stats.get("std")
            if base_mean is None and column in self.llm_ranges:
                base_mean = self.llm_ranges[column].get("mean")
                base_std = self.llm_ranges[column].get("std")
            if present and base_mean is not None and base_std:
                z = round((numeric - base_mean) / base_std, 3)

            if present and z is not None:
                qualitative = _qualitative_from_z(z)
            elif present:
                qualitative = "Observed"  # raw value, no normative reference
            else:
                qualitative = "Missing"

            out[column] = {
                "present": bool(present),
                "raw_label": raw_label,
                "z_score": z,
                "qualitative": qualitative,
                "reference": self.summary.get(column, ""),
            }
        return out

    def _participant_value(self, raw, stat_type, spec, invalid, column):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return "n/a", None
        text = str(raw).strip()
        if text.lower() in ("", "n/a", "nan"):
            return "n/a", None
        if stat_type == "numeric":
            try:
                val = float(text)
            except ValueError:
                return "n/a", None
            if invalid and val in invalid:
                return "n/a", None
            units = spec.get("units")
            label = f"{val:g}" + (f" {units}" if units else "")
            return label, val
        mapping = self.encoding.get(column, {})
        return text, mapping.get(text.lower())


def estimate_reference_ranges(llm, predictors: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """LLM-estimated {column: {mean, std}} priors for absolute (no-reference) mode.

    A single JSON call asks the model for a plausible population mean and sd for
    each numeric feature, given its label, description and units. This is the
    "clever" fallback that lets the engine still see High/Normal/Low context when
    no cohort or external norms exist. Categorical features are skipped.
    """
    numeric = [p for p in predictors if str(p.get("stat_type")) == "numeric"]
    if not numeric:
        return {}
    lines = "\n".join(
        f"- {p['column']} :: {p['label']} :: {p.get('description','')} :: units={p.get('units') or 'unknown'}"
        for p in numeric
    )
    system = (
        "You are a psychometrics and clinical measurement expert. For each measured feature, "
        "give a plausible healthy adult population MEAN and standard deviation (SD) from general "
        "knowledge of the instrument/scale. Return ONLY JSON."
    )
    user = (
        "Estimate {\"column\": {\"mean\": <float>, \"std\": <float>}} for these features. "
        "If truly unknown, use a reasonable scale-consistent guess.\n\n" + lines +
        "\n\nReturn exactly: {\"ranges\": {\"column\": {\"mean\": 0.0, \"std\": 1.0}}}"
    )
    obj = llm.chat_json(system, user, max_tokens=4000)
    ranges = obj.get("ranges", obj)
    out: Dict[str, Dict[str, float]] = {}
    for p in numeric:
        r = ranges.get(p["column"]) if isinstance(ranges, dict) else None
        if isinstance(r, dict) and "mean" in r and "std" in r:
            try:
                std = float(r["std"]) or 1.0
                out[p["column"]] = {"mean": float(r["mean"]), "std": std}
            except (TypeError, ValueError):
                continue
    return out
