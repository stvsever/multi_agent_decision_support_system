"""
Measurement scales for regression outputs answered by a decision model.

A decision model cannot write a number. It can place a participant on an ordered
set of levels (a Score) and return a probability for every level. COMPASS turns a
regression output into such levels in two passes:

1. Coarse pass: the output's full range is cut into at most 10 levels, each a
   numeric interval with a plain-language band ("about 1 SD above the reference
   mean"), because the model reads meaning better than raw numbers.
2. Refinement pass: the three adjacent coarse levels holding the most probability
   are cut into up to 10 finer levels and asked again.

The two distributions are combined into one piecewise-uniform density. Its mean
is the point estimate, so the estimate is continuous and is not snapped to a level
centre; its spread gives the standard deviation and quantiles.

Integer outputs with at most 10 possible values (for example a 0 to 6 item score)
get one level per value and need no refinement: the expectation over the exact
values is already full resolution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ScaleBin:
    """One level: an interval [lo, hi] represented by `center`."""

    lo: float
    hi: float
    center: float
    exact: bool = False  # True when the level is a single integer value

    @property
    def width(self) -> float:
        return 0.0 if self.exact else max(0.0, self.hi - self.lo)


@dataclass
class OutputScale:
    """Range and meaning of one regression output."""

    output: str
    minimum: float
    maximum: float
    integer: bool = False
    unit: str = ""
    description: str = ""
    low_meaning: str = ""
    high_meaning: str = ""
    reference_mean: Optional[float] = None
    reference_sd: Optional[float] = None
    source: str = "default"
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        lo, hi = float(self.minimum), float(self.maximum)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"scale for '{self.output}' needs finite bounds")
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            hi = lo + 1.0
        self.minimum, self.maximum = lo, hi
        if self.reference_sd is not None and (not math.isfinite(float(self.reference_sd)) or float(self.reference_sd) <= 0):
            self.reference_sd = None

    @property
    def span(self) -> float:
        return float(self.maximum - self.minimum)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------ grids
    def exact_integer_levels(self, max_levels: int) -> bool:
        if not self.integer:
            return False
        count = int(round(self.maximum)) - int(round(self.minimum)) + 1
        return 2 <= count <= int(max_levels)

    def coarse_grid(self, levels: int) -> List[ScaleBin]:
        levels = max(2, min(10, int(levels)))
        if self.exact_integer_levels(levels):
            start = int(round(self.minimum))
            stop = int(round(self.maximum))
            return [ScaleBin(lo=v - 0.5, hi=v + 0.5, center=float(v), exact=True) for v in range(start, stop + 1)]
        return equal_width_bins(self.minimum, self.maximum, levels)

    def fine_grid(self, lo: float, hi: float, levels: int) -> List[ScaleBin]:
        levels = max(2, min(10, int(levels)))
        if self.integer:
            first = int(math.ceil(lo - 1e-9))
            last = int(math.floor(hi + 1e-9))
            if first < last and (last - first + 1) <= levels:
                return [ScaleBin(lo=v - 0.5, hi=v + 0.5, center=float(v), exact=True) for v in range(first, last + 1)]
        return equal_width_bins(lo, hi, levels)

    # ----------------------------------------------------------------- labels
    def format_value(self, value: float, *, width: float) -> str:
        if self.integer and width >= 1.0:
            return str(int(round(value)))
        magnitude = abs(width)
        if magnitude >= 10:
            decimals = 0
        elif magnitude >= 1:
            decimals = 1
        elif magnitude >= 0.1:
            decimals = 2
        else:
            decimals = 3
        return f"{value:.{decimals}f}"

    def band(self, center: float) -> str:
        """Plain-language position of a value on this scale."""
        if self.reference_mean is not None and self.reference_sd:
            z = (center - float(self.reference_mean)) / float(self.reference_sd)
            if abs(z) < 0.35:
                text = "close to the reference mean"
            else:
                side = "above" if z > 0 else "below"
                size = abs(z)
                if size < 0.75:
                    text = f"about half a standard deviation {side} the reference mean"
                elif size < 1.25:
                    text = f"about 1 standard deviation {side} the reference mean"
                elif size < 1.75:
                    text = f"about 1.5 standard deviations {side} the reference mean"
                elif size < 2.5:
                    text = f"about 2 standard deviations {side} the reference mean"
                else:
                    text = f"far {side} the reference mean (more than 2.5 standard deviations)"
        else:
            frac = (center - self.minimum) / self.span
            if frac < 0.1:
                text = "at the very low end of the scale"
            elif frac < 0.3:
                text = "in the low part of the scale"
            elif frac < 0.45:
                text = "just below the middle of the scale"
            elif frac <= 0.55:
                text = "in the middle of the scale"
            elif frac <= 0.7:
                text = "just above the middle of the scale"
            elif frac <= 0.9:
                text = "in the high part of the scale"
            else:
                text = "at the very high end of the scale"
        return text

    def level_label(self, b: ScaleBin, *, index: int, total: int) -> str:
        unit = f" {self.unit}" if self.unit else ""
        if b.exact:
            core = f"{self.format_value(b.center, width=1.0)}{unit}"
        else:
            core = (
                f"{self.format_value(b.lo, width=b.width)} to "
                f"{self.format_value(b.hi, width=b.width)}{unit}"
            )
        parts = [core, self.band(b.center)]
        if index == 0 and self.low_meaning:
            parts.append(f"low end: {self.low_meaning}")
        if index == total - 1 and self.high_meaning:
            parts.append(f"high end: {self.high_meaning}")
        return "; ".join(parts)


def equal_width_bins(lo: float, hi: float, levels: int) -> List[ScaleBin]:
    lo, hi = float(lo), float(hi)
    width = (hi - lo) / float(levels)
    bins: List[ScaleBin] = []
    for i in range(levels):
        a = lo + i * width
        b = hi if i == levels - 1 else lo + (i + 1) * width
        bins.append(ScaleBin(lo=a, hi=b, center=(a + b) / 2.0))
    return bins


# --------------------------------------------------------------------- density
@dataclass
class DensityPiece:
    lo: float
    hi: float
    center: float
    mass: float
    exact: bool = False


def best_window(probs: Sequence[float], width: int = 3) -> Tuple[int, int]:
    """Indices [start, stop) of the `width` adjacent levels with the most mass."""
    n = len(probs)
    if n <= width:
        return 0, n
    best_start, best_mass = 0, -1.0
    for start in range(0, n - width + 1):
        mass = float(sum(probs[start : start + width]))
        if mass > best_mass + 1e-12:
            best_start, best_mass = start, mass
    return best_start, best_start + width


def combine_density(
    coarse_bins: Sequence[ScaleBin],
    coarse_probs: Sequence[float],
    *,
    window: Optional[Tuple[int, int]] = None,
    fine_bins: Optional[Sequence[ScaleBin]] = None,
    fine_probs: Optional[Sequence[float]] = None,
) -> List[DensityPiece]:
    """Replace the coarse window with the refined distribution, scaled to its mass."""
    pieces: List[DensityPiece] = []
    total = float(sum(coarse_probs)) or 1.0
    coarse = [float(p) / total for p in coarse_probs]
    if window is None or not fine_bins or not fine_probs:
        for b, p in zip(coarse_bins, coarse):
            pieces.append(DensityPiece(lo=b.lo, hi=b.hi, center=b.center, mass=p, exact=b.exact))
        return pieces
    start, stop = window
    window_mass = float(sum(coarse[start:stop]))
    fine_total = float(sum(fine_probs)) or 1.0
    for i, (b, p) in enumerate(zip(coarse_bins, coarse)):
        if i == start:
            for fb, fp in zip(fine_bins, fine_probs):
                pieces.append(
                    DensityPiece(
                        lo=fb.lo,
                        hi=fb.hi,
                        center=fb.center,
                        mass=window_mass * float(fp) / fine_total,
                        exact=fb.exact,
                    )
                )
        if start <= i < stop:
            continue
        pieces.append(DensityPiece(lo=b.lo, hi=b.hi, center=b.center, mass=p, exact=b.exact))
    return pieces


def density_summary(pieces: Sequence[DensityPiece]) -> Dict[str, float]:
    """Mean, standard deviation and quantiles of a piecewise-uniform density."""
    total = float(sum(p.mass for p in pieces)) or 1.0
    norm = [DensityPiece(p.lo, p.hi, p.center, p.mass / total, p.exact) for p in pieces]
    mean = sum(p.mass * p.center for p in norm)
    second = 0.0
    for p in norm:
        width = 0.0 if p.exact else (p.hi - p.lo)
        second += p.mass * (p.center ** 2 + (width ** 2) / 12.0)
    variance = max(0.0, second - mean ** 2)

    def quantile(q: float) -> float:
        ordered = sorted(norm, key=lambda x: (x.lo, x.hi))
        acc = 0.0
        for p in ordered:
            if p.mass <= 0:
                continue
            if acc + p.mass >= q - 1e-12:
                if p.exact or p.hi <= p.lo:
                    return p.center
                frac = (q - acc) / p.mass
                return p.lo + max(0.0, min(1.0, frac)) * (p.hi - p.lo)
            acc += p.mass
        last = ordered[-1]
        return last.hi if not last.exact else last.center

    return {
        "mean": float(mean),
        "sd": float(math.sqrt(variance)),
        "q05": float(quantile(0.05)),
        "q25": float(quantile(0.25)),
        "median": float(quantile(0.50)),
        "q75": float(quantile(0.75)),
        "q95": float(quantile(0.95)),
    }


def default_scale(output: str, *, unit: str = "") -> OutputScale:
    """Last-resort scale when neither the task nor the compiler gives a range."""
    return OutputScale(
        output=output,
        minimum=-3.0,
        maximum=3.0,
        integer=False,
        unit=unit or "standardized units",
        description=output.replace("_", " "),
        reference_mean=0.0,
        reference_sd=1.0,
        source="default",
        notes=["No range was provided or compiled; a standardized -3 to 3 scale was assumed."],
    )


def scale_from_mapping(output: str, data: Dict[str, Any], *, source: str, unit: str = "") -> Optional[OutputScale]:
    """Build a scale from a task-spec or compiler mapping; None when unusable."""
    if not isinstance(data, dict):
        return None

    def num(*keys: str) -> Optional[float]:
        for key in keys:
            value = data.get(key)
            if value is None or value == "":
                continue
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(out):
                return out
        return None

    lo = num("min", "minimum", "lo", "low")
    hi = num("max", "maximum", "hi", "high")
    if lo is None or hi is None or hi == lo:
        return None
    integer_raw = data.get("integer")
    if isinstance(integer_raw, str):
        integer = integer_raw.strip().lower() in ("1", "true", "yes")
    else:
        integer = bool(integer_raw)
    return OutputScale(
        output=output,
        minimum=lo,
        maximum=hi,
        integer=integer,
        unit=str(data.get("unit") or unit or "").strip(),
        description=str(data.get("description") or "").strip(),
        low_meaning=str(data.get("low_meaning") or "").strip(),
        high_meaning=str(data.get("high_meaning") or "").strip(),
        reference_mean=num("reference_mean", "mean"),
        reference_sd=num("reference_sd", "sd"),
        source=source,
    )
