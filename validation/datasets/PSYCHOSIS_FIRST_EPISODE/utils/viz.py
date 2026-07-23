"""Cohort and single-subject EEG visualizations.

Two entry points:
  build_cohort_figures()      group-level multi-panel PNGs (results/figures/cohort)
  build_subject_dashboard()   one representative-psychosis dashboard
                              (results/figures/example_subject)

Topographies need per-electrode values, but the feature matrix is region-averaged,
so a per-channel spectral summary is computed once from the harmonized checkpoints
and cached to results/channel_spectral_summary.csv.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Note: no backend is forced here so this module displays inline inside notebooks.
# Headless batch scripts set matplotlib.use("Agg") before importing this module.
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mne.time_frequency import psd_array_multitaper
from scipy.stats import gaussian_kde

from .config import (BAND_NAMES, BANDS, FIGURE_ROOT, RESULTS_ROOT, SUBJECT_ROOT,
                     load_config)
from .montage import COMMON_CORTICAL, NODES, REGIONS, scope_indices

mne.set_log_level("ERROR")

GROUP_COLORS = {"Control": "#2c7fb8", "Psychosis": "#d95f0e"}
BAND_PRETTY = {"delta_1_4_hz": "Delta (1-4 Hz)", "theta_4_8_hz": "Theta (4-8 Hz)",
               "alpha_8_13_hz": "Alpha (8-13 Hz)", "beta_13_30_hz": "Beta (13-30 Hz)",
               "low_gamma_30_45_hz": "Low gamma (30-45 Hz)"}
MS_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]


# --------------------------------------------------------------------------- #
# Shared data loading
# --------------------------------------------------------------------------- #
def montage_info() -> mne.Info:
    info = mne.create_info(list(COMMON_CORTICAL), sfreq=250.0, ch_types="eeg")
    info.set_montage("standard_1005")
    return info


_RECORDS_CACHE = None


def load_records_cached():
    """Discover records with preprocessing QC attached, cached within a session."""
    global _RECORDS_CACHE
    if _RECORDS_CACHE is None:
        from .extract import load_records
        _RECORDS_CACHE = load_records(load_config())
    return _RECORDS_CACHE


def load_tables() -> dict[str, Any]:
    eeg = pd.read_csv(RESULTS_ROOT / "eeg_features.csv")
    non_eeg = pd.read_csv(RESULTS_ROOT / "non_eeg_features.csv")
    z = pd.read_csv(RESULTS_ROOT / "eeg_features_zscores.csv")
    meta_cols = ["recording_id", "dataset_id",
                 "target__psychosis__case_control_label",
                 "covariate__demographics__age_years",
                 "covariate__demographics__sex_source_reported"]
    meta = non_eeg[meta_cols].rename(columns={
        "target__psychosis__case_control_label": "group",
        "covariate__demographics__age_years": "age",
        "covariate__demographics__sex_source_reported": "sex"})
    merged = eeg.merge(meta, on="recording_id", how="left")
    return {"eeg": eeg, "non_eeg": non_eeg, "z": z, "meta": meta, "merged": merged}


def build_channel_summary(records=None, cfg=None, force: bool = False) -> pd.DataFrame:
    """Per-channel relative/absolute band power, alpha peak, aperiodic exponent."""
    out_path = RESULTS_ROOT / "channel_spectral_summary.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    from fooof import FOOOF
    cfg = cfg or load_config()
    if records is None:
        from .extract import load_records
        records = load_records(cfg)
    rows = []
    eligible = [r for r in records
                if r.get("preprocessing_qc", {}).get("feature_eligible", False)]
    for k, record in enumerate(eligible, 1):
        raw = mne.io.read_raw_fif(SUBJECT_ROOT / record["recording_id"] / "clean_raw.fif",
                                  preload=True, verbose="ERROR")
        raw.reorder_channels(COMMON_CORTICAL)
        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose="ERROR")
        psd, freqs = psd_array_multitaper(epochs.get_data(copy=False), sfreq=250.0,
                                          fmin=1.0, fmax=45.0, bandwidth=1.5,
                                          adaptive=False, low_bias=True,
                                          normalization="full", output="power",
                                          n_jobs=1, verbose="ERROR")
        psd = np.nanmedian(psd, axis=0) * 1e12          # (49, n_freq) uV^2/Hz
        df = float(np.median(np.diff(freqs)))
        total = psd.sum(1) * df
        row = {"recording_id": record["recording_id"], "dataset_id": record["dataset_id"]}
        for b in BAND_NAMES:
            lo, hi = BANDS[b]
            mask = (freqs >= lo) & (freqs <= hi if b == BAND_NAMES[-1] else freqs < hi)
            bp = psd[:, mask].sum(1) * df
            for ci, ch in enumerate(COMMON_CORTICAL):
                row[f"abs__{b}__{ch}"] = float(np.log10(max(bp[ci], 1e-30)))
                row[f"rel__{b}__{ch}"] = float(bp[ci] / max(total[ci], 1e-30))
        for ci, ch in enumerate(COMMON_CORTICAL):
            fm = FOOOF(peak_width_limits=(1, 12), max_n_peaks=6, min_peak_height=0.05,
                       aperiodic_mode="fixed", verbose=False)
            try:
                fm.fit(freqs, psd[ci], (1, 45))
                row[f"exponent__{ch}"] = float(fm.aperiodic_params_[-1])
                peaks = np.atleast_2d(fm.get_params("peak_params"))
                alpha = peaks[(peaks[:, 0] >= 7) & (peaks[:, 0] <= 14)] if peaks.size else np.empty((0, 3))
                row[f"alpha_cf__{ch}"] = float(alpha[np.argmax(alpha[:, 1]), 0]) if alpha.size else np.nan
            except Exception:
                row[f"exponent__{ch}"] = np.nan
                row[f"alpha_cf__{ch}"] = np.nan
        rows.append(row)
        raw.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    return frame


def _group_channel_means(chan: pd.DataFrame, meta: pd.DataFrame, prefix: str,
                         band: str | None = None):
    m = chan.merge(meta[["recording_id", "group"]], on="recording_id", how="left")
    out = {}
    for group in ("Control", "Psychosis"):
        sub = m[m["group"] == group]
        if band:
            cols = [f"{prefix}__{band}__{ch}" for ch in COMMON_CORTICAL]
        else:
            cols = [f"{prefix}__{ch}" for ch in COMMON_CORTICAL]
        out[group] = sub[cols].mean().to_numpy()
    return out


def _topo(ax, values, info, title, cmap="RdBu_r", vlim=(None, None)):
    # sphere="eeglab" + a wide extrapolation makes the scalp map fill the axes (large,
    # no square raster edges); the axes frame is removed so only the round head shows.
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap, vlim=vlim,
                                 contours=4, sensors=True, extrapolate="head",
                                 outlines="head", sphere="eeglab")
    ax.set_title(title, fontsize=12, pad=2)
    ax.set_frame_on(False)
    return im


def _kde_by_group(ax, merged, column, xlabel, title):
    for group in ("Control", "Psychosis"):
        vals = pd.to_numeric(merged.loc[merged["group"] == group, column], errors="coerce").dropna()
        if len(vals) < 5:
            continue
        xs = np.linspace(vals.min(), vals.max(), 200)
        try:
            ax.plot(xs, gaussian_kde(vals)(xs), color=GROUP_COLORS[group], lw=2, label=group)
            ax.fill_between(xs, gaussian_kde(vals)(xs), color=GROUP_COLORS[group], alpha=0.15)
        except Exception:
            ax.hist(vals, bins=15, color=GROUP_COLORS[group], alpha=0.4, density=True, label=group)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)


def _cohens_d(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan


def _load_connectivity(records, meta):
    """Group-averaged wPLI matrices per band: {group: {band: 10x10}}."""
    acc = {g: {b: [] for b in BAND_NAMES} for g in ("Control", "Psychosis")}
    gmap = dict(zip(meta["recording_id"], meta["group"]))
    for record in records:
        npz = SUBJECT_ROOT / record["recording_id"] / "connectivity_matrices.npz"
        g = gmap.get(record["recording_id"])
        if not npz.exists() or g not in acc:
            continue
        data = np.load(npz)
        for b in BAND_NAMES:
            if b in data:
                acc[g][b].append(data[b])
    return {g: {b: (np.nanmean(v, axis=0) if v else np.full((10, 10), np.nan))
                for b, v in bands.items()} for g, bands in acc.items()}


# --------------------------------------------------------------------------- #
# Cohort figures
# --------------------------------------------------------------------------- #
def fig_cohort_overview(tables):
    meta, merged = tables["meta"], tables["merged"]
    qc = tables["qc"]
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    fig.suptitle("Cohort inventory and preprocessing quality", fontsize=15, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    counts = meta.groupby(["dataset_id", "group"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=ax, color=[GROUP_COLORS.get(c, "#888") for c in counts.columns])
    ax.set_title("Recordings by dataset and group"); ax.set_xlabel(""); ax.set_ylabel("n")
    ax.tick_params(axis="x", rotation=0); ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for group in ("Control", "Psychosis"):
        ages = pd.to_numeric(meta.loc[meta["group"] == group, "age"], errors="coerce").dropna()
        ax.hist(ages, bins=18, alpha=0.55, color=GROUP_COLORS[group], label=group)
    ax.set_title("Age distribution"); ax.set_xlabel("age (years)"); ax.set_ylabel("n")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    sex = meta.groupby(["group", "sex"]).size().unstack(fill_value=0)
    sex.plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1")
    ax.set_title("Sex by group"); ax.set_xlabel(""); ax.set_ylabel("n")
    ax.tick_params(axis="x", rotation=0); ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(qc["interpolated_channel_count"].dropna(), bins=range(0, 12),
            color="#7570b3", alpha=0.8)
    ax.set_title("Interpolated channels per recording"); ax.set_xlabel("channels"); ax.set_ylabel("n")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(qc["epoch_count_retained"].dropna(), bins=20, color="#1b9e77", alpha=0.8)
    ax.set_title("Retained 4 s epochs per recording"); ax.set_xlabel("epochs"); ax.set_ylabel("n")

    ax = fig.add_subplot(gs[1, 2])
    elig = qc.groupby(["dataset_id", "feature_eligible"]).size().unstack(fill_value=0)
    elig.plot(kind="bar", ax=ax, color=["#d95f0e", "#1b9e77"])
    ax.set_title("Feature eligibility"); ax.set_xlabel(""); ax.set_ylabel("n")
    ax.tick_params(axis="x", rotation=0); ax.legend(["excluded", "eligible"], frameon=False, fontsize=8)
    return fig


def fig_spectral_topography(tables, chan):
    meta = tables["meta"]
    info = montage_info()
    fig = plt.figure(figsize=(16, 7.5), constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1.15])
    fig.suptitle("Relative band power topography by group (group mean)", fontsize=15, fontweight="bold")
    for bi, b in enumerate(BAND_NAMES):
        gm = _group_channel_means(chan, meta, "rel", b)
        vmax = np.nanpercentile(np.concatenate([gm["Control"], gm["Psychosis"]]), 98)
        vmin = np.nanpercentile(np.concatenate([gm["Control"], gm["Psychosis"]]), 2)
        for gi, group in enumerate(("Control", "Psychosis")):
            ax = fig.add_subplot(gs[gi, bi])
            im = _topo(ax, gm[group], info, "" if gi else BAND_PRETTY[b],
                       cmap="viridis", vlim=(vmin, vmax))
            if bi == 0:
                ax.text(-0.25, 0.5, group, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold",
                        color=GROUP_COLORS[group])
        fig.colorbar(im, ax=[fig.axes[-1]], fraction=0.046, shrink=0.8)
    ax = fig.add_subplot(gs[:, 5])
    for group in ("Control", "Psychosis"):
        recs = meta.loc[meta["group"] == group, "recording_id"]
        cols = {b: [f"rel__{b}__{ch}" for ch in COMMON_CORTICAL] for b in BAND_NAMES}
        m = chan[chan["recording_id"].isin(recs)]
        band_means = [m[cols[b]].to_numpy().mean() for b in BAND_NAMES]
        ax.plot(range(len(BAND_NAMES)), band_means, "o-", color=GROUP_COLORS[group],
                lw=2, label=group)
    ax.set_xticks(range(len(BAND_NAMES)))
    ax.set_xticklabels([BAND_PRETTY[b].split(" ")[0] for b in BAND_NAMES], rotation=30, fontsize=8)
    ax.set_ylabel("mean relative power"); ax.set_title("Global spectral profile", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


def fig_alpha_aperiodic(tables, chan):
    meta, merged = tables["meta"], tables["merged"]
    info = montage_info()
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig)
    fig.suptitle("Alpha peak and aperiodic 1/f structure", fontsize=15, fontweight="bold")

    acf = _group_channel_means(chan, meta, "alpha_cf")
    vlim = (np.nanpercentile(np.concatenate(list(acf.values())), 5),
            np.nanpercentile(np.concatenate(list(acf.values())), 95))
    for gi, group in enumerate(("Control", "Psychosis")):
        ax = fig.add_subplot(gs[0, gi])
        im = _topo(ax, acf[group], info, f"Alpha peak freq: {group}", cmap="plasma", vlim=vlim)
    fig.colorbar(im, ax=[fig.axes[-1]], fraction=0.046, shrink=0.8, label="Hz")
    _kde_by_group(fig.add_subplot(gs[0, 2]), merged,
                  "B_alpha_peak__center_frequency_hz__occipital_left",
                  "Hz", "Occipital alpha peak frequency")
    _kde_by_group(fig.add_subplot(gs[0, 3]), merged,
                  "B_alpha_peak__power_log10_uv2_above_aperiodic__occipital_left",
                  "log10 uV^2", "Occipital alpha peak power")

    exp = _group_channel_means(chan, meta, "exponent")
    vlim = (np.nanpercentile(np.concatenate(list(exp.values())), 5),
            np.nanpercentile(np.concatenate(list(exp.values())), 95))
    for gi, group in enumerate(("Control", "Psychosis")):
        ax = fig.add_subplot(gs[1, gi])
        im = _topo(ax, exp[group], info, f"1/f exponent: {group}", cmap="cividis", vlim=vlim)
    fig.colorbar(im, ax=[fig.axes[-1]], fraction=0.046, shrink=0.8)
    _kde_by_group(fig.add_subplot(gs[1, 2]), merged,
                  "C_aperiodic__exponent__global", "exponent", "Global 1/f exponent")
    _kde_by_group(fig.add_subplot(gs[1, 3]), merged,
                  "C_aperiodic__offset_log10_uv2__global", "log10 uV^2", "Global 1/f offset")
    return fig


def fig_microstates(tables):
    merged = tables["merged"]
    templates = np.load(SUBJECT_ROOT.parent / "microstate_templates" / "group_templates.npz")
    maps, chans = templates["maps"], list(templates["channels"])
    info = mne.create_info(chans, 250.0, "eeg"); info.set_montage("standard_1005")
    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig)
    fig.suptitle("Microstates: group templates (Koenig A-D) and dynamics", fontsize=15, fontweight="bold")
    for i, letter in enumerate("ABCD"):
        ax = fig.add_subplot(gs[0, i])
        _topo(ax, maps[i], info, f"Microstate {letter}", cmap="RdBu_r",
              vlim=(-np.abs(maps[i]).max(), np.abs(maps[i]).max()))

    metrics = [("coverage_fraction", "Coverage fraction"),
               ("mean_duration_ms", "Mean duration (ms)"),
               ("occurrence_per_second", "Occurrence (per s)")]
    for mi, (metric, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, mi])
        x = np.arange(4); w = 0.38
        for gi, group in enumerate(("Control", "Psychosis")):
            vals = [pd.to_numeric(merged.loc[merged["group"] == group,
                    f"F_microstates__class_{s}__{metric}"], errors="coerce").mean()
                    for s in "abcd"]
            ax.bar(x + (gi - 0.5) * w, vals, w, color=GROUP_COLORS[group], label=group)
        ax.set_xticks(x); ax.set_xticklabels(list("ABCD"))
        ax.set_title(title, fontsize=10)
        if mi == 0:
            ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[1, 3])
    trans = np.zeros((4, 4))
    for i, s in enumerate("abcd"):
        for j, t in enumerate("abcd"):
            if i == j:
                continue
            trans[i, j] = pd.to_numeric(
                merged[f"F_microstates__transition_probability__class_{s}_to_class_{t}"],
                errors="coerce").mean()
    im = ax.imshow(trans, cmap="magma", vmin=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(list("ABCD"))
    ax.set_yticks(range(4)); ax.set_yticklabels(list("ABCD"))
    ax.set_title("Transition probability (cohort mean)", fontsize=10)
    ax.set_xlabel("to"); ax.set_ylabel("from")
    for i in range(4):
        for j in range(4):
            if i != j:
                ax.text(j, i, f"{trans[i,j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def fig_connectivity(tables, records):
    conn = _load_connectivity(records, tables["meta"])
    info = montage_info()
    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = GridSpec(2, 5, figure=fig)
    fig.suptitle("Debiased wPLI^2 connectivity between the 10 regions (group mean)",
                 fontsize=15, fontweight="bold")
    node_labels = [n.replace("_", " ") for n in NODES]
    vmax = np.nanmax([np.nanmax(conn[g][b]) for g in conn for b in BAND_NAMES])
    for bi, b in enumerate(BAND_NAMES):
        for gi, group in enumerate(("Control", "Psychosis")):
            ax = fig.add_subplot(gs[gi, bi])
            im = ax.imshow(conn[group][b], cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(BAND_PRETTY[b].split(" ")[0] if gi == 0 else "", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if bi == 0:
                ax.set_ylabel(group, color=GROUP_COLORS[group], fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=fig.axes, fraction=0.02, label="wPLI2 debiased")
    return fig


def fig_graph_metrics(tables):
    merged = tables["merged"]
    metrics = [("global_efficiency_auc_density_20_50_percent", "Global efficiency"),
               ("mean_clustering_coefficient_auc_density_20_50_percent", "Clustering"),
               ("characteristic_path_length_auc_density_20_50_percent", "Path length"),
               ("modularity_q_auc_density_20_50_percent", "Modularity Q"),
               ("small_world_propensity_auc_density_20_50_percent", "Small-world propensity")]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    fig.suptitle("Graph-theory metrics across frequency bands (group mean +/- SE)",
                 fontsize=15, fontweight="bold")
    x = np.arange(len(BAND_NAMES))
    for ax, (metric, title) in zip(axes, metrics):
        for group in ("Control", "Psychosis"):
            means, ses = [], []
            for b in BAND_NAMES:
                vals = pd.to_numeric(merged.loc[merged["group"] == group,
                        f"H_graph__global__{metric}__{b}"], errors="coerce").dropna()
                means.append(vals.mean()); ses.append(vals.std() / max(np.sqrt(len(vals)), 1))
            ax.errorbar(x, means, yerr=ses, marker="o", color=GROUP_COLORS[group],
                        capsize=3, lw=2, label=group)
        ax.set_xticks(x); ax.set_xticklabels([b.split("_")[0] for b in BAND_NAMES],
                                             rotation=30, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    return fig


def fig_complexity(tables):
    merged = tables["merged"]
    measures = [("D_entropy__sample_entropy", "Sample entropy"),
                ("D_entropy__permutation_entropy_normalized", "Permutation entropy"),
                ("D_entropy__lempel_ziv_complexity_normalized", "Lempel-Ziv complexity"),
                ("E_fractal__higuchi_fractal_dimension", "Higuchi dimension"),
                ("E_fractal__detrended_fluctuation_exponent", "DFA exponent")]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    fig.suptitle("Signal complexity and fractal scaling (global, by group)",
                 fontsize=15, fontweight="bold")
    for ax, (measure, title) in zip(axes, measures):
        data = [pd.to_numeric(merged.loc[merged["group"] == g, f"{measure}__global"],
                              errors="coerce").dropna() for g in ("Control", "Psychosis")]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc, group in zip(parts["bodies"], ("Control", "Psychosis")):
            pc.set_facecolor(GROUP_COLORS[group]); pc.set_alpha(0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Control", "Psychosis"], fontsize=8)
        ax.set_title(title, fontsize=10)
    return fig


FAMILY_COLORS = {"A": "#4e79a7", "B": "#f28e2b", "C": "#59a14f", "D": "#e15759",
                 "E": "#b07aa1", "F": "#76b7b2", "G": "#ff9da7", "H": "#9c755f"}
FAMILY_NAMES = {"A": "Spectral", "B": "Alpha peak", "C": "Aperiodic", "D": "Entropy",
                "E": "Fractal", "F": "Microstates", "G": "Connectivity", "H": "Graph"}


def _stars(q) -> str:
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _normality_ok(x, alpha=0.05, min_n=8) -> bool:
    """Shapiro-Wilk normality gate for choosing a parametric vs non-parametric test."""
    from scipy.stats import shapiro
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < min_n or np.allclose(x, x[0]):
        return False
    try:
        return float(shapiro(x)[1]) > alpha
    except Exception:
        return False


# Pre-specified psychosis resting-EEG hypothesis panel: individual, theory-driven
# features (one per "neural entity" - a band x region x measure) spanning every
# well-replicated psychosis signature. This is intentionally broad (50+ features) and
# NOT FDR-corrected across the whole panel: with this many correlated hypotheses a
# whole-panel FDR would inflate every q past significance and hide the real effects, so
# the panel is reported on nominal (uncorrected) p < 0.05, which is the appropriate read
# for a focused, pre-registered-style signature list.
def psychosis_signature_features(all_cols) -> list[str]:
    """Return the pre-specified psychosis-signature features present in ``all_cols``."""
    present, panel = set(all_cols), []

    def add(*needles):
        for c in all_cols:
            if c in present and c not in panel and all(n in c for n in needles):
                panel.append(c)

    posterior = ["occipital_left", "occipital_right", "parietal_left", "parietal_right"]
    anterior = ["frontal_left", "frontal_right", "global"]
    # Posterior alpha power deficit (relative + absolute), the canonical marker
    for meas in ("relative_power", "log10_absolute_power"):
        for s in posterior + ["global"]:
            add("A_spectral", meas, "alpha_8_13", s)
    # Frontal / global slow-wave (delta, theta) excess (relative + absolute)
    for meas in ("relative_power", "log10_absolute_power"):
        for band in ("delta_1_4", "theta_4_8"):
            for s in anterior:
                add("A_spectral", meas, band, s)
    # Spectral slowing ratios
    for ratio in ("theta_over_alpha", "alpha_over_delta"):
        for s in ["global"] + posterior:
            add("A_spectral", "natural_log_power_ratio", ratio, s)
    # Alpha-peak slowing (centre frequency) and peak power
    for s in ["global"] + posterior:
        add("B_alpha_peak", "center_frequency", s)
    add("B_alpha_peak", "power_log10", "global")
    # Aperiodic 1/f slope and offset
    for s in ("global", "frontal_left", "frontal_right", "occipital_left", "occipital_right"):
        add("C_aperiodic", "exponent", s)
    add("C_aperiodic", "offset", "global")
    # Signal complexity / entropy (all four measures) global + posterior
    for meas in ("sample_entropy", "permutation_entropy", "spectral_entropy", "lempel_ziv"):
        for s in ["global"] + posterior:
            add("D_entropy", meas, s)
    # Fractal long-range dynamics
    for meas in ("higuchi", "detrended_fluctuation"):
        for s in ("global", "occipital_left", "frontal_left"):
            add("E_fractal", meas, s)
    # Microstate coverage (A-D) and global dynamics
    for cl in ("class_a", "class_b", "class_c", "class_d"):
        add("F_microstates", cl, "coverage")
    add("F_microstates", "global", "transition_entropy")
    add("F_microstates", "global", "sequence")
    # Graph-theoretic network organisation (alpha band)
    for met in ("mean_edge_weight__alpha_8_13", "global_efficiency", "modularity_q",
                "characteristic_path_length", "small_world_propensity"):
        add("H_graph", "global", met)
    return panel


def signature_groups(all_cols) -> dict[str, list[str]]:
    """The psychosis-signature panel grouped by feature family (for the stats + figures)."""
    groups: dict[str, list[str]] = {}
    for c in psychosis_signature_features(all_cols):
        groups.setdefault(c.split("__")[0], []).append(c)
    return groups


def group_distribution_stats(tables, groups, correction: str = "fdr_family") -> pd.DataFrame:
    """Per-feature Control vs Psychosis distribution comparison.

    For each feature: Shapiro-Wilk normality is tested in each group; ONLY if BOTH
    groups pass is a Welch t-test used, otherwise the non-parametric Mann-Whitney U rank
    test (so a non-normal feature is never tested with a t-test). The effect size is
    Cohen's d (Psychosis minus Control); a rank-biserial correlation accompanies the
    non-parametric tests.

    ``correction`` controls the ``q`` column and the ``significant`` flag:
      - ``"fdr_family"``: Benjamini-Hochberg FDR within each feature family (exploratory
        836-feature sweep).
      - ``"fdr_global"``: one BH correction across all tested features.
      - ``"none"``: no correction; ``q = p`` and ``significant`` = nominal p < 0.05 (used
        for the broad pre-specified signature panel, where whole-panel FDR would be far
        too conservative for this many correlated hypotheses).
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    from statsmodels.stats.multitest import multipletests
    merged = tables["merged"]
    con = merged["group"] == "Control"
    psy = merged["group"] == "Psychosis"
    rows = []
    for fam, cols in groups.items():
        letter = fam[0]
        for c in cols:
            a = pd.to_numeric(merged.loc[con, c], errors="coerce").to_numpy(); a = a[np.isfinite(a)]
            b = pd.to_numeric(merged.loc[psy, c], errors="coerce").to_numpy(); b = b[np.isfinite(b)]
            rec = {"feature": c, "family": letter, "n_control": len(a), "n_psychosis": len(b),
                   "median_control": float(np.median(a)) if len(a) else np.nan,
                   "median_psychosis": float(np.median(b)) if len(b) else np.nan,
                   "cohens_d": _cohens_d(a, b), "rank_biserial": np.nan,
                   "test": "insufficient_n", "normal": False, "p": np.nan}
            if len(a) >= 8 and len(b) >= 8:
                normal = _normality_ok(a) and _normality_ok(b)
                rec["normal"] = normal
                if normal:
                    rec["test"] = "Welch t"
                    rec["p"] = float(ttest_ind(a, b, equal_var=False).pvalue)
                else:
                    rec["test"] = "Mann-Whitney U"
                    u, p = mannwhitneyu(a, b, alternative="two-sided")
                    rec["p"] = float(p)
                    rec["rank_biserial"] = float(2.0 * u / (len(a) * len(b)) - 1.0)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["q"] = np.nan
    df["nominal"] = df["p"] < 0.05  # uncorrected significance
    if correction == "none":
        df["q"] = df["p"]
    elif correction == "fdr_global":
        m = df["p"].notna()
        if m.any():
            df.loc[m, "q"] = multipletests(df.loc[m, "p"].to_numpy(), method="fdr_bh")[1]
    else:  # "fdr_family"
        for fam in df["family"].unique():
            sub = df[df["family"] == fam]
            m = sub["p"].notna()
            if m.any():
                q = multipletests(sub.loc[m, "p"].to_numpy(), method="fdr_bh")[1]
                df.loc[sub.index[m.to_numpy()], "q"] = q
    df["stars"] = df["q"].apply(_stars)
    df["significant"] = df["nominal"] if correction == "none" else (df["q"] < 0.05)
    return df


def fig_group_contrast(tables, groups, stats=None):
    """Manhattan of standardized group differences by family, with a clean ranked
    panel of the strongest features (no overlapping in-plot labels)."""
    merged = tables["merged"]
    is_control = merged["group"] == "Control"
    is_psych = merged["group"] == "Psychosis"
    star_of = dict(zip(stats["feature"], stats["stars"])) if stats is not None else {}
    fig = plt.figure(figsize=(16, 8.8), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.05], figure=fig)

    ax = fig.add_subplot(gs[0])
    x = 0; ticks = []; tick_labels = []; records = []
    for fam, cols in groups.items():
        letter = fam[0]
        xs, ds, sig = [], [], []
        for c in cols:
            d = _cohens_d(merged.loc[is_control, c], merged.loc[is_psych, c])
            records.append((abs(d) if np.isfinite(d) else 0.0, d, c, letter))
            xs.append(x); ds.append(d); sig.append(bool(star_of.get(c, "")))
            x += 1
        xs, ds, sig = np.array(xs), np.array(ds), np.array(sig)
        ax.scatter(xs[~sig], ds[~sig], s=11, color=FAMILY_COLORS[letter], alpha=0.45)
        ax.scatter(xs[sig], ds[sig], s=40, color=FAMILY_COLORS[letter], edgecolor="k",
                   lw=0.7, zorder=3)
        ticks.append(xs.mean()); tick_labels.append(letter)
        x += 6
    ax.axhline(0, color="k", lw=0.6)
    for thr in (0.5, -0.5, 0.8, -0.8):
        ax.axhline(thr, color="gray", ls="--", lw=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}\n{FAMILY_NAMES[t]}" for t in tick_labels], fontsize=8)
    ax.set_ylabel("Cohen's d (Psychosis - Control)")
    title = "Standardized group difference across all 836 features by family"
    if stats is not None:
        title += "\n(filled points = FDR-significant, q < 0.05)"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax2 = fig.add_subplot(gs[1])
    top = sorted(records, reverse=True)[:16][::-1]
    labels = [c.split("__", 1)[1].replace("__", " / ")[:52] for _, _, c, _ in top]
    vals = [d for _, d, _, _ in top]
    colors = [FAMILY_COLORS[l] for _, _, _, l in top]
    ax2.barh(range(len(vals)), vals, color=colors, edgecolor="k", lw=0.4)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, fontsize=7)
    ax2.axvline(0, color="k", lw=0.6)
    for thr in (0.5, -0.5, 0.8, -0.8):
        ax2.axvline(thr, color="gray", ls="--", lw=0.5)
    for i, (_, d, c, _) in enumerate(top):
        s = star_of.get(c, "")
        if s:
            ax2.text(d + (0.03 if d >= 0 else -0.03), i, s, va="center",
                     ha="left" if d >= 0 else "right", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Cohen's d (Psychosis - Control)")
    sub = "Top 16 features by effect size" + (
        "  (stars: * q<.05, ** q<.01, *** q<.001, FDR)" if stats is not None else "")
    ax2.set_title(sub, fontsize=10)
    return fig


def fig_group_stats_overview(stats):
    """Per-family significant-feature counts (with the parametric/non-parametric mix)
    and a volcano plot of effect size against FDR significance."""
    fig = plt.figure(figsize=(16, 5.8), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.25], figure=fig)
    fig.suptitle("Group distribution comparison: Control vs Psychosis (Benjamini-Hochberg FDR)",
                 fontsize=14, fontweight="bold")
    order = [l for l in "ABCDEFGH" if l in stats["family"].values]

    ax = fig.add_subplot(gs[0])
    total = stats.groupby("family")["feature"].count()
    sig = stats[stats["significant"]].groupby("family")["feature"].count()
    nonpar = stats[stats["test"] == "Mann-Whitney U"].groupby("family")["feature"].count()
    y = np.arange(len(order))
    ax.barh(y, [total.get(l, 0) for l in order], color="#dddddd", label="tested")
    ax.barh(y, [nonpar.get(l, 0) for l in order], color="#bbbbbb", label="non-parametric")
    ax.barh(y, [sig.get(l, 0) for l in order], color=[FAMILY_COLORS[l] for l in order],
            label="FDR-significant")
    for i, l in enumerate(order):
        ax.text(total.get(l, 0) + 1, i, f"{int(sig.get(l, 0))}/{int(total.get(l, 0))}",
                va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([f"{l} {FAMILY_NAMES[l]}" for l in order], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("features"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("significant features per family", fontsize=11)

    ax = fig.add_subplot(gs[1])
    d = stats.dropna(subset=["q"]).copy()
    d["logq"] = -np.log10(d["q"].clip(lower=1e-12))
    for l in order:
        s = d[d["family"] == l]
        ax.scatter(s["cohens_d"], s["logq"], s=16, color=FAMILY_COLORS[l], alpha=0.7,
                   label=l, edgecolor="none")
    ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.8)
    ax.text(ax.get_xlim()[1], -np.log10(0.05), " q=0.05", va="bottom", ha="right", fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    for thr in (-0.5, 0.5):
        ax.axvline(thr, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("Cohen's d (Psychosis - Control)")
    ax.set_ylabel("-log10 FDR q")
    ax.set_title("volcano: effect size vs corrected significance", fontsize=11)
    ax.legend(fontsize=7, frameon=False, ncol=2, title="family")
    return fig


def fig_significant_distributions(tables, stats, n=6):
    """Processed value distributions (histogram plus fitted KDE) for the strongest
    significantly-different features, annotated with the test, q-value and stars."""
    merged = tables["merged"]
    sig = stats[stats["significant"]].copy()
    if sig.empty:
        sig = stats.dropna(subset=["q"]).copy()
    sig["absd"] = sig["cohens_d"].abs()
    sig = sig.sort_values(["q", "absd"], ascending=[True, False]).head(n)
    ncol = 3
    nrow = int(np.ceil(len(sig) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.5 * nrow),
                             constrained_layout=True)
    fig.suptitle("Strongest significant Control vs Psychosis distribution differences",
                 fontsize=14, fontweight="bold")
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, sig.iterrows()):
        c = r["feature"]
        for g in ("Control", "Psychosis"):
            v = pd.to_numeric(merged.loc[merged["group"] == g, c], errors="coerce").dropna()
            if len(v) < 3:
                continue
            ax.hist(v, bins=14, density=True, alpha=0.35, color=GROUP_COLORS[g])
            try:
                xs = np.linspace(v.min(), v.max(), 200)
                ax.plot(xs, gaussian_kde(v)(xs), color=GROUP_COLORS[g], lw=2, label=g)
                ax.axvline(v.median(), color=GROUP_COLORS[g], ls="--", lw=1)
            except Exception:
                pass
        short = c.split("__", 1)[1].replace("__", " / ")[:46]
        ax.set_title(f"{r['family']} {short}\n{r['test']}, q={r['q']:.1e} {r['stars']}, "
                     f"d={r['cohens_d']:+.2f}", fontsize=8.5)
        ax.set_ylabel("density"); ax.legend(fontsize=7, frameon=False)
    for ax in axes[len(sig):]:
        ax.axis("off")
    return fig


# --------------------------------------------------------------------------- #
# Representative subject dashboard
# --------------------------------------------------------------------------- #
def pick_representative_psychosis(tables) -> str:
    """Median-profile eligible psychosis recording (closest to psychosis centroid)."""
    merged = tables["merged"]
    z = tables["z"]
    zcols = [c for c in z.columns if c.startswith("Z_control__")]
    psy = merged[merged["group"] == "Psychosis"]["recording_id"]
    zp = z[z["recording_id"].isin(psy)].set_index("recording_id")[zcols]
    zp = zp.dropna(axis=1, how="any")
    if zp.empty:
        return psy.iloc[0]
    centroid = zp.mean(0)
    dist = ((zp - centroid) ** 2).sum(1)
    return dist.idxmin()


def build_subject_dashboard(recording_id, tables, cfg=None):
    from fooof import FOOOF
    cfg = cfg or load_config()
    info = montage_info()
    subject_dir = SUBJECT_ROOT / recording_id
    raw = mne.io.read_raw_fif(subject_dir / "clean_raw.fif", preload=True, verbose="ERROR")
    raw.reorder_channels(COMMON_CORTICAL)
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose="ERROR")
    data = epochs.get_data(copy=False)
    psd, freqs = psd_array_multitaper(data, sfreq=250.0, fmin=1.0, fmax=45.0, bandwidth=1.5,
                                      adaptive=False, low_bias=True, normalization="full",
                                      output="power", n_jobs=1, verbose="ERROR")
    psd = np.nanmedian(psd, axis=0) * 1e12
    conn = np.load(subject_dir / "connectivity_matrices.npz") if (subject_dir / "connectivity_matrices.npz").exists() else None
    templates = np.load(SUBJECT_ROOT.parent / "microstate_templates" / "group_templates.npz")
    row = tables["merged"].set_index("recording_id").loc[recording_id]

    fig = plt.figure(figsize=(24, 15), constrained_layout=True)
    # Wide figure with tall topomap rows so the six band-power maps and the microstate
    # maps render large and legible (each scalp map fills its cell), not thumbnails.
    gs = GridSpec(4, 6, figure=fig, height_ratios=[0.75, 1.3, 1.3, 1.0])
    fig.suptitle(f"Representative psychosis subject dashboard: {recording_id} "
                 f"(age {row['age']:.0f}, {row['sex']})", fontsize=16, fontweight="bold")

    # Row 0: raw trace + region PSDs
    ax = fig.add_subplot(gs[0, :3])
    seg = raw.copy().crop(30, 34).get_data(picks=["Fz", "Cz", "Pz", "Oz", "O1", "O2"]) * 1e6
    t = np.arange(seg.shape[1]) / 250.0
    for i, ch in enumerate(["Fz", "Cz", "Pz", "Oz", "O1", "O2"]):
        ax.plot(t, seg[i] + i * 60, lw=0.6, color="k")
        ax.text(-0.15, i * 60, ch, fontsize=7, va="center")
    ax.set_title("Cleaned EEG (4 s, midline + occipital)", fontsize=10)
    ax.set_xlabel("s"); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3:])
    for scope, color in [("frontal_left", "#4e79a7"), ("central_left", "#59a14f"),
                         ("occipital_left", "#e15759")]:
        idx = scope_indices(COMMON_CORTICAL, scope)
        ax.semilogy(freqs, psd[idx].mean(0), color=color, lw=1.6, label=scope.replace("_", " "))
    for b in BAND_NAMES:
        ax.axvspan(*BANDS[b], alpha=0.05, color="gray")
    ax.set_title("Regional power spectra", fontsize=10)
    ax.set_xlabel("Hz"); ax.set_ylabel("uV^2/Hz"); ax.legend(fontsize=7, frameon=False)

    # Row 1: band-power topomaps
    total = psd.sum(1) * float(np.median(np.diff(freqs)))
    for bi, b in enumerate(BAND_NAMES):
        lo, hi = BANDS[b]
        mask = (freqs >= lo) & (freqs <= hi if b == BAND_NAMES[-1] else freqs < hi)
        rel = psd[:, mask].sum(1) * float(np.median(np.diff(freqs))) / total
        ax = fig.add_subplot(gs[1, bi])
        _topo(ax, rel, info, BAND_PRETTY[b].split(" ")[0], cmap="viridis")
    ax = fig.add_subplot(gs[1, 5])
    exps = []
    for ci in range(len(COMMON_CORTICAL)):
        fm = FOOOF(peak_width_limits=(1, 12), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
        try:
            fm.fit(freqs, psd[ci], (1, 45)); exps.append(fm.aperiodic_params_[-1])
        except Exception:
            exps.append(np.nan)
    _topo(ax, np.array(exps), info, "1/f exponent", cmap="cividis")

    # Row 2: FOOOF fit on occipital + microstate templates
    ax = fig.add_subplot(gs[2, :2])
    occ = scope_indices(COMMON_CORTICAL, "occipital_left")
    occ_psd = psd[occ].mean(0)
    fm = FOOOF(peak_width_limits=(1, 12), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fm.fit(freqs, occ_psd, (1, 45))
    ax.plot(fm.freqs, fm.power_spectrum, "k", lw=1.5, label="log power")
    ax.plot(fm.freqs, fm._ap_fit, "--", color="#e15759", lw=1.5, label="aperiodic")
    ax.plot(fm.freqs, fm.fooofed_spectrum_, color="#4e79a7", lw=1.2, label="full fit")
    ax.set_title(f"Occipital FOOOF fit (exp={fm.aperiodic_params_[-1]:.2f})", fontsize=10)
    ax.set_xlabel("Hz"); ax.set_ylabel("log10 power"); ax.legend(fontsize=7, frameon=False)

    for i, letter in enumerate("ABCD"):
        ax = fig.add_subplot(gs[2, 2 + i])
        m = templates["maps"][i]
        _topo(ax, m, info, f"MS {letter}: cov={row[f'F_microstates__class_{letter.lower()}__coverage_fraction']:.2f}",
              cmap="RdBu_r", vlim=(-np.abs(m).max(), np.abs(m).max()))

    # Row 3: connectivity matrix (alpha) + graph strengths + microstate bars
    ax = fig.add_subplot(gs[3, :2])
    if conn is not None and "alpha_8_13_hz" in conn:
        im = ax.imshow(conn["alpha_8_13_hz"], cmap="inferno", vmin=0)
        ax.set_xticks(range(10)); ax.set_xticklabels([n[:4] for n in NODES], rotation=90, fontsize=6)
        ax.set_yticks(range(10)); ax.set_yticklabels([n[:4] for n in NODES], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Alpha wPLI connectivity", fontsize=10)

    ax = fig.add_subplot(gs[3, 2:4])
    strengths = [row[f"H_graph__node__strength_normalized__alpha_8_13_hz__{n}"] for n in NODES]
    ax.barh(range(10), strengths, color="#59a14f")
    ax.set_yticks(range(10)); ax.set_yticklabels([n.replace("_", " ") for n in NODES], fontsize=7)
    ax.invert_yaxis(); ax.set_title("Alpha node strength", fontsize=10)

    ax = fig.add_subplot(gs[3, 4:])
    x = np.arange(4)
    ax.bar(x - 0.2, [row[f"F_microstates__class_{s}__coverage_fraction"] for s in "abcd"],
           0.4, color="#4e79a7", label="coverage")
    ax2 = ax.twinx()
    ax2.plot(x, [row[f"F_microstates__class_{s}__mean_duration_ms"] for s in "abcd"],
             "o-", color="#e15759", label="duration")
    ax.set_xticks(x); ax.set_xticklabels(list("ABCD"))
    ax.set_ylabel("coverage"); ax2.set_ylabel("duration (ms)", color="#e15759")
    ax.set_title("Microstate profile", fontsize=10)
    raw.close()
    return fig


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _qc_frame():
    from .extract import load_records
    rows = []
    for r in load_records(load_config()):
        qc = r.get("preprocessing_qc", {})
        rows.append({"recording_id": r["recording_id"], "dataset_id": r["dataset_id"],
                     "feature_eligible": bool(qc.get("feature_eligible", False)),
                     "interpolated_channel_count": qc.get("interpolated_channel_count", np.nan),
                     "epoch_count_retained": qc.get("epoch_count_retained", np.nan)})
    return pd.DataFrame(rows)


def build_cohort_figures(save=True):
    from .extract import load_records
    from .features import build_schema
    cfg = load_config()
    records = load_records(cfg)
    tables = load_tables()
    tables["qc"] = _qc_frame()
    groups, _ = build_schema()
    chan = build_channel_summary(records, cfg)
    cohort_dir = FIGURE_ROOT / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_cohort_overview": lambda: fig_cohort_overview(tables),
        "02_spectral_topography": lambda: fig_spectral_topography(tables, chan),
        "03_alpha_aperiodic": lambda: fig_alpha_aperiodic(tables, chan),
        "04_microstates": lambda: fig_microstates(tables),
        "05_connectivity": lambda: fig_connectivity(tables, records),
        "06_graph_metrics": lambda: fig_graph_metrics(tables),
        "07_complexity": lambda: fig_complexity(tables),
        "08_group_contrast": lambda: fig_group_contrast(tables, groups),
    }
    manifest = []
    figs = {}
    for name, builder in builders.items():
        fig = builder()
        figs[name] = fig
        if save:
            path = cohort_dir / f"{name}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            manifest.append({"figure": name, "path": str(path.relative_to(RESULTS_ROOT.parent))})
    if save:
        (RESULTS_ROOT / "figure_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    return figs, tables


def build_example_dashboard(save=True, recording_id=None):
    tables = load_tables()
    if recording_id is None:
        recording_id = pick_representative_psychosis(tables)
    fig = build_subject_dashboard(recording_id, tables)
    if save:
        out = FIGURE_ROOT / "example_subject"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{recording_id}_dashboard.png", dpi=130, bbox_inches="tight")
    return fig, recording_id
