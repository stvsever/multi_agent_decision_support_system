"""The eight canonical resting-EEG feature families (A-H), 836 features total.

Every feature is computed on the shared 49-channel montage, so a column such as
``...__alpha_8_13_hz__occipital_left`` means the same electrodes in both datasets.

  A  spectral         multitaper band power (absolute, relative), band ratios,
                      left/right power asymmetry                          (168)
  B  alpha peak       FOOOF periodic alpha peak: centre freq, power, width  (33)
  C  aperiodic 1/f    FOOOF aperiodic exponent and offset                   (22)
  D  entropy          sample / permutation / spectral / Lempel-Ziv          (44)
  E  fractal          Higuchi dimension, detrended fluctuation exponent     (22)
  F  microstates      pycrostates ModK group maps relabelled to Koenig A-D  (32)
  G  connectivity     mne-connectivity debiased wPLI^2 between 10 regions  (225)
  H  graph theory     density-thresholded graph metrics, AUC over density  (290)

Groups B/C use FOOOF (specparam), F uses pycrostates, G uses mne-connectivity;
these replace the bespoke fits in the previous pipeline so each value matches the
canonical definition its column name promises.
"""

from __future__ import annotations

import hashlib
import math
from itertools import combinations, product
from typing import Any

import warnings as _warnings

import antropy as ant
import mne
import networkx as nx
import numpy as np

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", DeprecationWarning)
    from fooof import FOOOF
from mne.time_frequency import psd_array_multitaper
from mne_connectivity import spectral_connectivity_epochs
from scipy.integrate import trapezoid

from .config import BANDS, BAND_NAMES
from .montage import (ASYMMETRY_REGIONS, COMMON_CORTICAL, NODES, REGIONS, SCOPES,
                      scope_indices)

RATIOS = {
    "theta_over_alpha": ("theta_4_8_hz", "alpha_8_13_hz"),
    "theta_over_beta": ("theta_4_8_hz", "beta_13_30_hz"),
    "alpha_over_delta": ("alpha_8_13_hz", "delta_1_4_hz"),
}
MICROSTATE_CLASSES = ["a", "b", "c", "d"]
_TINY = np.finfo(float).tiny


# --------------------------------------------------------------------------- #
# Feature-name schema (the exact, ordered 836 column names)
# --------------------------------------------------------------------------- #
def build_schema() -> tuple[dict[str, list[str]], list[str]]:
    groups: dict[str, list[str]] = {}

    a: list[str] = []
    for measure in ["log10_absolute_power_uv2", "relative_power_fraction_of_1_45_hz"]:
        a += [f"A_spectral__{measure}__{band}__{scope}"
              for band, scope in product(BAND_NAMES, SCOPES)]
    a += [f"A_spectral__natural_log_power_ratio__{ratio}__{scope}"
          for ratio, scope in product(RATIOS, SCOPES)]
    a += [f"A_spectral__log10_power_asymmetry_right_minus_left__{band}__{region}"
          for band, region in product(BAND_NAMES, ASYMMETRY_REGIONS)]
    groups["A_spectral"] = a

    groups["B_alpha_peak"] = [
        f"B_alpha_peak__{measure}__{scope}"
        for measure, scope in product(
            ["center_frequency_hz", "power_log10_uv2_above_aperiodic", "bandwidth_hz"],
            SCOPES)]

    groups["C_aperiodic"] = [
        f"C_aperiodic__{measure}__{scope}"
        for measure, scope in product(
            ["exponent", "offset_log10_uv2"], SCOPES)]

    groups["D_entropy"] = [
        f"D_entropy__{measure}__{scope}"
        for measure, scope in product(
            ["sample_entropy", "permutation_entropy_normalized",
             "spectral_entropy_normalized", "lempel_ziv_complexity_normalized"],
            SCOPES)]

    groups["E_fractal"] = [
        f"E_fractal__{measure}__{scope}"
        for measure, scope in product(
            ["higuchi_fractal_dimension", "detrended_fluctuation_exponent"], SCOPES)]

    f = [f"F_microstates__class_{state}__{measure}"
         for state, measure in product(
             MICROSTATE_CLASSES,
             ["mean_duration_ms", "coverage_fraction", "occurrence_per_second",
              "global_explained_variance_fraction"])]
    f += [f"F_microstates__transition_probability__class_{s}_to_class_{t}"
          for s in MICROSTATE_CLASSES for t in MICROSTATE_CLASSES if s != t]
    f += ["F_microstates__global__global_explained_variance_fraction",
          "F_microstates__global__transition_entropy_normalized",
          "F_microstates__global__sequence_lempel_ziv_complexity_normalized",
          "F_microstates__global__mean_global_field_power_uv"]
    groups["F_microstates"] = f

    edges = [f"{l}_to_{r}" for l, r in combinations(NODES, 2)]
    groups["G_connectivity"] = [
        f"G_connectivity__wpli2_debiased__{band}__{edge}"
        for band, edge in product(BAND_NAMES, edges)]

    global_measures = [
        "mean_edge_weight",
        "global_efficiency_auc_density_20_50_percent",
        "characteristic_path_length_auc_density_20_50_percent",
        "mean_clustering_coefficient_auc_density_20_50_percent",
        "transitivity_auc_density_20_50_percent",
        "modularity_q_auc_density_20_50_percent",
        "assortativity_auc_density_20_50_percent",
        "small_world_propensity_auc_density_20_50_percent",
    ]
    node_measures = [
        "strength_normalized",
        "local_efficiency_auc_density_20_50_percent",
        "betweenness_centrality_auc_density_20_50_percent",
        "eigenvector_centrality_auc_density_20_50_percent",
        "participation_coefficient_auc_density_20_50_percent",
    ]
    h = [f"H_graph__global__{m}__{band}" for m, band in product(global_measures, BAND_NAMES)]
    h += [f"H_graph__node__{m}__{band}__{node}"
          for m, band, node in product(node_measures, BAND_NAMES, NODES)]
    groups["H_graph"] = h

    expected = {"A_spectral": 168, "B_alpha_peak": 33, "C_aperiodic": 22,
                "D_entropy": 44, "E_fractal": 22, "F_microstates": 32,
                "G_connectivity": 225, "H_graph": 290}
    got = {k: len(v) for k, v in groups.items()}
    assert got == expected, f"schema counts {got} != {expected}"
    names = [n for v in groups.values() for n in v]
    assert len(names) == 836 and len(set(names)) == 836
    return groups, names


# --------------------------------------------------------------------------- #
# Scope aggregation
# --------------------------------------------------------------------------- #
def _scope_median(values: np.ndarray) -> dict[str, float]:
    """Median of a per-channel vector within each scope (channels in the 49 order)."""
    out: dict[str, float] = {}
    for scope in SCOPES:
        idx = scope_indices(COMMON_CORTICAL, scope)
        sel = values[idx]
        out[scope] = float(np.nanmedian(sel)) if np.isfinite(sel).any() else float("nan")
    return out


def _band_mask(freqs: np.ndarray, band: str) -> np.ndarray:
    lo, hi = BANDS[band]
    if band == BAND_NAMES[-1]:
        return (freqs >= lo) & (freqs <= hi)
    return (freqs >= lo) & (freqs < hi)


# --------------------------------------------------------------------------- #
# A / B / C  from a multitaper spectrum
# --------------------------------------------------------------------------- #
def extract_spectral(epoch_data: np.ndarray, sfreq: float,
                     cfg: dict[str, Any]) -> dict[str, float]:
    """Groups A (spectral), B (FOOOF alpha peak), C (FOOOF aperiodic)."""
    feats: dict[str, float] = {}
    psd, freqs = psd_array_multitaper(
        epoch_data, sfreq=sfreq, fmin=1.0, fmax=45.0,
        bandwidth=float(cfg["features"]["multitaper_bandwidth_hz"]),
        adaptive=False, low_bias=True, normalization="full", output="power",
        n_jobs=1, verbose="ERROR")
    psd_uv2 = psd * 1e12                              # V^2/Hz -> uV^2/Hz
    df = float(np.median(np.diff(freqs)))
    total = np.sum(psd_uv2, axis=-1) * df             # (n_ep, n_ch)

    band_epoch = {b: np.sum(psd_uv2[..., _band_mask(freqs, b)], axis=-1) * df
                  for b in BAND_NAMES}               # each (n_ep, n_ch)
    band_channel = {b: np.nanmedian(v, axis=0) for b, v in band_epoch.items()}

    scope_log_abs: dict[str, dict[str, float]] = {}
    for b in BAND_NAMES:
        abs_log = np.log10(np.maximum(band_channel[b], _TINY))
        rel = np.nanmedian(band_epoch[b] / np.maximum(total, _TINY), axis=0)
        s_abs = _scope_median(abs_log)
        s_rel = _scope_median(rel)
        scope_log_abs[b] = s_abs
        for scope in SCOPES:
            feats[f"A_spectral__log10_absolute_power_uv2__{b}__{scope}"] = s_abs[scope]
            feats[f"A_spectral__relative_power_fraction_of_1_45_hz__{b}__{scope}"] = s_rel[scope]

    for ratio, (num, den) in RATIOS.items():
        val = np.log(np.maximum(band_epoch[num], _TINY) / np.maximum(band_epoch[den], _TINY))
        s = _scope_median(np.nanmedian(val, axis=0))
        for scope in SCOPES:
            feats[f"A_spectral__natural_log_power_ratio__{ratio}__{scope}"] = s[scope]

    for b in BAND_NAMES:
        for region in ASYMMETRY_REGIONS:
            val = scope_log_abs[b][f"{region}_right"] - scope_log_abs[b][f"{region}_left"]
            feats[f"A_spectral__log10_power_asymmetry_right_minus_left__{b}__{region}"] = float(val)

    # B / C via FOOOF on the epoch-median spectrum of each channel.
    channel_psd = np.nanmedian(psd_uv2, axis=0)       # (n_ch, n_freqs)
    fcfg = cfg["features"]
    lo, hi = fcfg["fooof_fit_range_hz"]
    a_lo, a_hi = fcfg["alpha_peak_search_hz"]
    exponent = np.full(len(COMMON_CORTICAL), np.nan)
    offset = np.full(len(COMMON_CORTICAL), np.nan)
    a_cf = np.full(len(COMMON_CORTICAL), np.nan)
    a_pw = np.full(len(COMMON_CORTICAL), np.nan)
    a_bw = np.full(len(COMMON_CORTICAL), np.nan)
    for ci in range(len(COMMON_CORTICAL)):
        fm = FOOOF(peak_width_limits=tuple(fcfg["fooof_peak_width_limits_hz"]),
                   max_n_peaks=int(fcfg["fooof_max_n_peaks"]),
                   min_peak_height=float(fcfg["fooof_min_peak_height"]),
                   peak_threshold=float(fcfg["fooof_peak_threshold_sd"]),
                   aperiodic_mode=str(fcfg["fooof_aperiodic_mode"]), verbose=False)
        try:
            fm.fit(freqs, channel_psd[ci], freq_range=(lo, hi))
        except Exception:
            continue
        ap = fm.aperiodic_params_
        offset[ci] = float(ap[0])
        exponent[ci] = float(ap[-1])
        peaks = fm.get_params("peak_params")
        peaks = np.atleast_2d(peaks) if peaks is not None and len(peaks) else np.empty((0, 3))
        alpha = peaks[(peaks[:, 0] >= a_lo) & (peaks[:, 0] <= a_hi)] if peaks.size else peaks
        if alpha.size:
            best = alpha[np.argmax(alpha[:, 1])]
            a_cf[ci], a_pw[ci], a_bw[ci] = float(best[0]), float(best[1]), float(best[2])

    for measure, vals in [("center_frequency_hz", a_cf),
                          ("power_log10_uv2_above_aperiodic", a_pw),
                          ("bandwidth_hz", a_bw)]:
        s = _scope_median(vals)
        for scope in SCOPES:
            feats[f"B_alpha_peak__{measure}__{scope}"] = s[scope]
    for measure, vals in [("exponent", exponent), ("offset_log10_uv2", offset)]:
        s = _scope_median(vals)
        for scope in SCOPES:
            feats[f"C_aperiodic__{measure}__{scope}"] = s[scope]
    return feats


# --------------------------------------------------------------------------- #
# D / E  entropy, complexity, fractal
# --------------------------------------------------------------------------- #
def _spectral_entropy(sig: np.ndarray, sfreq: float) -> float:
    try:
        return float(ant.spectral_entropy(sig, sf=sfreq, method="welch",
                                          nperseg=min(len(sig), int(sfreq * 2)),
                                          normalize=True))
    except Exception:
        return float("nan")


def extract_complexity(epoch_data: np.ndarray, continuous: np.ndarray,
                       sfreq: float, cfg: dict[str, Any]) -> dict[str, float]:
    """Groups D (entropy family, per-epoch median) and E (Higuchi per-epoch,
    DFA on the continuous cleaned signal)."""
    feats: dict[str, float] = {}
    n_ch = len(COMMON_CORTICAL)
    order = int(cfg["features"]["entropy_permutation_order"])
    se_ord = int(cfg["features"]["entropy_sample_entropy_order"])
    se_tol = float(cfg["features"]["entropy_sample_entropy_tolerance_sd"])
    measures = {m: np.full(n_ch, np.nan) for m in
                ["sample_entropy", "permutation_entropy_normalized",
                 "spectral_entropy_normalized", "lempel_ziv_complexity_normalized",
                 "higuchi_fractal_dimension", "detrended_fluctuation_exponent"]}
    for ci in range(n_ch):
        samp, perm, spec, lz, hfd = [], [], [], [], []
        for sig in epoch_data[:, ci, :]:
            s = sig - np.nanmean(sig)
            sd = np.nanstd(s)
            if not np.isfinite(sd) or sd <= 1e-12:
                continue
            s = s / sd
            try:
                samp.append(float(ant.sample_entropy(s, order=se_ord)))
            except Exception:
                samp.append(np.nan)
            try:
                perm.append(float(ant.perm_entropy(s, order=order, normalize=True)))
            except Exception:
                perm.append(np.nan)
            spec.append(_spectral_entropy(s, sfreq))
            try:
                binary = "".join((s > np.median(s)).astype(int).astype(str))
                lz.append(float(ant.lziv_complexity(binary, normalize=True)))
            except Exception:
                lz.append(np.nan)
            try:
                hfd.append(float(ant.higuchi_fd(s, kmax=10)))
            except Exception:
                hfd.append(np.nan)
        measures["sample_entropy"][ci] = np.nanmedian(samp) if samp else np.nan
        measures["permutation_entropy_normalized"][ci] = np.nanmedian(perm) if perm else np.nan
        measures["spectral_entropy_normalized"][ci] = np.nanmedian(spec) if spec else np.nan
        measures["lempel_ziv_complexity_normalized"][ci] = np.nanmedian(lz) if lz else np.nan
        measures["higuchi_fractal_dimension"][ci] = np.nanmedian(hfd) if hfd else np.nan
        # DFA (long-range temporal correlations) on the continuous cleaned trace.
        sig = continuous[ci]
        sig = sig[np.isfinite(sig)]
        if len(sig) >= int(cfg["features"]["dfa_minimum_contiguous_seconds"] * sfreq):
            try:
                measures["detrended_fluctuation_exponent"][ci] = float(
                    ant.detrended_fluctuation(sig))
            except Exception:
                pass

    name_map = {
        "sample_entropy": "D_entropy__sample_entropy",
        "permutation_entropy_normalized": "D_entropy__permutation_entropy_normalized",
        "spectral_entropy_normalized": "D_entropy__spectral_entropy_normalized",
        "lempel_ziv_complexity_normalized": "D_entropy__lempel_ziv_complexity_normalized",
        "higuchi_fractal_dimension": "E_fractal__higuchi_fractal_dimension",
        "detrended_fluctuation_exponent": "E_fractal__detrended_fluctuation_exponent",
    }
    for key, vals in measures.items():
        s = _scope_median(vals)
        for scope in SCOPES:
            feats[f"{name_map[key]}__{scope}"] = s[scope]
    return feats


# --------------------------------------------------------------------------- #
# F  microstates (group templates supplied by build_group_microstate_templates)
# --------------------------------------------------------------------------- #
def extract_microstates(raw: mne.io.BaseRaw, modk, cfg: dict[str, Any]) -> dict[str, float]:
    from pycrostates.preprocessing import extract_gfp_peaks  # local, heavy import
    feats: dict[str, float] = {}
    sfreq = float(raw.info["sfreq"])
    half_win = max(1, int(round(0.030 * sfreq / 2)))
    min_seg = max(1, int(round(float(cfg["features"]["microstate_min_segment_ms"]) * sfreq / 1000.0)))
    mraw = raw.copy().filter(*cfg["features"]["microstate_filter_hz"], picks="eeg",
                             verbose="ERROR")
    seg = modk.predict(mraw, factor=10, half_window_size=half_win,
                       min_segment_length=min_seg, reject_edges=True,
                       reject_by_annotation=True, verbose="ERROR")
    params = seg.compute_parameters(norm_gfp=True)
    for state in MICROSTATE_CLASSES:
        up = state.upper()
        feats[f"F_microstates__class_{state}__mean_duration_ms"] = float(params[f"{up}_meandurs"] * 1000.0)
        feats[f"F_microstates__class_{state}__coverage_fraction"] = float(params[f"{up}_timecov"])
        feats[f"F_microstates__class_{state}__occurrence_per_second"] = float(params[f"{up}_occurrences"])
        feats[f"F_microstates__class_{state}__global_explained_variance_fraction"] = float(params[f"{up}_gev"])

    # Transition probabilities conditioned on a state switch (self-transitions removed).
    trans = np.asarray(seg.compute_transition_matrix(stat="probability", ignore_repetitions=True),
                       dtype=float)
    for i, s in enumerate(MICROSTATE_CLASSES):
        row = trans[i].copy()
        row[i] = 0.0
        denom = row.sum()
        for j, t in enumerate(MICROSTATE_CLASSES):
            if i == j:
                continue
            feats[f"F_microstates__transition_probability__class_{s}_to_class_{t}"] = (
                float(row[j] / denom) if denom > 0 else float("nan"))

    feats["F_microstates__global__global_explained_variance_fraction"] = float(
        sum(params[f"{s.upper()}_gev"] for s in MICROSTATE_CLASSES))
    # Transition entropy over the 12 off-diagonal transition probabilities.
    off = trans.copy()
    np.fill_diagonal(off, 0.0)
    flat = off[off > 0]
    flat = flat / flat.sum() if flat.size else flat
    feats["F_microstates__global__transition_entropy_normalized"] = (
        float(-np.sum(flat * np.log(flat)) / np.log(12.0)) if flat.size else float("nan"))
    labels = np.asarray(seg.labels)
    label_string = "".join(str(int(v)) for v in labels[labels >= 0])
    feats["F_microstates__global__sequence_lempel_ziv_complexity_normalized"] = (
        float(ant.lziv_complexity(label_string, normalize=True)) if label_string else float("nan"))
    gfp = np.std(raw.get_data(picks="eeg") * 1e6, axis=0)
    feats["F_microstates__global__mean_global_field_power_uv"] = float(np.mean(gfp))
    return feats


# --------------------------------------------------------------------------- #
# G / H  connectivity and graph theory
# --------------------------------------------------------------------------- #
def _region_node_signals(epoch_data: np.ndarray) -> np.ndarray:
    """First within-region principal component per epoch, shape (n_ep, 10, n_t)."""
    nodes = []
    for region in NODES:
        idx = [COMMON_CORTICAL.index(c) for c in REGIONS[region]]
        vals = epoch_data[:, idx, :]                              # (n_ep, k, n_t)
        flat = np.moveaxis(vals, 1, 0).reshape(len(idx), -1)      # (k, n_ep*n_t)
        flat = flat - flat.mean(axis=1, keepdims=True)
        cov = flat @ flat.T / max(1, flat.shape[1] - 1)
        w, v = np.linalg.eigh(cov)
        weights = v[:, np.argmax(w)]
        if np.corrcoef(weights @ flat, flat.mean(0))[0, 1] < 0:
            weights = -weights
        nodes.append(np.einsum("c,ect->et", weights, vals))
    return np.stack(nodes, axis=1)


def _wpli_matrices(epoch_data: np.ndarray, sfreq: float,
                   cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    nodes = _region_node_signals(epoch_data)
    fmin = tuple(BANDS[b][0] for b in BAND_NAMES)
    fmax = tuple(BANDS[b][1] for b in BAND_NAMES)
    con = spectral_connectivity_epochs(
        nodes, method="wpli2_debiased", mode="multitaper", sfreq=sfreq,
        fmin=fmin, fmax=fmax, faverage=True,
        mt_bandwidth=2.0 * float(cfg["features"]["multitaper_bandwidth_hz"]),
        mt_adaptive=False, verbose="ERROR")
    dense = con.get_data(output="dense")                          # (10,10,5)
    matrices = {}
    for bi, b in enumerate(BAND_NAMES):
        m = dense[:, :, bi].copy()
        m = np.clip(m, 0.0, 1.0)
        m = m + m.T                     # dense returns lower triangle only
        np.fill_diagonal(m, 0.0)
        matrices[b] = m
    return matrices


def _density_graph(matrix: np.ndarray, density: float) -> nx.Graph:
    n = matrix.shape[0]
    edges = sorted([(i, j, float(matrix[i, j])) for i, j in combinations(range(n), 2)
                    if np.isfinite(matrix[i, j]) and matrix[i, j] > 0.0],
                   key=lambda e: e[2], reverse=True)
    target = max(n - 1, int(math.floor(density * n * (n - 1) / 2 + 0.5)))
    if len(edges) < target:
        raise RuntimeError("insufficient positive edges for requested density")
    complete = nx.Graph()
    complete.add_nodes_from(range(n))
    for i, j, w in edges:
        complete.add_edge(i, j, weight=w, distance=1.0 / w)
    if not nx.is_connected(complete):
        raise RuntimeError("positive graph support is disconnected")
    tree = nx.maximum_spanning_tree(complete, weight="weight")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(tree.edges(data=True))
    for i, j, w in edges:
        if graph.number_of_edges() >= target:
            break
        if not graph.has_edge(i, j):
            graph.add_edge(i, j, weight=w, distance=1.0 / w)
    return graph


def _weighted_efficiency(graph: nx.Graph, nodes=None) -> float:
    sel = list(graph.nodes if nodes is None else nodes)
    if len(sel) < 2:
        return float("nan")
    sub = graph.subgraph(sel)
    lengths = dict(nx.all_pairs_dijkstra_path_length(sub, weight="distance"))
    vals = []
    for i, j in combinations(sel, 2):
        d = lengths.get(i, {}).get(j, np.inf)
        vals.append(1.0 / d if np.isfinite(d) and d > 0 else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def _char_path_length(graph: nx.Graph) -> float:
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="distance"))
    return float(np.mean([lengths[i][j] for i, j in combinations(graph.nodes, 2)]))


def _local_efficiency(graph: nx.Graph, node: int) -> float:
    nbrs = list(graph.neighbors(node))
    return _weighted_efficiency(graph, nbrs) if len(nbrs) >= 2 else 0.0


def _participation(graph: nx.Graph, communities) -> dict[int, float]:
    member = {n: i for i, c in enumerate(communities) for n in c}
    out = {}
    for node in graph.nodes:
        total = sum(d["weight"] for _, _, d in graph.edges(node, data=True))
        if total <= 0:
            out[node] = 0.0
            continue
        by_mod: dict[int, float] = {}
        for _, nb, d in graph.edges(node, data=True):
            by_mod[member[nb]] = by_mod.get(member[nb], 0.0) + d["weight"]
        out[node] = float(1.0 - sum((v / total) ** 2 for v in by_mod.values()))
    return out


def _lattice(weights, n) -> nx.Graph:
    pairs = sorted(combinations(range(n), 2),
                   key=lambda e: min((e[1] - e[0]) % n, (e[0] - e[1]) % n))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), w in zip(pairs, sorted(weights, reverse=True)):
        g.add_edge(i, j, weight=w, distance=1.0 / max(w, 1e-12))
    return g


def _small_world_propensity(graph: nx.Graph, seed: int, replicates: int) -> float:
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    weights = [d["weight"] for _, _, d in graph.edges(data=True)]
    c_obs = float(np.mean(list(nx.clustering(graph, weight="weight").values())))
    l_obs = _char_path_length(graph)
    latt = _lattice(weights, n)
    c_latt = float(np.mean(list(nx.clustering(latt, weight="weight").values())))
    l_latt = _char_path_length(latt)
    rng = np.random.default_rng(seed)
    rc, rl = [], []
    for _ in range(replicates):
        rand = None
        for _try in range(30):
            cand = nx.Graph()
            cand.add_nodes_from(graph.nodes)
            cand.add_edges_from(graph.edges)
            try:
                nx.double_edge_swap(cand, nswap=max(1, 5 * m),
                                    max_tries=max(100, 100 * m),
                                    seed=int(rng.integers(0, 2**31 - 1)))
            except (nx.NetworkXAlgorithmError, nx.NetworkXError):
                continue
            if nx.is_connected(cand):
                rand = cand
                break
        if rand is None:
            continue
        shuffled = rng.permutation(weights)
        for (i, j), w in zip(rand.edges(), shuffled):
            rand[i][j]["weight"] = float(w)
            rand[i][j]["distance"] = 1.0 / max(float(w), 1e-12)
        rc.append(float(np.mean(list(nx.clustering(rand, weight="weight").values()))))
        rl.append(_char_path_length(rand))
    if not rc:
        return float("nan")
    c_rand, l_rand = float(np.mean(rc)), float(np.mean(rl))
    cd, ld = c_latt - c_rand, l_latt - l_rand
    if abs(cd) < 1e-12 or abs(ld) < 1e-12:
        return float("nan")
    delta_c = np.clip((c_latt - c_obs) / cd, 0.0, 1.0)
    delta_l = np.clip((l_obs - l_rand) / ld, 0.0, 1.0)
    return float(np.clip(1.0 - np.sqrt((delta_c**2 + delta_l**2) / 2.0), 0.0, 1.0))


def _auc(values, densities) -> float:
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return float("nan")
    return float(trapezoid(v[ok], np.asarray(densities)[ok]) / (max(densities) - min(densities)))


def _graph_band(matrix: np.ndarray, band: str, cfg: dict[str, Any], seed_text: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    tri = matrix[np.triu_indices_from(matrix, k=1)]
    feats[f"H_graph__global__mean_edge_weight__{band}"] = float(np.nanmean(tri))
    densities = [float(d) for d in cfg["features"]["graph_densities"]]
    g_vals = {k: [] for k in ["global_efficiency", "characteristic_path_length",
                              "mean_clustering_coefficient", "transitivity",
                              "modularity_q", "assortativity", "small_world_propensity"]}
    n_vals = {k: {i: [] for i in range(len(NODES))}
              for k in ["local_efficiency", "betweenness_centrality",
                        "eigenvector_centrality", "participation_coefficient"]}
    for di, density in enumerate(densities):
        graph = _density_graph(matrix, density)
        comms = list(nx.community.louvain_communities(
            graph, weight="weight",
            seed=int(hashlib.sha256(f"{seed_text}|{density}".encode()).hexdigest()[:8], 16)))
        g_vals["global_efficiency"].append(_weighted_efficiency(graph))
        g_vals["characteristic_path_length"].append(_char_path_length(graph))
        g_vals["mean_clustering_coefficient"].append(
            float(np.mean(list(nx.clustering(graph, weight="weight").values()))))
        g_vals["transitivity"].append(float(nx.transitivity(graph)))
        g_vals["modularity_q"].append(float(nx.community.modularity(graph, comms, weight="weight")))
        with np.errstate(all="ignore"):
            g_vals["assortativity"].append(float(nx.degree_assortativity_coefficient(graph, weight="weight")))
        sw_seed = int(hashlib.sha256(f"{seed_text}|sw|{di}".encode()).hexdigest()[:8], 16)
        g_vals["small_world_propensity"].append(_small_world_propensity(
            graph, sw_seed, int(cfg["features"]["small_world_random_replicates"])))
        betw = nx.betweenness_centrality(graph, weight="distance", normalized=True)
        try:
            eig = nx.eigenvector_centrality_numpy(graph, weight="weight")
        except Exception:
            eig = {node: np.nan for node in graph.nodes}
        part = _participation(graph, comms)
        for node in graph.nodes:
            n_vals["local_efficiency"][node].append(_local_efficiency(graph, node))
            n_vals["betweenness_centrality"][node].append(float(betw[node]))
            n_vals["eigenvector_centrality"][node].append(float(eig[node]))
            n_vals["participation_coefficient"][node].append(float(part[node]))
    g_map = {
        "global_efficiency": "global_efficiency_auc_density_20_50_percent",
        "characteristic_path_length": "characteristic_path_length_auc_density_20_50_percent",
        "mean_clustering_coefficient": "mean_clustering_coefficient_auc_density_20_50_percent",
        "transitivity": "transitivity_auc_density_20_50_percent",
        "modularity_q": "modularity_q_auc_density_20_50_percent",
        "assortativity": "assortativity_auc_density_20_50_percent",
        "small_world_propensity": "small_world_propensity_auc_density_20_50_percent",
    }
    for internal, out in g_map.items():
        feats[f"H_graph__global__{out}__{band}"] = _auc(g_vals[internal], densities)
    strengths = np.nansum(matrix, axis=1) / (len(NODES) - 1)
    n_map = {
        "local_efficiency": "local_efficiency_auc_density_20_50_percent",
        "betweenness_centrality": "betweenness_centrality_auc_density_20_50_percent",
        "eigenvector_centrality": "eigenvector_centrality_auc_density_20_50_percent",
        "participation_coefficient": "participation_coefficient_auc_density_20_50_percent",
    }
    for ni, node in enumerate(NODES):
        feats[f"H_graph__node__strength_normalized__{band}__{node}"] = float(strengths[ni])
        for internal, out in n_map.items():
            feats[f"H_graph__node__{out}__{band}__{node}"] = _auc(n_vals[internal][ni], densities)
    return feats


def extract_connectivity_graph(epoch_data: np.ndarray, sfreq: float, cfg: dict[str, Any],
                               recording_id: str) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    feats: dict[str, float] = {}
    matrices = _wpli_matrices(epoch_data, sfreq, cfg)
    max_density = max(float(d) for d in cfg["features"]["graph_densities"])
    required = int(math.floor(len(NODES) * (len(NODES) - 1) / 2 * max_density + 0.5))
    for band in BAND_NAMES:
        m = matrices[band]
        for i, j in combinations(range(len(NODES)), 2):
            feats[f"G_connectivity__wpli2_debiased__{band}__{NODES[i]}_to_{NODES[j]}"] = float(m[i, j])
        support = nx.from_numpy_array((m > 0.0).astype(int))
        if nx.is_connected(support) and support.number_of_edges() >= required:
            feats.update(_graph_band(m, band, cfg, f"{recording_id}|{band}"))
    return feats, matrices
