"""Locked eeg_fep_rest_v1 feature schema - adapted from spec section 8.2.

Generates the 804 delivered feature names across 7 groups (A,B,C,D,E,G,H).
Group F (microstates, 32 features) from the original spec is intentionally
dropped by user decision; reviving it would be a v2 schema, not an edit here.
"""

import hashlib
import json
from itertools import combinations, product
from pathlib import Path

SCOPES = [
    "global",
    "frontal_left", "frontal_right",
    "central_left", "central_right",
    "temporal_left", "temporal_right",
    "parietal_left", "parietal_right",
    "occipital_left", "occipital_right",
]

NODES = SCOPES[1:]

BANDS = [
    "delta_1_4_hz",
    "theta_4_8_hz",
    "alpha_8_13_hz",
    "beta_13_30_hz",
    "low_gamma_30_45_hz",
]

RATIOS = ["theta_over_alpha", "theta_over_beta", "alpha_over_delta"]
ASYMMETRY_REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]


def canonical_feature_names() -> tuple[dict[str, list[str]], list[str]]:
    groups = {}

    a = []
    for measure in ["log10_absolute_power_uv2", "relative_power_fraction_of_1_45_hz"]:
        a += [f"A_spectral_features__{measure}__{band}__{scope}" for band, scope in product(BANDS, SCOPES)]
    a += [
        f"A_spectral_features__natural_log_power_ratio__{ratio}__{scope}"
        for ratio, scope in product(RATIOS, SCOPES)
    ]
    a += [
        f"A_spectral_features__log10_power_asymmetry_right_minus_left__{band}__{region}"
        for band, region in product(BANDS, ASYMMETRY_REGIONS)
    ]
    groups["A_spectral"] = a

    groups["B_alpha_peak"] = [
        f"B_peak_alpha_frequency__{measure}__{scope}"
        for measure, scope in product(
            ["strongest_alpha_peak_frequency_hz", "alpha_center_of_gravity_hz", "alpha_peak_prominence_db"],
            SCOPES,
        )
    ]

    groups["C_aperiodic"] = [
        f"C_spectral_slope_1_f__{measure}__{scope}"
        for measure, scope in product(["aperiodic_exponent", "aperiodic_offset_log10_power"], SCOPES)
    ]

    groups["D_entropy"] = [
        f"D_entropy_complexity__{measure}__{scope}"
        for measure, scope in product(
            [
                "sample_entropy",
                "permutation_entropy_normalized",
                "spectral_entropy_normalized",
                "lempel_ziv_complexity_normalized",
            ],
            SCOPES,
        )
    ]

    groups["E_fractal"] = [
        f"E_fractal_features__{measure}__{scope}"
        for measure, scope in product(
            ["higuchi_fractal_dimension", "detrended_fluctuation_analysis_exponent"], SCOPES
        )
    ]

    # Group F (microstates) intentionally omitted from v1 - dropped by user
    # decision (would require sourcing MICROSTATELAB canonical templates).
    # Reviving it is a v2 schema, never an in-place edit here. This takes the
    # delivered feature count from 836 to 804.

    edges = [f"{left}_to_{right}" for left, right in combinations(NODES, 2)]
    groups["G_connectivity"] = [
        f"G_functional_connectivity__wpli2_debiased__{band}__{edge}" for band, edge in product(BANDS, edges)
    ]

    global_graph_measures = [
        "mean_edge_weight",
        "global_efficiency_auc_density_20_50_percent",
        "characteristic_path_length_auc_density_20_50_percent",
        "mean_clustering_coefficient_auc_density_20_50_percent",
        "transitivity_auc_density_20_50_percent",
        "modularity_q_auc_density_20_50_percent",
        "assortativity_auc_density_20_50_percent",
        "small_world_propensity_auc_density_20_50_percent",
    ]
    node_graph_measures = [
        "strength_normalized",
        "local_efficiency_auc_density_20_50_percent",
        "betweenness_centrality_auc_density_20_50_percent",
        "eigenvector_centrality_auc_density_20_50_percent",
        "participation_coefficient_auc_density_20_50_percent",
    ]
    h = [f"H_graph_theory__global__{measure}__{band}" for measure, band in product(global_graph_measures, BANDS)]
    h += [
        f"H_graph_theory__node__{measure}__{band}__{node}"
        for measure, band, node in product(node_graph_measures, BANDS, NODES)
    ]
    groups["H_graph"] = h

    expected = {
        "A_spectral": 168, "B_alpha_peak": 33, "C_aperiodic": 22, "D_entropy": 44,
        "E_fractal": 22, "G_connectivity": 225, "H_graph": 290,
    }
    actual = {key: len(value) for key, value in groups.items()}
    if actual != expected:
        raise AssertionError(f"Feature group counts mismatch: expected {expected}, got {actual}")

    names = [name for group in groups.values() for name in group]
    if len(names) != 804:
        raise AssertionError(f"Expected 804 total features, got {len(names)}")
    if len(set(names)) != 804:
        raise AssertionError("Duplicate feature name detected")

    return groups, names


def write_feature_manifest(output_path: Path) -> str:
    """Writes the locked column order + a hash of it; returns the hash."""
    _, names = canonical_feature_names()
    columns = ["participant_id", "dataset"] + names
    manifest_hash = hashlib.sha256(json.dumps(columns).encode()).hexdigest()
    output_path.write_text(
        json.dumps(
            {"schema_version": "eeg_fep_rest_v1", "n_features": 804, "columns": columns, "columns_hash": manifest_hash},
            indent=2,
        )
    )
    return manifest_hash
