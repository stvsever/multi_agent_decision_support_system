"""Group H: graph theory (290) - spec section 7.8.

Built on Group G's wpli2_debiased matrices. Negative values clipped to 0 for
graph weights only (the signed value stays in Group G). Seven fixed-density
proportional graphs (9/11/14/16/18/20/23 of 45 possible edges), each
guaranteed connected via a maximum-spanning-tree core plus the next-strongest
remaining edges, nested (each higher density is a superset of the lower).
Global/node measures computed per density, then AUC-integrated (trapezoidal,
divided by the density-range width) across the 7 densities into one value.

Two measures have no library implementation and are custom here - the
highest-risk code in this group, validated against toy graphs before use:
  - participation_coefficient: community-based, from Louvain communities.
  - small_world_propensity (Muldoon et al. 2016): compares the actual
    network's weighted clustering/path-length to "random" and "lattice"
    nulls built by redistributing the SAME edge weights onto random node
    pairs (random null) or onto a ring ordered by decreasing weight with
    ring-distance (lattice null) - not classic Watts-Strogatz rewiring,
    which operates on binary topology rather than a fixed weight set.
"""

import networkx as nx
import numpy as np

from pipeline_v1 import regions

DENSITY_PCTS = [20, 25, 30, 35, 40, 45, 50]
EDGE_COUNTS = [9, 11, 14, 16, 18, 20, 23]  # of 45 possible edges, nearest-integer-half-up
N_NULL_SURROGATES = 100
RANDOM_SEED = 97
EPS = 1e-12


def _weight_graph(matrix: np.ndarray, node_names: list[str]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(node_names)
    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):
            w = max(matrix[i, j], 0.0)  # clip negative wPLI to 0 for graph weight only
            if w > 0:
                g.add_edge(node_names[i], node_names[j], weight=w)
    return g


def _length_graph(weight_graph: nx.Graph) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(weight_graph.nodes)
    for u, v, data in weight_graph.edges(data=True):
        g.add_edge(u, v, weight=1.0 / max(data["weight"], EPS))
    return g


def _nested_density_graphs(weight_graph: nx.Graph, node_names: list[str]) -> dict[int, nx.Graph]:
    """Seven fixed-density graphs, connected via MST core, nested by edge count."""
    full_edges = sorted(weight_graph.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
    if weight_graph.number_of_edges() == 0:
        return {pct: nx.Graph() for pct in DENSITY_PCTS}

    mst = nx.maximum_spanning_tree(weight_graph, weight="weight")
    mst_edge_set = {frozenset((u, v)) for u, v in mst.edges()}
    remaining_sorted = [e for e in full_edges if frozenset((e[0], e[1])) not in mst_edge_set]

    graphs = {}
    for pct, target_count in zip(DENSITY_PCTS, EDGE_COUNTS):
        g = nx.Graph()
        g.add_nodes_from(node_names)
        g.add_edges_from(mst.edges(data=True))
        n_extra_needed = max(target_count - g.number_of_edges(), 0)
        for u, v, data in remaining_sorted[:n_extra_needed]:
            g.add_edge(u, v, **data)
        graphs[pct] = g
    return graphs


def _local_efficiency_per_node(length_graph: nx.Graph) -> dict[str, float]:
    result = {}
    for node in length_graph.nodes:
        neighbors = list(length_graph.neighbors(node))
        if len(neighbors) < 2:
            result[node] = 0.0
            continue
        sub = length_graph.subgraph(neighbors)
        result[node] = nx.global_efficiency(sub)
    return result


def _participation_coefficient(weight_graph: nx.Graph, communities: list[set]) -> dict[str, float]:
    node_to_comm = {n: ci for ci, comm in enumerate(communities) for n in comm}
    pc = {}
    for node in weight_graph.nodes:
        k_i = sum(d["weight"] for _, _, d in weight_graph.edges(node, data=True))
        if k_i == 0:
            pc[node] = 0.0
            continue
        comm_strengths = {}
        for _, nbr, d in weight_graph.edges(node, data=True):
            c = node_to_comm.get(nbr, -1)
            comm_strengths[c] = comm_strengths.get(c, 0.0) + d["weight"]
        pc[node] = 1.0 - sum((s / k_i) ** 2 for s in comm_strengths.values())
    return pc


def _weighted_char_path_length(length_graph: nx.Graph) -> float:
    if length_graph.number_of_nodes() < 2:
        return np.nan
    lengths = dict(nx.all_pairs_dijkstra_path_length(length_graph, weight="weight"))
    vals = [d for src, targets in lengths.items() for tgt, d in targets.items() if src != tgt]
    return float(np.mean(vals)) if vals else np.nan


def _small_world_propensity(weight_graph: nx.Graph, rng: np.random.Generator) -> float:
    """Muldoon et al. 2016. See module docstring for the null-model approximation used."""
    nodes = list(weight_graph.nodes)
    n = len(nodes)
    edges = [(u, v, d["weight"]) for u, v, d in weight_graph.edges(data=True)]
    if len(edges) < 2 or n < 4:
        return np.nan
    weights_sorted_desc = sorted((w for _, _, w in edges), reverse=True)

    def clustering_and_path(g: nx.Graph) -> tuple[float, float]:
        c = nx.average_clustering(g, weight="weight")
        length_g = _length_graph(g)
        if not nx.is_connected(length_g):
            components = list(nx.connected_components(length_g))
            largest = max(components, key=len)
            length_g = length_g.subgraph(largest)
        length = _weighted_char_path_length(length_g)
        return c, length

    c_actual, l_actual = clustering_and_path(weight_graph)

    all_pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]

    c_rands, l_rands = [], []
    for _ in range(N_NULL_SURROGATES):
        chosen_pairs = rng.choice(len(all_pairs), size=len(edges), replace=False)
        g_rand = nx.Graph()
        g_rand.add_nodes_from(nodes)
        for w, idx in zip(weights_sorted_desc, chosen_pairs):
            u, v = all_pairs[idx]
            g_rand.add_edge(u, v, weight=w)
        c_r, l_r = clustering_and_path(g_rand)
        c_rands.append(c_r)
        l_rands.append(l_r)
    c_rand, l_rand = float(np.nanmean(c_rands)), float(np.nanmean(l_rands))

    # Lattice null: strongest weights go to the shortest ring-distance pairs.
    ring_pairs = sorted(all_pairs, key=lambda p: min(abs(nodes.index(p[0]) - nodes.index(p[1])), n - abs(nodes.index(p[0]) - nodes.index(p[1]))))
    g_latt = nx.Graph()
    g_latt.add_nodes_from(nodes)
    for w, (u, v) in zip(weights_sorted_desc, ring_pairs[: len(edges)]):
        g_latt.add_edge(u, v, weight=w)
    c_latt, l_latt = clustering_and_path(g_latt)

    delta_c = (c_latt - c_actual) / (c_latt - c_rand) if (c_latt - c_rand) != 0 else np.nan
    delta_l = (l_actual - l_rand) / (l_latt - l_rand) if (l_latt - l_rand) != 0 else np.nan
    delta_c = float(np.clip(delta_c, 0, 1)) if not np.isnan(delta_c) else np.nan
    delta_l = float(np.clip(delta_l, 0, 1)) if not np.isnan(delta_l) else np.nan
    if np.isnan(delta_c) or np.isnan(delta_l):
        return np.nan
    return float(1.0 - np.sqrt((delta_c**2 + delta_l**2) / 2.0))


def _auc_across_densities(values_by_density: list[float]) -> float | None:
    valid = [(d, v) for d, v in zip(DENSITY_PCTS, values_by_density) if v is not None and not np.isnan(v)]
    if len(valid) < 2:
        return None
    xs = np.array([d for d, _ in valid], dtype=float) / 100.0
    ys = np.array([v for _, v in valid], dtype=float)
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def _empty_band_features(band: str) -> dict:
    """All-None features for a band when fewer than 2 nodes have connectivity data."""
    features = {}
    for measure in [
        "mean_edge_weight", "global_efficiency_auc_density_20_50_percent",
        "characteristic_path_length_auc_density_20_50_percent",
        "mean_clustering_coefficient_auc_density_20_50_percent",
        "transitivity_auc_density_20_50_percent", "modularity_q_auc_density_20_50_percent",
        "assortativity_auc_density_20_50_percent", "small_world_propensity_auc_density_20_50_percent",
    ]:
        features[f"H_graph_theory__global__{measure}__{band}"] = None
    for measure in [
        "strength_normalized", "local_efficiency_auc_density_20_50_percent",
        "betweenness_centrality_auc_density_20_50_percent", "eigenvector_centrality_auc_density_20_50_percent",
        "participation_coefficient_auc_density_20_50_percent",
    ]:
        for node in regions.NODES:
            features[f"H_graph_theory__node__{measure}__{band}__{node}"] = None
    return features


def graph_features(connectivity_matrices: dict[str, np.ndarray] | None, valid_names: list[str]) -> dict:
    """connectivity_matrices: {band: (n_valid, n_valid) symmetric matrix, zero diagonal} from
    connectivity.compute_connectivity_matrices, or None if fewer than 2 nodes had data. Always
    emits the full 290-key schema (all 10 nodes x 5 bands); missing nodes/bands are None.
    """
    if connectivity_matrices is None or len(valid_names) < 2:
        features = {}
        for band in ["delta_1_4_hz", "theta_4_8_hz", "alpha_8_13_hz", "beta_13_30_hz", "low_gamma_30_45_hz"]:
            features.update(_empty_band_features(band))
        return features

    rng = np.random.default_rng(RANDOM_SEED)
    features = {}
    node_names = valid_names
    missing_nodes = [n for n in regions.NODES if n not in valid_names]

    for band, matrix in connectivity_matrices.items():
        wg = _weight_graph(matrix, node_names)
        density_graphs = _nested_density_graphs(wg, node_names)

        global_series = {m: [] for m in [
            "mean_edge_weight", "global_efficiency", "characteristic_path_length",
            "mean_clustering_coefficient", "transitivity", "modularity_q",
            "assortativity", "small_world_propensity",
        ]}
        node_series = {m: {n: [] for n in node_names} for m in [
            "strength_normalized", "local_efficiency", "betweenness_centrality",
            "eigenvector_centrality", "participation_coefficient",
        ]}

        for pct in DENSITY_PCTS:
            g = density_graphs[pct]
            lg = _length_graph(g)
            n = g.number_of_nodes()

            weights = [d["weight"] for _, _, d in g.edges(data=True)]
            global_series["mean_edge_weight"].append(float(np.mean(weights)) if weights else 0.0)
            global_series["global_efficiency"].append(nx.global_efficiency(lg))
            global_series["characteristic_path_length"].append(_weighted_char_path_length(lg))
            global_series["mean_clustering_coefficient"].append(nx.average_clustering(g, weight="weight"))
            global_series["transitivity"].append(nx.transitivity(g))  # binary topology, see module note

            try:
                communities = nx.community.louvain_communities(g, weight="weight", seed=RANDOM_SEED)
                modularity_q = nx.community.modularity(g, communities, weight="weight")
            except Exception:
                communities, modularity_q = [set(node_names)], np.nan
            global_series["modularity_q"].append(modularity_q)

            try:
                assortativity = nx.degree_pearson_correlation_coefficient(g, weight="weight")
            except Exception:
                assortativity = np.nan
            global_series["assortativity"].append(assortativity)

            global_series["small_world_propensity"].append(_small_world_propensity(g, rng))

            strength = dict(g.degree(weight="weight"))
            local_eff = _local_efficiency_per_node(lg)
            try:
                betweenness = nx.betweenness_centrality(lg, weight="weight")
            except Exception:
                betweenness = {n_: np.nan for n_ in node_names}
            try:
                eigenvector = nx.eigenvector_centrality(g, weight="weight", max_iter=1000)
            except Exception:
                eigenvector = {n_: np.nan for n_ in node_names}
            participation = _participation_coefficient(g, communities)

            for node in node_names:
                node_series["strength_normalized"][node].append(strength.get(node, 0.0) / max(n - 1, 1))
                node_series["local_efficiency"][node].append(local_eff.get(node, np.nan))
                node_series["betweenness_centrality"][node].append(betweenness.get(node, np.nan))
                node_series["eigenvector_centrality"][node].append(eigenvector.get(node, np.nan))
                node_series["participation_coefficient"][node].append(participation.get(node, np.nan))

        for measure, series in global_series.items():
            key = f"H_graph_theory__global__{_global_measure_token(measure)}__{band}"
            features[key] = _auc_across_densities(series) if measure != "mean_edge_weight" else float(
                np.mean(series)
            )

        for measure, per_node in node_series.items():
            for node in node_names:
                key = f"H_graph_theory__node__{_node_measure_token(measure)}__{band}__{node}"
                features[key] = _auc_across_densities(per_node[node])
            for node in missing_nodes:
                key = f"H_graph_theory__node__{_node_measure_token(measure)}__{band}__{node}"
                features[key] = None

    return features


def _global_measure_token(measure: str) -> str:
    tokens = {
        "mean_edge_weight": "mean_edge_weight",
        "global_efficiency": "global_efficiency_auc_density_20_50_percent",
        "characteristic_path_length": "characteristic_path_length_auc_density_20_50_percent",
        "mean_clustering_coefficient": "mean_clustering_coefficient_auc_density_20_50_percent",
        "transitivity": "transitivity_auc_density_20_50_percent",
        "modularity_q": "modularity_q_auc_density_20_50_percent",
        "assortativity": "assortativity_auc_density_20_50_percent",
        "small_world_propensity": "small_world_propensity_auc_density_20_50_percent",
    }
    return tokens[measure]


def _node_measure_token(measure: str) -> str:
    tokens = {
        "strength_normalized": "strength_normalized",
        "local_efficiency": "local_efficiency_auc_density_20_50_percent",
        "betweenness_centrality": "betweenness_centrality_auc_density_20_50_percent",
        "eigenvector_centrality": "eigenvector_centrality_auc_density_20_50_percent",
        "participation_coefficient": "participation_coefficient_auc_density_20_50_percent",
    }
    return tokens[measure]
