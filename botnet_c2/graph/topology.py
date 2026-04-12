"""Scenario-level structural metrics for botnet graph analysis.

Computes a flat dict of graph topology metrics for a single scenario's
botnet-only subgraph. All values are JSON-safe (Python int/float/list, no
numpy types).

**Important design boundary:**
This module operates exclusively on botnet-only subgraphs (caller passes
split_flows output). These metrics are for structural EDA and Framing A
visualization ONLY — they must never be used as ML features, because they
are derived from labeled data and would constitute data leakage.
See features/windows.py for the ML feature extraction path.

**Domain finding — clustering coefficient:**
avg_clustering = 0.0 for all 13 CTU-13 scenarios. This is expected for
hub-and-spoke C2 topology: infected hosts connect only to the controller,
so no triangles form and the clustering coefficient is zero by construction.
Confirmed via sanity check on neris_50 LCC before finalizing this module.
local_clustering is therefore excluded from the ML feature set (see
features/windows.py). flow_count is used as a proxy for communication
intensity instead.
sigma (small-world coefficient) is computed and stored but will also be 0
as a cascading consequence of zero clustering.

Public API:
    compute_topology(G, scenario_id) -> dict
    compute_robustness(G, steps) -> dict
"""

from __future__ import annotations

import logging
import random
from typing import Any

import networkx as nx
import numpy as np

from botnet_c2.graph.builder import to_simple_undirected

logger = logging.getLogger(__name__)

# Minimum positive-degree nodes required for a meaningful power-law fit.
# rbot_51 (44 nodes), rbot_52 (12 nodes), sogou_48 (18 nodes) fall below this.
_POWERLAW_MIN_N = 50

# Number of steps for robustness curve computation.
_ROBUSTNESS_STEPS = 20

# Number of nodes to sample for betweenness approximation in large graphs.
_BETWEENNESS_K = 50


def compute_topology(G: nx.DiGraph, scenario_id: str) -> dict[str, Any]:
    """Compute scenario-level structural metrics from a botnet flow graph.

    The input graph should be built from botnet-only flows (split_flows output)
    for structural analysis. These metrics are used in EDA notebooks and
    Framing A visualization — never as ML features.

    Metrics returned:
        scenario_id, nodes, edges, density, gc_fraction,
        avg_clustering, avg_path_length, sigma, assortativity,
        max_kcore, powerlaw_gamma (None if n < 50),
        max_in_degree, max_out_degree,
        top5_pagerank (list of [node, score]),
        robustness_auc_targeted, robustness_auc_random,
        degree_ccdf_x (list), degree_ccdf_y (list)

    All values are JSON-safe: int, float, list, str, or None. No numpy types.

    Args:
        G: Directed flow graph (botnet subgraph).
        scenario_id: Scenario key, stored in the output dict for traceability.

    Returns:
        Flat dict of metrics, ready for json.dumps().
    """
    U = to_simple_undirected(G)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # ── Giant component (weakly connected, on directed graph) ─────────────────
    if n_nodes == 0:
        gc_fraction = 0.0
        lcc_U = nx.Graph()
    else:
        wcc = list(nx.weakly_connected_components(G))
        lcc_nodes = max(wcc, key=len)
        gc_fraction = float(len(lcc_nodes) / n_nodes)
        lcc_U = to_simple_undirected(G.subgraph(lcc_nodes).copy())

    # ── Clustering (expected 0.0 for hub-and-spoke — see module docstring) ────
    avg_clustering = float(nx.average_clustering(U)) if n_nodes > 0 else 0.0

    # ── Average shortest path length (LCC only; undefined for disconnected) ───
    if lcc_U.number_of_nodes() > 1:
        avg_path_length = float(nx.average_shortest_path_length(lcc_U))
    else:
        avg_path_length = 0.0

    # ── Small-world sigma ─────────────────────────────────────────────────────
    # sigma = (C/C_rand) / (L/L_rand). Will be ~0 when clustering is 0.
    sigma = _compute_sigma(lcc_U, avg_clustering, avg_path_length)

    # ── Assortativity ─────────────────────────────────────────────────────────
    try:
        assortativity = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        assortativity = float("nan")

    # ── k-core (undirected) ───────────────────────────────────────────────────
    if U.number_of_nodes() > 0:
        core_numbers = nx.core_number(U)
        max_kcore = int(max(core_numbers.values()))
    else:
        max_kcore = 0

    # ── Power-law gamma ───────────────────────────────────────────────────────
    powerlaw_gamma = _compute_powerlaw_gamma(U)

    # ── Degree stats (directed) ───────────────────────────────────────────────
    if n_nodes > 0:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        max_in_degree = int(max(in_degrees.values()))
        max_out_degree = int(max(out_degrees.values()))
    else:
        max_in_degree = 0
        max_out_degree = 0

    # ── PageRank top-5 ────────────────────────────────────────────────────────
    top5_pagerank: list[list] = []
    if n_nodes > 0:
        pr = nx.pagerank(G)
        top5 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_pagerank = [[str(node), float(score)] for node, score in top5]

    # ── Degree CCDF ───────────────────────────────────────────────────────────
    degree_ccdf_x, degree_ccdf_y = _compute_degree_ccdf(U)

    # ── Robustness curves ─────────────────────────────────────────────────────
    robustness = compute_robustness(G, steps=_ROBUSTNESS_STEPS)

    return {
        "scenario_id": scenario_id,
        "nodes": int(n_nodes),
        "edges": int(n_edges),
        "density": float(nx.density(G)),
        "gc_fraction": gc_fraction,
        "avg_clustering": avg_clustering,
        "avg_path_length": avg_path_length,
        "sigma": sigma,
        "assortativity": assortativity,
        "max_kcore": max_kcore,
        "powerlaw_gamma": powerlaw_gamma,
        "max_in_degree": max_in_degree,
        "max_out_degree": max_out_degree,
        "top5_pagerank": top5_pagerank,
        "robustness_auc_targeted": robustness["auc_targeted"],
        "robustness_auc_random": robustness["auc_random"],
        "degree_ccdf_x": degree_ccdf_x,
        "degree_ccdf_y": degree_ccdf_y,
    }


def compute_robustness(
    G: nx.DiGraph,
    steps: int = _ROBUSTNESS_STEPS,
) -> dict[str, Any]:
    """Compute targeted and random node removal robustness curves.

    Measures how the giant connected component (GCC) fraction shrinks as nodes
    are removed. Targeted removal removes highest-degree nodes first; random
    removal removes nodes uniformly at random.

    Args:
        G: Directed flow graph.
        steps: Number of equally-spaced removal fractions to sample (0 to 1).

    Returns:
        Dict with keys:
            targeted: list of gc_fraction values at each removal step
            random:   list of gc_fraction values at each removal step
            auc_targeted: area under targeted curve (float)
            auc_random:   area under random curve (float)
    """
    U = to_simple_undirected(G)
    n = U.number_of_nodes()
    if n == 0:
        empty: list[float] = [0.0] * steps
        return {
            "targeted": empty,
            "random": empty,
            "auc_targeted": 0.0,
            "auc_random": 0.0,
        }

    targeted_curve = _removal_curve(U, strategy="targeted", steps=steps)
    random_curve = _removal_curve(U, strategy="random", steps=steps)

    x = np.linspace(0, 1, steps)
    auc_targeted = float(np.trapezoid(targeted_curve, x))
    auc_random = float(np.trapezoid(random_curve, x))

    return {
        "targeted": [float(v) for v in targeted_curve],
        "random": [float(v) for v in random_curve],
        "auc_targeted": auc_targeted,
        "auc_random": auc_random,
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _removal_curve(
    U: nx.Graph,
    strategy: str,
    steps: int,
) -> list[float]:
    """Return gc_fraction at each removal step for a given strategy."""
    H = U.copy()
    n = H.number_of_nodes()
    curve: list[float] = []

    if strategy == "targeted":
        # Sort nodes by degree descending once; remove in that order.
        removal_order = sorted(H.nodes(), key=lambda v: H.degree(v), reverse=True)
    else:
        removal_order = list(H.nodes())
        random.shuffle(removal_order)

    removal_fractions = np.linspace(0, 1, steps)
    removed_count = 0

    for frac in removal_fractions:
        target_removed = int(frac * n)
        while removed_count < target_removed and removal_order:
            node = removal_order[removed_count]
            if H.has_node(node):
                H.remove_node(node)
            removed_count += 1

        remaining = H.number_of_nodes()
        if remaining == 0:
            curve.append(0.0)
        else:
            wcc = nx.connected_components(H)
            gcc_size = max((len(c) for c in wcc), default=0)
            curve.append(float(gcc_size / n))

    return curve


def _compute_powerlaw_gamma(U: nx.Graph) -> float | None:
    """Fit a power-law to the degree sequence, or return None if n < 50.

    rbot_51 (44 nodes) and rbot_52 (12 nodes) produced gamma = 14.24 and 1.76
    in the previous codebase — artifacts of tiny sample size, not real power-law
    behavior. This guard prevents those nonsense values.
    """
    try:
        import powerlaw  # optional dependency, only needed here
    except ImportError:
        logger.warning("powerlaw package not installed; skipping gamma fit")
        return None

    degrees = [d for _, d in U.degree() if d > 0]
    if len(degrees) < _POWERLAW_MIN_N:
        logger.debug(
            "Only %d positive-degree nodes; skipping power-law fit (min=%d)",
            len(degrees),
            _POWERLAW_MIN_N,
        )
        return None

    try:
        fit = powerlaw.Fit(degrees, verbose=False)
        return float(fit.power_law.alpha)
    except Exception as exc:
        logger.warning("powerlaw.Fit failed: %s", exc)
        return None


def _compute_degree_ccdf(U: nx.Graph) -> tuple[list[float], list[float]]:
    """Compute the complementary cumulative degree distribution."""
    if U.number_of_nodes() == 0:
        return [], []

    degrees = sorted([d for _, d in U.degree()], reverse=True)
    n = len(degrees)
    unique_degrees = sorted(set(degrees))

    ccdf_x: list[float] = []
    ccdf_y: list[float] = []

    for k in unique_degrees:
        ccdf_x.append(float(k))
        ccdf_y.append(float(sum(1 for d in degrees if d >= k) / n))

    return ccdf_x, ccdf_y


def _compute_sigma(
    lcc_U: nx.Graph,
    avg_clustering: float,
    avg_path_length: float,
) -> float:
    """Estimate small-world sigma = (C/C_rand) / (L/L_rand).

    For hub-and-spoke graphs where avg_clustering = 0, sigma will be 0.
    This is documented as a domain finding, not a bug.
    """
    n = lcc_U.number_of_nodes()
    if n < 3 or avg_clustering == 0.0 or avg_path_length == 0.0:
        return 0.0

    m = lcc_U.number_of_edges()
    if m == 0:
        return 0.0

    # Random graph reference values (Erdős–Rényi approximations)
    p = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
    c_rand = p  # expected clustering of ER graph
    l_rand = np.log(n) / np.log(n * p) if (n * p) > 1 else float("nan")

    if c_rand == 0.0 or np.isnan(l_rand) or l_rand == 0.0:
        return 0.0

    return float((avg_clustering / c_rand) / (avg_path_length / l_rand))
