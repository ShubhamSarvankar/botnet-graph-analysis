"""Per-(time_bin, ip) graph feature extraction.

Builds the ML feature matrix by splitting a full-traffic flow DataFrame into
5-minute time windows, constructing a graph per window, and computing node-level
structural features for every IP active in that window.

**Critical design constraint — no label filtering:**
This module receives full-traffic flows (all labels). It never filters by Label
before building window graphs. Filtering before feature extraction would be data
leakage: the model must operate without knowing which flows are botnet in advance.
Label attachment happens exclusively in engineering.attach_labels(), always after
this module.

**Vectorized graph construction:**
Uses groupby + nx.from_pandas_edgelist per window slice. iterrows is banned.

**Betweenness approximation:**
For windows with > 200 nodes, betweenness uses k=min(50, len(G)) sampled pivots
to keep runtime manageable on large captures (neris_50 full-traffic).

**Domain note — local_clustering excluded:**
avg_clustering = 0 for all 13 CTU-13 botnet subgraphs (hub-and-spoke topology,
no triangles by construction). local_clustering is therefore not a useful ML
feature and is excluded. flow_count (total flows through a node, from the 'flows'
edge attribute) is used as a proxy for communication intensity instead.
Confirmed via sanity check on neris_50 before finalizing this module.

Public API:
    build_window_features(df, scenario_id, window) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd

from botnet_c2.graph.builder import build_flow_graph, to_simple_undirected

logger = logging.getLogger(__name__)

# Time window size for binning flows.
_DEFAULT_WINDOW = "5min"

# Threshold above which betweenness is approximated rather than exact.
_BETWEENNESS_EXACT_MAX = 200

# Number of pivot nodes for approximate betweenness.
_BETWEENNESS_K = 50


def build_window_features(
    df: pd.DataFrame,
    scenario_id: str,
    window: str = _DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Build the per-(time_bin, ip) feature matrix from full-traffic flows.

    Each row represents one IP address active in one time window. The returned
    DataFrame does NOT contain is_bot — that is attached by
    engineering.attach_labels() after this function returns.

    Args:
        df: Full-traffic flow DataFrame (all labels, not filtered). Must contain
            SrcAddr, DstAddr, TotBytes, StartTime columns.
        scenario_id: Scenario key stored in the output for traceability.
        window: Pandas offset string for time binning. Default "5min".

    Returns:
        DataFrame indexed by (time_bin, ip) with columns:
            scenario_id, time_bin, ip,
            degree, in_degree, out_degree, kcore, pagerank,
            betweenness, flow_count, window_node_count

    Raises:
        ValueError: If StartTime column is missing.
    """
    if "StartTime" not in df.columns:
        raise ValueError("df must contain a 'StartTime' column")

    if df.empty:
        return _empty_feature_df()

    df = df.copy()
    df["time_bin"] = df["StartTime"].dt.floor(window)

    grouped = df.groupby("time_bin", sort=True)
    logger.info(
        "Building window features for %s: %d flows, %d time bins",
        scenario_id,
        len(df),
        grouped.ngroups,
    )

    records: list[dict[str, Any]] = []
    for time_bin, window_df in grouped:
        window_records = _process_window(window_df, time_bin, scenario_id)
        records.extend(window_records)

    if not records:
        return _empty_feature_df()

    result = pd.DataFrame(records)
    result["time_bin"] = pd.to_datetime(result["time_bin"])
    result = result.set_index(["time_bin", "ip"])

    logger.info(
        "Feature matrix for %s: %d rows, %d columns",
        scenario_id,
        len(result),
        len(result.columns),
    )
    return result


def _process_window(
    window_df: pd.DataFrame,
    time_bin: pd.Timestamp,
    scenario_id: str,
) -> list[dict[str, Any]]:
    """Compute node-level features for all IPs in a single time window."""
    G_dir = build_flow_graph(window_df, weight_col="TotBytes")

    if G_dir.number_of_nodes() == 0:
        return []

    G_undir = to_simple_undirected(G_dir)
    n_nodes = G_dir.number_of_nodes()

    # ── Degree features ───────────────────────────────────────────────────────
    in_deg: dict[str, int] = dict(G_dir.in_degree())
    out_deg: dict[str, int] = dict(G_dir.out_degree())
    total_deg: dict[str, int] = dict(G_undir.degree())

    # ── k-core (undirected) ───────────────────────────────────────────────────
    core_numbers: dict[str, int] = nx.core_number(G_undir)

    # ── PageRank (directed) ───────────────────────────────────────────────────
    pagerank: dict[str, float] = nx.pagerank(G_dir)

    # ── Betweenness (undirected, approximated for large windows) ─────────────
    if n_nodes > _BETWEENNESS_EXACT_MAX:
        k = min(_BETWEENNESS_K, n_nodes)
        betweenness: dict[str, float] = nx.betweenness_centrality(
            G_undir, k=k, normalized=True
        )
    else:
        betweenness = nx.betweenness_centrality(G_undir, normalized=True)

    # ── flow_count: total flows through each node (in + out) ─────────────────
    # Sum the 'flows' edge attribute for all edges incident to each node.
    flow_count = _compute_flow_count(G_dir)

    # ── Assemble per-node records ─────────────────────────────────────────────
    records = []
    for node in G_dir.nodes():
        records.append(
            {
                "scenario_id": scenario_id,
                "time_bin": time_bin,
                "ip": node,
                "degree": int(total_deg.get(node, 0)),
                "in_degree": int(in_deg.get(node, 0)),
                "out_degree": int(out_deg.get(node, 0)),
                "kcore": int(core_numbers.get(node, 0)),
                "pagerank": float(pagerank.get(node, 0.0)),
                "betweenness": float(betweenness.get(node, 0.0)),
                "flow_count": int(flow_count.get(node, 0)),
                "window_node_count": int(n_nodes),
                "in_degree_norm": float(in_deg.get(node, 0)) / max(n_nodes, 1),
                "flow_count_norm": float(flow_count.get(node, 0)) / max(n_nodes, 1),
            }
        )
    return records


def _compute_flow_count(G_dir: nx.DiGraph) -> dict[str, int]:
    """Sum the 'flows' edge attribute for all edges incident to each node.

    Counts both outgoing and incoming flows for each IP, giving a measure
    of total communication activity regardless of direction.
    """
    counts: dict[str, int] = dict.fromkeys(G_dir.nodes(), 0)
    for src, dst, data in G_dir.edges(data=True):
        f = int(data.get("flows", 1))
        counts[src] = counts.get(src, 0) + f
        counts[dst] = counts.get(dst, 0) + f
    return counts


def _empty_feature_df() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    columns = [
        "scenario_id",
        "degree",
        "in_degree",
        "out_degree",
        "kcore",
        "pagerank",
        "betweenness",
        "flow_count",
        "window_node_count",
    ]
    df = pd.DataFrame(columns=columns)
    df.index = pd.MultiIndex.from_tuples([], names=["time_bin", "ip"])
    return df
