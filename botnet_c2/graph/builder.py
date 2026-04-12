"""Flow graph construction.

Builds directed weighted graphs from flow DataFrames using vectorized
construction — groupby aggregation followed by nx.from_pandas_edgelist.
iterrows() is explicitly banned here: for scenarios with 26k+ flows the
row-iteration pattern is ~100× slower.

Public API:
    build_flow_graph(flows, weight_col) -> nx.DiGraph
    to_simple_undirected(G) -> nx.Graph
"""

from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from botnet_c2.exceptions import GraphError

logger = logging.getLogger(__name__)

_SRC_COL = "SrcAddr"
_DST_COL = "DstAddr"


def build_flow_graph(
    flows: pd.DataFrame,
    weight_col: str = "TotBytes",
) -> nx.DiGraph:
    """Build a directed weighted graph from a flow DataFrame.

    Each unique (SrcAddr, DstAddr) pair becomes a directed edge. The edge
    carries two attributes:
        weight  — sum of weight_col across all flows on that edge
        flows   — count of flows on that edge

    The caller is responsible for passing the appropriate flows (botnet-only
    for structural analysis; all flows for ML feature extraction). This
    function performs no label filtering.

    Args:
        flows: DataFrame containing at minimum SrcAddr, DstAddr, and weight_col.
        weight_col: Column to aggregate as edge weight. Defaults to "TotBytes".

    Returns:
        Directed graph with nodes = IP strings, edges weighted by total bytes
        and annotated with flow count.

    Raises:
        GraphError: If required columns are missing or flows is empty after
            dropping NaN addresses.
    """
    required = {_SRC_COL, _DST_COL, weight_col}
    missing = required - set(flows.columns)
    if missing:
        raise GraphError(
            f"build_flow_graph requires columns {required}; missing: {missing}"
        )

    clean = flows.dropna(subset=[_SRC_COL, _DST_COL])
    if clean.empty:
        logger.warning("build_flow_graph: all rows dropped after NaN address filter")
        return nx.DiGraph()

    # Vectorized aggregation — no iterrows
    edge_df = (
        clean.groupby([_SRC_COL, _DST_COL], sort=False)[weight_col]
        .agg(weight="sum", flows="count")
        .reset_index()
    )

    G = nx.from_pandas_edgelist(
        edge_df,
        source=_SRC_COL,
        target=_DST_COL,
        edge_attr=["weight", "flows"],
        create_using=nx.DiGraph(),
    )

    logger.debug(
        "Built directed graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def to_simple_undirected(G: nx.DiGraph) -> nx.Graph:
    """Convert a directed graph to a simple undirected graph.

    Steps:
        1. Convert directed → undirected (anti-parallel edges are merged).
        2. Remove self-loops (a node connected only to itself contributes no
           triangles and inflates degree counts).

    This is the canonical conversion used by topology.py for k-core and
    betweenness computation, and by features/windows.py for undirected
    graph features.

    Args:
        G: Any nx.DiGraph.

    Returns:
        nx.Graph with no self-loops.
    """
    U = G.to_undirected()
    U.remove_edges_from(nx.selfloop_edges(U))
    return U
