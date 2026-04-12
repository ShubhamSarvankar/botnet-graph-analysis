"""Tests for botnet_c2.graph.builder."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from botnet_c2.exceptions import GraphError
from botnet_c2.graph.builder import build_flow_graph, to_simple_undirected

# ── build_flow_graph ──────────────────────────────────────────────────────────


def test_build_flow_graph_returns_digraph(mini_flow_df):
    G = build_flow_graph(mini_flow_df)
    assert isinstance(G, nx.DiGraph)


def test_build_flow_graph_has_expected_node_count(mini_flow_df):
    G = build_flow_graph(mini_flow_df)
    # 20 unique SrcAddr + 10 unique DstAddr; some may overlap
    assert G.number_of_nodes() > 0


def test_build_flow_graph_has_edges(mini_flow_df):
    G = build_flow_graph(mini_flow_df)
    assert G.number_of_edges() > 0


def test_build_flow_graph_edge_has_weight_and_flows(mini_flow_df):
    G = build_flow_graph(mini_flow_df)
    for _, _, data in G.edges(data=True):
        assert "weight" in data
        assert "flows" in data


def test_build_flow_graph_weight_is_sum_of_totbytes(mini_flow_df):
    """Edge weight should aggregate TotBytes, not just count flows."""
    G = build_flow_graph(mini_flow_df)
    # Pick any edge and verify weight >= flows (bytes >= count)
    for _src, _dst, data in G.edges(data=True):
        assert data["weight"] >= data["flows"]


def test_build_flow_graph_aggregates_parallel_flows():
    """Multiple flows between the same pair should be collapsed to one edge."""
    df = pd.DataFrame(
        {
            "SrcAddr": ["1.1.1.1", "1.1.1.1", "1.1.1.1"],
            "DstAddr": ["2.2.2.2", "2.2.2.2", "2.2.2.2"],
            "TotBytes": [100.0, 200.0, 300.0],
        }
    )
    G = build_flow_graph(df)
    assert G.number_of_edges() == 1
    assert G["1.1.1.1"]["2.2.2.2"]["weight"] == pytest.approx(600.0)
    assert G["1.1.1.1"]["2.2.2.2"]["flows"] == 3


def test_build_flow_graph_directed_not_symmetric():
    """A→B and B→A should be separate directed edges."""
    df = pd.DataFrame(
        {
            "SrcAddr": ["1.1.1.1", "2.2.2.2"],
            "DstAddr": ["2.2.2.2", "1.1.1.1"],
            "TotBytes": [100.0, 50.0],
        }
    )
    G = build_flow_graph(df)
    assert G.number_of_edges() == 2
    assert G.has_edge("1.1.1.1", "2.2.2.2")
    assert G.has_edge("2.2.2.2", "1.1.1.1")


def test_build_flow_graph_drops_nan_addresses():
    """Rows with NaN SrcAddr or DstAddr should be silently dropped."""
    df = pd.DataFrame(
        {
            "SrcAddr": ["1.1.1.1", None, "3.3.3.3"],
            "DstAddr": ["2.2.2.2", "2.2.2.2", None],
            "TotBytes": [100.0, 200.0, 300.0],
        }
    )
    G = build_flow_graph(df)
    # Only the first row produces a valid edge
    assert G.number_of_edges() == 1


def test_build_flow_graph_empty_after_nan_drop():
    """All rows NaN → empty graph, no error."""
    df = pd.DataFrame({"SrcAddr": [None], "DstAddr": [None], "TotBytes": [100.0]})
    G = build_flow_graph(df)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_build_flow_graph_missing_column_raises():
    df = pd.DataFrame({"SrcAddr": ["1.1.1.1"], "DstAddr": ["2.2.2.2"]})
    with pytest.raises(GraphError):
        build_flow_graph(df)


def test_build_flow_graph_custom_weight_col():
    df = pd.DataFrame(
        {
            "SrcAddr": ["1.1.1.1"],
            "DstAddr": ["2.2.2.2"],
            "TotPkts": [42.0],
        }
    )
    G = build_flow_graph(df, weight_col="TotPkts")
    assert G["1.1.1.1"]["2.2.2.2"]["weight"] == pytest.approx(42.0)


def test_build_flow_graph_no_iterrows(mini_flow_df, monkeypatch):
    """Verify iterrows is not called during graph construction."""
    called = []
    original_iterrows = pd.DataFrame.iterrows

    def spy_iterrows(self):
        called.append(True)
        return original_iterrows(self)

    monkeypatch.setattr(pd.DataFrame, "iterrows", spy_iterrows)
    build_flow_graph(mini_flow_df)
    assert not called, "build_flow_graph must not use iterrows"


# ── to_simple_undirected ──────────────────────────────────────────────────────


def test_to_simple_undirected_returns_graph(star_graph):
    U = to_simple_undirected(star_graph)
    assert isinstance(U, nx.Graph)
    assert not isinstance(U, nx.DiGraph)


def test_to_simple_undirected_no_self_loops(star_graph):
    # Add a self-loop to verify removal
    star_graph.add_edge("0.0.0.1", "0.0.0.1", weight=1.0, flows=1)
    U = to_simple_undirected(star_graph)
    assert nx.number_of_selfloops(U) == 0


def test_to_simple_undirected_preserves_node_count(star_graph):
    U = to_simple_undirected(star_graph)
    assert U.number_of_nodes() == star_graph.number_of_nodes()


def test_to_simple_undirected_empty_graph():
    G = nx.DiGraph()
    U = to_simple_undirected(G)
    assert U.number_of_nodes() == 0
    assert U.number_of_edges() == 0


def test_to_simple_undirected_star_has_fewer_edges(star_graph):
    """Undirected star has same edges — directed star edges are uni-directional."""
    U = to_simple_undirected(star_graph)
    # 9 leaf→hub edges become 9 undirected edges (no anti-parallel to merge)
    assert U.number_of_edges() == 9
