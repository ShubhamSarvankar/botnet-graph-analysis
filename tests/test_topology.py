"""Tests for botnet_c2.graph.topology."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from botnet_c2.graph.topology import compute_robustness, compute_topology

# ── compute_topology ──────────────────────────────────────────────────────────


def test_compute_topology_returns_dict(star_graph):
    result = compute_topology(star_graph, "test_scenario")
    assert isinstance(result, dict)


def test_compute_topology_scenario_id_present(star_graph):
    result = compute_topology(star_graph, "neris_42")
    assert result["scenario_id"] == "neris_42"


def test_compute_topology_required_keys_present(star_graph):
    result = compute_topology(star_graph, "test")
    required_keys = {
        "scenario_id",
        "nodes",
        "edges",
        "density",
        "gc_fraction",
        "avg_clustering",
        "avg_path_length",
        "sigma",
        "assortativity",
        "max_kcore",
        "powerlaw_gamma",
        "max_in_degree",
        "max_out_degree",
        "top5_pagerank",
        "robustness_auc_targeted",
        "robustness_auc_random",
        "degree_ccdf_x",
        "degree_ccdf_y",
    }
    assert required_keys.issubset(result.keys())


def test_compute_topology_json_serializable(star_graph):
    """All values must be JSON-safe — no numpy types."""
    result = compute_topology(star_graph, "test")
    # Should not raise
    serialized = json.dumps(result)
    assert len(serialized) > 0


def test_compute_topology_json_serializable_mesh(mesh_graph):
    result = compute_topology(mesh_graph, "mesh_test")
    json.dumps(result)  # must not raise


def test_compute_topology_node_count_correct(star_graph):
    result = compute_topology(star_graph, "test")
    assert result["nodes"] == star_graph.number_of_nodes()


def test_compute_topology_edge_count_correct(star_graph):
    result = compute_topology(star_graph, "test")
    assert result["edges"] == star_graph.number_of_edges()


def test_compute_topology_gc_fraction_star(star_graph):
    """Star graph is fully connected — gc_fraction should be 1.0."""
    result = compute_topology(star_graph, "test")
    assert result["gc_fraction"] == pytest.approx(1.0)


def test_compute_topology_gc_fraction_disconnected():
    """Two disconnected stars → gc_fraction < 1.0."""
    G = nx.DiGraph()
    # Star 1
    for i in range(2, 6):
        G.add_edge(f"0.0.0.{i}", "0.0.0.1", weight=100.0, flows=1)
    # Star 2 (isolated)
    for i in range(12, 16):
        G.add_edge(f"0.0.1.{i}", "0.0.1.11", weight=100.0, flows=1)
    result = compute_topology(G, "disconnected")
    assert result["gc_fraction"] < 1.0


def test_compute_topology_avg_clustering_star_is_zero(star_graph):
    """Hub-and-spoke topology: no triangles → clustering = 0."""
    result = compute_topology(star_graph, "test")
    assert result["avg_clustering"] == pytest.approx(0.0)


def test_compute_topology_avg_clustering_mesh_nonzero(mesh_graph):
    """Near-complete graph should have nonzero clustering."""
    result = compute_topology(mesh_graph, "mesh")
    assert result["avg_clustering"] > 0.0


def test_compute_topology_max_in_degree_star(star_graph):
    """Hub has in_degree = 9 in a 10-node star (9 leaves → hub)."""
    result = compute_topology(star_graph, "test")
    assert result["max_in_degree"] == 9


def test_compute_topology_max_out_degree_star(star_graph):
    """Leaves have out_degree = 1; hub has out_degree = 0."""
    result = compute_topology(star_graph, "test")
    assert result["max_out_degree"] == 1


def test_compute_topology_top5_pagerank_is_list(star_graph):
    result = compute_topology(star_graph, "test")
    assert isinstance(result["top5_pagerank"], list)
    assert len(result["top5_pagerank"]) <= 5


def test_compute_topology_top5_pagerank_hub_first(star_graph):
    """Hub receives all traffic — should have highest PageRank."""
    result = compute_topology(star_graph, "test")
    top_node = result["top5_pagerank"][0][0]
    assert top_node == "0.0.0.1"


def test_compute_topology_powerlaw_gamma_none_small_graph():
    """Graph with < 50 nodes → powerlaw_gamma must be None."""
    G = nx.DiGraph()
    # 10-node star, well below the 50-node threshold
    for i in range(2, 11):
        G.add_edge(f"0.0.0.{i}", "0.0.0.1", weight=100.0, flows=1)
    result = compute_topology(G, "small")
    assert result["powerlaw_gamma"] is None


def test_compute_topology_degree_ccdf_lists(star_graph):
    result = compute_topology(star_graph, "test")
    assert isinstance(result["degree_ccdf_x"], list)
    assert isinstance(result["degree_ccdf_y"], list)
    assert len(result["degree_ccdf_x"]) == len(result["degree_ccdf_y"])


def test_compute_topology_degree_ccdf_values_in_range(star_graph):
    """CCDF y-values should be in [0, 1]."""
    result = compute_topology(star_graph, "test")
    for y in result["degree_ccdf_y"]:
        assert 0.0 <= y <= 1.0


def test_compute_topology_empty_graph():
    """Empty graph should not raise; returns zero-valued metrics."""
    G = nx.DiGraph()
    result = compute_topology(G, "empty")
    assert result["nodes"] == 0
    assert result["edges"] == 0
    assert result["gc_fraction"] == 0.0
    json.dumps(result)  # must be JSON-serializable


def test_compute_topology_single_node():
    """Single isolated node should not raise."""
    G = nx.DiGraph()
    G.add_node("1.1.1.1")
    result = compute_topology(G, "single")
    assert result["nodes"] == 1
    assert result["edges"] == 0
    json.dumps(result)


# ── compute_robustness ────────────────────────────────────────────────────────


def test_compute_robustness_returns_dict(star_graph):
    result = compute_robustness(star_graph)
    assert isinstance(result, dict)


def test_compute_robustness_required_keys(star_graph):
    result = compute_robustness(star_graph)
    assert "targeted" in result
    assert "random" in result
    assert "auc_targeted" in result
    assert "auc_random" in result


def test_compute_robustness_curve_length(star_graph):
    steps = 10
    result = compute_robustness(star_graph, steps=steps)
    assert len(result["targeted"]) == steps
    assert len(result["random"]) == steps


def test_compute_robustness_targeted_auc_leq_random_auc(star_graph):
    """Targeted removal should degrade the network faster than random removal.
    AUC under targeted curve should be <= AUC under random curve."""
    result = compute_robustness(star_graph, steps=20)
    # This holds for hub-and-spoke graphs (removing the hub destroys connectivity)
    assert result["auc_targeted"] <= result["auc_random"] + 0.05  # small tolerance


def test_compute_robustness_auc_in_range(star_graph):
    result = compute_robustness(star_graph, steps=20)
    assert 0.0 <= result["auc_targeted"] <= 1.0
    assert 0.0 <= result["auc_random"] <= 1.0


def test_compute_robustness_empty_graph():
    G = nx.DiGraph()
    result = compute_robustness(G, steps=5)
    assert result["auc_targeted"] == pytest.approx(0.0)
    assert result["auc_random"] == pytest.approx(0.0)
