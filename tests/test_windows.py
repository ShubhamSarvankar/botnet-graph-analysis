"""Tests for botnet_c2.features.windows."""

from __future__ import annotations

import pandas as pd
import pytest

from botnet_c2.features.windows import build_window_features

# ── build_window_features ─────────────────────────────────────────────────────


def test_returns_dataframe(mini_flow_df):
    result = build_window_features(mini_flow_df, "test_scenario")
    assert isinstance(result, pd.DataFrame)


def test_index_names(mini_flow_df):
    result = build_window_features(mini_flow_df, "test_scenario")
    assert result.index.names == ["time_bin", "ip"]


def test_expected_feature_columns_present(mini_flow_df):
    result = build_window_features(mini_flow_df, "test_scenario")
    expected = {
        "scenario_id",
        "degree",
        "in_degree",
        "out_degree",
        "kcore",
        "pagerank",
        "betweenness",
        "flow_count",
        "window_node_count",
    }
    assert expected.issubset(set(result.columns))


def test_no_is_bot_column(mini_flow_df):
    """is_bot must never be present before attach_labels() is called."""
    result = build_window_features(mini_flow_df, "test_scenario")
    assert "is_bot" not in result.columns


def test_no_label_column(mini_flow_df):
    """Label column must not leak into the feature matrix."""
    result = build_window_features(mini_flow_df, "test_scenario")
    assert "Label" not in result.columns


def test_scenario_id_stored(mini_flow_df):
    result = build_window_features(mini_flow_df, "neris_42")
    assert (result["scenario_id"] == "neris_42").all()


def test_two_time_bins_produced(mini_flow_df):
    """mini_flow_df spans two 5-minute bins — both should appear."""
    result = build_window_features(mini_flow_df, "test")
    n_bins = result.index.get_level_values("time_bin").nunique()
    assert n_bins == 2


def test_no_nan_in_core_columns(mini_flow_df):
    result = build_window_features(mini_flow_df, "test")
    core_cols = [
        "degree",
        "in_degree",
        "out_degree",
        "kcore",
        "pagerank",
        "betweenness",
        "flow_count",
        "window_node_count",
    ]
    for col in core_cols:
        assert result[col].isna().sum() == 0, f"NaN found in {col}"


def test_degree_nonnegative(mini_flow_df):
    result = build_window_features(mini_flow_df, "test")
    assert (result["degree"] >= 0).all()


def test_in_out_degree_sum_consistency(mini_flow_df):
    """For undirected degree, in+out can exceed degree due to direction merging,
    but both must be non-negative."""
    result = build_window_features(mini_flow_df, "test")
    assert (result["in_degree"] >= 0).all()
    assert (result["out_degree"] >= 0).all()


def test_kcore_nonnegative(mini_flow_df):
    result = build_window_features(mini_flow_df, "test")
    assert (result["kcore"] >= 0).all()


def test_pagerank_sums_to_approx_one_per_bin(mini_flow_df):
    """PageRank values in each time bin should sum to approximately 1.0."""
    result = build_window_features(mini_flow_df, "test")
    for time_bin, group in result.groupby("time_bin"):
        total = group["pagerank"].sum()
        assert abs(total - 1.0) < 0.01, f"PageRank sum={total} for bin {time_bin}"


def test_betweenness_in_range(mini_flow_df):
    result = build_window_features(mini_flow_df, "test")
    assert (result["betweenness"] >= 0.0).all()
    assert (result["betweenness"] <= 1.0).all()


def test_flow_count_positive(mini_flow_df):
    result = build_window_features(mini_flow_df, "test")
    assert (result["flow_count"] > 0).all()


def test_window_node_count_consistent(mini_flow_df):
    """All nodes in the same time bin must share the same window_node_count."""
    result = build_window_features(mini_flow_df, "test")
    for _time_bin, group in result.groupby("time_bin"):
        assert group["window_node_count"].nunique() == 1


def test_botnet_flow_df_produces_features(mini_botnet_flow_df):
    """Full-traffic df including botnet flows must work without error."""
    result = build_window_features(mini_botnet_flow_df, "test")
    assert len(result) > 0


def test_no_label_filtering_applied(mini_botnet_flow_df):
    """All IPs (bot and normal) must appear in the feature matrix."""
    result = build_window_features(mini_botnet_flow_df, "test")
    # C2 IP appears in botnet flows — it must be in the feature matrix
    all_ips = result.index.get_level_values("ip")
    assert "10.0.0.1" in all_ips


def test_empty_dataframe_returns_empty():
    empty_df = pd.DataFrame(
        columns=["StartTime", "SrcAddr", "DstAddr", "TotBytes", "Label"]
    )
    result = build_window_features(empty_df, "test")
    assert len(result) == 0


def test_missing_starttime_raises():
    df = pd.DataFrame(
        {"SrcAddr": ["1.1.1.1"], "DstAddr": ["2.2.2.2"], "TotBytes": [100.0]}
    )
    with pytest.raises(ValueError, match="StartTime"):
        build_window_features(df, "test")


def test_no_iterrows_used(mini_flow_df, monkeypatch):
    """Vectorized construction must not use iterrows anywhere in windows.py."""
    called = []
    original = pd.DataFrame.iterrows

    def spy(self):
        called.append(True)
        return original(self)

    monkeypatch.setattr(pd.DataFrame, "iterrows", spy)
    build_window_features(mini_flow_df, "test")
    assert not called, "build_window_features must not use iterrows"
