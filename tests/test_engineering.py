"""Tests for botnet_c2.features.engineering."""

from __future__ import annotations

import pytest

from botnet_c2.features.engineering import (
    LEAKAGE_COLUMNS,
    add_delta_features,
    attach_labels,
    compute_window_bot_fraction,
    get_feature_columns,
)
from botnet_c2.features.windows import build_window_features

# ── get_feature_columns ───────────────────────────────────────────────────────


def test_get_feature_columns_returns_list():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_get_feature_columns_contains_required():
    cols = get_feature_columns()
    required = {
        "degree",
        "in_degree",
        "out_degree",
        "kcore",
        "pagerank",
        "betweenness",
        "flow_count",
        "delta_degree",
        "window_node_count",
    }
    assert required == set(cols)


def test_get_feature_columns_no_leakage():
    cols = get_feature_columns()
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in cols, (
            f"Leakage column '{leak_col}' found in feature columns"
        )


def test_get_feature_columns_no_is_bot():
    cols = get_feature_columns()
    assert "is_bot" not in cols


def test_leakage_columns_defined():
    assert isinstance(LEAKAGE_COLUMNS, list)
    assert "window_bot_fraction" in LEAKAGE_COLUMNS


# ── add_delta_features ────────────────────────────────────────────────────────


def test_add_delta_features_adds_column(mini_flow_df):
    features = build_window_features(mini_flow_df, "test")
    result = add_delta_features(features)
    assert "delta_degree" in result.columns


def test_add_delta_features_dtype_int(mini_flow_df):
    features = build_window_features(mini_flow_df, "test")
    result = add_delta_features(features)
    assert result["delta_degree"].dtype == "int64"


def test_add_delta_features_first_appearance_is_zero(mini_flow_df):
    """IPs appearing for the first time must have delta_degree = 0."""
    features = build_window_features(mini_flow_df, "test")
    result = add_delta_features(features)
    # Get each IP's first time_bin
    flat = result.reset_index()
    first_appearances = flat.groupby("ip")["time_bin"].min()
    for ip, first_bin in first_appearances.items():
        delta = flat[(flat["ip"] == ip) & (flat["time_bin"] == first_bin)][
            "delta_degree"
        ].iloc[0]
        assert delta == 0, (
            f"IP {ip} first appearance has delta_degree={delta}, expected 0"
        )


def test_add_delta_features_no_new_nan(mini_flow_df):
    features = build_window_features(mini_flow_df, "test")
    result = add_delta_features(features)
    assert result["delta_degree"].isna().sum() == 0


def test_add_delta_features_preserves_other_columns(mini_flow_df):
    features = build_window_features(mini_flow_df, "test")
    original_cols = set(features.columns)
    result = add_delta_features(features)
    assert original_cols.issubset(set(result.columns))


def test_add_delta_features_empty_df():
    from botnet_c2.features.windows import _empty_feature_df

    empty = _empty_feature_df()
    result = add_delta_features(empty)
    assert "delta_degree" in result.columns
    assert len(result) == 0


def test_add_delta_features_does_not_mutate_input(mini_flow_df):
    features = build_window_features(mini_flow_df, "test")
    original_cols = list(features.columns)
    add_delta_features(features)
    assert list(features.columns) == original_cols


# ── attach_labels ─────────────────────────────────────────────────────────────


def test_attach_labels_adds_is_bot_column(mini_botnet_flow_df):
    features = build_window_features(mini_botnet_flow_df, "test")
    result = attach_labels(features, mini_botnet_flow_df)
    assert "is_bot" in result.columns


def test_attach_labels_dtype_bool(mini_botnet_flow_df):
    features = build_window_features(mini_botnet_flow_df, "test")
    result = attach_labels(features, mini_botnet_flow_df)
    assert result["is_bot"].dtype == bool


def test_attach_labels_c2_ip_is_true(mini_botnet_flow_df):
    """The C2 IP (10.0.0.1) must be labeled is_bot=True."""
    features = build_window_features(mini_botnet_flow_df, "test")
    result = attach_labels(features, mini_botnet_flow_df)
    flat = result.reset_index()
    c2_rows = flat[flat["ip"] == "10.0.0.1"]
    assert len(c2_rows) > 0, "C2 IP not found in feature matrix"
    assert c2_rows["is_bot"].all(), "C2 IP must be labeled is_bot=True"


def test_attach_labels_normal_ip_is_false(mini_flow_df):
    """All IPs in background-only data must be labeled is_bot=False."""
    features = build_window_features(mini_flow_df, "test")
    result = attach_labels(features, mini_flow_df)
    assert not result["is_bot"].any()


def test_attach_labels_no_nan(mini_botnet_flow_df):
    features = build_window_features(mini_botnet_flow_df, "test")
    result = attach_labels(features, mini_botnet_flow_df)
    assert result["is_bot"].isna().sum() == 0


def test_attach_labels_does_not_mutate_input(mini_botnet_flow_df):
    features = build_window_features(mini_botnet_flow_df, "test")
    original_cols = list(features.columns)
    attach_labels(features, mini_botnet_flow_df)
    assert list(features.columns) == original_cols


def test_attach_labels_called_after_features(mini_botnet_flow_df):
    """Verify full pipeline order: build → delta → labels."""
    features = build_window_features(mini_botnet_flow_df, "test")
    features = add_delta_features(features)
    features = attach_labels(features, mini_botnet_flow_df)
    assert "delta_degree" in features.columns
    assert "is_bot" in features.columns


# ── compute_window_bot_fraction ───────────────────────────────────────────────


def test_window_bot_fraction_in_leakage_columns():
    assert "window_bot_fraction" in LEAKAGE_COLUMNS


def test_compute_window_bot_fraction_adds_column(mini_botnet_flow_df):
    features = build_window_features(mini_botnet_flow_df, "test")
    features = attach_labels(features, mini_botnet_flow_df)
    result = compute_window_bot_fraction(features)
    assert "window_bot_fraction" in result.columns


def test_compute_window_bot_fraction_in_range(mini_botnet_flow_df):
    features = build_window_features(mini_botnet_flow_df, "test")
    features = attach_labels(features, mini_botnet_flow_df)
    result = compute_window_bot_fraction(features)
    assert (result["window_bot_fraction"] >= 0.0).all()
    assert (result["window_bot_fraction"] <= 1.0).all()


def test_compute_window_bot_fraction_requires_is_bot(mini_flow_df):
    """Calling without is_bot column must raise ValueError."""
    features = build_window_features(mini_flow_df, "test")
    with pytest.raises(ValueError, match="attach_labels"):
        compute_window_bot_fraction(features)
