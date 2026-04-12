"""Feature engineering: delta features, label attachment, and leakage guards.

This module is always called after build_window_features() — never before.
It is the only place in the codebase where the is_bot label is joined onto
the feature matrix.

**Label attachment order:**
    1. build_window_features(df, ...)      # full-traffic flows, no labels
    2. add_delta_features(features)        # temporal delta_degree
    3. attach_labels(features, df)         # is_bot joined LAST
    4. trainer.py receives the result      # enforces LEAKAGE_COLUMNS absent

**Leakage columns:**
    window_bot_fraction — fraction of window nodes that are bot IPs.
    Computed for analysis in notebook 03 but must never enter the model.
    Defined here, enforced in trainer.py.

Public API:
    add_delta_features(features) -> pd.DataFrame
    attach_labels(features, df) -> pd.DataFrame
    get_feature_columns() -> list[str]
    LEAKAGE_COLUMNS: list[str]
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must never be passed to the model.
# Defined here as the single source of truth; enforced in trainer.py.
LEAKAGE_COLUMNS: list[str] = ["window_bot_fraction"]


def get_feature_columns() -> list[str]:
    """Return the canonical ordered list of ML feature column names.

    This is the single source of truth for the feature set used by both
    trainer.py and the notebooks. Order is fixed for reproducibility.

    Returns:
        List of column names that form the model input matrix X.
    """
    return [
        "degree",
        "in_degree",
        "out_degree",
        "kcore",
        "pagerank",
        "betweenness",
        "flow_count",
        "delta_degree",
        "window_node_count",
    ]


def add_delta_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add delta_degree: the change in degree from the previous time bin.

    For each IP, delta_degree = degree(t) - degree(t-1). On first appearance
    (no previous bin), delta_degree = 0.

    The feature captures temporal dynamics: a C2 node that suddenly acquires
    many new connections in a window will have a large positive delta_degree.

    Args:
        features: DataFrame indexed by (time_bin, ip) as returned by
            build_window_features(). Must contain a 'degree' column.

    Returns:
        Copy of features with 'delta_degree' column added.
    """
    if features.empty:
        features = features.copy()
        features["delta_degree"] = pd.Series(dtype="int64")
        return features

    features = features.copy()

    # Sort index to ensure time order within each IP group
    features = features.sort_index()

    # Group by ip (second level of MultiIndex) and diff within each group
    # reset_index to work with groupby, then restore index
    flat = features.reset_index()
    flat = flat.sort_values(["ip", "time_bin"])
    flat["delta_degree"] = flat.groupby("ip")["degree"].diff().fillna(0).astype("int64")
    flat = flat.sort_values(["time_bin", "ip"])
    features = flat.set_index(["time_bin", "ip"])

    logger.debug("Added delta_degree feature (%d rows)", len(features))
    return features


def attach_labels(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach is_bot label to the feature matrix.

    An IP is labeled is_bot=True if it appears in ANY Botnet-labeled flow
    in the full scenario DataFrame. This is a scenario-level label: an IP
    is either a bot or not for the entire capture, regardless of which
    time windows it appears in.

    **Must be called after all feature computation** — this is the last step
    before the feature matrix is passed to trainer.py.

    Args:
        features: DataFrame indexed by (time_bin, ip) with feature columns.
        df: Full scenario flow DataFrame (from load_capture). Must contain
            'SrcAddr', 'DstAddr', and 'Label' columns.

    Returns:
        Copy of features with 'is_bot' bool column appended.
    """
    if features.empty:
        features = features.copy()
        features["is_bot"] = pd.Series(dtype="bool")
        return features

    # Build set of bot IPs from the full scenario (not filtered to window)
    is_botnet_flow = df["Label"].str.contains("Botnet", na=False)
    botnet_flows = df[is_botnet_flow]

    bot_ips: set[str] = set(botnet_flows["SrcAddr"].dropna()).union(
        set(botnet_flows["DstAddr"].dropna())
    )

    logger.info("attach_labels: %d unique bot IPs found in scenario", len(bot_ips))

    features = features.copy()
    # ip is the second level of the MultiIndex
    ip_index = features.index.get_level_values("ip")
    features["is_bot"] = ip_index.isin(bot_ips)

    n_bot_rows = features["is_bot"].sum()
    logger.info(
        "attach_labels: %d/%d rows labeled is_bot=True (%.1f%%)",
        n_bot_rows,
        len(features),
        100 * n_bot_rows / max(len(features), 1),
    )
    return features


def compute_window_bot_fraction(features: pd.DataFrame) -> pd.DataFrame:
    """Compute window_bot_fraction for EDA use only — never for model input.

    window_bot_fraction = fraction of IPs in each (scenario, time_bin) window
    that are bot IPs. This is in LEAKAGE_COLUMNS and must never be passed
    to trainer.py.

    Args:
        features: DataFrame indexed by (time_bin, ip) with is_bot column
            already attached by attach_labels().

    Returns:
        Copy of features with 'window_bot_fraction' column added.
    """
    if "is_bot" not in features.columns:
        raise ValueError(
            "attach_labels() must be called before compute_window_bot_fraction()"
        )

    features = features.copy()
    window_fraction = (
        features.groupby("time_bin")["is_bot"]
        .transform("mean")
        .rename("window_bot_fraction")
    )
    features["window_bot_fraction"] = window_fraction
    return features
