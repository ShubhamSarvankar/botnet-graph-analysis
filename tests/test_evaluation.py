"""Tests for botnet_c2.models.evaluation and botnet_c2.models.baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from botnet_c2.models.baseline import ThresholdClassifier
from botnet_c2.models.evaluation import EvalResult, evaluate, leave_one_family_out, pr_auc_label


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_features():
    """Small synthetic feature DataFrame with two families."""
    np.random.seed(42)
    n = 200
    rows = []
    for i in range(n):
        family = "Neris" if i < 100 else "Rbot"
        is_bot = i % 10 == 0  # 10% positive rate
        window_node_count = 50
        in_deg = 40 + (180 if is_bot else 0) + np.random.randint(0, 10)
        flow_cnt = 100 + (500 if is_bot else 0)
        rows.append(
            {
                "family": family,
                "time_bin": pd.Timestamp("2011-08-10") + pd.Timedelta(minutes=5 * (i % 20)),
                "degree": 50 + (200 if is_bot else 0) + np.random.randint(0, 10),
                "in_degree": in_deg,
                "in_degree_norm": float(in_deg) / window_node_count,
                "out_degree": 10 + np.random.randint(0, 5),
                "kcore": 3 + (10 if is_bot else 0) + np.random.randint(0, 2),
                "pagerank": 0.01 + (0.1 if is_bot else 0),
                "betweenness": 0.01 + (0.05 if is_bot else 0),
                "flow_count": flow_cnt,
                "flow_count_norm": float(flow_cnt) / window_node_count,
                "delta_degree": np.random.randint(-5, 5),
                "is_bot": is_bot,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def feature_cols():
    from botnet_c2.features.engineering import get_feature_columns
    return get_feature_columns()


# ── ThresholdClassifier ───────────────────────────────────────────────────────


def test_threshold_classifier_fit_returns_self(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    result = clf.fit(X, y)
    assert result is clf


def test_threshold_classifier_sets_thresholds(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    assert hasattr(clf, "tau_deg_")
    assert hasattr(clf, "tau_core_")
    assert clf.tau_deg_ > 0
    assert clf.tau_core_ > 0


def test_threshold_classifier_thresholds_from_normal_only(simple_features, feature_cols):
    """Thresholds must be based on normal-IP values only."""
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier(percentile=99)
    clf.fit(X, y)
    # Normal degree values are ~50-60; bot degree values are ~250+
    # So the 99th percentile of normals should be well below bot values
    assert clf.tau_deg_ < 200


def test_threshold_classifier_predict_returns_bool_array(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.dtype == bool or preds.dtype == np.bool_


def test_threshold_classifier_predict_proba_shape(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_threshold_classifier_predict_proba_sums_to_one(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_threshold_classifier_bot_scores_higher(simple_features, feature_cols):
    """Bot IPs should receive higher scores than normal IPs on average."""
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    bot_scores = proba[y, 1]
    normal_scores = proba[~y, 1]
    assert bot_scores.mean() > normal_scores.mean()


def test_threshold_classifier_not_fitted_raises(simple_features, feature_cols):
    from sklearn.exceptions import NotFittedError
    X = simple_features[feature_cols]
    clf = ThresholdClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(X)


# ── evaluate ──────────────────────────────────────────────────────────────────


def test_evaluate_returns_eval_result(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    result = evaluate(clf, X, y)
    assert isinstance(result, EvalResult)


def test_evaluate_pr_auc_in_range(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    result = evaluate(clf, X, y)
    assert 0.0 <= result.pr_auc <= 1.0


def test_evaluate_roc_auc_in_range(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    result = evaluate(clf, X, y)
    assert 0.0 <= result.roc_auc <= 1.0


def test_evaluate_tpr_at_fpr_keys(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    result = evaluate(clf, X, y)
    assert "tpr_at_fpr01pct" in result.tpr_at_fpr
    assert "tpr_at_fpr05pct" in result.tpr_at_fpr
    assert "tpr_at_fpr10pct" in result.tpr_at_fpr


def test_evaluate_n_pos_correct(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    result = evaluate(clf, X, y)
    assert result.n_pos == int(y.sum())
    assert result.n_total == len(y)


def test_evaluate_label_assigned(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    result = evaluate(clf, X, y)
    assert result.label in {"Strong", "Acceptable", "Null result"}


def test_evaluate_curve_lengths_consistent(simple_features, feature_cols):
    X = simple_features[feature_cols]
    y = simple_features["is_bot"].values
    clf = ThresholdClassifier()
    clf.fit(X, y)
    result = evaluate(clf, X, y)
    # precision and recall have one more point than thresholds (sklearn convention)
    assert len(result.precision_curve) == len(result.recall_curve)
    assert len(result.thresholds) == len(result.precision_curve) - 1


# ── pr_auc_label ──────────────────────────────────────────────────────────────


def test_pr_auc_label_strong():
    assert pr_auc_label(0.80) == "Strong"
    assert pr_auc_label(0.75) == "Strong"


def test_pr_auc_label_acceptable():
    assert pr_auc_label(0.65) == "Acceptable"
    assert pr_auc_label(0.55) == "Acceptable"


def test_pr_auc_label_null():
    assert pr_auc_label(0.50) == "Null result"
    assert pr_auc_label(0.10) == "Null result"


def test_pr_auc_label_nan():
    assert pr_auc_label(float("nan")) == "N/A"


# ── leave_one_family_out ──────────────────────────────────────────────────────


def test_lofo_returns_dataframe(simple_features, feature_cols):
    result = leave_one_family_out(simple_features, feature_cols)
    assert isinstance(result, pd.DataFrame)


def test_lofo_one_row_per_family(simple_features, feature_cols):
    result = leave_one_family_out(simple_features, feature_cols)
    n_families = simple_features["family"].nunique()
    assert len(result) == n_families


def test_lofo_required_columns(simple_features, feature_cols):
    result = leave_one_family_out(simple_features, feature_cols)
    required = {"family", "pr_auc", "roc_auc", "n_bot_ips", "n_observations", "reliable", "label"}
    assert required.issubset(set(result.columns))


def test_lofo_reliable_flag(simple_features, feature_cols):
    """Families with < 100 observations must be flagged unreliable."""
    # Make Neris tiny
    tiny = simple_features[simple_features["family"] == "Neris"].head(50).copy()
    rbot = simple_features[simple_features["family"] == "Rbot"].copy()
    combined = pd.concat([tiny, rbot], ignore_index=True)
    result = leave_one_family_out(combined, feature_cols)
    neris_row = result[result["family"] == "Neris"].iloc[0]
    assert neris_row["reliable"] is False or neris_row["reliable"] == False


def test_lofo_no_leakage_column_raises(simple_features, feature_cols):
    from botnet_c2.exceptions import ModelError
    df_with_leakage = simple_features.copy()
    df_with_leakage["window_bot_fraction"] = 0.1
    with pytest.raises(ModelError):
        leave_one_family_out(df_with_leakage, feature_cols)