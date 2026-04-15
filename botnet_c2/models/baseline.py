"""Rule-based threshold classifier baseline.

ThresholdClassifier is an sklearn-compatible estimator that detects C2 nodes
by thresholding on degree and k-core number. It fits by computing the 99th
percentile of normal-IP values for each feature, then predicts is_bot=True
if either threshold is exceeded.

This baseline is intentionally simple — it operationalizes the domain intuition
that C2 nodes have anomalously high degree and k-core relative to normal hosts.
Its value is as a competitive floor: if LightGBM cannot beat it meaningfully,
the added complexity is not justified.

The classifier implements predict_proba() so it can be plotted on the same
PR curve machinery as LightGBM. The probability score is the max of the
normalized degree and kcore values (scaled by their thresholds), capped at 1.

Public API:
    ThresholdClassifier  — sklearn-compatible estimator
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Rule-based C2 detector using degree and k-core thresholds.

    Fits by computing tau_deg and tau_core at the 99th percentile of
    normal-IP (is_bot=False) feature values. Predicts is_bot=True if
    degree > tau_deg OR kcore > tau_core.

    Implements predict_proba() for PR curve evaluation — the score is
    max(degree / tau_deg, kcore / tau_core), capped at 1.0, giving a
    continuous ranking consistent with the hard threshold rule.

    Parameters:
        percentile: Percentile of normal-IP distribution used as threshold.
            Default 99 (top 1% of normal hosts triggers the rule).

    Attributes (set after fit):
        tau_deg_:  degree threshold
        tau_core_: k-core threshold
        classes_:  [False, True]
        feature_names_in_: column names from training DataFrame
    """

    def __init__(self, percentile: float = 99.0) -> None:
        self.percentile = percentile

    def fit(self, X: np.ndarray, y: np.ndarray) -> ThresholdClassifier:
        """Fit thresholds from normal-IP rows.

        Args:
            X: Feature matrix. Must contain 'degree' and 'kcore' columns
               if passed as a DataFrame, or columns at indices matching
               the position of degree and kcore in get_feature_columns().
            y: Binary labels (True/1 = bot, False/0 = normal).

        Returns:
            self
        """

        y = np.asarray(y, dtype=bool)
        self.classes_ = np.array([False, True])

        # Work with both DataFrame and ndarray inputs
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
            X_normal = X[~y]
            degree_vals = X_normal["degree"].values
            kcore_vals = X_normal["kcore"].values
        else:
            # ndarray: use get_feature_columns() to find indices
            from botnet_c2.features.engineering import get_feature_columns

            cols = get_feature_columns()
            self.feature_names_in_ = np.array(cols)
            deg_idx = cols.index("degree")
            kcore_idx = cols.index("kcore")
            X_normal = X[~y]
            degree_vals = X_normal[:, deg_idx]
            kcore_vals = X_normal[:, kcore_idx]

        if len(degree_vals) == 0:
            # Edge case: all rows are bots — set thresholds to 0
            self.tau_deg_ = 0.0
            self.tau_core_ = 0.0
        else:
            self.tau_deg_ = float(np.percentile(degree_vals, self.percentile))
            self.tau_core_ = float(np.percentile(kcore_vals, self.percentile))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict is_bot labels.

        Returns True if degree > tau_deg OR kcore > tau_core.
        Equivalent to predict_proba score > 0.5 (squashed score of 1.0).
        """
        check_is_fitted(self, ["tau_deg_", "tau_core_"])
        scores = self._raw_scores(X)
        return scores >= 1.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for PR curve evaluation.

        Column 0: P(normal), Column 1: P(bot).
        Score = max(degree / tau_deg, kcore / tau_core), squashed through
        x/(1+x) to map [0, inf) -> [0, 1) while preserving strict monotonicity.

        This replaces the previous hard np.clip(scores, 0, 1) which collapsed
        all above-threshold nodes to score=1.0, destroying ranking resolution
        at the top of the distribution and artificially depressing PR-AUC.
        """
        check_is_fitted(self, ["tau_deg_", "tau_core_"])
        scores = self._raw_scores(X)
        # Monotonic squash: x/(1+x) maps 0->0, 1->0.5, inf->1
        # Nodes that exceed the threshold (score > 1) get scores > 0.5
        # Nodes below threshold get scores < 0.5
        # Relative ordering is fully preserved throughout
        p_bot = scores / (1.0 + scores)
        return np.column_stack([1.0 - p_bot, p_bot])

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous anomaly scores (may exceed 1.0)."""
        if hasattr(X, "columns"):
            degree_vals = X["degree"].values.astype(float)
            kcore_vals = X["kcore"].values.astype(float)
        else:
            from botnet_c2.features.engineering import get_feature_columns

            cols = get_feature_columns()
            deg_idx = cols.index("degree")
            kcore_idx = cols.index("kcore")
            degree_vals = X[:, deg_idx].astype(float)
            kcore_vals = X[:, kcore_idx].astype(float)

        # Avoid division by zero when threshold is 0
        deg_score = np.where(self.tau_deg_ > 0, degree_vals / self.tau_deg_, 0.0)
        core_score = np.where(self.tau_core_ > 0, kcore_vals / self.tau_core_, 0.0)
        return np.maximum(deg_score, core_score)