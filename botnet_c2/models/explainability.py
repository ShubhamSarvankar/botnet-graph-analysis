"""SHAP-based feature importance for the LightGBM pipeline.

Thin wrapper around shap.TreeExplainer. Extracts the LightGBM classifier
from the sklearn Pipeline before computing SHAP values (TreeExplainer
requires the raw model, not the Pipeline wrapper).

Public API:
    compute_shap(pipeline, X) -> np.ndarray
    top_features(shap_values, feature_names, n) -> pd.DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def compute_shap(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for a fitted LightGBM pipeline.

    Extracts the LightGBM model from the pipeline, transforms X through
    the scaler step, then runs TreeExplainer on the transformed features.

    Args:
        pipeline: Fitted sklearn Pipeline with 'scaler' and 'clf' steps.
        X: Feature DataFrame (unscaled). Same columns used during training.

    Returns:
        SHAP values array of shape (n_samples, n_features).
        For binary classification, returns values for the positive class.
    """
    import shap

    # Extract scaler and model from pipeline
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["clf"]

    # Transform features through scaler (TreeExplainer needs consistent input)
    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # For binary classification, LightGBM returns a list [neg_class, pos_class]
    # or a single 2D array depending on version. Normalize to pos class only.
    if isinstance(shap_values, list):
        # Older SHAP: list of arrays, index 1 = positive class
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        # Newer SHAP: shape (n_samples, n_features, n_classes), take pos class
        shap_values = shap_values[:, :, 1]

    logger.info("SHAP values computed: shape %s", shap_values.shape)
    return shap_values


def top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 10,
) -> pd.DataFrame:
    """Return top N features by mean absolute SHAP value.

    Args:
        shap_values: SHAP values array of shape (n_samples, n_features).
        feature_names: Ordered list of feature names matching shap_values columns.
        n: Number of top features to return.

    Returns:
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)

    return df.head(n).reset_index(drop=True)
