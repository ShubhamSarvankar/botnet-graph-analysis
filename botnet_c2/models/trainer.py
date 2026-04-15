"""LightGBM training pipeline.

Builds a StandardScaler -> LightGBMClassifier sklearn Pipeline, trains with
cross-validation, and serializes the fitted model alongside a
feature_names.json for schema validation at inference time.

trainer.py enforces the leakage guard: if any column from LEAKAGE_COLUMNS
is present in the input DataFrame, it raises ModelError immediately.

**Cross-validation strategy:**
Uses GroupKFold grouped by scenario_id rather than plain StratifiedKFold.
Row-level stratified CV inflates scores because the same IP appears in both
train and validation folds across multiple time windows. Scenario-grouped CV
holds out all windows of one scenario at a time, giving a realistic estimate
of within-distribution generalization. The number of folds equals min(cv,
n_unique_scenarios) to avoid empty folds when few scenarios are present.

Public API:
    build_pipeline() -> Pipeline
    train(X, y, groups, cv) -> tuple[Pipeline, dict]
    save_model(pipeline, path)
    load_model(path) -> Pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from botnet_c2.exceptions import ModelError
from botnet_c2.features.engineering import LEAKAGE_COLUMNS, get_feature_columns

logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Build a StandardScaler -> LightGBMClassifier pipeline.

    Uses class_weight='balanced' to handle the severe class imbalance
    in CTU-13 (bot rows are 0.1% - 4.7% of observations).

    Returns:
        Unfitted sklearn Pipeline.
    """
    from lightgbm import LGBMClassifier

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LGBMClassifier(
                    class_weight="balanced",
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=63,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )


def train(
    X: Any,
    y: Any,
    groups: Any = None,
    cv: int = 5,
) -> tuple[Pipeline, dict]:
    """Train the pipeline with scenario-grouped cross-validation.

    Enforces leakage guard before training: raises ModelError if any
    column from LEAKAGE_COLUMNS is present in X.

    Uses GroupKFold so that all rows from the same scenario are always in
    the same fold. This prevents optimistic leakage from the same IP
    appearing in both train and validation folds across time windows —
    a problem with plain StratifiedKFold on node-window data.

    Args:
        X: Feature DataFrame. Must not contain LEAKAGE_COLUMNS.
           If it contains a 'scenario_id' column, that column is used as
           groups for GroupKFold and then dropped before fitting.
        y: Binary labels (True/1 = bot).
        groups: Optional array of group labels (scenario IDs) for GroupKFold.
           If None and X has a 'scenario_id' column, that column is used.
           If neither is available, falls back to StratifiedKFold.
        cv: Maximum number of CV folds. Actual folds = min(cv,
           n_unique_groups). Default 5.

    Returns:
        (fitted_pipeline, cv_metrics) where cv_metrics contains:
            cv_pr_auc_mean, cv_pr_auc_std, cv_roc_auc_mean, cv_roc_auc_std,
            cv_folds, cv_strategy

    Raises:
        ModelError: If leakage columns are present in X.
    """
    import pandas as pd
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate

    _check_no_leakage(X)

    # Ensure X is a DataFrame so feature names survive CV fold creation
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    feature_cols = get_feature_columns()

    # Extract groups from scenario_id column if present and not passed
    if groups is None and "scenario_id" in X.columns:
        groups = X["scenario_id"].values

    # Drop non-feature columns before fitting — scenario_id must not enter
    # the model even if it was used for grouping
    cols_to_drop = [c for c in X.columns if c not in feature_cols]
    X_features = X.drop(columns=cols_to_drop) if cols_to_drop else X

    pipeline = build_pipeline()

    if groups is not None:
        unique_groups = np.unique(groups)
        n_folds = min(cv, len(unique_groups))
        cv_splitter = GroupKFold(n_splits=n_folds)
        cv_strategy = f"GroupKFold(n_splits={n_folds}, grouped_by=scenario_id)"
        logger.info(
            "Using GroupKFold CV: %d folds over %d scenarios",
            n_folds,
            len(unique_groups),
        )
        cv_results = cross_validate(
            pipeline,
            X_features,
            y,
            cv=cv_splitter,
            groups=groups,
            scoring=["average_precision", "roc_auc"],
            return_train_score=False,
        )
    else:
        # Fallback: no scenario grouping available
        logger.warning(
            "No scenario_id groups available — falling back to StratifiedKFold. "
            "CV scores may be optimistic due to within-scenario IP leakage."
        )
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_strategy = f"StratifiedKFold(n_splits={cv})"
        cv_results = cross_validate(
            pipeline,
            X_features,
            y,
            cv=cv_splitter,
            scoring=["average_precision", "roc_auc"],
            return_train_score=False,
        )

    cv_metrics = {
        "cv_pr_auc_mean": float(np.mean(cv_results["test_average_precision"])),
        "cv_pr_auc_std": float(np.std(cv_results["test_average_precision"])),
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "cv_folds": int(len(cv_results["test_average_precision"])),
        "cv_strategy": cv_strategy,
    }

    logger.info(
        "CV results (%s): PR-AUC=%.3f±%.3f, ROC-AUC=%.3f±%.3f",
        cv_strategy,
        cv_metrics["cv_pr_auc_mean"],
        cv_metrics["cv_pr_auc_std"],
        cv_metrics["cv_roc_auc_mean"],
        cv_metrics["cv_roc_auc_std"],
    )

    # Fit on full training set (feature columns only)
    pipeline.fit(X_features, y)
    logger.info("Pipeline fitted on full training set (%d rows)", len(y))

    return pipeline, cv_metrics


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    """Serialize the fitted pipeline to disk.

    Saves two files:
        <path>.pkl            -- the fitted pipeline (joblib)
        <path>_feature_names.json -- ordered feature names for schema validation

    Args:
        pipeline: Fitted sklearn Pipeline.
        path: Output path (without extension).
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_path = path.with_suffix(".pkl")
    joblib.dump(pipeline, model_path)
    logger.info("Model saved to %s", model_path)

    feature_names = get_feature_columns()
    names_path = path.parent / f"{path.stem}_feature_names.json"
    names_path.write_text(json.dumps(feature_names, indent=2))
    logger.info("Feature names saved to %s", names_path)


def load_model(path: str | Path) -> Pipeline:
    """Load a fitted pipeline from disk.

    Args:
        path: Path to the .pkl file.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ModelError: If the feature names file is missing.
    """
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    names_path = path.parent / f"{path.stem}_feature_names.json"
    if not names_path.exists():
        raise ModelError(
            f"Feature names file not found: {names_path}. "
            "Model may have been saved without schema metadata."
        )

    pipeline = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return pipeline


def _check_no_leakage(X: Any) -> None:
    """Raise ModelError if any leakage column is present in X."""
    if not hasattr(X, "columns"):
        return  # ndarray input -- cannot check by name

    present_leakage = [col for col in LEAKAGE_COLUMNS if col in X.columns]
    if present_leakage:
        raise ModelError(
            f"Leakage columns detected in training data: {present_leakage}. "
            f"Remove these columns before calling train(). "
            f"LEAKAGE_COLUMNS = {LEAKAGE_COLUMNS}"
        )
