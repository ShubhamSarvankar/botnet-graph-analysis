"""Model evaluation: PR-AUC, operating points, and LOFO cross-validation.

Implements the evaluation criteria fixed in pre_coding_docs.md:
  - Primary metric: PR-AUC (not ROC-AUC — class imbalance is severe)
  - Success thresholds: >= 0.75 strong, 0.55-0.74 acceptable, < 0.55 null
  - LOFO: train on all other families, evaluate on held-out family
  - Reliable = False for families with < 100 node-window observations
    (rbot_51, rbot_52, sogou_48)

These thresholds and exclusion criteria are fixed and must not be adjusted
after seeing model results (anti-p-hacking rule from pre_coding_docs.md).

Public API:
    evaluate(model, X_test, y_test) -> EvalResult
    leave_one_family_out(features_by_scenario, labels_by_scenario) -> pd.DataFrame
    pr_auc_label(pr_auc) -> str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Minimum node-window observations for a scenario to be considered reliable.
# Fixed in pre_coding_docs.md — do not change after seeing results.
_RELIABLE_MIN_OBSERVATIONS = 100

# PR-AUC success thresholds — fixed, not adjusted after seeing results.
_THRESHOLD_STRONG = 0.75
_THRESHOLD_ACCEPTABLE = 0.55


@dataclass
class EvalResult:
    """Evaluation results for a single model on a single test set.

    Attributes:
        pr_auc: Area under the precision-recall curve.
        roc_auc: Area under the ROC curve (secondary metric).
        tpr_at_fpr: TPR at fixed FPR operating points (1%, 5%, 10%).
        precision_curve: Precision values along the PR curve.
        recall_curve: Recall values along the PR curve.
        thresholds: Decision thresholds along the PR curve.
        n_pos: Number of positive (bot) samples in test set.
        n_total: Total number of samples in test set.
        label: Interpretive label ("Strong", "Acceptable", or "Null result").
    """

    pr_auc: float
    roc_auc: float
    tpr_at_fpr: dict[str, float]
    precision_curve: list[float]
    recall_curve: list[float]
    thresholds: list[float]
    n_pos: int
    n_total: int
    label: str = field(init=False)

    def __post_init__(self) -> None:
        self.label = pr_auc_label(self.pr_auc)


def evaluate(model: object, X_test: pd.DataFrame, y_test: np.ndarray) -> EvalResult:
    """Evaluate a fitted model on a test set.

    Computes PR-AUC, ROC-AUC, and TPR at FPR = 1%, 5%, 10%.

    Args:
        model: Fitted sklearn-compatible model with predict_proba().
        X_test: Feature matrix.
        y_test: True binary labels.

    Returns:
        EvalResult dataclass with all evaluation metrics.
    """
    y_test = np.asarray(y_test, dtype=bool)
    y_scores = model.predict_proba(X_test)[:, 1]

    pr_auc = float(average_precision_score(y_test, y_scores))

    try:
        roc_auc = float(roc_auc_score(y_test, y_scores))
    except ValueError:
        roc_auc = float("nan")

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    tpr_at_fpr = _compute_tpr_at_fpr(y_test, y_scores, fpr_targets=[0.01, 0.05, 0.10])

    n_pos = int(y_test.sum())
    n_total = len(y_test)

    logger.info(
        "Evaluation: PR-AUC=%.3f (%s), ROC-AUC=%.3f, n_pos=%d/%d (%.1f%%)",
        pr_auc,
        pr_auc_label(pr_auc),
        roc_auc,
        n_pos,
        n_total,
        100 * n_pos / max(n_total, 1),
    )

    return EvalResult(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        tpr_at_fpr=tpr_at_fpr,
        precision_curve=precision.tolist(),
        recall_curve=recall.tolist(),
        thresholds=thresholds.tolist(),
        n_pos=n_pos,
        n_total=n_total,
    )


def leave_one_family_out(
    all_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Run leave-one-family-out (LOFO) cross-validation.

    For each family: trains on all other families, evaluates on the held-out
    family. Returns a summary table.

    The LOFO design (which families, which exclusion criterion) is fixed in
    pre_coding_docs.md and must not be changed after seeing results.

    Args:
        all_features: Concatenated feature DataFrame with columns including
            'family', 'is_bot', and all feature columns. Must NOT contain
            any LEAKAGE_COLUMNS.
        feature_cols: Ordered list of feature column names from
            get_feature_columns().

    Returns:
        DataFrame with columns:
            family, pr_auc, roc_auc, n_bot_ips, n_windows,
            n_observations, reliable, label
        One row per family.
    """
    from botnet_c2.models.trainer import _check_no_leakage, build_pipeline
    from botnet_c2.data.registry import SCENARIOS, SMALL_SCENARIOS

    _check_no_leakage(all_features)

    families = sorted(all_features["family"].unique())
    rows = []

    for held_out_family in families:
        train_mask = all_features["family"] != held_out_family
        test_mask = all_features["family"] == held_out_family

        X_train = all_features.loc[train_mask, feature_cols]
        y_train = all_features.loc[train_mask, "is_bot"].values
        X_test = all_features.loc[test_mask, feature_cols]
        y_test = all_features.loc[test_mask, "is_bot"].values

        n_observations = int(test_mask.sum())
        n_bot_ips = int(y_test.sum())
        n_windows = int(
            all_features.loc[test_mask, "time_bin"].nunique()
            if "time_bin" in all_features.columns
            else 0
        )

        # A family is reliable if it has enough observations AND is not
        # composed entirely of small scenarios (rbot_51, rbot_52, sogou_48).
        # Previously this only checked row count, which was always True since
        # families aggregate multiple scenarios into large row counts.
        held_out_scenario_ids = [
            sid for sid, meta in SCENARIOS.items()
            if meta["family"] == held_out_family
        ]
        all_small = all(sid in SMALL_SCENARIOS for sid in held_out_scenario_ids)
        reliable = not all_small and n_observations >= _RELIABLE_MIN_OBSERVATIONS

        if y_train.sum() == 0 or y_test.sum() == 0:
            logger.warning(
                "LOFO %s: no positive samples in train or test — skipping",
                held_out_family,
            )
            rows.append(
                {
                    "family": held_out_family,
                    "pr_auc": float("nan"),
                    "roc_auc": float("nan"),
                    "n_bot_ips": n_bot_ips,
                    "n_windows": n_windows,
                    "n_observations": n_observations,
                    "reliable": reliable,
                    "label": "Skipped (no positives)",
                }
            )
            continue

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            result = evaluate(pipeline, X_test, y_test)

        rows.append(
            {
                "family": held_out_family,
                "pr_auc": result.pr_auc,
                "roc_auc": result.roc_auc,
                "n_bot_ips": n_bot_ips,
                "n_windows": n_windows,
                "n_observations": n_observations,
                "reliable": reliable,
                "label": result.label,
            }
        )
        logger.info(
            "LOFO %s: PR-AUC=%.3f (%s), reliable=%s",
            held_out_family,
            result.pr_auc,
            result.label,
            reliable,
        )

    return pd.DataFrame(rows)


def pr_auc_label(pr_auc: float) -> str:
    """Return interpretive label for a PR-AUC value.

    Thresholds fixed in pre_coding_docs.md — do not adjust after seeing results.
    """
    if np.isnan(pr_auc):
        return "N/A"
    if pr_auc >= _THRESHOLD_STRONG:
        return "Strong"
    if pr_auc >= _THRESHOLD_ACCEPTABLE:
        return "Acceptable"
    return "Null result"


def _compute_tpr_at_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fpr_targets: list[float],
) -> dict[str, float]:
    """Compute TPR at fixed FPR operating points.

    Args:
        y_true: True binary labels.
        y_scores: Predicted scores/probabilities.
        fpr_targets: List of FPR values at which to compute TPR.

    Returns:
        Dict mapping "tpr_at_fprX" -> TPR value.
    """
    from sklearn.metrics import roc_curve

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)

    result = {}
    for target in fpr_targets:
        # Find the TPR at the closest FPR <= target
        idx = np.searchsorted(fpr_curve, target, side="right") - 1
        idx = max(0, min(idx, len(tpr_curve) - 1))
        key = f"tpr_at_fpr{int(target * 100):02d}pct"
        result[key] = float(tpr_curve[idx])

    return result