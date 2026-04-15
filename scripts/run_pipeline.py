"""c2detect -- CLI entry point for the botnet C2 detection pipeline.

Built with typer. Each pipeline stage is independently runnable.
Per-scenario failures are caught, logged, and do not abort the run.

Commands:
    c2detect download              # download all captures to data/
    c2detect topology              # compute structural metrics -> results/metrics/
    c2detect features              # build feature matrix -> results/features/
    c2detect train                 # train model -> results/models/
    c2detect evaluate              # run evaluation -> results/evaluation/
    c2detect run-all --skip-download
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="c2detect",
    help="Graph-topology-based botnet C2 node detection pipeline.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("c2detect")

ScenarioOption = Annotated[
    str | None,
    typer.Option("--scenario", "-s", help="Single scenario ID. Omit for all."),
]
DataDirOption = Annotated[Path, typer.Option("--data-dir", help="Raw capture files.")]
ResultsDirOption = Annotated[
    Path, typer.Option("--results-dir", help="Pipeline outputs.")
]


# -- download ------------------------------------------------------------------


@app.command()
def download(
    scenario: ScenarioOption = None,
    data_dir: DataDirOption = Path("data"),
) -> None:
    """Download CTU-13 capture files. Skips files already present on disk."""
    from botnet_c2.data.download import download_all, download_capture
    from botnet_c2.data.registry import SCENARIOS
    from botnet_c2.exceptions import DownloadError

    if scenario is not None:
        if scenario not in SCENARIOS:
            typer.echo(f"Unknown scenario '{scenario}'.", err=True)
            raise typer.Exit(1)
        try:
            path = download_capture(scenario, data_dir)
            typer.echo(f"OK {scenario} -> {path}")
        except DownloadError as exc:
            typer.echo(f"FAIL {scenario}: {exc}", err=True)
            raise typer.Exit(1) from exc
    else:
        results = download_all(data_dir)
        n_total = len(SCENARIOS)
        n_ok = len(results)
        typer.echo(f"\nDownloaded {n_ok}/{n_total} scenarios to {data_dir}/")
        if n_ok < n_total:
            raise typer.Exit(1)


# -- topology ------------------------------------------------------------------


@app.command()
def topology(
    scenario: ScenarioOption = None,
    data_dir: DataDirOption = Path("data"),
    results_dir: ResultsDirOption = Path("results"),
) -> None:
    """Compute scenario-level structural metrics -> results/metrics/."""
    from botnet_c2.data.loader import load_capture, split_flows
    from botnet_c2.data.registry import SCENARIOS
    from botnet_c2.graph.builder import build_flow_graph
    from botnet_c2.graph.topology import compute_topology

    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    errors_log = results_dir / "errors.log"
    scenario_ids = [scenario] if scenario else list(SCENARIOS.keys())
    n_ok = n_fail = 0

    for sid in scenario_ids:
        meta = SCENARIOS[sid]
        capture_path = data_dir / meta["local_filename"]
        if not capture_path.exists():
            msg = f"{sid}: file not found -- run 'c2detect download' first"
            typer.echo(f"FAIL {msg}", err=True)
            _log_error(errors_log, msg)
            n_fail += 1
            continue
        try:
            df = load_capture(capture_path)
            botnet_df, _ = split_flows(df)
            if botnet_df.empty:
                logger.warning("%s: no botnet flows found -- skipping", sid)
                n_fail += 1
                continue
            G = build_flow_graph(botnet_df)
            metrics = compute_topology(G, sid)
            out_path = metrics_dir / f"{sid}_topology.json"
            out_path.write_text(json.dumps(metrics, indent=2))
            typer.echo(f"OK {sid} -> {out_path.relative_to(results_dir.parent)}")
            n_ok += 1
        except Exception as exc:
            msg = f"{sid}: {type(exc).__name__}: {exc}"
            typer.echo(f"FAIL {msg}", err=True)
            _log_error(errors_log, msg)
            logger.exception("Error processing %s", sid)
            n_fail += 1

    typer.echo(f"\nTopology: {n_ok} succeeded, {n_fail} failed")
    if n_fail > 0:
        raise typer.Exit(1)


# -- features ------------------------------------------------------------------


@app.command()
def features(
    scenario: ScenarioOption = None,
    data_dir: DataDirOption = Path("data"),
    results_dir: ResultsDirOption = Path("results"),
    label_strategy: Annotated[
        str,
        typer.Option(
            "--label-strategy",
            "-l",
            help=(
                "Labeling strategy for is_bot. One of: "
                "both_src_dst (default, CTU-13 spec), "
                "src_only (infected hosts only), "
                "cc_dst_only (C2 servers only — purest), "
                "cc_src_and_dst (both CC endpoints)."
            ),
        ),
    ] = "both_src_dst",
) -> None:
    """Build per-(time_bin, ip) feature matrix -> results/features/."""
    from botnet_c2.data.loader import load_capture
    from botnet_c2.data.registry import SCENARIOS
    from botnet_c2.features.engineering import (
        LABEL_STRATEGIES,
        add_delta_features,
        attach_labels,
    )
    from botnet_c2.features.windows import build_window_features

    if label_strategy not in LABEL_STRATEGIES:
        typer.echo(
            f"Unknown label strategy '{label_strategy}'. "
            f"Valid: {', '.join(LABEL_STRATEGIES)}",
            err=True,
        )
        raise typer.Exit(1)

    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    errors_log = results_dir / "errors.log"
    scenario_ids = [scenario] if scenario else list(SCENARIOS.keys())
    n_ok = n_fail = 0

    for sid in scenario_ids:
        meta = SCENARIOS[sid]
        capture_path = data_dir / meta["local_filename"]
        if not capture_path.exists():
            msg = f"{sid}: file not found -- run 'c2detect download' first"
            typer.echo(f"FAIL {msg}", err=True)
            _log_error(errors_log, msg)
            n_fail += 1
            continue
        try:
            df = load_capture(capture_path)
            feat = build_window_features(df, scenario_id=sid)
            if feat.empty:
                logger.warning("%s: empty feature matrix -- skipping", sid)
                n_fail += 1
                continue
            feat = add_delta_features(feat)
            feat = attach_labels(feat, df, strategy=label_strategy)
            feat["family"] = meta["family"]
            feat["label_strategy"] = label_strategy
            out_path = features_dir / f"{sid}_features.parquet"
            feat.reset_index().to_parquet(out_path, index=False)
            n_bot = feat["is_bot"].sum()
            n_total = len(feat)
            typer.echo(
                f"OK {sid} -> {out_path.relative_to(results_dir.parent)} "
                f"({n_total} rows, {n_bot} bot [{100 * n_bot / n_total:.1f}%])"
            )
            n_ok += 1
        except Exception as exc:
            msg = f"{sid}: {type(exc).__name__}: {exc}"
            typer.echo(f"FAIL {msg}", err=True)
            _log_error(errors_log, msg)
            logger.exception("Error processing %s", sid)
            n_fail += 1

    typer.echo(f"\nFeatures: {n_ok} succeeded, {n_fail} failed")
    if n_fail > 0:
        raise typer.Exit(1)


# -- train ---------------------------------------------------------------------


@app.command()
def train(
    results_dir: ResultsDirOption = Path("results"),
    held_out: Annotated[
        str,
        typer.Option("--held-out", help="Scenario ID to hold out as test set."),
    ] = "neris_50",
) -> None:
    """Train LightGBM on all scenarios except held-out -> results/models/."""
    import pandas as pd

    from botnet_c2.data.registry import SCENARIOS
    from botnet_c2.features.engineering import get_feature_columns
    from botnet_c2.models.trainer import save_model
    from botnet_c2.models.trainer import train as train_model

    features_dir = results_dir / "features"
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_columns()

    # Load all scenarios except held-out
    dfs = []
    for sid in SCENARIOS:
        path = features_dir / f"{sid}_features.parquet"
        if not path.exists():
            logger.warning("%s: parquet not found -- skipping", sid)
            continue
        if sid == held_out:
            logger.info("Holding out %s for test evaluation", sid)
            continue
        df = pd.read_parquet(path)
        dfs.append(df)

    if not dfs:
        typer.echo("No feature files found. Run 'c2detect features' first.", err=True)
        raise typer.Exit(1)

    all_train = pd.concat(dfs, ignore_index=True)
    # Include scenario_id for GroupKFold CV — train() will use it for
    # fold assignment and drop it before fitting the model.
    X_train = all_train[feature_cols + ["scenario_id"]]
    y_train = all_train["is_bot"].values

    n_scenarios = all_train["scenario_id"].nunique()
    typer.echo(
        f"Training on {len(all_train):,} rows "
        f"({int(y_train.sum()):,} bot, {100 * y_train.mean():.1f}%) "
        f"across {n_scenarios} scenarios -- held out: {held_out}"
    )

    pipeline, cv_metrics = train_model(X_train, y_train)

    typer.echo(
        f"CV PR-AUC: {cv_metrics['cv_pr_auc_mean']:.3f} "
        f"+/- {cv_metrics['cv_pr_auc_std']:.3f}"
    )

    model_path = models_dir / "lgbm_model"
    save_model(pipeline, model_path)

    cv_path = models_dir / "cv_metrics.json"
    cv_path.write_text(json.dumps(cv_metrics, indent=2))

    typer.echo(f"OK Model saved to {models_dir}/")


# -- evaluate ------------------------------------------------------------------


@app.command()
def evaluate(
    results_dir: ResultsDirOption = Path("results"),
    held_out: Annotated[
        str,
        typer.Option("--held-out", help="Held-out test scenario ID."),
    ] = "neris_50",
) -> None:
    """Run full evaluation -> results/evaluation/."""
    import pandas as pd

    from botnet_c2.data.registry import SCENARIOS
    from botnet_c2.features.engineering import get_feature_columns
    from botnet_c2.models.baseline import ThresholdClassifier
    from botnet_c2.models.evaluation import evaluate as eval_model
    from botnet_c2.models.evaluation import leave_one_family_out
    from botnet_c2.models.trainer import load_model

    features_dir = results_dir / "features"
    models_dir = results_dir / "models"
    eval_dir = results_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_feature_columns()

    # Load held-out test set
    test_path = features_dir / f"{held_out}_features.parquet"
    if not test_path.exists():
        typer.echo(f"Held-out parquet not found: {test_path}", err=True)
        raise typer.Exit(1)
    test_df = pd.read_parquet(test_path)
    X_test = test_df[feature_cols]
    y_test = test_df["is_bot"].values

    # Load all training scenarios for LOFO
    all_dfs = []
    for sid in SCENARIOS:
        path = features_dir / f"{sid}_features.parquet"
        if path.exists():
            all_dfs.append(pd.read_parquet(path))
    all_features = pd.concat(all_dfs, ignore_index=True)

    # -- Baseline evaluation
    # Train baseline on the same data as LightGBM: all scenarios except
    # the held-out scenario (by scenario ID, not by family). Previously the
    # baseline excluded all scenarios of the held-out family, which
    # disadvantaged it vs LightGBM and made the comparison unfair.
    typer.echo("Evaluating ThresholdClassifier baseline...")
    train_mask = all_features["scenario_id"] != held_out
    X_train = all_features[train_mask][feature_cols]
    y_train = all_features[train_mask]["is_bot"].values
    baseline = ThresholdClassifier()
    baseline.fit(X_train, y_train)
    baseline_result = eval_model(baseline, X_test, y_test)
    typer.echo(
        f"  Baseline PR-AUC: {baseline_result.pr_auc:.3f} ({baseline_result.label})"
    )

    # -- LightGBM evaluation
    model_path = models_dir / "lgbm_model.pkl"
    if not model_path.exists():
        typer.echo("Model not found -- run 'c2detect train' first.", err=True)
        raise typer.Exit(1)
    lgbm = load_model(model_path)
    lgbm_result = eval_model(lgbm, X_test, y_test)
    typer.echo(f"  LightGBM PR-AUC: {lgbm_result.pr_auc:.3f} ({lgbm_result.label})")
    typer.echo(
        f"  TPR@FPR1%={lgbm_result.tpr_at_fpr['tpr_at_fpr01pct']:.3f}  "
        f"TPR@FPR5%={lgbm_result.tpr_at_fpr['tpr_at_fpr05pct']:.3f}  "
        f"TPR@FPR10%={lgbm_result.tpr_at_fpr['tpr_at_fpr10pct']:.3f}"
    )

    # -- LOFO
    typer.echo("Running leave-one-family-out evaluation...")
    lofo_df = leave_one_family_out(all_features, feature_cols)
    typer.echo(
        lofo_df[["family", "pr_auc", "reliable", "label"]].to_string(index=False)
    )

    # -- Save results
    eval_summary = {
        "held_out_scenario": held_out,
        "baseline": {
            "pr_auc": baseline_result.pr_auc,
            "roc_auc": baseline_result.roc_auc,
            "label": baseline_result.label,
            "tpr_at_fpr": baseline_result.tpr_at_fpr,
        },
        "lgbm": {
            "pr_auc": lgbm_result.pr_auc,
            "roc_auc": lgbm_result.roc_auc,
            "label": lgbm_result.label,
            "tpr_at_fpr": lgbm_result.tpr_at_fpr,
        },
    }
    (eval_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
    lofo_df.to_csv(eval_dir / "lofo_table.csv", index=False)

    # Save PR curve data for notebooks
    pr_curves = {
        "baseline": {
            "precision": baseline_result.precision_curve,
            "recall": baseline_result.recall_curve,
        },
        "lgbm": {
            "precision": lgbm_result.precision_curve,
            "recall": lgbm_result.recall_curve,
        },
    }
    (eval_dir / "pr_curves.json").write_text(json.dumps(pr_curves))

    typer.echo(f"\nOK Evaluation saved to {eval_dir}/")


# -- run-all -------------------------------------------------------------------


@app.command(name="run-all")
def run_all(
    skip_download: Annotated[
        bool,
        typer.Option("--skip-download", help="Skip download step."),
    ] = False,
    label_strategy: Annotated[
        str,
        typer.Option(
            "--label-strategy",
            "-l",
            help="Labeling strategy. See 'c2detect features --help'.",
        ),
    ] = "both_src_dst",
    data_dir: DataDirOption = Path("data"),
    results_dir: ResultsDirOption = Path("results"),
) -> None:
    """Run full pipeline: download -> topology -> features -> train -> evaluate."""
    step = 1
    total = 5 if not skip_download else 4

    if not skip_download:
        typer.echo(f"-- Step {step}/{total}: download --")
        download(scenario=None, data_dir=data_dir)
        step += 1

    typer.echo(f"-- Step {step}/{total}: topology --")
    topology(scenario=None, data_dir=data_dir, results_dir=results_dir)
    step += 1

    typer.echo(f"-- Step {step}/{total}: features [{label_strategy}] --")
    features(
        scenario=None,
        data_dir=data_dir,
        results_dir=results_dir,
        label_strategy=label_strategy,
    )
    step += 1

    typer.echo(f"-- Step {step}/{total}: train --")
    train(results_dir=results_dir)
    step += 1

    typer.echo(f"-- Step {step}/{total}: evaluate --")
    evaluate(results_dir=results_dir)

    typer.echo("\nPipeline complete.")


# -- helpers -------------------------------------------------------------------


def _log_error(errors_log: Path, message: str) -> None:
    import datetime

    errors_log.parent.mkdir(parents=True, exist_ok=True)
    with errors_log.open("a") as fh:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        fh.write(f"[{ts}] {message}\n")


if __name__ == "__main__":
    app()