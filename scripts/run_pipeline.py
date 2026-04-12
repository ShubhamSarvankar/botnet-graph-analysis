"""c2detect — CLI entry point for the botnet C2 detection pipeline.

Built with typer. Each pipeline stage is independently runnable.
Per-scenario failures are caught, logged, and do not abort the run.

Commands:
    c2detect download              # download all captures to data/
    c2detect download --scenario neris_42

    c2detect topology              # compute structural metrics -> results/metrics/
    c2detect topology --scenario neris_42

    c2detect features              # build feature matrix -> results/features/
    c2detect features --scenario neris_42

    c2detect run-all               # end-to-end
    c2detect run-all --skip-download

Usage:
    uv run c2detect --help
    uv run c2detect features
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

# -- Logging setup -------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("c2detect")


# -- Shared option types -------------------------------------------------------

ScenarioOption = Annotated[
    str | None,
    typer.Option(
        "--scenario",
        "-s",
        help="Single scenario ID to process (e.g. neris_42). Omit for all.",
    ),
]

DataDirOption = Annotated[
    Path,
    typer.Option("--data-dir", help="Directory for raw capture files."),
]

ResultsDirOption = Annotated[
    Path,
    typer.Option("--results-dir", help="Directory for pipeline outputs."),
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
            typer.echo(
                f"Unknown scenario '{scenario}'. Valid IDs: {sorted(SCENARIOS.keys())}",
                err=True,
            )
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
    """Compute scenario-level structural metrics -> results/metrics/<id>_topology.json."""
    from botnet_c2.data.loader import load_capture, split_flows
    from botnet_c2.data.registry import SCENARIOS
    from botnet_c2.graph.builder import build_flow_graph
    from botnet_c2.graph.topology import compute_topology

    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    errors_log = results_dir / "errors.log"

    scenario_ids = [scenario] if scenario else list(SCENARIOS.keys())
    n_ok = 0
    n_fail = 0

    for sid in scenario_ids:
        meta = SCENARIOS[sid]
        capture_path = data_dir / meta["local_filename"]

        if not capture_path.exists():
            msg = f"{sid}: capture file not found at {capture_path} -- run 'c2detect download' first"
            typer.echo(f"FAIL {msg}", err=True)
            _log_error(errors_log, msg)
            n_fail += 1
            continue

        try:
            df = load_capture(capture_path)
            botnet_df, _ = split_flows(df)

            if botnet_df.empty:
                logger.warning("%s: no botnet flows found -- skipping topology", sid)
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
            logger.exception("Unexpected error processing %s", sid)
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
) -> None:
    """Build per-(time_bin, ip) feature matrix -> results/features/<id>_features.parquet."""
    from botnet_c2.data.loader import load_capture
    from botnet_c2.data.registry import SCENARIOS
    from botnet_c2.features.engineering import add_delta_features, attach_labels
    from botnet_c2.features.windows import build_window_features

    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    errors_log = results_dir / "errors.log"

    scenario_ids = [scenario] if scenario else list(SCENARIOS.keys())
    n_ok = 0
    n_fail = 0

    for sid in scenario_ids:
        meta = SCENARIOS[sid]
        capture_path = data_dir / meta["local_filename"]

        if not capture_path.exists():
            msg = f"{sid}: capture file not found at {capture_path} -- run 'c2detect download' first"
            typer.echo(f"FAIL {msg}", err=True)
            _log_error(errors_log, msg)
            n_fail += 1
            continue

        try:
            # Load full-traffic data -- no label filtering before feature extraction
            df = load_capture(capture_path)

            # Build feature matrix from all flows
            feat = build_window_features(df, scenario_id=sid)

            if feat.empty:
                logger.warning("%s: empty feature matrix -- skipping", sid)
                n_fail += 1
                continue

            # Add temporal delta feature
            feat = add_delta_features(feat)

            # Attach labels LAST -- never before feature computation
            feat = attach_labels(feat, df)

            # Store family from registry for grouping in notebooks
            feat["family"] = meta["family"]

            out_path = features_dir / f"{sid}_features.parquet"
            feat.reset_index().to_parquet(out_path, index=False)

            n_bot = feat["is_bot"].sum()
            n_total_rows = len(feat)
            typer.echo(
                f"OK {sid} -> {out_path.relative_to(results_dir.parent)} "
                f"({n_total_rows} rows, {n_bot} bot [{100 * n_bot / n_total_rows:.1f}%])"
            )
            n_ok += 1

        except Exception as exc:
            msg = f"{sid}: {type(exc).__name__}: {exc}"
            typer.echo(f"FAIL {msg}", err=True)
            _log_error(errors_log, msg)
            logger.exception("Unexpected error processing %s", sid)
            n_fail += 1

    typer.echo(f"\nFeatures: {n_ok} succeeded, {n_fail} failed")
    if n_fail > 0:
        raise typer.Exit(1)


# -- run-all -------------------------------------------------------------------


@app.command(name="run-all")
def run_all(
    skip_download: Annotated[
        bool,
        typer.Option(
            "--skip-download",
            help="Skip the download step (assumes data/ is populated).",
        ),
    ] = False,
    data_dir: DataDirOption = Path("data"),
    results_dir: ResultsDirOption = Path("results"),
) -> None:
    """Run the full pipeline end-to-end (download -> topology -> features)."""
    step = 1
    total = 3 if not skip_download else 2

    if not skip_download:
        typer.echo(f"-- Step {step}/{total}: download --")
        download(scenario=None, data_dir=data_dir)
        step += 1

    typer.echo(f"-- Step {step}/{total}: topology --")
    topology(scenario=None, data_dir=data_dir, results_dir=results_dir)
    step += 1

    typer.echo(f"-- Step {step}/{total}: features --")
    features(scenario=None, data_dir=data_dir, results_dir=results_dir)

    typer.echo("\nPipeline complete.")


# -- helpers -------------------------------------------------------------------


def _log_error(errors_log: Path, message: str) -> None:
    """Append a timestamped error line to the errors log."""
    import datetime

    errors_log.parent.mkdir(parents=True, exist_ok=True)
    with errors_log.open("a") as fh:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        fh.write(f"[{ts}] {message}\n")


if __name__ == "__main__":
    app()
