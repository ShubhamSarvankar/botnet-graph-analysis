"""Idempotent capture file downloader.

Downloads CTU-13 .binetflow files from the public dataset server.
Each download is streamed in chunks and saved to a local data directory.
If the file already exists on disk, the download is skipped.

Public API:
    download_capture(scenario_id, data_dir) -> Path
    download_all(data_dir) -> dict[str, Path]
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from tqdm import tqdm

from botnet_c2.data.registry import SCENARIOS
from botnet_c2.exceptions import DownloadError

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8192  # bytes per stream chunk


def download_capture(
    scenario_id: str,
    data_dir: str | Path = "data",
) -> Path:
    """Download a single CTU-13 capture file, skipping if already present.

    Args:
        scenario_id: Key into SCENARIOS registry (e.g. "neris_42").
        data_dir: Local directory to save the file. Created if absent.

    Returns:
        Path to the local file (whether freshly downloaded or already present).

    Raises:
        KeyError: If scenario_id is not in the SCENARIOS registry.
        DownloadError: If the HTTP request fails or the response is not 200.
    """
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. Valid IDs: {sorted(SCENARIOS.keys())}"
        )

    meta = SCENARIOS[scenario_id]
    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / meta["local_filename"]

    if dest_path.exists():
        logger.info("Already present, skipping download: %s", dest_path)
        return dest_path

    url = meta["url"]
    logger.info("Downloading %s → %s", url, dest_path)

    try:
        response = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as exc:
        raise DownloadError(scenario_id, url, str(exc)) from exc

    if response.status_code != 200:
        raise DownloadError(
            scenario_id,
            url,
            f"HTTP {response.status_code}",
        )

    total_bytes = int(response.headers.get("content-length", 0)) or None
    tmp_path = dest_path.with_suffix(".tmp")

    try:
        with (
            tmp_path.open("wb") as fh,
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=scenario_id,
                leave=False,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                fh.write(chunk)
                progress.update(len(chunk))
        tmp_path.rename(dest_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise DownloadError(scenario_id, url, f"Write failed: {exc}") from exc

    logger.info("Saved %s (%.1f MB)", dest_path, dest_path.stat().st_size / 1e6)
    return dest_path


def download_all(
    data_dir: str | Path = "data",
) -> dict[str, Path]:
    """Download all CTU-13 capture files.

    Skips files that are already present. Per-scenario failures are logged
    and collected but do not abort the run — the returned dict omits failed
    scenarios and the caller can check the errors dict separately.

    Args:
        data_dir: Local directory to save files.

    Returns:
        Mapping of scenario_id → local Path for each successfully downloaded file.
    """
    results: dict[str, Path] = {}
    errors: dict[str, str] = {}

    for scenario_id in SCENARIOS:
        try:
            path = download_capture(scenario_id, data_dir)
            results[scenario_id] = path
        except DownloadError as exc:
            logger.error("Download failed for %s: %s", scenario_id, exc)
            errors[scenario_id] = str(exc)

    if errors:
        logger.warning(
            "%d scenario(s) failed to download: %s",
            len(errors),
            list(errors.keys()),
        )

    return results
