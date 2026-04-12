"""CTU-13 .binetflow file loader and flow splitter.

Loads a single capture CSV into a validated, typed DataFrame. Schema validation
runs on every load — missing required columns raise SchemaError immediately
rather than producing a silent KeyError ten steps later.

Public API:
    load_capture(path) -> pd.DataFrame
    split_flows(df) -> tuple[pd.DataFrame, pd.DataFrame]

Column spec for CTU-13 .binetflow files:
    Required: SrcAddr, DstAddr, TotBytes, Label, StartTime
    Optional (used if present): Proto, Sport, Dport, Dir, TotPkts, Dur, State

The returned DataFrame has:
    - StartTime parsed to datetime64[ns, UTC] (or naive datetime64 if no tz info)
    - TotBytes coerced to float (some rows have '0' strings or missing values)
    - Label kept as object/string — callers use str.contains("Botnet")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from botnet_c2.exceptions import SchemaError

logger = logging.getLogger(__name__)

# Columns that must be present for any downstream processing to work.
REQUIRED_COLUMNS: list[str] = ["SrcAddr", "DstAddr", "TotBytes", "Label", "StartTime"]

# Dtypes applied after load. Only columns that are reliably present are listed.
# StartTime is handled separately (date parsing).
_DTYPE_MAP: dict[str, str] = {
    "TotBytes": "float64",
}


def load_capture(path: str | Path) -> pd.DataFrame:
    """Load a CTU-13 .binetflow CSV into a validated DataFrame.

    Args:
        path: Path to the .binetflow file.

    Returns:
        DataFrame with at minimum REQUIRED_COLUMNS present and correctly typed.
        StartTime is parsed to datetime64. TotBytes is float64. All other columns
        are kept as-is.

    Raises:
        FileNotFoundError: If path does not exist.
        SchemaError: If any column in REQUIRED_COLUMNS is absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Capture file not found: {path}")

    logger.info("Loading capture: %s", path)

    df = pd.read_csv(
        path,
        skipinitialspace=True,  # CTU-13 files have leading spaces in header
        low_memory=False,
    )

    # Strip whitespace from column names (some files have ' Label' etc.)
    df.columns = df.columns.str.strip()

    _validate_schema(path, df)
    df = _coerce_dtypes(df)

    logger.info(
        "Loaded %d rows, %d columns from %s",
        len(df),
        len(df.columns),
        path.name,
    )
    return df


def _validate_schema(path: Path, df: pd.DataFrame) -> None:
    """Raise SchemaError if any required column is missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(str(path), missing)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dtype coercions in-place and return the DataFrame."""
    # Parse StartTime to datetime. Use utc=True to handle mixed-tz strings
    # gracefully; falls back to naive if the column has no timezone info.
    try:
        df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)
    except Exception:
        df["StartTime"] = pd.to_datetime(df["StartTime"])

    # TotBytes: coerce to float, replacing unparseable values with NaN
    df["TotBytes"] = pd.to_numeric(df["TotBytes"], errors="coerce").astype("float64")

    return df


def split_flows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a capture DataFrame into botnet and background flows.

    Uses the Label column: rows containing "Botnet" (case-sensitive) are
    classified as botnet. Everything else — Background and Normal — is background.

    Args:
        df: DataFrame returned by load_capture().

    Returns:
        (botnet_df, background_df) — two non-overlapping DataFrames that together
        cover all rows of df.
    """
    is_botnet = df["Label"].str.contains("Botnet", na=False)
    botnet_df = df[is_botnet].copy()
    background_df = df[~is_botnet].copy()

    logger.info(
        "Split: %d botnet rows (%.1f%%), %d background rows",
        len(botnet_df),
        100 * len(botnet_df) / max(len(df), 1),
        len(background_df),
    )
    return botnet_df, background_df
