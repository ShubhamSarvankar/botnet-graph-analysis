"""Tests for botnet_c2.data.loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from botnet_c2.data.loader import REQUIRED_COLUMNS, load_capture, split_flows
from botnet_c2.exceptions import SchemaError

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.binetflow"
    p.write_text(textwrap.dedent(content).strip())
    return p


VALID_CSV = """\
    StartTime,SrcAddr,DstAddr,TotBytes,Label,Proto
    2011-08-10 10:00:00,192.168.1.1,10.0.0.1,1000,flow=Background,tcp
    2011-08-10 10:01:00,192.168.1.2,10.0.0.1,2000,flow=From-Botnet/CC  Botnet,tcp
    2011-08-10 10:02:00,192.168.1.3,192.168.1.4,500,flow=Background,udp
"""


# ── load_capture ──────────────────────────────────────────────────────────────


def test_load_capture_happy_path(tmp_path):
    path = _write_csv(tmp_path, VALID_CSV)
    df = load_capture(path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_load_capture_starttime_is_datetime(tmp_path):
    path = _write_csv(tmp_path, VALID_CSV)
    df = load_capture(path)
    assert pd.api.types.is_datetime64_any_dtype(df["StartTime"])


def test_load_capture_totbytes_is_float(tmp_path):
    path = _write_csv(tmp_path, VALID_CSV)
    df = load_capture(path)
    assert df["TotBytes"].dtype == "float64"


def test_load_capture_strips_column_whitespace(tmp_path):
    """Column names with leading/trailing spaces should be stripped."""
    csv = " StartTime, SrcAddr, DstAddr, TotBytes, Label\n2011-08-10,1.1.1.1,2.2.2.2,100,flow=Background\n"
    path = tmp_path / "test.binetflow"
    path.write_text(csv)
    df = load_capture(path)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_load_capture_missing_column_raises_schema_error(tmp_path):
    csv = "StartTime,SrcAddr,DstAddr,TotBytes\n2011-08-10,1.1.1.1,2.2.2.2,100\n"
    path = tmp_path / "test.binetflow"
    path.write_text(csv)
    with pytest.raises(SchemaError) as exc_info:
        load_capture(path)
    assert "Label" in exc_info.value.missing_columns


def test_load_capture_multiple_missing_columns_raises_schema_error(tmp_path):
    csv = "StartTime,SrcAddr\n2011-08-10,1.1.1.1\n"
    path = tmp_path / "test.binetflow"
    path.write_text(csv)
    with pytest.raises(SchemaError) as exc_info:
        load_capture(path)
    assert len(exc_info.value.missing_columns) >= 3


def test_load_capture_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_capture("/nonexistent/path/capture.binetflow")


def test_load_capture_totbytes_coerces_unparseable(tmp_path):
    """Non-numeric TotBytes should become NaN, not raise."""
    csv = "StartTime,SrcAddr,DstAddr,TotBytes,Label\n2011-08-10,1.1.1.1,2.2.2.2,INVALID,flow=Background\n"
    path = tmp_path / "test.binetflow"
    path.write_text(csv)
    df = load_capture(path)
    assert pd.isna(df["TotBytes"].iloc[0])


# ── split_flows ───────────────────────────────────────────────────────────────


def test_split_flows_returns_two_dataframes(mini_botnet_flow_df):
    botnet, background = split_flows(mini_botnet_flow_df)
    assert isinstance(botnet, pd.DataFrame)
    assert isinstance(background, pd.DataFrame)


def test_split_flows_partitions_all_rows(mini_botnet_flow_df):
    botnet, background = split_flows(mini_botnet_flow_df)
    assert len(botnet) + len(background) == len(mini_botnet_flow_df)


def test_split_flows_botnet_contains_botnet_label(mini_botnet_flow_df):
    botnet, _ = split_flows(mini_botnet_flow_df)
    assert botnet["Label"].str.contains("Botnet").all()


def test_split_flows_background_contains_no_botnet_label(mini_botnet_flow_df):
    _, background = split_flows(mini_botnet_flow_df)
    assert not background["Label"].str.contains("Botnet").any()


def test_split_flows_all_background(mini_flow_df):
    botnet, background = split_flows(mini_flow_df)
    assert len(botnet) == 0
    assert len(background) == len(mini_flow_df)


def test_split_flows_returns_copies(mini_botnet_flow_df):
    """Mutations to the split DataFrames should not affect the original."""
    botnet, _ = split_flows(mini_botnet_flow_df)
    original_len = len(mini_botnet_flow_df)
    botnet.drop(botnet.index, inplace=True)
    assert len(mini_botnet_flow_df) == original_len
