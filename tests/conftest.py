"""Shared pytest fixtures for botnet_c2 tests.

All fixtures use synthetic data only — no real CTU-13 capture files are
required to run the test suite. This keeps CI fast and dependency-free.

Fixtures:
    star_graph()        10-node star DiGraph (representative of C2 hub-and-spoke)
    mesh_graph()        10-node near-complete DiGraph (p2p topology)
    mini_flow_df()      50-row DataFrame with SrcAddr, DstAddr, TotBytes, Label, StartTime
    mini_botnet_flow_df()  same but includes Botnet-labeled rows
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

# ── Graph fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def star_graph() -> nx.DiGraph:
    """10-node directed star graph with hub '0.0.0.1'.

    Topology: all leaf nodes (0.0.0.2 .. 0.0.0.10) send flows to the hub.
    This mimics botnet C2: infected hosts communicate exclusively with the
    controller. Hub has in_degree=9, leaves have out_degree=1.
    """
    G = nx.DiGraph()
    hub = "0.0.0.1"
    leaves = [f"0.0.0.{i}" for i in range(2, 11)]
    for leaf in leaves:
        G.add_edge(leaf, hub, weight=1000.0, flows=5)
    return G


@pytest.fixture
def mesh_graph() -> nx.DiGraph:
    """10-node near-complete directed graph (p2p topology).

    Every node communicates with every other node. Clustering should be
    nonzero here (contrast with star_graph where clustering = 0).
    """
    G = nx.DiGraph()
    nodes = [f"10.0.0.{i}" for i in range(1, 11)]
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i != j:
                G.add_edge(src, dst, weight=500.0, flows=3)
    return G


# ── Flow DataFrame fixtures ───────────────────────────────────────────────────

_BASE_TIME = pd.Timestamp("2011-08-10 10:00:00", tz="UTC")
_FIVE_MIN = pd.Timedelta("5min")


@pytest.fixture
def mini_flow_df() -> pd.DataFrame:
    """50-row flow DataFrame with only background/normal traffic.

    Contains all required columns: SrcAddr, DstAddr, TotBytes, Label, StartTime.
    Flows span two 5-minute time bins to support windowing tests.
    """
    n = 50
    rows = []
    for i in range(n):
        bin_offset = pd.Timedelta(minutes=5 * (i // 25))  # 2 bins of 25 rows each
        rows.append(
            {
                "StartTime": _BASE_TIME + bin_offset + pd.Timedelta(seconds=i),
                "SrcAddr": f"192.168.1.{(i % 20) + 1}",
                "DstAddr": f"192.168.2.{(i % 10) + 1}",
                "TotBytes": float(1000 + i * 100),
                "Label": "flow=Background",
                "Proto": "tcp",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def mini_botnet_flow_df() -> pd.DataFrame:
    """50-row flow DataFrame with a mix of botnet and background flows.

    10 botnet flows all originate from infected hosts (192.168.1.x) to
    the C2 controller (10.0.0.1). The remaining 40 are background.
    Spans two 5-minute time bins.
    """
    rows = []
    c2_ip = "10.0.0.1"

    # 10 botnet flows in bin 0
    for i in range(10):
        rows.append(
            {
                "StartTime": _BASE_TIME + pd.Timedelta(seconds=i),
                "SrcAddr": f"192.168.1.{i + 1}",
                "DstAddr": c2_ip,
                "TotBytes": float(2000 + i * 50),
                "Label": "flow=From-Botnet/CC  Botnet",
                "Proto": "tcp",
            }
        )

    # 40 background flows across both bins
    for i in range(40):
        bin_offset = pd.Timedelta(minutes=5 * (i // 20))
        rows.append(
            {
                "StartTime": _BASE_TIME + bin_offset + pd.Timedelta(seconds=i + 10),
                "SrcAddr": f"192.168.2.{(i % 20) + 1}",
                "DstAddr": f"192.168.3.{(i % 10) + 1}",
                "TotBytes": float(500 + i * 30),
                "Label": "flow=Background",
                "Proto": "tcp",
            }
        )

    return pd.DataFrame(rows)
