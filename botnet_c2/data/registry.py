"""Scenario registry for CTU-13 botnet captures.

Single source of truth for all scenario metadata. No imports from the rest
of the package — this module must remain dependency-free so it can be imported
anywhere without side effects.

Dataset: CTU-13 (Czech Technical University, 2011)
Source: https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/
13 captures spanning 7 botnet families. All captures include bidirectional
flow labels with ground-truth Botnet / Background / Normal annotations.

Scenario notes:
  rbot_51  (44 nodes)  ── too small for reliable power-law fit or LOFO eval
  rbot_52  (12 nodes)  ── too small for reliable power-law fit or LOFO eval
  sogou_48 (18 nodes)  ── too small for reliable power-law fit or LOFO eval
  neris_50             ── largest scenario; used as held-out test set in notebook 04
"""

from __future__ import annotations

from typing import TypedDict


class ScenarioMeta(TypedDict):
    url: str
    local_filename: str
    family: str
    has_timestamps: bool


SCENARIOS: dict[str, ScenarioMeta] = {
    # ── Neris ────────────────────────────────────────────────────────────────
    "neris_42": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow",
        "local_filename": "capture20110810.binetflow",
        "family": "Neris",
        "has_timestamps": True,
    },
    "neris_43": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/detailed-bidirectional-flow-labels/capture20110811.binetflow",
        "local_filename": "capture20110811.binetflow",
        "family": "Neris",
        "has_timestamps": True,
    },
    "neris_50": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-50/detailed-bidirectional-flow-labels/capture20110817.binetflow",
        "local_filename": "capture20110817.binetflow",
        "family": "Neris",
        "has_timestamps": True,
    },
    # ── Rbot ─────────────────────────────────────────────────────────────────
    "rbot_44": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-44/detailed-bidirectional-flow-labels/capture20110812.binetflow",
        "local_filename": "capture20110812.binetflow",
        "family": "Rbot",
        "has_timestamps": True,
    },
    "rbot_45": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-45/detailed-bidirectional-flow-labels/capture20110815.binetflow",
        "local_filename": "capture20110815.binetflow",
        "family": "Rbot",
        "has_timestamps": True,
    },
    "rbot_51": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/detailed-bidirectional-flow-labels/capture20110818.binetflow",
        "local_filename": "capture20110818.binetflow",
        "family": "Rbot",
        "has_timestamps": True,
    },
    "rbot_52": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow",
        "local_filename": "capture20110818-2.binetflow",
        "family": "Rbot",
        "has_timestamps": True,
    },
    # ── Virut ─────────────────────────────────────────────────────────────────
    "virut_46": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46/detailed-bidirectional-flow-labels/capture20110815-2.binetflow",
        "local_filename": "capture20110815-2.binetflow",
        "family": "Virut",
        "has_timestamps": True,
    },
    "virut_54": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-54/detailed-bidirectional-flow-labels/capture20110815-3.binetflow",
        "local_filename": "capture20110815-3.binetflow",
        "family": "Virut",
        "has_timestamps": True,
    },
    # ── Single-scenario families ───────────────────────────────────────────────
    "donbot_47": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/detailed-bidirectional-flow-labels/capture20110816.binetflow",
        "local_filename": "capture20110816.binetflow",
        "family": "Donbot",
        "has_timestamps": True,
    },
    "murlo_49": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49/detailed-bidirectional-flow-labels/capture20110816-3.binetflow",
        "local_filename": "capture20110816-3.binetflow",
        "family": "Murlo",
        "has_timestamps": True,
    },
    "sogou_48": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-48/detailed-bidirectional-flow-labels/capture20110816-2.binetflow",
        "local_filename": "capture20110816-2.binetflow",
        "family": "Sogou",
        "has_timestamps": True,
    },
    "nsis_53": {
        "url": "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-53/detailed-bidirectional-flow-labels/capture20110819.binetflow",
        "local_filename": "capture20110819.binetflow",
        "family": "Nsis",
        "has_timestamps": True,
    },
}

# Scenarios too small for reliable power-law fit and excluded from LOFO ranking.
# Threshold: < 100 node-window observations in the feature matrix.
SMALL_SCENARIOS: frozenset[str] = frozenset({"rbot_51", "rbot_52", "sogou_48"})

# Held-out test scenario for notebook 04 single-split evaluation.
# Largest scenario by node count — best test of scale generalization.
HELD_OUT_SCENARIO: str = "neris_50"

# Ordered list of all families for consistent grouping in plots and tables.
FAMILIES: list[str] = sorted({meta["family"] for meta in SCENARIOS.values()})
