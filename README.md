# Botnet C2 Detection via Graph Topology

[![CI](https://github.com/ShubhamSarvankar/botnet-graph-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/ShubhamSarvankar/botnet-graph-analysis/actions/workflows/ci.yml)

Can graph topology alone — no ports, no payload, no protocol — identify command-and-control nodes in botnet network traffic, and does that signal generalize to botnet families the model has never seen?

This project builds a rigorous, reproducible pipeline to test that question against the [CTU-13 dataset](https://www.stratosphereips.org/datasets-ctu13): 13 captures spanning 7 botnet families, evaluated under leave-one-family-out cross-validation with pre-registered success thresholds.

---

## Results

The project finds that **topology signal is real but conditional**. Whether it crosses the threshold for reliable detection depends critically on what "positive" means.

The project tested four labeling strategies — four different definitions of what a C2-adjacent node is:

| Labeling strategy | CV PR-AUC | Held-out PR-AUC | LOFO avg | Best LOFO |
|---|---|---|---|---|
| `both_src_dst` — any endpoint of any botnet flow | 0.143 | 0.424 | 0.113 | Virut 0.148 |
| `src_only` — infected hosts only | 0.216 | 0.466 | 0.143 | **Donbot 0.872 ★** |
| `cc_src_and_dst` — both endpoints of CC flows | 0.098 | 0.222 | 0.107 | Donbot 0.468 |
| `cc_dst_only` — C2 server IPs only | 0.001 | 0.001 | 0.006 | Donbot 0.031 |

Pre-registered thresholds: **≥ 0.75 = Strong**, 0.55–0.74 = Acceptable, **< 0.55 = Null result**.

The **Donbot 0.872 under `src_only`** is the headline finding. When the positive class is clean — one infected host, clear hub-and-spoke topology — a LightGBM classifier trained entirely on graph features from other botnet families detects the infected host with Strong PR-AUC on an unseen family. The topology signal generalizes.

The remaining families are Null under every strategy. Cross-family generalization is harder when botnet behavior is structurally atypical (multi-bot scenarios, port-scanning, P2P elements) or when the positive class is noisy (thousands of innocent destination servers labeled positive alongside actual C2 nodes).

The `cc_dst_only` collapse to PR-AUC 0.001 is itself a finding: the C2 server IP, while the highest in-degree node in the *botnet subgraph*, is indistinguishable from ordinary high-degree external servers (DNS resolvers, CDN nodes, mail relays) in the full-traffic graph. Topology-only detection of the C2 server specifically requires protocol-level or behavioral features beyond what graph structure alone provides.

---

## What This Demonstrates

**The topology signal exists.** ROC-AUC of 0.904–0.948 across labeling strategies confirms the model consistently ranks bot-adjacent nodes above random. SHAP analysis identifies `in_degree_norm` and `flow_count_norm` as the dominant features — theoretically grounded in hub-and-spoke C2 communication patterns.

**The evaluation methodology is stricter than prior work.** Papers like BotSward (2023, 99% accuracy) and Chowdhury et al. (2017) evaluate per-scenario — the model sees training data from the same botnet family it's tested on. Our leave-one-family-out protocol trains on all other families and tests on the held-out family, which is the correct test for generalization. Under this stricter evaluation, performance drops substantially — the signal is real but narrow.

**Label construction is the dominant source of variance.** The `both_src_dst` strategy (which follows the CTU-13 authors' own specification) labels ~14,000 innocent destination servers per scenario as positive alongside actual C2 nodes. This dilution explains why the original result appeared null: 76–95% of positive rows have degree ≤ 1 and are topologically indistinguishable from background traffic. The `src_only` strategy, which labels only the initiating infected hosts, produces the cleanest signal and the only Strong result.

---

## Dataset

[CTU-13](https://www.stratosphereips.org/datasets-ctu13) — Czech Technical University, 2011. 13 botnet captures across 7 families: Neris, Rbot, Virut, Donbot, Murlo, Sogou, Nsis. Bidirectional NetFlow `.binetflow` files with ground-truth Botnet / Background / Normal labels.

```
Scenario       Family   Duration   #Flows      #Bots
neris_42       Neris    6.15 hrs   2,824,542   1
neris_43       Neris    3.85 hrs   1,808,057   1
neris_50 *     Neris    5.63 hrs   2,087,433   10
rbot_44        Rbot     66.85 hrs  4,709,774   1
rbot_45        Rbot     4.49 hrs   1,121,021   1
rbot_51        Rbot     5.13 hrs   1,309,732   10
rbot_52        Rbot     0.27 hrs   107,233     3
virut_46       Virut    0.50 hrs   129,818     1
virut_54       Virut    16.36 hrs  1,924,940   1
donbot_47      Donbot   2.15 hrs   558,888     1
murlo_49       Murlo    19.49 hrs  2,953,984   1
sogou_48 †     Sogou    0.35 hrs   114,067     1
nsis_53        Nsis     1.72 hrs   325,452     3

* held-out test set for single-split evaluation
† unreliable for LOFO — only scenario in family, in SMALL_SCENARIOS
```

---

## Pipeline

```
c2detect download              # download all 13 captures to data/
c2detect topology              # scenario-level structural metrics → results/metrics/
c2detect features              # per-(time_bin, ip) feature matrix → results/features/
c2detect features --label-strategy src_only   # switch labeling strategy
c2detect train                 # LightGBM with scenario-grouped CV → results/models/
c2detect evaluate              # full evaluation → results/evaluation/
c2detect run-all --skip-download              # end-to-end
```

Four labeling strategies available via `--label-strategy`: `both_src_dst` (default), `src_only`, `cc_dst_only`, `cc_src_and_dst`.

---

## Features

Per-(time_bin, ip) graph features extracted from 5-minute windows of full-traffic flows:

| Feature | Description | Graph |
|---|---|---|
| `degree` | Total degree | Undirected |
| `in_degree` | In-degree | Directed |
| `in_degree_norm` | in_degree / window_node_count | Directed |
| `out_degree` | Out-degree | Directed |
| `kcore` | k-core number | Undirected |
| `pagerank` | PageRank score | Directed |
| `betweenness` | Betweenness centrality (approximated for windows > 200 nodes) | Undirected |
| `flow_count` | Total incident flows (in + out) | Directed |
| `flow_count_norm` | flow_count / window_node_count | Directed |
| `delta_degree` | degree(t) − degree(t−1), 0 on first appearance | Undirected |

`local_clustering` excluded: confirmed 0.0 across all 13 scenarios. Hub-and-spoke C2 topology produces no triangles by construction — a domain finding documented in the EDA notebook.

`window_node_count` excluded from model input after SHAP analysis revealed it acted as a scenario memorization artifact (3rd in SHAP importance, 1.07× bot/normal ratio). Retained in parquet for analysis.

---

## Evaluation

**Primary metric:** PR-AUC. ROC-AUC is reported as secondary. Accuracy is not reported — class imbalance is severe (0.003–4.7% positive rate depending on strategy and scenario), making accuracy misleading.

**Cross-validation:** GroupKFold grouped by `scenario_id` (12 training scenarios, 5 folds). Plain StratifiedKFold was found to inflate CV scores 3× by allowing the same IP to appear in both train and validation folds across time windows.

**Held-out test:** `neris_50` (largest scenario, 10 infected hosts). Excluded from all training.

**LOFO:** For each family: train on all other families, test on held-out family. One row per family.

**Pre-registered thresholds** (fixed before any training, not adjusted after seeing results):

| Label | PR-AUC | Interpretation |
|---|---|---|
| Strong | ≥ 0.75 | Topology reliably identifies C2 nodes |
| Acceptable | 0.55–0.74 | Meaningful signal, topology partially useful |
| Null result | < 0.55 | Topology alone insufficient at this confidence level |

---

## Reproducing

```bash
# Install
uv sync

# Download data (requires ~800MB)
uv run c2detect download

# Run full pipeline with default labeling
uv run c2detect run-all --skip-download

# Run src_only experiment
uv run c2detect features --label-strategy src_only
uv run c2detect train
uv run c2detect evaluate

# Tests
uv run pytest
```

Requires Python 3.12. All dependencies in `pyproject.toml`. Data files are gitignored; download fresh with `c2detect download`.

---

## Notebooks

| Notebook | Contents |
|---|---|
| [01_data_and_graphs](notebooks/01_data_and_graphs.ipynb) | Load neris_42, split flows, build graph, visualize giant component |
| [02_eda](notebooks/02_eda.ipynb) | Structural metrics across all 13 scenarios, degree distributions, robustness curves, clustering=0 finding |
| [03_features](notebooks/03_features.ipynb) | Feature distributions, class imbalance, delta_degree time series, feature set justification |
| [04_detection_ml](notebooks/04_detection_ml.ipynb) | Baseline, logistic regression, LightGBM, SHAP, operating points, LOFO |
| [05_generalization](notebooks/05_generalization.ipynb) | Structural similarity matrix, PCA of scenario vectors, feature stability ranking |

---

## Stack

| Tool | Purpose |
|---|---|
| uv | Package management |
| NetworkX | Graph construction and topology metrics |
| LightGBM | Primary classifier |
| scikit-learn | Pipeline, GroupKFold CV, evaluation utilities |
| SHAP | Feature importance |
| Plotly | Notebook visualizations |
| pyarrow | Parquet I/O for feature matrices |
| typer | CLI |
| pytest | Tests (synthetic fixtures only — no real captures required) |
| ruff | Linting and formatting |

---

## Related Work

**Chowdhury et al. (2017)** — *Botnet Detection Using Graph-Based Feature Clustering* — applies topological features (in-degree, out-degree, betweenness, eigenvector centrality) with unsupervised clustering on CTU-13. Our supervised approach is a natural extension; their feature set closely matches ours.

**BotSward (Shinan et al., 2023)** — applies centrality measures with ML classifiers on CTU-13, reporting 99% accuracy. Evaluated per-scenario (model trained on same family it's tested on). Our cross-family LOFO evaluation is strictly harder and not directly comparable to their accuracy figures.

**Sinha et al. (2019)** — *Tracking Temporal Evolution of Network Activity for Botnet Detection* — time-windowed graph feature extraction with LSTM on CTU-13, 96.2% accuracy. Closest architectural analog to our pipeline; their `botnet-surf` repo is publicly available.

**Zhou et al. (2020)** — *Automating Botnet Detection with Graph Neural Networks* — topology-only GNN botnet node detection with cross-topology generalization. The GNN approach preserves relational context that scalar per-node features collapse. Our null result on most families is consistent with their finding that deeper architectures are needed for structurally complex botnets.

---

## Citation

CTU-13 dataset:
> García S, Grill M, Stiborek J, Zunino A. An empirical comparison of botnet detection methods. *Computers & Security*, 45:100–123, 2014. https://doi.org/10.1016/j.cose.2014.05.011