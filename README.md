# base-stat-aug-ad

Repository with five anomaly-detection models for temporal (JODIE-style) graphs
(Wikipedia, Reddit, MOOC) and their statistics-augmented variants
(`_stats`, `_stats_mlp_ks_{8,64}`). All models read their raw inputs from the
repo-root `raw_data/` folder via relative paths and provide a uniform set of
pipeline scripts (`preprocess_all.sh` -> `train_all.sh` -> `test_all.sh`).

## Download the raw data

Put the raw CSVs and the statistics-augmented files into `raw_data/`:

https://myshare.uni-osnabrueck.de/f/21e614e0b1f5480daf06/?dl=1

## Environments

| Model | conda env | torch | requirements |
|---|---|---|---|
| AER | `aer` | 2.5.1+cu121 | `AER/requirements.txt` |
| DyGLib | `sad` | 1.10.1+cu111 | `DyGLib/requirements.txt` |
| GraphSAGE | `hogat` | 2.0.1+cu118 | `GraphSAGE/requirements.txt` |
| SAD | `sad` | 1.10.1+cu111 | `SAD/requirements.txt` |
| StrGNN | `sad` | 1.10.1+cu111 | `StrGNN/requirements.txt` |

```
conda activate <env>
pip install -r <model>/requirements.txt
```

## Pipeline overview

| Model | Preprocess | Train | Evaluate |
|---|---|---|---|
| AER | `AER/preprocess_all.sh` | `AER/train_all.sh` | `AER/test_all.sh` |
| DyGLib (TGAT) | `DyGLib/preprocess_data/preprocess_all.sh` | `DyGLib/tgat_train_all.sh` | `DyGLib/tgat_test_all.sh` |
| DyGLib (TGN) | `DyGLib/preprocess_data/preprocess_all.sh` | `DyGLib/tgn_train_all.sh` | `DyGLib/tgn_test_all.sh` |
| DyGLib (DyGFormer) | `DyGLib/preprocess_data/preprocess_all.sh` | `DyGLib/dygformer_train_all.sh` | `DyGLib/dygformer_test_all.sh` |
| GraphSAGE | `GraphSAGE/preprocess_all.sh` | `GraphSAGE/train_all.sh` | `GraphSAGE/test_all.sh` |
| SAD | `SAD/preprocess_all.sh` | `SAD/train_all.sh` | `SAD/test_all.sh` |
| StrGNN | `StrGNN/detection/preprocess_all.sh` | `StrGNN/detection/train_all.sh` | `StrGNN/detection/test_all.sh` |

All `*_all.sh` scripts `cd` into their model folder, so they are run from anywhere.

---

## AER

Env: `aer`. Build the C++ graph library once (`graph/graph.so` is gitignored):

```
cd AER && g++ -shared -fPIC -O2 -o graph/graph.so graph/graph.cpp
```
