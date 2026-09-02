# base-stat-aug-ad

Repository with five anomaly-detection models for temporal (JODIE-style) graphs
(Wikipedia, Reddit, MOOC) and their statistics-augmented variants
(`_stats`, `_stats_mlp_ks_{8,64}`). All models read their raw inputs from the
repo-root `raw_data/` folder via relative paths and provide a uniform set of
pipeline scripts (`preprocess_all.sh` -> `train_all.sh` -> `test_all.sh`).

## Download the raw data

Put the raw CSVs and the statistics-augmented files into `raw_data/`:

https://myshare.uni-osnabrueck.de/f/21e614e0b1f5480daf06/?dl=1

Expected layout:

```
raw_data/
  {reddit,wikipedia,mooc}.csv                          # user_id,item_id,timestamp,state_label,<feats>
  {reddit,wikipedia,mooc}_stats.csv                    # statistics-augmented features
  {reddit,wikipedia,mooc}_stats_mlp_ks_{64,64,8}.csv   # ML-encoded features
  <dataset>_stats_mlp_ks_*.pt                          # raw-mode encoders (SHAP)
```

Each model is trained on the 9 dataset variants
(`{reddit,wikipedia,mooc}` x `{raw,stats,stats_mlp_ks_*}`) across multiple seeds.

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
| AER | `AER/preprocess_all.sh` | `AER/train_all.sh` | `AER/test_all.sh` + `AER/make_table.py` |
| DyGLib (TGAT) | `DyGLib/preprocess_data/preprocess_all.sh` | `DyGLib/tgat_train_all.sh` | `DyGLib/tgat_test_all.sh` |
| DyGLib (TGN) | `DyGLib/preprocess_data/preprocess_all.sh` | `DyGLib/tgn_train_all.sh` | `DyGLib/tgn_test_all.sh` |
| DyGLib (DyGFormer) | `DyGLib/preprocess_data/preprocess_all.sh` | `DyGLib/dygformer_train_all.sh` | `DyGLib/dygformer_test_all.sh` |
| GraphSAGE | `GraphSAGE/preprocess_all.sh` | `GraphSAGE/train_all.sh` | `GraphSAGE/test_all.sh` |
| SAD | `SAD/preprocess_all.sh` | `SAD/train_all.sh` | `SAD/test_all.sh` |
| StrGNN | `StrGNN/detection/preprocess_all.sh` | `StrGNN/detection/train_all.sh` | `StrGNN/detection/test_all.sh` |

All `*_all.sh` scripts `cd` into their model folder, so they are run from anywhere.

## Run everything at once

`run.sh` drives the whole pipeline (preprocess -> train -> test -> LaTeX tables) for
all models on all dataset variants with a single seed, switching conda envs per model:

```bash
bash run.sh                      # all models, seed 0
SEED=3 bash run.sh               # one seed
MODELS="aer dyglib" bash run.sh  # subset
CLEAN=0 bash run.sh              # keep existing checkpoints/results
```

It appends every model's ROC/AUC (and SHAP) LaTeX table to `pipeline_tables.log`
at the repo root and the full pipeline output to `run_all.log`. By default it
removes stale generated checkpoints/results first so the tables only reflect the
current run.

---

## AER

Env: `aer`. Build the C++ graph library once (`graph/graph.so` is gitignored):

```
cd AER && g++ -shared -fPIC -O2 -o graph/graph.so graph/graph.cpp
```

### Preprocess

```bash
cd AER && bash preprocess_all.sh     # reddit/mooc/wikipedia x raw/stats/statsdim
```

Wraps the single `preprocess.py` (raw CSVs -> AER format + synthetic anomalies in `./data/<dataset>/`).

### Train

```bash
cd AER && bash train_all.sh                        # 3 datasets x 3 feets x seeds 0-3
WANDB_LOGGING=true bash train_all.sh               # enable wandb (off by default)
```

Trains on GPU (`main.py`, model moved via `to(device)`) and saves the best model
to `./saved_models/<runtime_id>best-model.pth`.

### Test (AUC + SHAP) and LaTeX table

```bash
cd AER && bash test_all.sh          # AUC + SHAP per dataset x feets x seed
python make_table.py                # LaTeX AUC table + max gain + SHAP shares
```

`test_all.sh` runs `calcshape.py` per (dataset, feature set): for each saved best
model it scores the test split (ROC-AUC / PR-AUC / accuracy) and computes
expected-gradients SHAP importances split into embedding vs statistic shares
(for `stats`/`statsdim`). Results are written to
`./results/shap/<dataset>-<feets>/result.json`. `make_table.py` emits the LaTeX
test-AUC table (`raw`/`stats`/`statsdim`, mean +- std, max gain =
`best(stats, statsdim) - raw`) and the SHAP statistic-share table.

---

## DyGLib

Env: `sad`. Models: TGAT, TGN, DyGFormer (node classification).

### Preprocess

```bash
cd DyGLib/preprocess_data && bash preprocess_all.sh
```

### Train (per model, all variants x seeds 0-3)

```bash
cd DyGLib
bash tgat_train_all.sh
bash tgn_train_all.sh
bash dygformer_train_all.sh
```

Single run:

```bash
python train_node_classification.py --dataset_name wikipedia --model_name DyGFormer \
    --patch_size 2 --max_input_sequence_length 64 --num_runs 5
```

### Test / evaluate

```bash
cd DyGLib
bash tgat_test_all.sh
bash tgn_test_all.sh
bash dygformer_test_all.sh
```

Writes `result.json` (ROC-AUC / PR-AUC) per checkpoint; `*_test_all.sh` also run
the SHAP pipelines (`shap_tgat.py` / `shap_tgn.py` / `shap_dygformer.py`) on the
`_stats` checkpoints. Single run:

```bash
python test.py --dirs node_classification_TGN_wikipedia_seed0 --compute_importances
```

---

## GraphSAGE

Env: `hogat` (torch 2.0.1 + torch-geometric).

### Preprocess (JODIE snapshots)

```bash
cd GraphSAGE && bash preprocess_all.sh   # writes ./data/<data_set>.snapshots.npz + .json
```

### Train

```bash
cd GraphSAGE && bash train_all.sh        # all variants x seeds 0-3 (dims 64 reddit/wikipedia, 8 mooc)
```

Best model per run -> `checkpoints/<wandb_run_name>/model.pt` + `config.json`.

### Test / evaluate

```bash
cd GraphSAGE && bash test_all.sh         # writes result.json (roc_auc, pr_auc) per run
```

SHAP importances:

```bash
python -m graphsage.torch_test --dirs <run_dir> --compute_importances \
    --shap_max_samples 2000 --shap_background_samples 50 --shap_nsamples 30
```

`*_mlp_ks_*` checkpoints evaluate automatically in raw mode (raw features embedded
on the fly via the saved encoder; `--raw_data_dir ../raw_data/`).

---

## SAD

Env: `sad`.

### Preprocess

```bash
cd SAD && bash preprocess_all.sh     # writes ml2_*.csv/.npy/_node.npy into ./data
```

### Train

```bash
cd SAD && bash train_all.sh         # all datasets x seeds 0-3
```

Key options in `option.py`: `--mask_label`, `--mask_ratio`, `--anomaly_alpha`,
`--supc_alpha`, `--train_split` / `--val_split` / `--test_split`, `--seed`.

### Test / evaluate

```bash
cd SAD && bash test_all.sh          # result.json (roc_auc, pr_auc) per checkpoint
```

SHAP importances:

```bash
python test.py --dirs <run_dir> --compute_importances \
    --shap_max_samples 2000 --shap_background_samples 50 --shap_nsamples 30
```

`*_mlp_ks_64` checkpoints evaluate automatically in raw mode
(`--raw_data_dir ../raw_data/`).

---

## StrGNN

Env: `sad`.

### Install (once)

```bash
cd StrGNN/detection && bash install.sh     # builds pytorch_DGCNN/lib
```

### Preprocess

```bash
cd StrGNN/detection && bash preprocess_all.sh
```

Builds the accumulated (`acc_<ds>.npy`) and time-evolving (`sta_<ds>.npy`)
snapshot matrices + edge features and syncs them into `StrGNN/detection/data`
and `data_sta`.

### Train

```bash
cd StrGNN/detection && bash train_all.sh          # time-evolving graphs (default)
GRAPHS="acc sta" bash train_all.sh                # accumulated + time-evolving
```

Per-dataset hyperparameter groups (hidden / latent / edge-proj / dense / dropout)
are preserved; defaults: epochs 100, batch 128, lr 1e-4, hop 20, seeds 1-3.

### Test / evaluate

```bash
cd StrGNN/detection && bash test_all.sh   # ROC-AUC / PR-AUC + SHAP on *_stats* checkpoints
```

---

## Notes

- Generated artifacts (`./data`, `./checkpoints`, `./results`, `./wandb`,
  `graph/graph.so`, StrGNN `pytorch_DGCNN/lib`) are gitignored and must be
  regenerated / rebuilt on a fresh clone.
- The AUC + SHAP evaluators need trained best models first (`train_all.sh`
  produces the checkpoints).
- Raw data files are read with relative paths (repo-root `raw_data/`), so the
  repository is self-contained.