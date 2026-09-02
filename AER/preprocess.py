"""AER end-to-end data preprocessing pipeline (single entry point).

Step 1: convert the raw JODIE CSVs (repo root raw_data/) into the AER input
        format (./data/<dataset>/...).
Step 2: generate the synthetic anomaly (negative) samples used by the training
        scripts (./data/<dataset>/AER/<dataset>_cont_full.csv + features).

Run (preprocess_all.sh wraps this for all datasets/feature sources):
    python preprocess.py
    python preprocess.py --datasets reddit mooc --feets raw stats
    python preprocess.py --validation-rate 0.7 --test-rate 0.85
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import utils

RAW_DATA_DIR = "../raw_data/"
DATA_DIR = "./data"
FEETS_SUFFIX = {"raw": "", "stats": "_stats", "statsdim": "_statsdim"}
DATASETS = ["reddit", "mooc", "wikipedia"]
FEETS = ["raw", "stats", "statsdim"]


# --------------------------------------------------------------------- step 1
def get_in_path(dataset, feets):
    if feets == "raw":
        return os.path.join(RAW_DATA_DIR, f"{dataset}.csv")
    if feets == "stats":
        return os.path.join(RAW_DATA_DIR, f"{dataset}_stats.csv")
    # statsdim: Dateiname enthält die ks-Größe (_8/_64), daher glob
    candidates = sorted(glob.glob(os.path.join(RAW_DATA_DIR, f"{dataset}_stats_mlp_ks*.csv")))
    if not candidates:
        raise FileNotFoundError(
            f"no {dataset}_stats_mlp_ks*.csv found in {RAW_DATA_DIR}"
        )
    print(f"statsdim: use {candidates[0]} (available: {candidates})")
    return candidates[0]


def run_preprocessing(datasets, feets):
    """Convert the raw CSVs into the AER input format."""
    for dataset in datasets:
        in_path = get_in_path(dataset, feets)
        out_dir = os.path.join(DATA_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        suffix = FEETS_SUFFIX[feets]

        data = np.loadtxt(in_path, skiprows=1, delimiter=",")
        # expected cols: ['user_id','item_id','timestamp','state_label', ...features...]
        features = data[:, 4:].astype(float)
        np.save(os.path.join(out_dir, f"features{suffix}.npy"), features)
        np.save(os.path.join(out_dir, f"ml_{dataset}{suffix}.npy"), features)

        reduced = np.column_stack((
            data[:, 0].astype(int),
            data[:, 1].astype(int),
            data[:, 2],               # timestamp lieber als float
            data[:, 3].astype(int),
        ))

        np.savetxt(
            os.path.join(out_dir, f"{dataset}{suffix}.csv"),
            reduced,
            delimiter=",",
            fmt=["%d", "%d", "%.6f", "%d"],
            header="u,i,ts,label",
            comments=""
        )
        print(f"[ok] {dataset} processed ({feets}): {out_dir}/{dataset}{suffix}.csv "
              f"+ features{suffix}.npy ({features.shape})")


# ------------------------------------------------------ step 2: synthetic negatives
def add(users, g_df, radio, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start):
    for idx, uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid]) / radio)
        or_ids = g_df[g_df.u == uid].idx.values
        e_id_u = np.random.randint(g_df.idx.min(), g_df.idx.max(), size=N)
        for eid in or_ids:
            tid = g_df.loc[g_df.idx == eid, "i"].values[0]
            ts = g_df[g_df.idx == eid].ts.values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid, "i"].values[0]
            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c) == 0:
                continue
            ts_min, ts_max = ts_c.min(), ts_c.max()
            ts = ts_min if ts_min >= ts_max else np.random.randint(ts_min, ts_max)
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(0)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start


def cut(users, g_df, radio, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start):
    for idx, uid in enumerate(users):
        ulen = len(g_df[g_df.u == uid].idx.values)
        N = int(ulen / radio)
        e_id_u = np.random.choice(list(g_df[g_df.u == uid].idx.values), size=N)
        g_df_new = g_df[g_df.u == uid].copy()
        g_df_new = g_df_new[g_df_new.idx.isin(e_id_u) == False]
        tmax = g_df_new.ts.max()
        index = g_df_new[g_df_new.ts == tmax].index
        g_df_new.loc[index, "label"] = 1
        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid, "i"].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start


def replicate(users, g_df, radio, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start):
    for idx, uid in enumerate(users):
        or_ids = g_df[g_df.u == uid].idx.values
        for eid in or_ids:
            tid = g_df.loc[g_df.idx == eid, "i"].values[0]
            ts = g_df[g_df.idx == eid].ts.values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        N = int(len(g_df[g_df.u == uid]) / radio)
        user_df = g_df[g_df.u == uid]
        idx_min, idx_max = user_df.idx.min(), user_df.idx.max()
        if idx_min >= idx_max or N == 0:
            u_idx_start += 1
            continue
        available_ids = user_df.idx.values
        e_id_u = available_ids if len(available_ids) <= N else np.random.choice(available_ids, size=N, replace=False)
        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid, "i"].values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c) == 0:
                continue
            ts_min, ts_max = ts_c.min(), ts_c.max()
            ts = ts_min if ts_min >= ts_max else np.random.randint(ts_min, ts_max)
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start


def reorder(users, g_df, radio, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start):
    for idx, uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid]) / radio)
        g_df_new = g_df[g_df.u == uid].copy()
        for i in range(N):
            e_id_u = np.random.choice(list(g_df_new.index), size=2)
            t0 = g_df_new.loc[e_id_u[0], "ts"]
            t1 = g_df_new.loc[e_id_u[1], "ts"]
            g_df_new.loc[e_id_u[0], "ts"] = t1
            g_df_new.loc[e_id_u[1], "ts"] = t0
        tmax = g_df_new.ts.max()
        index = g_df_new[g_df_new.ts == tmax].index
        g_df_new.loc[index, "label"] = 1
        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid, "i"].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start


def replicate_cut(users, g_df, radio, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start):
    for idx, uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid]) / radio)
        e_id_u = np.random.randint(
            g_df[g_df.u == uid].idx.min(), g_df[g_df.u == uid].idx.max(), size=N
        )
        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid, "i"].values[0]
            label = g_df[g_df.idx == eid].label.values[0]
            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c) == 0:
                continue
            ts_min, ts_max = ts_c.min(), ts_c.max()
            ts = ts_min if ts_min >= ts_max else np.random.randint(ts_min, ts_max)
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        g_df_new = g_df[g_df.u == uid].copy()
        e_id_u = np.random.choice(list(g_df_new.idx.values), size=N)
        g_df_new = g_df_new[g_df_new.idx.isin(e_id_u) == False]
        tmax = g_df_new.ts.max()
        index = g_df_new[g_df_new.ts == tmax].index
        g_df_new.loc[index, "label"] = 1
        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid, "i"].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start


def add_cut(users, g_df, radio, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start):
    for idx, uid in enumerate(users):
        N = int(len(g_df[g_df.u == uid]) / radio)
        e_id_u = np.random.randint(g_df.idx.min(), g_df.idx.max(), size=N)
        for eid in e_id_u:
            tid = g_df.loc[g_df.idx == eid, "i"].values[0]
            ts_c = g_df[g_df.u == uid].ts.values
            if len(ts_c) == 0:
                continue
            ts_min, ts_max = ts_c.min(), ts_c.max()
            ts = ts_min if ts_min >= ts_max - 1 else np.random.randint(ts_min, ts_max - 1)
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(0)
            edic[e_idx_start] = eid
            e_idx_start += 1
        g_df_new = g_df[g_df.u == uid].copy()
        e_id_u = np.random.choice(list(g_df_new.idx.values), size=N)
        g_df_new = g_df_new[g_df_new.idx.isin(e_id_u) == False]
        tmax = g_df_new.ts.max()
        index = g_df_new[g_df_new.ts == tmax].index
        g_df_new.loc[index, "label"] = 1
        e_id_u = list(g_df_new.idx.values)
        for eid in e_id_u:
            tid = g_df_new.loc[g_df_new.idx == eid, "i"].values[0]
            ts = g_df_new[g_df_new.idx == eid].ts.values[0]
            label = g_df_new[g_df_new.idx == eid].label.values[0]
            src_l.append(int(u_idx_start))
            dst_l.append(int(tid))
            ts_l.append(ts)
            e_idx_l.append(e_idx_start)
            label_l.append(label)
            edic[e_idx_start] = eid
            e_idx_start += 1
        u_idx_start += 1
    return e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start


def generate_negatives(dataset, feets, validation_rate=0.0, test_rate=0.5):
    """Generate the synthetic anomaly samples for one dataset/feature source."""
    data_root = DATA_DIR
    suffix = FEETS_SUFFIX[feets]
    print(f"Processing dataset: {dataset} in {data_root}")

    g_df = pd.read_csv(f"{data_root}/{dataset}/{dataset}{suffix}.csv")
    g_df.columns = ["u", "i", "ts", "label"]
    print(f"Loaded {len(g_df)} interactions from {data_root}/{dataset}/{dataset}{suffix}.csv")

    split_path = f"{data_root}/{dataset}/{dataset}_split{suffix}.npy"
    if os.path.isfile(split_path):
        print(f"load {split_path}")
        src_data = np.load(split_path)
    else:
        print("Random splitting data (stratified: anomaly and normal users interleaved)")
        src_data = utils.create_stratified_split(g_df)
        print(len(src_data))
        print(f"save {split_path}")
        os.makedirs(f"{data_root}/{dataset}", exist_ok=True)
        np.save(split_path, src_data)

    val_rate = float(validation_rate)
    test_rate = float(test_rate)
    val_idx, test_idx = int(len(src_data) * val_rate), int(len(src_data) * test_rate)
    if val_rate == 0:
        val_idx = test_idx
    print(f"val_idx {val_idx} test_idx {test_idx} len(src_data) {len(src_data)}")
    src_train, src_val, src_test = (
        src_data[:val_idx],
        src_data[val_idx:test_idx],
        src_data[test_idx:],
    )

    src_l, dst_l, e_idx_l, label_l, ts_l = [], [], [], [], []

    g_df["idx"] = list(range(len(g_df)))
    e_idx_start = max(list(g_df.idx.values)) + 1
    u_idx_start = max(g_df.u.values.max(), g_df.i.values.max()) + 1
    print(u_idx_start)
    users = set(g_df[g_df.label == 1].u.values) & set(src_train)

    rnum = len(src_train) - len(users)
    print("normal user {}, abnormal user {}".format(rnum, len(users)))
    gnum = rnum - len(users)
    if gnum > 6 * len(users):
        gnum = 6 * len(users)
    if gnum < len(users) * 0.1:
        print("no need to generate data still doing it")
        os.makedirs(f"{data_root}/{dataset}/AER", exist_ok=True)
        features = np.load(f"{data_root}/{dataset}/features{suffix}.npy")
        np.save(f"{data_root}/{dataset}/AER/features_cont_full{suffix}.npy", features)
        g_df.to_csv(f"{data_root}/{dataset}/AER/{dataset}_cont_full{suffix}.csv", index=False)
        assert g_df["idx"].max() < len(features), (
            f"[early-exit] max edge id {g_df['idx'].max()} >= features rows {len(features)}"
            f" in {data_root}/{dataset}{suffix}.csv"
        )
        return

    print("generate {} users".format(gnum))
    edic = dict()

    e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start = add(
        users, g_df, 1, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start
    )
    e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start = cut(
        users, g_df, 1, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start
    )
    e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start = replicate(
        users, g_df, 1, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start
    )
    e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start = reorder(
        users, g_df, 1, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start
    )
    e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start = replicate_cut(
        users, g_df, 2, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start
    )
    e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start = add_cut(
        users, g_df, 2, e_idx_start, src_l, dst_l, ts_l, e_idx_l, label_l, edic, u_idx_start
    )

    new_df = pd.DataFrame({"u": src_l, "i": dst_l, "ts": ts_l, "label": label_l, "idx": e_idx_l})

    users = set(new_df[new_df.label == 1].u.values)
    print("user number ", len(users))
    if os.path.isfile(f"{data_root}/{dataset}/features{suffix}.npy"):
        e_feat = np.load(f"{data_root}/{dataset}/features{suffix}.npy")
        feat_new = np.zeros((e_idx_start + 1, len(e_feat[0])))
        for i in range(len(e_feat)):
            feat_new[i] = e_feat[i]
        for u, i in edic.items():
            feat_new[u] = e_feat[i]
        os.makedirs(f"{data_root}/{dataset}/AER", exist_ok=True)
        assert new_df.idx.max() < feat_new.shape[0], (
            f"[generate] max edge id {new_df.idx.max()} >= features_cont_full rows {feat_new.shape[0]} "
            f"({data_root}/{dataset}{suffix}.csv has {len(g_df)} edges, features has {len(e_feat)})"
        )
        np.save(f"{data_root}/{dataset}/AER/features_cont_full{suffix}.npy", feat_new)

    os.makedirs(f"{data_root}/{dataset}/AER", exist_ok=True)
    new_df.to_csv(f"{data_root}/{dataset}/AER/{dataset}_cont_full{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="AER preprocessing pipeline")
    parser.add_argument("--datasets", nargs="*", default=DATASETS,
                        help="datasets to process (default: %(default)s)")
    parser.add_argument("--feets", nargs="*", default=FEETS, choices=FEETS,
                        help="feature sources to process (default: %(default)s)")
    parser.add_argument("--validation-rate", type=float, default=0.0,
                        help="train/validation split for synthetic-anomaly generation")
    parser.add_argument("--test-rate", type=float, default=0.5,
                        help="train+val/test split for synthetic-anomaly generation")
    args = parser.parse_args()

    # step 1: convert the raw CSVs into the AER input format
    for feets in args.feets:
        run_preprocessing(args.datasets, feets)

    # step 2: generate the synthetic anomaly (negative) samples
    for dataset in args.datasets:
        for feets in args.feets:
            generate_negatives(dataset, feets, args.validation_rate, args.test_rate)

    print("preprocessing done.")


if __name__ == "__main__":
    main()