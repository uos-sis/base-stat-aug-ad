#!/usr/bin/env python3
"""Single-ended preprocessing for the SAD pipeline.

Reads the raw JODIE-style CSV for one or all datasets and writes the final files
that SAD consumes (datasets.py / Collate / train.py / test.py):

    {dir_data}/{prefix}_{data_set}.csv        -> u, i, ts, label, idx
    {dir_data}/{prefix}_{data_set}.npy        -> edge features, row 0 zero-padded
    {dir_data}/{prefix}_{data_set}_node.npy   -> zero node features (max_idx+1, d)

Replaces the previous two-stage scripts process.py + build_dataset_graph.py (and the
shell wrappers).  Raw files must look like
user_id,item_id,timestamp,state_label,<features...> (first line is a header).

Examples:
    python preprocess.py --data_set all
    python preprocess.py --data_set wikipedia
    python preprocess.py --data_set reddit --no_item_shift
"""
import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_DATASETS = ["reddit", "reddit_stats", "reddit_stats_mlp_ks_64",
                    "wikipedia", "wikipedia_stats", "wikipedia_stats_mlp_ks_64",
                    "mooc", "mooc_stats", "mooc_stats_mlp_ks_8"]
DEFAULT_RAW_DATA_DIR = "../raw_data/"
DEFAULT_DIR_DATA = "./data/"


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess raw JODIE CSVs into SAD-ready files")
    p.add_argument("--raw_data_dir", type=str, default=DEFAULT_RAW_DATA_DIR)
    p.add_argument("--dir_data", type=str, default=DEFAULT_DIR_DATA)
    p.add_argument("--data_set", type=str, default="all",
                  help="dataset(s) to process; 'all' loops over %s" % DEFAULT_DATASETS)
    p.add_argument("--prefix", type=str, default="ml2", help="output file prefix")
    p.add_argument("--item_shift", dest="item_shift", action="store_true", default=True,
                  help="shift item ids beyond the user ids (default: on)")
    p.add_argument("--no_item_shift", dest="item_shift", action="store_false",
                  help="disable the item-id shift")
    return p.parse_args()


def preprocess(path):
    """Read raw csv; return (DataFrame[u,i,ts,label,idx], feature array [N, d])."""
    u_list, i_list, ts_list, label_list, idx_list, feat_l = [], [], [], [], [], []
    with open(path) as f:
        next(f)  # skip header
        for idx, line in enumerate(f):
            e = line.strip().split(",")
            u_list.append(int(e[0]))
            i_list.append(int(e[1]))
            ts_list.append(float(e[2]))
            label_list.append(int(e[3]))
            idx_list.append(idx)
            feat_l.append(np.array([float(x) for x in e[4:]]))

    df = pd.DataFrame(
        {
            "u": pd.array(u_list, dtype="Int64"),
            "i": pd.array(i_list, dtype="Int64"),
            "ts": ts_list,
            "label": pd.array(label_list, dtype="Int64"),
            "idx": pd.array(idx_list, dtype="Int64"),
        }
    )
    df["u"] = df["u"].astype(np.int64)
    df["i"] = df["i"].astype(np.int64)
    df["label"] = df["label"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.float64)
    df["idx"] = df["idx"].astype(np.int64)
    return df, np.array(feat_l)


def reindex(df, item_shift=True):
    """Shift item ids beyond users (if ids are contiguous), then make everything 1-based."""
    new_df = df.copy()
    if item_shift:
        u_contig = new_df.u.max() - new_df.u.min() + 1 == len(new_df.u.unique())
        i_contig = new_df.i.max() - new_df.i.min() + 1 == len(new_df.i.unique())
        if u_contig and i_contig:
            new_df.i = new_df.i + new_df.u.max() + 1
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    return new_df


def run(data_set, args):
    in_path = os.path.join(args.raw_data_dir, f"{data_set}.csv")
    if not os.path.exists(in_path):
        print(f"[skip] raw file not found: {in_path}")
        return

    df, feat = preprocess(in_path)
    new_df = reindex(df, item_shift=args.item_shift)

    d = feat.shape[1]
    feat = np.vstack([np.zeros((1, d)), feat])

    max_idx = int(max(new_df.u.max(), new_df.i.max()))
    node_feat = np.zeros((max_idx + 1, d), dtype=np.float64)

    os.makedirs(args.dir_data, exist_ok=True)
    out_df = os.path.join(args.dir_data, f"{args.prefix}_{data_set}.csv")
    out_feat = os.path.join(args.dir_data, f"{args.prefix}_{data_set}.npy")
    out_node = os.path.join(args.dir_data, f"{args.prefix}_{data_set}_node.npy")

    new_df.to_csv(out_df, index=False)
    np.save(out_feat, feat.astype(np.float64))
    np.save(out_node, node_feat)

    print(
        f"[ok] {data_set}: rows={len(new_df)} u_max={new_df.u.max()} i_max={new_df.i.max()} "
        f"feat_dim={d} -> {out_df}"
    )


def main():
    args = parse_args()
    data_sets = DEFAULT_DATASETS if args.data_set == "all" else [args.data_set]
    for ds in data_sets:
        run(ds, args)


if __name__ == "__main__":
    main()
