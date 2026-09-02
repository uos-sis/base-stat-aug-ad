from __future__ import print_function

import argparse
import csv
import json
import os

import numpy as np


def _find_col(header, names):
    lowered = [c.lower() for c in header]
    for n in names:
        if n.lower() in lowered:
            return lowered.index(n.lower())
    return None

def parse_raw(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        iu = _find_col(header, ["user_id", "u", "src"])
        ii = _find_col(header, ["item_id", "i", "dst", "target"])
        it = _find_col(header, ["timestamp", "ts", "time"])
        il = _find_col(header, ["state_label", "label"])
        if iu is None or ii is None or it is None or il is None:
            raise ValueError("Required columns not found in %s: %s" % (path, header))
        us, items, ts, ls, fl = [], [], [], [], []
        for row in reader:
            if not row:
                continue
            us.append(int(row[iu]))
            items.append(int(row[ii]))
            ts.append(float(row[it]))
            ls.append(int(row[il]))
            fl.append([float(x) for x in row[4:]])
    if not us:
        raise ValueError("No data rows in %s" % path)
    d = len(fl[0])
    for fv in fl:
        if len(fv) != d:
            raise ValueError("Inconsistent feature dimension: %d != %d" % (len(fv), d))
    return (np.asarray(us, dtype=np.int64), np.asarray(items, dtype=np.int64),
            np.asarray(ts, dtype=np.float64), np.asarray(ls, dtype=np.int64),
            np.asarray(fl, dtype=np.float32), d)


def remap_nodes(us, its):
    users = sorted(set(us.tolist()))
    items = sorted(set(its.tolist()))
    user_map = {raw: k for k, raw in enumerate(users)}
    item_map = {raw: len(users) + k for k, raw in enumerate(items)}
    u_mapped = np.asarray([user_map[x] for x in us.tolist()], dtype=np.int64)
    i_mapped = np.asarray([item_map[x] for x in its.tolist()], dtype=np.int64)
    id_map = {str(raw): val for raw, val in list(user_map.items()) + list(item_map.items())}
    return u_mapped, i_mapped, len(users) + len(items), id_map


def build_snapshot(su, si, sl, sf, d, max_degree, rng):
    user_ids = np.unique(su)
    item_ids = np.unique(si)
    local_user = {int(g): k for k, g in enumerate(user_ids)}
    item_off = len(user_ids)
    local_item = {int(g): item_off + k for k, g in enumerate(item_ids)}
    M = len(user_ids) + len(item_ids)
    gids = np.concatenate([user_ids.astype(np.int64), item_ids.astype(np.int64)])
    users_local = np.arange(len(user_ids), dtype=np.int64)
    labels = np.zeros(len(user_ids), dtype=np.int64)
    feats = np.zeros((M, d), dtype=np.float32)
    sums = np.zeros((M, d), dtype=np.float64)
    cnts = np.zeros(M, dtype=np.int64)
    neigh = [[] for _ in range(M)]
    for e in range(len(su)):
        lu = local_user[int(su[e])]
        li = local_item[int(si[e])]
        neigh[lu].append(li)
        neigh[li].append(lu)
        labels[lu] = max(labels[lu], int(sl[e]))
        feats[lu] = sf[e]
        sums[li] += sf[e]
        cnts[li] += 1
    for li in range(len(item_ids)):
        if cnts[item_off + li] > 0:
            feats[item_off + li] = (sums[item_off + li] / cnts[item_off + li]).astype(np.float32)
    deg = np.zeros(M, dtype=np.int32)
    adj = np.full((M, max_degree), M, dtype=np.int32)
    for n in range(M):
        nb = neigh[n]
        deg[n] = len(nb)
        if len(nb) == 0:
            continue
        if len(nb) < max_degree:
            vals = rng.choice(nb, max_degree, replace=True)
        else:
            vals = rng.choice(nb, max_degree, replace=False)
        adj[n] = np.asarray(vals, dtype=np.int32)
    feats_pad = np.vstack([feats, np.zeros((1, d), dtype=np.float32)])
    return gids, users_local, labels, feats_pad, adj, deg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data_dir", default="../raw_data/")
    p.add_argument("--data_set", required=True, help="csv basename without extension; snapshots are "
                  "written to <out_dir>/<data_set>.snapshots.npz / <out_dir>/<data_set>.json")
    p.add_argument("--out_dir", default="./data/",
                  help="directory where the <data_set>.snapshots.npz / <data_set>.json are written")
    p.add_argument("--window_size", type=int, default=2000)
    p.add_argument("--max_degree", type=int, default=25)
    p.add_argument("--split_list", default="0.7,0.15,0.15")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    rng = np.random.RandomState(args.seed)
    s_list = [float(x) for x in args.split_list.split(",")]
    assert len(s_list) == 3 and abs(sum(s_list) - 1.0) < 1e-9

    path = os.path.join(args.raw_data_dir, args.data_set + ".csv")
    if not os.path.exists(path):
        raise SystemExit("raw file not found: %s" % path)
    us, its, ts, ls, feats, d = parse_raw(path)
    ls = np.where(ls != 0, 1, 0).astype(np.int64)
    u_mapped, i_mapped, num_nodes, id_map = remap_nodes(us, its)
    feats = feats.astype(np.float32)
    order = np.argsort(ts, kind="mergesort")
    uc, ic, tc, lc, fc = u_mapped[order], i_mapped[order], ts[order], ls[order], feats[order]

    snap_ts = []
    snap_data = []
    maxM = 0
    for s in range(0, len(uc), args.window_size):
        a, b = s, min(s + args.window_size, len(uc))
        snap_ts.append(float(tc[b - 1]))
        data = build_snapshot(uc[a:b], ic[a:b], lc[a:b], fc[a:b], d, args.max_degree, rng)
        snap_data.append(data)
        maxM = max(maxM, data[3].shape[0] - 1)
    S = len(snap_ts)
    print("parsed %d edges, %d dims, %d nodes, %d snapshots, maxM=%d" % (len(uc), d, num_nodes, S, maxM))

    snap_ts_arr = np.asarray(snap_ts)
    val_time, test_time = np.percentile(snap_ts_arr, [s_list[0] * 100, (s_list[0] + s_list[1]) * 100])
    split = np.zeros(S, dtype=np.int64)
    split[snap_ts_arr > val_time] = 1
    split[snap_ts_arr > test_time] = 2
    str_counts = {0: int(np.sum(split == 0)), 1: int(np.sum(split == 1)), 2: int(np.sum(split == 2))}
    print("split (train/val/test):", str_counts)

    out = os.path.join(args.out_dir, args.data_set)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    save = {
        "feat_dim": np.asarray([d]),
        "num_nodes": np.asarray([num_nodes]),
        "max_degree": np.asarray([args.max_degree]),
        "window_size": np.asarray([args.window_size]),
        "num_snapshots": np.asarray([S]),
        "split": split,
        "snap_ts": snap_ts_arr,
    }
    for k, (gids, users_local, labels, feats_pad, adj, deg) in enumerate(snap_data):
        save["s%d_nodes" % k] = gids
        save["s%d_users" % k] = users_local
        save["s%d_labels" % k] = labels
        save["s%d_feats" % k] = feats_pad
        save["s%d_adj" % k] = adj
        save["s%d_deg" % k] = deg
    npz_path = out + ".snapshots.npz"
    np.savez_compressed(npz_path, **save)
    print("wrote", npz_path)

    meta_path = out + ".json"
    with open(meta_path, "w") as f:
        json.dump({
            "data_set": args.data_set,
            "window_size": args.window_size,
            "num_edges": int(len(uc)),
            "num_nodes": int(num_nodes),
            "feat_dim": int(d),
            "num_snapshots": int(S),
            "split": str_counts,
            "id_map": id_map,
        }, f, indent=2)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
