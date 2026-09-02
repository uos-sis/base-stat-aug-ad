#!/usr/bin/env python3
"""
Preprocess MODUS2 raw temporal edge lists (Wikipedia, Reddit, MOOC) into the
input format expected by StrGNN (detection/Main.py and detection/Main_statistic.py).

Each dataset variant (e.g. reddit, reddit_stats, reddit_stats_mlp_ks_64) is
processed independently so their results can be compared. Outputs per variant:

  acc_<ds>.npy        object array (T,) of scipy sparse CSR (N,N) accumulated snapshots
  sta_<ds>.npy        object array (T,) of scipy sparse CSR (N,N) time-evolving snapshots
  acc_<ds>_efeat.npy  per-snapshot {(u,v): feat} maps for the accumulated graphs
  sta_<ds>_efeat.npy  per-snapshot {(u,v): feat} maps for the time-evolving graphs
  <ds>.npz            train/val/test pos/neg edges and snapshot ids (identical copy for sta)

Positive samples are edges with label == 1 (anomalous); negatives are subsampled
edges with label == 0 (normal). Only edges in snapshots >= window - 1 are used.
The train/val/test split is chronological, by time quantiles of the usable events
(defaults: 70% train, 15% validation, 15% test).
"""

import argparse
import csv
import os
import shutil

import numpy as np
from scipy import sparse

DEFAULT_DATASETS = [
    'mooc', 'mooc_stats', 'mooc_stats_mlp_ks_8',
    'wikipedia', 'wikipedia_stats', 'wikipedia_stats_mlp_ks_64',
    'reddit', 'reddit_stats', 'reddit_stats_mlp_ks_64',
]


def parse_args():
    p = argparse.ArgumentParser(description='Preprocess raw edge streams into StrGNN input format')
    p.add_argument('--datasets', default=','.join(DEFAULT_DATASETS),
                   help='comma-separated dataset stems (file name without .csv)')
    p.add_argument('--input-dir', default='../../raw_data')
    p.add_argument('--output-dir', default='../data')
    p.add_argument('--snapshots', type=int, default=120, help='number of snapshots T')
    p.add_argument('--window', type=int, default=5, help='temporal window (model is hardcoded to 5)')
    p.add_argument('--train-ratio', type=float, default=0.7,
                   help='fraction of usable events used for training (chronological)')
    p.add_argument('--val-ratio', type=float, default=0.15,
                   help='fraction of usable events used for validation (chronological)')
    p.add_argument('--neg-ratio', type=float, default=1.0,
                   help='number of negative samples per positive sample')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no-normalize', action='store_true', default=False,
                   help='do not standardize edge features (default: z-score per column '
                        'using train-split event statistics)')
    p.add_argument('--sync-repo', action='store_true',
                   help='copy outputs into StrGNN/detection/data and .../data_sta')
    return p.parse_args()


def load_edges(path):
    with open(path, newline='') as f:
        rd = csv.reader(f)
        header = next(rd)
        col = {name: i for i, name in enumerate(header)}
        if 'src' in col and 'target' in col:
            u_col, v_col, t_col, l_col = col['src'], col['target'], col['time'], col['label']
        else:
            u_col = col['user_id']
            v_col = col['item_id']
            t_col = col['timestamp']
            l_col = col['state_label']
        rows = []
        for r in rd:
            if len(r) <= max(u_col, v_col, t_col, l_col):
                continue
            u, v = int(r[u_col]), int(r[v_col])
            if u == v:
                continue
            t = float(r[t_col])
            lbl = int(r[l_col])
            feat = np.array([float(x) for x in r[4:]], dtype=np.float32)
            rows.append((u, v, t, lbl, feat))
    return rows


def remap(rows):
    users = sorted({r[0] for r in rows})
    items = sorted({r[1] for r in rows})
    u_map = {x: i for i, x in enumerate(users)}
    v_map = {x: i + len(users) for i, x in enumerate(items)}
    n = len(users) + len(items)
    out = [(u_map[u], v_map[v], t, lbl, feat) for u, v, t, lbl, feat in rows]
    return out, n


def to_snapshots(rows, n_snap):
    rows = sorted(rows, key=lambda r: r[2])
    t0, t1 = rows[0][2], rows[-1][2]
    span = t1 - t0

    def bin_idx(ts):
        if span <= 0:
            return 0
        return min(int((ts - t0) * n_snap / span), n_snap - 1)

    bins = [[] for _ in range(n_snap)]
    for u, v, ts, lbl, feat in rows:
        bins[bin_idx(ts)].append((u, v, ts, lbl, feat))
    return bins


def build_matrices(bins, n):
    acc, sta = [], []
    cur = sparse.csr_matrix((n, n), dtype=np.float64)
    for b in bins:
        if b:
            rows = np.array([x[0] for x in b], dtype=np.int64)
            cols = np.array([x[1] for x in b], dtype=np.int64)
            data = np.ones(len(b), dtype=np.float64)
            m = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            m = sparse.csr_matrix((n, n), dtype=np.float64)
        sta.append(m)
        cur = cur + m
        acc.append(cur.copy())
    return acc, sta


def _edge_key(u, v, n):
    lo, hi = (u, v) if u < v else (v, u)
    return lo * n + hi


def build_edge_features(bins, n, accumulate):
    out = []
    state = {}
    for b in bins:
        agg = {}
        for u, v, _, _, feat in b:
            key = _edge_key(u, v, n)
            if accumulate:
                s, c = state.get(key, (0, 0))
                state[key] = (s + feat, c + 1)
            else:
                s, c = agg.get(key, (0, 0))
                agg[key] = (s + feat, c + 1)
        src = state if accumulate else agg
        d = {}
        for key, (s, c) in src.items():
            d[key] = (s / c).astype(np.float32)
        out.append(d)
    return out


def compute_quantiles(bins, window_start, train_ratio, val_ratio):
    times = []
    for n in range(window_start, len(bins)):
        for u, v, ts, lbl, _ in bins[n]:
            times.append(ts)
    if not times:
        return None, None
    times = np.array(times, dtype=np.float64)
    return float(np.quantile(times, train_ratio)), float(np.quantile(times, train_ratio + val_ratio))


def standardize_features(bins, window_start, q_tr, eps=1e-8):
    feats = []
    for n in range(window_start, len(bins)):
        for u, v, ts, lbl, feat in bins[n]:
            if q_tr is None or ts <= q_tr:
                feats.append(feat)
    if not feats:
        return bins
    mat = np.stack(feats).astype(np.float64)
    mean = mat.mean(0)
    std = mat.std(0)
    std[std < eps] = 1.0
    new_bins = []
    for b in bins:
        nb = []
        for u, v, ts, lbl, feat in b:
            feat = ((feat.astype(np.float64) - mean) / std).astype(np.float32)
            nb.append((u, v, ts, lbl, feat))
        new_bins.append(nb)
    return new_bins


def sample_edges(bins, window_start, q_tr, q_va, neg_ratio, seed):
    rng = np.random.RandomState(seed)
    events = []
    for n in range(window_start, len(bins)):
        for u, v, ts, lbl, _ in bins[n]:
            events.append((ts, u, v, n, lbl))
    if not events:
        return {'train': ([], []), 'val': ([], []), 'test': ([], [])}
    events = sorted(events, key=lambda e: e[0])
    splits = {'train': ([], []), 'val': ([], []), 'test': ([], [])}
    for ts, u, v, n, lbl in events:
        if q_tr is None or ts <= q_tr:
            part = 'train'
        elif ts <= q_va:
            part = 'val'
        else:
            part = 'test'
        (splits[part][0] if lbl == 1 else splits[part][1]).append((u, v, n))
    out = {}
    for part in ('train', 'val', 'test'):
        pos, neg = splits[part]
        k = min(len(neg), int(len(pos) * neg_ratio))
        if k < len(neg):
            idx = rng.choice(len(neg), size=k, replace=False)
            neg = [neg[i] for i in idx]
        out[part] = (pos, neg)
    return out


def to_arrays(samples):
    if not samples:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    arr = np.array(samples, dtype=np.int64)
    edges = arr[:, :2].T
    ids = arr[:, 2]
    return edges, ids


def save_matrices(path, mats):
    arr = np.empty(len(mats), dtype=object)
    for i, m in enumerate(mats):
        arr[i] = m
    np.save(path, arr)


def process(args, ds):
    src = os.path.join(args.input_dir, ds + '.csv')
    if not os.path.exists(src):
        print('skip %s: file not found' % ds)
        return
    rows = load_edges(src)
    rows, n_nodes = remap(rows)
    bins = to_snapshots(rows, args.snapshots)

    window_start = args.window - 1
    q_tr, q_va = compute_quantiles(bins, window_start, args.train_ratio, args.val_ratio)
    if not args.no_normalize:
        bins = standardize_features(bins, window_start, q_tr)

    acc, sta = build_matrices(bins, n_nodes)
    splits = sample_edges(bins, window_start, q_tr, q_va, args.neg_ratio, args.seed)

    train_pos, train_pos_id = to_arrays(splits['train'][0])
    train_neg, train_neg_id = to_arrays(splits['train'][1])
    val_pos, val_pos_id = to_arrays(splits['val'][0])
    val_neg, val_neg_id = to_arrays(splits['val'][1])
    test_pos, test_pos_id = to_arrays(splits['test'][0])
    test_neg, test_neg_id = to_arrays(splits['test'][1])

    assert train_pos.shape == train_neg.shape
    assert val_pos.shape == val_neg.shape
    assert test_pos.shape == test_neg.shape
    assert train_pos.shape[0] == 2 and train_pos_id.shape[0] == train_pos.shape[1]
    for arr in (train_pos_id, val_pos_id, test_pos_id):
        if arr.size:
            assert arr.min() >= window_start

    feat_dim = len(rows[0][4])
    acc_feats = build_edge_features(bins, n_nodes, accumulate=True)
    sta_feats = build_edge_features(bins, n_nodes, accumulate=False)

    os.makedirs(args.output_dir, exist_ok=True)
    save_matrices(os.path.join(args.output_dir, 'acc_%s.npy' % ds), acc)
    save_matrices(os.path.join(args.output_dir, 'sta_%s.npy' % ds), sta)
    save_feature_path(os.path.join(args.output_dir, 'acc_%s_efeat.npy' % ds), acc_feats)
    save_feature_path(os.path.join(args.output_dir, 'sta_%s_efeat.npy' % ds), sta_feats)
    np.savez(os.path.join(args.output_dir, '%s.npz' % ds),
             train_pos_id=train_pos_id, train_neg_id=train_neg_id,
             val_pos_id=val_pos_id, val_neg_id=val_neg_id,
             test_pos_id=test_pos_id, test_neg_id=test_neg_id,
             train_pos=train_pos, train_neg=train_neg,
             val_pos=val_pos, val_neg=val_neg,
             test_pos=test_pos, test_neg=test_neg)

    print('%s: nodes=%d snapshots=%d feat_dim=%d | train=%d/%d val=%d/%d test=%d/%d (pos/neg)'
          % (ds, n_nodes, args.snapshots, feat_dim,
             len(splits['train'][0]), len(splits['train'][1]),
             len(splits['val'][0]), len(splits['val'][1]),
             len(splits['test'][0]), len(splits['test'][1])))

    if args.sync_repo:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for sub in ('data', 'data_sta'):
            d = os.path.join(repo, 'detection', sub)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(args.output_dir, '%s.npz' % ds), d)
            graph = 'acc_%s.npy' % ds if sub == 'data' else 'sta_%s.npy' % ds
            shutil.copy(os.path.join(args.output_dir, graph), d)
            shutil.copy(os.path.join(args.output_dir, graph.replace('.npy', '_efeat.npy')), d)


def save_feature_path(path, dicts):
    arr = np.empty(len(dicts), dtype=object)
    for i, d in enumerate(dicts):
        arr[i] = d
    np.save(path, arr)


def main():
    args = parse_args()
    for ds in args.datasets.split(','):
        ds = ds.strip()
        if not ds:
            continue
        process(args, ds)


if __name__ == '__main__':
    main()
