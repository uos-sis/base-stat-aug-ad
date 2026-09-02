from __future__ import division
from __future__ import print_function

import numpy as np
import torch


class SnapshotData(object):
    """A single temporal snapshot converted into a small PyTorch Geometric-style graph.

    The raw npz layout (written by prepare_jodie_snapshots.py) is:
        s{k}_feats   : [M+1, d] float32, last row is the dummy padding node
        s{k}_adj     : [M, max_degree] int32, padded with index ``M``
        s{k}_users   : [n_users] local indices of the (labeled) user nodes
        s{k}_labels  : [n_users] int binary labels
    User nodes are always the first ``n_users`` rows (0 .. n_users-1), item
    nodes follow.  Labels are only defined for user nodes.
    """

    def __init__(self, feats, adj, users, labels, snap_ts):
        self.snap_ts = float(snap_ts)
        self.users = users.astype(np.int64)
        self.labels = labels.astype(np.float32)

        M = int(adj.shape[0])
        x = feats[:M].astype(np.float32)

        mask = adj < M
        src = np.repeat(np.arange(M, dtype=np.int64), mask.sum(axis=1))
        dst = adj[mask].astype(np.int64)
        if src.size:
            edge_index = np.stack([src, dst])
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        self.x = torch.from_numpy(x)
        self.edge_index = torch.from_numpy(edge_index)
        self.user_idx = torch.from_numpy(self.users)
        self.y = torch.from_numpy(self.labels)


class SnapshotDataset(object):
    """Loads the ``*.snapshots.npz`` file into per-snapshot SnapshotData."""

    def __init__(self, snapshot_file):
        data = np.load(snapshot_file)
        self.feat_dim = int(data["feat_dim"][0])
        self.num_nodes = int(data["num_nodes"][0])
        self.max_degree = int(data["max_degree"][0])
        self.split = data["split"].astype(np.int64)
        self.snap_ts = data["snap_ts"]
        self.num_snapshots = int(data["num_snapshots"][0])

        self.snapshots = []
        for k in range(self.num_snapshots):
            self.snapshots.append(SnapshotData(
                data["s%d_feats" % k],
                data["s%d_adj" % k],
                data["s%d_users" % k],
                data["s%d_labels" % k],
                data["snap_ts"][k],
            ))

        self.train_ids = [k for k in range(self.num_snapshots) if self.split[k] == 0]
        self.val_ids = [k for k in range(self.num_snapshots) if self.split[k] == 1]
        self.test_ids = [k for k in range(self.num_snapshots) if self.split[k] == 2]

    def split_counts(self):
        return {0: len(self.train_ids), 1: len(self.val_ids), 2: len(self.test_ids)}
