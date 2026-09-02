from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEModel(nn.Module):
    """Supervised node-classification GraphSAGE over fixed temporal snapshots.

    Mirrors the original TensorFlow supervised GraphSAGE (mean aggregation, concat of
    self/neighbour transforms, L2-normalised node embeddings, linear readout to a
    single logit, BCE with sigmoid) using torch_geometric's SAGEConv.
    """

    def __init__(self, feat_dim, dim_1, dim_2, dropout=0.0):
        super(GraphSAGEModel, self).__init__()
        self.feat_dim = int(feat_dim)
        self.dim_1 = int(dim_1)
        self.dim_2 = int(dim_2)
        self.dropout = float(dropout)

        self.conv1 = SAGEConv(self.feat_dim, self.dim_1)
        self.conv2 = SAGEConv(self.dim_1, self.dim_2)
        self.node_pred = nn.Linear(self.dim_2, 1)

    def forward(self, x, edge_index, user_idx=None):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.normalize(h, p=2, dim=1)
        logits = self.node_pred(h)
        if user_idx is not None:
            return logits[user_idx]
        return logits
