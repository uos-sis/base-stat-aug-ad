#!/usr/bin/env python3
"""Evaluate trained StrGNN checkpoints on the test split.

Run from the detection directory (anomaly_detection/StrGNN/detection):

    python test.py                        # iterate all checkpoints under ./checkpoints
    python test.py --dirs sta_wikipedia_stats_mlp_ks_64_seed0
    python test.py --checkpoint-root /path/to/checkpoints
    python test.py --compute-shap         # also compute per-dim SHAP importance

For each model a result.json is written into the checkpoint dir with the test
metrics (roc_auc, pr_auc, avg_precision, acc, loss) and, when --compute-shap
is given, a per-dimension mean-absolute SHAP importance of the edge features
per class (importance.class_{all,ano,normal}.mean_abs_shap_per_feature).
"""
import argparse
import json
import math
import os
import pickle
import sys
import types

import numpy as np
import torch

sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import Classifier, loop_dataset, cmd_args  # noqa: E402

DEFAULT_CHECKPOINT_ROOT = './checkpoints'


def parse_args():
    p = argparse.ArgumentParser(description='Test StrGNN checkpoints on the test split')
    p.add_argument('--checkpoint-root', type=str, default=DEFAULT_CHECKPOINT_ROOT,
                   help='root directory holding per-run checkpoint folders')
    p.add_argument('--dirs', type=str, default=None, nargs='*',
                   help='specific checkpoint dirs (else all under --checkpoint-root)')
    p.add_argument('--config-file', type=str, default='config.json')
    p.add_argument('--model-file', type=str, default='model.pt')
    p.add_argument('--data-root', type=str, default=None,
                   help='directory containing data/ and data_sta/ (default: this script dir)')
    p.add_argument('--gpu', type=int, default=0, help='cuda device id, -1 for cpu')
    p.add_argument('--seed', type=int, default=1)
    # SHAP importance
    p.add_argument('--compute-shap', action='store_true',
                   help='compute per-dimension SHAP feature importances (optional)')
    p.add_argument('--shap-max-samples', type=int, default=2000,
                   help='max # correctly classified samples used for the SHAP summary')
    p.add_argument('--shap-background-samples', type=int, default=50)
    p.add_argument('--shap-nsamples', type=int, default=30)
    p.add_argument('--split-dim', type=int, default=None,
                   help='feature dim split: embedding=[0:split_dim], statistic=[split_dim:]; '
                        'default: D - 10 for *_stats datasets, else D')
    p.add_argument('--pred-threshold', type=float, default=0.5)
    return p.parse_args()


def get_device(args):
    if args.gpu < 0 or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda:%d' % args.gpu)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(model_dir, config_file):
    with open(os.path.join(model_dir, config_file)) as f:
        return types.SimpleNamespace(**json.load(f))


def data_dir_of(graph, data_root):
    if graph.startswith('sta_'):
        return os.path.join(data_root, 'data_sta')
    return os.path.join(data_root, 'data')


def build_classifier(config, train_graphs, val_graphs, test_graphs, max_n_label, device):
    cmd_args.gm = 'DGCNN'
    cmd_args.sortpooling_k = float(getattr(config, 'sortpooling_k', 0.6))
    cmd_args.latent_dim = [int(x) for x in getattr(config, 'latent_dim', '32-32-32-1').split('-')]
    cmd_args.hidden = int(getattr(config, 'hidden', 128))
    cmd_args.out_dim = 0
    cmd_args.dropout = not getattr(config, 'no_dropout', False)
    cmd_args.num_class = 2
    cmd_args.mode = 'cpu' if device.type == 'cpu' else 'gpu'
    cmd_args.batch_size = int(getattr(config, 'batch_size', 32))
    cmd_args.feat_dim = max_n_label + 1
    cmd_args.attr_dim = 0
    cmd_args.edge_feat_dim = 0
    for g_list_i in train_graphs:
        for g in g_list_i:
            if g.edge_features is not None:
                cmd_args.edge_feat_dim = g.edge_features.shape[1]
                break
        if cmd_args.edge_feat_dim:
            break
    cmd_args.window = 5
    cmd_args.edge_proj_dim = int(getattr(config, 'edge_proj_dim', 0))
    cmd_args.dense_dim = int(getattr(config, 'dense_dim', 256))
    cmd_args.mlp_dropout = float(getattr(config, 'mlp_dropout', 0.5))
    if cmd_args.sortpooling_k <= 1:
        all_graphs = []
        for g_list_i in train_graphs:
            all_graphs.append(g_list_i[-1])
        for g_list_i in val_graphs:
            all_graphs.append(g_list_i[-1])
        for g_list_i in test_graphs:
            all_graphs.append(g_list_i[-1])
        num_nodes_list = sorted([g.num_nodes for g in all_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    return classifier


def _import_shap():
    """Import shap, patching numpy>=1.24 removed aliases that shap still uses."""
    if not hasattr(np, 'float'):
        np.float = float
        np.int = int
        np.bool = bool
        np.object = object
        np.complex = complex
        np.str = str
        np.unicode = str
    import shap
    return shap


def collect_predictions(classifier, test_graphs, bsize=32):
    """Return (pred probs for class 1, labels) over the test groups."""
    m_pred = []
    m_label = []
    with torch.no_grad():
        for i in range(0, len(test_graphs), bsize):
            batch = test_graphs[i:i + bsize]
            logits, _, _ = classifier(batch)
            m_pred.append(torch.exp(logits[:, 1]).detach().cpu().numpy())
            m_label.extend([g[-1].label for g in batch])
    return np.concatenate(m_pred), np.asarray(m_label, dtype=np.int64)


class ShapStrGNN(torch.nn.Module):
    """Wrapper so shap.GradientExplainer can attribute StrGNN logits w.r.t. the
    edge-feature matrix of a fixed window of subgraphs.

    forward receives X of shape [B, E, D] (B perturbed copies of one sample's
    window edge features, E = sum over the 5 subgraphs of 2*num_edges) and
    returns logits [B, 1].
    """

    def __init__(self, classifier, device):
        super(ShapStrGNN, self).__init__()
        self.classifier = classifier
        self.device = device
        self.subgraphs = None
        self.base_node_feat = None

    def set_context(self, subgraphs, base_node_feat):
        self.subgraphs = subgraphs
        self.base_node_feat = base_node_feat

    def forward(self, X):
        if self.subgraphs is None or self.base_node_feat is None:
            raise RuntimeError('ShapStrGNN.set_context() must be called first')
        B, E, D = X.shape
        node_feat = self.base_node_feat.repeat(B, 1).to(self.device)
        edge_feat = X.reshape(-1, D).to(self.device)
        batch = []
        for b in range(B):
            batch += self.subgraphs
        embed = self.classifier.gnn(batch, node_feat, edge_feat)
        logits = self.classifier.mlp(embed)
        return logits[:, 1].unsqueeze(-1)


def build_shap_context(test_graphs, idx, feat_dim):
    """Return (subgraphs, base_node_feat_tensor, edge_feat_matrix) for a test group."""
    subgraphs = test_graphs[idx]
    tags = []
    blocks = []
    for g in subgraphs:
        tags += list(g.node_tags)
        if g.edge_features is not None:
            blocks.append(g.edge_features)
    if not blocks:
        return None, None, None
    feat = np.concatenate(blocks, 0).astype(np.float32)  # [E, D]
    node_tags = np.asarray(tags, dtype=np.int64)
    base_node_feat = np.zeros((len(tags), feat_dim), dtype=np.float32)
    base_node_feat[np.arange(len(tags)), node_tags] = 1.0
    return subgraphs, torch.from_numpy(base_node_feat), feat


def explain_class(classifier, device, sample_idx, test_graphs, bg_by_E, rng, args, feat_dim):
    """Mean |SHAP| per feature dim over the given test samples (one class)."""
    shap = _import_shap()
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False  # GRU backward in eval mode needs the native implementation
    agg = None
    agg_count = 0
    sm = ShapStrGNN(classifier, device)
    explainer_cache = {}
    for idx in sample_idx:
        subgraphs, base_node_feat, feat = build_shap_context(test_graphs, int(idx), feat_dim)
        if subgraphs is None:
            continue
        E, D = feat.shape
        pool = bg_by_E.get(E, [feat])
        S = min(args.shap_background_samples, len(pool))
        bg_idx = rng.choice(len(pool), size=S, replace=False)
        bg = torch.stack([torch.from_numpy(pool[i]) for i in bg_idx])
        X = torch.from_numpy(feat).float().unsqueeze(0)
        # When the (per-#edges) pool holds only this sample's own features, using it
        # as the SHAP baseline makes (x - baseline) = 0, so GradientExplainer returns
        # ~0 importances. Fall back to a mean-row baseline.
        if S == 1 and np.array_equal(pool[bg_idx[0]], feat):
            bg = torch.from_numpy(feat.mean(axis=0, keepdims=True)).float().expand(1, E, D).contiguous()
        sm.set_context(subgraphs, base_node_feat)
        if E not in explainer_cache:
            explainer_cache[E] = shap.GradientExplainer(sm, bg)
        explainer = explainer_cache[E]
        sv = explainer.shap_values(X, nsamples=args.shap_nsamples)
        arr = np.abs(np.asarray(sv)[0]).reshape(-1, D)
        if agg is None:
            agg = arr.sum(0)
        else:
            agg += arr.sum(0)
        agg_count += arr.shape[0]

    torch.backends.cudnn.enabled = cudnn_was_enabled
    if agg is None:
        return None
    return agg / float(agg_count)


def compute_shap_importances(classifier, test_graphs, split_dim, args, device):
    """Select correctly classified samples per class and compute SHAP summaries."""
    m_pred, m_label = collect_predictions(classifier, test_graphs)
    pos = np.arange(len(m_label))
    correct = (m_pred >= args.pred_threshold).astype(np.int64) == m_label
    correct_normal = pos[np.logical_and(m_label == 0, correct)]
    correct_anomaly = pos[np.logical_and(m_label == 1, correct)]

    rng = np.random.RandomState(args.seed)

    selected_anomaly = correct_anomaly[: args.shap_max_samples]
    n_normal = max(0, args.shap_max_samples - len(selected_anomaly))
    selected_normal = rng.permutation(correct_normal)[:n_normal] if n_normal > 0 \
        else np.zeros(0, dtype=np.int64)
    all_idx = np.concatenate([selected_normal, selected_anomaly]).astype(np.int64)

    bg_by_E = {}
    D = 0
    for idx in all_idx:
        subgraphs, _, feat = build_shap_context(test_graphs, int(idx), cmd_args.feat_dim)
        if subgraphs is None:
            continue
        bg_by_E.setdefault(feat.shape[0], []).append(feat)
        if D == 0:
            D = feat.shape[1]

    emb_end = min(split_dim, D) if D else split_dim
    imp = {
        'enabled': True,
        'limit': args.shap_max_samples,
        'split_dim': split_dim,
        'feature_dim': D,
        'split': {'embedding': [0, emb_end], 'statistic': [emb_end, D]},
    }

    def summarize(per_feature, extra=None):
        arr = np.asarray(per_feature)
        emb = arr[:emb_end]
        stat = arr[emb_end:]
        emb_imp = float(emb.mean()) if len(emb) else 0.0
        stat_imp = float(stat.mean()) if len(stat) else 0.0
        total = emb_imp + stat_imp
        emb_share = emb_imp / total if total > 0 else 0.0
        stat_share = stat_imp / total if total > 0 else 0.0
        out = {
            'embedding_importance': emb_imp,
            'statistic_importance': stat_imp,
            'embedding_min': float(emb.min()) if len(emb) else 0.0,
            'embedding_max': float(emb.max()) if len(emb) else 0.0,
            'embedding_std': float(emb.std()) if len(emb) else 0.0,
            'statistic_min': float(stat.min()) if len(stat) else 0.0,
            'statistic_max': float(stat.max()) if len(stat) else 0.0,
            'statistic_std': float(stat.std()) if len(stat) else 0.0,
            'embedding_share': round(emb_share, 4),
            'statistic_share': round(stat_share, 4),
            'score': 'embedding: %.0f%%, statistic: %.0f%%' % (emb_share * 100, stat_share * 100),
            'mean_abs_shap_per_feature': arr.tolist(),
        }
        out.update(extra)
        return out

    if len(selected_normal) > 0:
        v = explain_class(classifier, device, selected_normal, test_graphs, bg_by_E, rng, args, cmd_args.feat_dim)
        if v is not None:
            imp['class_normal'] = summarize(v, {'num_correct_normals': int(len(correct_normal))})
    if len(selected_anomaly) > 0:
        v = explain_class(classifier, device, selected_anomaly, test_graphs, bg_by_E, rng, args, cmd_args.feat_dim)
        if v is not None:
            imp['class_ano'] = summarize(v, {'num_correct_anomalies': int(len(correct_anomaly))})
    v = explain_class(classifier, device, all_idx, test_graphs, bg_by_E, rng, args, cmd_args.feat_dim)
    if v is not None:
        imp['class_all'] = summarize(v, {'num_correct_samples': int(len(correct_normal)) + int(len(correct_anomaly))})
    return imp


def evaluate_model_dir(model_dir, args):
    cfg_path = os.path.join(model_dir, args.config_file)
    mod_path = os.path.join(model_dir, args.model_file)
    if not (os.path.exists(cfg_path) and os.path.exists(mod_path)):
        print('[skip] %s missing model/config' % model_dir)
        return

    config = load_config(model_dir, args.config_file)
    device = get_device(args)
    set_seed(getattr(config, 'seed', args.seed))

    data_root = args.data_root or os.path.dirname(os.path.realpath(__file__))
    d_dir = data_dir_of(config.graph, data_root)
    efeat_path = os.path.join(d_dir, config.graph.replace('.npy', '_efeat.npy'))
    cache_name = (config.split + 'h' + str(config.hop)
                  + ('f' if os.path.exists(efeat_path) else '') + 'v')
    cache_path = os.path.join(d_dir, cache_name)
    if not os.path.exists(cache_path):
        print('[skip] %s missing subgraph cache %s' % (model_dir, cache_path))
        return

    with open(cache_path, 'rb') as f:
        train_graphs, val_graphs, test_graphs, max_n_label = pickle.load(f)

    classifier = build_classifier(config, train_graphs, val_graphs, test_graphs, max_n_label, device)
    classifier.load_state_dict(torch.load(mod_path, map_location=device))
    classifier.eval()

    metrics = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
    roc_auc, pr_auc = metrics[2], metrics[4]

    result = {
        'checkpoint': os.path.basename(model_dir),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'avg_precision': metrics[3],
        'acc': metrics[1],
        'loss': metrics[0],
    }

    if args.compute_shap:
        try:
            shap = _import_shap()
        except Exception as e:
            print('[skip shap] %s: %s' % (model_dir, e))
            shap = None
        if shap is not None:
            split_dim = args.split_dim
            if split_dim is None:
                split_dim = (cmd_args.edge_feat_dim - 10
                             if '_stats' in getattr(config, 'split', '') else cmd_args.edge_feat_dim)
            result['importance'] = compute_shap_importances(
                classifier, test_graphs, split_dim, args, device)

    with open(os.path.join(model_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('[ok] %s -> roc_auc=%.4f pr_auc=%.4f acc=%.4f (result.json written)'
          % (os.path.basename(model_dir), roc_auc, pr_auc, metrics[1]))


def main():
    args = parse_args()
    if args.dirs:
        model_dirs = [d if os.path.isdir(d) else os.path.join(args.checkpoint_root, d) for d in args.dirs]
    else:
        model_dirs = [os.path.join(args.checkpoint_root, d)
                      for d in sorted(os.listdir(args.checkpoint_root))
                      if os.path.isdir(os.path.join(args.checkpoint_root, d))]
    if not model_dirs:
        print('no model dirs found in %s' % args.checkpoint_root)
        return
    for d in model_dirs:
        evaluate_model_dir(d, args)


if __name__ == '__main__':
    main()