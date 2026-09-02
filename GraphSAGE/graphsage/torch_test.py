from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import types

import numpy as np
import torch
import torch.nn as nn

from graphsage.snapshots_data import SnapshotDataset
from graphsage.torch_model import GraphSAGEModel


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PyG GraphSAGE checkpoints on the test split")
    p.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    p.add_argument("--dirs", nargs="*", default=None,
                  help="explicit model dir(s); default: all dirs in checkpoint_root")
    p.add_argument("--config_file", type=str, default="config.json")
    p.add_argument("--model_file", type=str, default="model.pt")
    p.add_argument("--compute_importances", action="store_true",
                  help="compute SHAP feature importances (optional, off by default)")
    p.add_argument("--split_dim", type=int, default=None,
                  help="input feature dim split: embedding=[0:split_dim], statistic=[split_dim:]; "
                       "default: feat_dim minus the number of statistic columns (from "
                       "the raw header) for *_stats datasets, else feat_dim")
    p.add_argument("--raw_mode", action="store_true",
                  help="for *_mlp_ks_64 checkpoints: use the raw snapshot file and embed the "
                       "features on the fly with raw_data/<data_set>_mlp_ks_64.pt, so "
                       "importances attribute the RAW input features (split 0:171 / 172:n). "
                       "Auto-enabled when data_set ends in _mlp_ks_64.")
    p.add_argument("--raw_data_dir", type=str, default="../raw_data/")
    p.add_argument("--data_dir", type=str, default="./data/",
                  help="directory of the <data_set>.snapshots.npz files produced by "
                       "prepare_jodie_snapshots.py")
    p.add_argument("--encoder_file", type=str, default="",
                  help="encoder .pt name within raw_data_dir (default: <raw>_mlp_ks_64.pt)")
    p.add_argument("--raw_snapshot_prefix", type=str, default=None,
                  help="raw snapshot npz prefix used to rebuild node features with the encoder "
                       "(default: <data_dir>/<raw_data_set>)")
    p.add_argument("--raw_feature_npy", type=str, default=None,
                  help="full raw feature matrix (.npy, leading zero row) used to refit the "
                       "encoder scaler/latent statistics; default: {raw_data_dir}/<raw>.csv")
    p.add_argument("--shap_max_samples", type=int, default=5000)
    p.add_argument("--shap_background_samples", type=int, default=40)
    p.add_argument("--shap_nsamples", type=int, default=20)
    p.add_argument("--shap_max_users", type=int, default=256,
                  help="max user rows attributed per snapshot (caps the expensive SHAP "
                       "per-output gradient loop); lower = faster")
    p.add_argument("--pred_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_config(model_dir, config_file):
    with open(os.path.join(model_dir, config_file)) as f:
        return types.SimpleNamespace(**json.load(f))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def mlp_ks_raw_data_set(data_set):
    for suffix in ("_mlp_ks_64", "_mlp_ks_16", "_mlp_llr_64", "_mlp_ks_8",
                  "_mlp_combined_64", "_mlp_llr", "_mlp_ks"):
        if data_set.endswith(suffix):
            return data_set[: -len(suffix)]
    return None


_RAW_ENC_CACHE = {}


def get_raw_feature_encoder(config, args, device):
    """Return the FeatureEncoder (on ``device``) for an encoded data_set.

    Encoder files follow the reduced-dataset naming ``{data_set}.pt`` (e.g.
    ``reddit_stats_mlp_ks_64.pt`` or ``mooc_stats_mlp_ks_8.pt`` under
    ``--raw_data_dir``); ``--encoder_file`` overrides.  Returns ``None`` when the
    data_set is not an encoded one or the encoder/raw features are missing.
    """
    from graphsage.feature_encoder import FeatureEncoder
    data_set = getattr(config, "data_set", "")
    if mlp_ks_raw_data_set(data_set) is None:
        return None
    enc_path = args.encoder_file or os.path.join(args.raw_data_dir, "%s.pt" % data_set)
    if not os.path.exists(enc_path):
        print("[skip raw mode] encoder not found: %s" % enc_path)
        return None
    if args.raw_feature_npy:
        feat_path = args.raw_feature_npy
    else:
        csv_path = os.path.join(args.raw_data_dir, "%s.csv" % mlp_ks_raw_data_set(data_set))
        if not os.path.exists(csv_path):
            print("[skip raw mode] raw features not found: %s" % csv_path)
            return None
        feat_path = csv_path
    key = (enc_path, feat_path)
    if key in _RAW_ENC_CACHE:
        return _RAW_ENC_CACHE[key][0]
    raw_dim = int(torch.load(enc_path, map_location="cpu")["0.weight"].shape[1])
    with_leading_zero = feat_path.endswith(".npy")
    if with_leading_zero:
        feats = np.load(feat_path).astype(np.float32)
        fit = feats[1:]
    else:
        with open(feat_path) as f:
            next(f)
            fit = np.asarray([[float(x) for x in line.strip().split(",")[4:4 + raw_dim]]
                             for line in f], dtype=np.float32)
    if fit.shape[1] != raw_dim:
        raise ValueError("raw feature dim %d from %s != encoder input dim %d" %
                       (fit.shape[1], feat_path, raw_dim))
    fe, _ = FeatureEncoder.from_checkpoint(enc_path, fit)
    fe = fe.to(device)
    _RAW_ENC_CACHE[key] = (fe,)
    return fe


def raw_stat_feature_count(data_set, args, default=10):
    """Number of trailing statistic columns in the raw feature vector.

    ``reduce_features_mlp.py`` names the statistics right after the embedding token in the
    raw CSV header; the header is malformed (a single trailing token spans all remaining
    columns), so the count is taken from the header tokens.
    """
    if args.raw_feature_npy:
        return default
    raw_name = mlp_ks_raw_data_set(data_set)
    if raw_name is None:
        return default
    csv_path = os.path.join(args.raw_data_dir, "%s.csv" % raw_name)
    if not os.path.exists(csv_path):
        return default
    with open(csv_path) as f:
        tokens = f.readline().strip().split(",")
    return max(0, len(tokens) - 5)


class _LazySnapshots(object):
    def __init__(self, build_fn):
        self._build = build_fn
        self._cache = {}

    def __getitem__(self, k):
        if k not in self._cache:
            self._cache[k] = self._build(int(k))
        return self._cache[k]

    def get(self, k, default=None):
        return self._cache.get(k, default)


class RawSnapshotDataset(object):
    """Wraps the RAW snapshot file (feat_dim 181) and the stored *_mlp_ks_64
    snapshot file (feat_dim 64) so the model sees faithful encoded node features while
    SHAP attributes the RAW user features.

    Each snapshot's ``x`` is rebuilt as [ encoder(raw user rows),
    encoded item rows, encoded padding row ] -- exactly the layout the model was trained
    on -- because ``encoder`` does not commute with the item mean aggregation.
    Snapshots are built lazily on first access (and on the GPU) so only the ones
    actually used are ever materialised.
    """

    def __init__(self, encoder, enc_dataset, raw_dataset, device):
        self.encoder = encoder
        self.enc = enc_dataset
        self.raw = raw_dataset
        self.device = device
        for name in ("feat_dim", "num_nodes", "max_degree", "num_snapshots",
                    "train_ids", "val_ids", "test_ids", "snap_ts", "split", "test_ids"):
            setattr(self, name, getattr(enc_dataset, name))
        self.snapshots = _LazySnapshots(self._make_snapshot)

    def _make_snapshot(self, k):
        enc = self.enc.snapshots[k]
        raw = self.raw.snapshots[k]
        n_users = int(enc.user_idx.shape[0])
        users_enc = self.encoder(raw.x[:n_users].to(self.device))
        x = torch.cat([users_enc, enc.x[n_users:].to(self.device)], 0)
        return types.SimpleNamespace(
            x=x, edge_index=enc.edge_index, user_idx=enc.user_idx, labels=enc.labels,
            users=enc.users)

    def split_counts(self):
        return self.enc.split_counts()


def build_model(config, dataset, device):
    feat_dim = int(getattr(config, "feat_dim", dataset.feat_dim))
    model = GraphSAGEModel(feat_dim,
                          int(getattr(config, "dim_1", 128)),
                          int(getattr(config, "dim_2", 128)),
                          float(getattr(config, "dropout", 0.0)))
    return model


def eval_test(model, dataset, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for k in dataset.test_ids:
            snap = dataset.snapshots[k]
            x = snap.x.to(device)
            edge_index = snap.edge_index.to(device)
            user_idx = snap.user_idx.to(device)
            logits = model(x, edge_index, user_idx)
            preds.append(logits.sigmoid().cpu().numpy().flatten())
            labels.append(snap.labels.flatten())
    m_pred = np.asarray(np.concatenate(preds))
    m_label = np.asarray(np.concatenate(labels))
    keep = m_label >= 0
    m_pred, m_label = m_pred[keep], m_label[keep].astype(np.int64)
    import sklearn.metrics as skm
    roc_auc = float(skm.roc_auc_score(m_label, m_pred))
    pr_auc = float(skm.average_precision_score(m_label, m_pred))
    return roc_auc, pr_auc, m_pred, m_label


class ShapGraphSAGEModel(nn.Module):
    """Wrapper so shap.GradientExplainer can attribute GraphSAGE logits w.r.t. the
    user-node feature matrix of a fixed per-snapshot graph context.

    forward receives X of shape [B, n_users, D] (B perturbed copies of one
    snapshot's user features) and returns logits [B, n_users, 1].  Each copy is
    an independent subgraph with node ids offset by its index, so per-row outputs are
    independent.
    """

    def __init__(self, model, device, encoder=None, max_users=None):
        super().__init__()
        self.model = model
        self.device = device
        self.encoder = encoder
        self.max_users = max_users  # cap on #user rows attributed per sample
        self.ctx = None

    def set_context(self, ctx):
        self.ctx = ctx

    def forward(self, X):
        if self.ctx is None:
            raise RuntimeError("ShapGraphSAGEModel.set_context() must be called first")
        X = X.float().to(self.device)
        if self.encoder is not None:
            X = self.encoder(X)  # raw [..., raw_dim] -> reduced [..., 64]
        squeeze_out = X.ndim == 2
        if squeeze_out:
            X = X.unsqueeze(0)
        B, n_users, D = X.shape
        K = min(n_users, self.max_users or n_users)
        M = self.ctx["M"]
        items = self.ctx["x"][self.ctx["n_users"]:]  # [M - n_users, D]
        x_cat = torch.cat([X, items.expand(B, M - n_users, D)], dim=1).view(B * M, D)
        edge = self.ctx["edge_index"]
        offsets = (torch.arange(B, device=self.device) * M).repeat_interleave(edge.shape[1])
        edge_index = edge.repeat(1, B) + offsets.view(1, -1)
        logits = self.model(x_cat, edge_index)  # [B*M, 1]
        user_pos = (torch.arange(B, device=self.device) * M).view(B, 1) + \
            torch.arange(K, device=self.device).view(1, K)
        out = logits[user_pos.view(-1)].view(B, K, 1)
        if squeeze_out:
            out = out[0]
        return out


def build_context(dataset, k, device):
    if hasattr(dataset, "raw"):
        enc_snap = dataset.snapshots[k]
        raw_snap = dataset.raw.snapshots[k]
        n_users = raw_snap.user_idx.shape[0]
        M = enc_snap.x.shape[0]
        return {
            "x": enc_snap.x.to(device),
            "edge_index": enc_snap.edge_index.to(device),
            "n_users": int(n_users),
            "M": int(M),
        }, raw_snap.x[raw_snap.user_idx].to(device)
    snap = dataset.snapshots[k]
    n_users = snap.user_idx.shape[0]
    M = snap.x.shape[0]
    return {
        "x": snap.x.to(device),
        "edge_index": snap.edge_index.to(device),
        "n_users": int(n_users),
        "M": int(M),
    }, snap.x[snap.user_idx].to(device)


def explain_samples(model, device, sample_idx, dataset, bg_by_n, rng, args, split_dim):
    import shap
    agg = None
    agg_count = 0
    sm = ShapGraphSAGEModel(model, device, encoder=getattr(dataset, "encoder", None),
                          max_users=args.shap_max_users)
    explainer_cache = {}
    for k in sample_idx:
        ctx, feat = build_context(dataset, int(k), device)
        n_users = feat.shape[0]
        D = feat.shape[1]
        pool = bg_by_n.get(n_users, [feat])
        S = min(args.shap_background_samples, len(pool))
        bg_idx = rng.choice(len(pool), size=S, replace=False)
        bg = torch.stack([pool[i] for i in bg_idx]).to(device)
        X = feat.reshape(1, n_users, D)
        # When the (per-#users) pool holds only this snapshot's own features, using it as
        # the SHAP baseline makes (x - baseline) = 0, so GradientExplainer returns ~0
        # importances. Fall back to a mean-row baseline (deviation from the average user row).
        if S == 1 and torch.equal(bg[0], feat):
            bg = feat.mean(dim=0, keepdim=True).expand(1, n_users, D).contiguous()

        sm.set_context(ctx)
        explainer = explainer_cache.get(n_users)
        if explainer is None:
            explainer = shap.GradientExplainer(sm, bg)
            explainer_cache[n_users] = explainer
        sv = explainer.shap_values(X, nsamples=args.shap_nsamples)
        arr = np.abs(np.asarray(sv)[0]).reshape(-1, D)
        if agg is None:
            agg = arr.sum(0)
        else:
            agg += arr.sum(0)
        agg_count += arr.shape[0]

    if agg is None:
        return None
    per_feature = agg / float(agg_count)

    n_emb = min(len(per_feature), max(0, split_dim))
    embedding_part = per_feature[:n_emb]
    statistic_part = per_feature[n_emb:]

    def _stats(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        return float(arr.min()), float(arr.max()), float(arr.std())

    embedding_min, embedding_max, embedding_std = _stats(embedding_part)
    statistic_min, statistic_max, statistic_std = _stats(statistic_part)

    embedding_importance = float(embedding_part.mean()) if len(embedding_part) > 0 else 0.0
    statistic_importance = float(statistic_part.mean()) if len(statistic_part) > 0 else 0.0
    total = embedding_importance + statistic_importance
    if total > 0:
        emb_share = embedding_importance / total
        stat_share = statistic_importance / total
    else:
        emb_share, stat_share = 0.0, 0.0

    return {
        "num_samples_used": len(sample_idx),
        "entry_count": int(agg_count),
        "embedding_importance": embedding_importance,
        "statistic_importance": statistic_importance,
        "embedding_min": embedding_min,
        "embedding_max": embedding_max,
        "embedding_std": embedding_std,
        "statistic_min": statistic_min,
        "statistic_max": statistic_max,
        "statistic_std": statistic_std,
        "embedding_share": round(emb_share, 4),
        "statistic_share": round(stat_share, 4),
        "score": "embedding: %.0f%%, statistic: %.0f%%" % (emb_share * 100, stat_share * 100),
        "mean_abs_shap_per_feature": per_feature.tolist(),
    }


def compute_importances(model, device, dataset, args, m_pred, m_label, split_dim):
    pos = np.arange(len(m_label))
    correct = (m_pred >= args.pred_threshold).astype(np.int64) == m_label
    # m_label/m_pred are concatenated over test snapshots in temporal order; map each
    # test user entry back to its snapshot id so we can attribute at snapshot level.
    snaps = np.repeat(np.asarray(dataset.test_ids),
                     [int(dataset.snapshots[k].labels.shape[0]) for k in dataset.test_ids])
    correct_normal = pos[np.logical_and(m_label == 0, correct)]
    correct_anomaly = pos[np.logical_and(m_label == 1, correct)]

    rng = np.random.RandomState(args.seed)

    selected_anomaly = correct_anomaly[: args.shap_max_samples]
    n_normal = max(0, args.shap_max_samples - len(selected_anomaly))
    selected_normal = rng.permutation(correct_normal)[:n_normal] if n_normal > 0 \
        else np.zeros(0, dtype=np.int64)

    all_idx = np.concatenate([selected_normal, selected_anomaly]).astype(np.int64)
    snapshot_idx = np.unique(snaps[all_idx]).astype(np.int64)

    # background pool of user feature matrices, keyed by the snapshot's #user nodes
    # (each SnapshotData has a different node count, like SAD keys by edge count).
    bg_by_n = {}
    D = 0
    for k in snapshot_idx:
        snap = dataset.snapshots[k]
        feat = build_context(dataset, int(k), device)[1] if hasattr(dataset, "raw") \
            else snap.x[snap.user_idx].to(device)
        bg_by_n.setdefault(feat.shape[0], []).append(feat)
        if D == 0:
            D = feat.shape[1]

    emb_end = min(split_dim, D)
    imp = {
        "enabled": True,
        "limit": args.shap_max_samples,
        "split_dim": split_dim,
        "feature_dim": D,
        "split": {"embedding": [0, emb_end], "statistic": [emb_end, D]},
    }
    if len(all_idx) > 0:
        if len(selected_normal) > 0:
            cls0 = explain_samples(model, device,
                               np.unique(snaps[selected_normal]), dataset, bg_by_n,
                               rng, args, split_dim)
            if cls0 is not None:
                cls0["num_correct_normals"] = int(len(correct_normal))
                imp["class_normal"] = cls0
        if len(selected_anomaly) > 0:
            cls1 = explain_samples(model, device,
                               np.unique(snaps[selected_anomaly]), dataset, bg_by_n,
                               rng, args, split_dim)
            if cls1 is not None:
                cls1["num_correct_anomalies"] = int(len(correct_anomaly))
                imp["class_ano"] = cls1
        cls_all = explain_samples(model, device, snapshot_idx, dataset, bg_by_n,
                              rng, args, split_dim)
        if cls_all is not None:
            cls_all["num_correct_samples"] = int(len(correct_normal)) + int(len(correct_anomaly))
            imp["class_all"] = cls_all
    return imp


def evaluate_model_dir(model_dir, args):
    cfg_path = os.path.join(model_dir, args.config_file)
    mod_path = os.path.join(model_dir, args.model_file)
    if not (os.path.exists(cfg_path) and os.path.exists(mod_path)):
        print("[skip] %s missing model/config" % model_dir)
        return

    config = load_config(model_dir, args.config_file)
    device = get_device()
    set_seed(getattr(config, "seed", args.seed))

    raw_mode = args.raw_mode or mlp_ks_raw_data_set(getattr(config, "data_set", "")) is not None
    snap_name = getattr(config, "data_set", "data")
    configured_prefix = getattr(config, "snapshot_prefix", None)
    snap_prefix = os.path.join(args.data_dir, snap_name)
    if configured_prefix and os.path.exists(configured_prefix + ".snapshots.npz"):
        snap_prefix = configured_prefix
    raw_name = mlp_ks_raw_data_set(snap_name)
    if raw_mode:
        encoder = get_raw_feature_encoder(config, args, device)
        if encoder is None:
            print("[skip raw mode] no raw features/encoder available for %s" % config.data_set)
            return
        if args.raw_snapshot_prefix:
            raw_prefix = args.raw_snapshot_prefix
        elif configured_prefix and os.path.exists(configured_prefix + "_stats_raw.snapshots.npz"):
            raw_prefix = configured_prefix + "_stats_raw"
        else:
            raw_prefix = os.path.join(args.data_dir, raw_name if raw_name else snap_name)
        dataset = RawSnapshotDataset(
            encoder,
            SnapshotDataset(snap_prefix + ".snapshots.npz"),
            SnapshotDataset(raw_prefix + ".snapshots.npz"),
            device,
        )
    else:
        dataset = SnapshotDataset(snap_prefix + ".snapshots.npz")

    model = build_model(config, dataset, device)  # feat_dim from checkpoint (64)
    model.load_state_dict(torch.load(mod_path, map_location=device))
    model.to(device).eval()

    roc_auc, pr_auc, m_pred, m_label = eval_test(model, dataset, device)

    result = {"checkpoint": os.path.basename(model_dir), "roc_auc": roc_auc, "pr_auc": pr_auc}

    if args.compute_importances:
        if raw_mode:
            split_dim = args.split_dim if args.split_dim is not None else \
                int(encoder.raw_dim) - raw_stat_feature_count(getattr(config, "data_set", ""), args)
        else:
            split_dim = args.split_dim if args.split_dim is not None else \
                (int(dataset.feat_dim) - raw_stat_feature_count(getattr(config, "data_set", ""), args)
                 if "stats" in getattr(config, "data_set", "") else int(dataset.feat_dim))
        result["importance"] = compute_importances(model, device, dataset, args, m_pred, m_label, split_dim)

    with open(os.path.join(model_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[ok] %s -> roc_auc=%.4f pr_auc=%.4f (result.json written)" %
          (os.path.basename(model_dir), roc_auc, pr_auc))


def main():
    args = parse_args()
    if args.dirs:
        model_dirs = [d if os.path.isdir(d) else os.path.join(args.checkpoint_root, d)
                     for d in args.dirs]
    else:
        model_dirs = [os.path.join(args.checkpoint_root, d)
                     for d in sorted(os.listdir(args.checkpoint_root))
                     if os.path.isdir(os.path.join(args.checkpoint_root, d))]
    if not model_dirs:
        print("no model dirs found in %s" % args.checkpoint_root)
        return
    for d in model_dirs:
        evaluate_model_dir(d, args)


if __name__ == "__main__":
    main()
