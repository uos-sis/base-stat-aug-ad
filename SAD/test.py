#!/usr/bin/env python3
"""Evaluate trained SAD checkpoints on the test split.

Run from the SAD directory (anomaly_detection/SAD):

    # iterate all saved models, compute ROC-AUC / PR-AUC only
    python test.py

    # also compute (optional) SHAP feature importances
    python test.py --compute_importances

    # explain a specific saved model dir
    python test.py --dirs SAD_reddit_0.9472_seed42

    # evaluate a *_mlp_ks_64 checkpoint on the RAW dataset (embedded on the fly by
    # prepending raw_data/<data_set>.pt); importances then attribute the RAW features,
    # split into embedding ([0:split_dim]) and statistic ([split_dim:]).
    python test.py --dirs SAD_reddit_stats_mlp_ks_64_... --compute_importances

For each model, a result.json is written into the model's checkpoint dir with the
scores (roc_auc, pr_auc) and, if importances are computed, per-class summary
importance split into embedding ([0:split_dim]) and statistic ([split_dim:]).
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.utils.data

# option.py (imported transitively by datasets.py) calls argparse.parse_args()
# immediately at import time.  Feed it a placeholder argv so it uses defaults and
# does not choke on this script's own flags.
_argv_backup = sys.argv
sys.argv = ["sad_test"]
import datasets as dataset  # noqa: E402
from model.tgat import TGAT  # noqa: E402
sys.argv = _argv_backup

from feature_encoder import FeatureEncoder  # noqa: E402

import sklearn.metrics  # noqa: E402


_RAW_ENC_CACHE = {}


def _mlp_ks_raw_data_set(data_set):
    """raw data_set underlying an encoded one, e.g. reddit_stats_mlp_ks_64 -> reddit_stats."""
    for suffix in ("_mlp_ks_64", "_mlp_ks_16", "_mlp_llr_64", "_mlp_ks_8", "_mlp_combined_64", "_mlp_llr", "_mlp_ks"):
        if data_set.endswith(suffix):
            return data_set[: -len(suffix)]
    return None


def _load_raw_feats(path, with_leading_zero, raw_dim=181):
    """Raw feature matrix aligned to the *_mlp_ks files.

    An ``.npy`` already carries the leading zero row; a raw ``.csv`` does not, so a
    leading zero row (phantom edge/node idx 0) is prepended to keep row alignment.
    """
    if path.endswith(".npy"):
        feats = np.load(path, mmap_mode="r").astype(np.float32)
        if with_leading_zero:
            return np.asarray(feats)
        return np.asarray(np.vstack([np.zeros((1, feats.shape[1]), np.float32), feats]))
    with open(path) as f:
        next(f)
        rows = [[float(x) for x in line.strip().split(",")[4:4 + raw_dim]] for line in f]
    feats = np.asarray(rows, dtype=np.float32)
    if with_leading_zero:
        return np.vstack([np.zeros((1, feats.shape[1]), np.float32), feats])
    return feats


def get_raw_feature_encoder(config, args):
    """Return the (FeatureEncoder, aligned raw features) for an encoded data_set.

    Encoder files follow the reduced-dataset naming ``{data_set}.pt`` (e.g.
    ``reddit_stats_mlp_ks_64.pt`` or ``mooc_stats_mlp_ks_8.pt`` under
    ``--raw_data_dir``); ``--encoder_file`` overrides.  Returns ``None`` when the
    data_set is not an encoded one or the encoder/raw features are missing.
    """
    data_set = getattr(config, "data_set", "")
    if _mlp_ks_raw_data_set(data_set) is None:
        return None
    raw_data_dir = args.raw_data_dir or getattr(config, "raw_data_dir", "") or "../raw_data/"
    if not args.encoder_file:
        enc_path = os.path.join(raw_data_dir, "%s.pt" % data_set)
    else:
        enc_path = args.encoder_file if os.path.isabs(args.encoder_file or "") else os.path.join(raw_data_dir, args.encoder_file)
    if not os.path.exists(enc_path):
        print("[skip raw mode] encoder not found: %s" % enc_path)
        return None
    if args.raw_feature_npy:
        feat_path = args.raw_feature_npy
    else:
        npy = os.path.join(getattr(config, "dir_data", ""), "ml2_%s.npy" % _mlp_ks_raw_data_set(data_set))
        feat_path = npy if os.path.exists(npy) else os.path.join(raw_data_dir, "%s.csv" % _mlp_ks_raw_data_set(data_set))
    key = (enc_path, feat_path)
    if key in _RAW_ENC_CACHE:
        return _RAW_ENC_CACHE[key]
    with_leading_zero = feat_path.endswith(".npy")
    raw_dim = int(torch.load(enc_path, map_location="cpu")["0.weight"].shape[1])
    feats = _load_raw_feats(feat_path, with_leading_zero, raw_dim)
    if feats.shape[1] != raw_dim:
        raise ValueError("raw feature file %s has %d dims != encoder input dim %d" %
                         (feat_path, feats.shape[1], raw_dim))
    fit = feats[1:] if with_leading_zero else feats
    fe, _ = FeatureEncoder.from_checkpoint(enc_path, fit)
    _RAW_ENC_CACHE[key] = (fe, feats)
    return _RAW_ENC_CACHE[key]


def raw_stat_feature_count(data_set, args, default=10):
    """Number of trailing statistic columns in the raw feature vector."""
    if args.raw_feature_npy:
        return default
    raw_name = _mlp_ks_raw_data_set(data_set)
    if raw_name is None:
        return default
    raw_data_dir = args.raw_data_dir or "../raw_data/"
    csv_path = os.path.join(raw_data_dir, "%s.csv" % raw_name)
    if not os.path.exists(csv_path):
        return default
    with open(csv_path) as f:
        tokens = f.readline().strip().split(",")
    return max(0, len(tokens) - 5)


class EncTGAT(torch.nn.Module):
    """Prepend the raw->reduced feature encoder to a TGAT model, so the whole
    forward (and SHAP) run on RAW edge features."""

    def __init__(self, encoder, tgat):
        super().__init__()
        self.encoder = encoder
        self.tgat = tgat

    def forward(self, src_edge_feat, src_edge_to_time, src_center_node_idx, src_neigh_edge,
               src_node_features, current_time, label):
        return self.tgat(self.encoder(src_edge_feat), src_edge_to_time, src_center_node_idx,
                      src_neigh_edge, src_node_features, current_time, label)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SAD checkpoints on the test split")
    p.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    p.add_argument("--dirs", nargs="*", default=None,
                   help="explicit model dir(s); default: all dirs in checkpoint_root")
    p.add_argument("--model_file", type=str, default="model.pt")
    p.add_argument("--config_file", type=str, default="config.json")
    p.add_argument("--raw_mode", action="store_true",
                   help="for *_mlp_ks_64 checkpoints: use the raw data_set (features) and embed "
                        "on the fly with the encoder, so importances attribute the raw features "
                        "(split 0:171 embedding / 172:n statistics). Auto-enabled when data_set "
                        "ends in _mlp_ks_64.")
    p.add_argument("--raw_data_dir", type=str, default=None,
                   help="directory with the raw csv/npy and the encoder .pt (default: config.raw_data_dir)")
    p.add_argument("--encoder_file", type=str, default="",
                   help="encoder .pt name within raw_data_dir (default: <raw>_mlp_ks_64.pt)")
    p.add_argument("--raw_feature_npy", type=str, default=None,
                   help="optional preprocessed raw feature .npy (row-aligned, leading zero row); "
                        "default: {dir_data}/ml2_<raw>.npy, else {raw_data_dir}/<raw>.csv")
    p.add_argument("--compute_importances", action="store_true",
                   help="compute SHAP feature importances (optional, off by default)")
    p.add_argument("--shap_max_samples", type=int, default=2000,
                   help="max # correctly classified samples used for SHAP summary")
    p.add_argument("--shap_background_samples", type=int, default=50)
    p.add_argument("--shap_nsamples", type=int, default=30)
    p.add_argument("--split_dim", type=int, default=None,
                   help="input feature dim split: embedding=[0:split_dim], "
                        "statistic=[split_dim:]; default: feat_dim minus the number of "
                        "statistic columns (from the raw header) for *_stats datasets, "
                        "else feat_dim (raw mode: encoder raw dim minus statistic columns)")
    p.add_argument("--pred_threshold", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0, help="cuda device id, -1 for cpu")
    p.add_argument("--num_data_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_device(args):
    if args.gpu is None or args.gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:%d" % args.gpu)


def load_config(model_dir, config_file):
    with open(os.path.join(model_dir, config_file)) as f:
        return types.SimpleNamespace(**json.load(f))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def eval_test(model, test_dataset, collate, config, args, device):
    """Full test-split evaluation; returns roc_auc, pr_auc, pred scores, labels."""
    batch_size = getattr(config, "batch_size", 256)
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_data_workers,
        collate_fn=collate.dyg_collate_fn,
    )
    m_pred, m_label = [], []
    with torch.no_grad():
        for batch in loader:
            x = model(
                batch["src_edge_feat"].to(device),
                batch["src_edge_to_time"].to(device),
                batch["src_center_node_idx"].to(device),
                batch["src_neigh_edge"].to(device),
                batch["src_node_features"].to(device),
                batch["current_time"].to(device),
                batch["labels"].to(device),
            )
            y = batch["labels"].to(device).float()
            m_pred = np.concatenate([m_pred, x["logits"].sigmoid().cpu().numpy().flatten()])
            m_label = np.concatenate([m_label, y.cpu().numpy().flatten()])

    m_pred = np.asarray(m_pred)
    m_label = np.asarray(m_label).astype(np.int64)
    keep = m_label >= 0
    m_pred, m_label = m_pred[keep], m_label[keep]

    roc_auc = sklearn.metrics.roc_auc_score(m_label, m_pred)
    pr_auc = sklearn.metrics.average_precision_score(m_label, m_pred)
    return float(roc_auc), float(pr_auc), m_pred, m_label


class SADShapModel(torch.nn.Module):
    """Wrapper so shap.GradientExplainer can attribute TGAT logits w.r.t. the
    edge-feature matrix of a fixed per-sample graph context.

    forward receives X of shape [B, E, D] (B perturbed copies of one sample's
    context) and returns logits [B, 1].  Each copy is an independent subgraph
    with node ids offset by its index, so per-row outputs are independent.
    """

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.ctx = None

    def set_context(self, ctx):
        self.ctx = ctx

    def forward(self, X):
        if self.ctx is None:
            raise RuntimeError("SADShapModel.set_context() must be called first")
        B = X.shape[0]
        D = X.shape[2]
        src_edge_feat = X.reshape(-1, D)
        ctx = self.ctx
        N = ctx["node_feat"].shape[0]

        neigh_edges = [ctx["neigh_edge"] + j * N for j in range(B)]
        node_feats = [ctx["node_feat"] for _ in range(B)]
        centers = [ctx["center"] + j * N for j in range(B)]
        times = [ctx["current_time"] for _ in range(B)]
        edge_times = [ctx["edge_to_time"] for _ in range(B)]
        labels = torch.ones(B, dtype=torch.float32, device=self.device)

        out = self.model(
            src_edge_feat,
            torch.cat(edge_times, 0),
            torch.cat(centers, 0),
            torch.cat(neigh_edges, 0),
            torch.cat(node_feats, 0),
            torch.cat(times, 0),
            labels,
        )
        return out["logits"].unsqueeze(-1)


def build_context(dataset, collate, idx, device):
    item = collate.dyg_collate_fn([dataset[idx]])
    src_edge_feat = item["src_edge_feat"].to(device)
    ctx = {
        "edge_to_time": item["src_edge_to_time"].to(device),
        "neigh_edge": item["src_neigh_edge"].to(device),
        "node_feat": item["src_node_features"].to(device),
        "center": item["src_center_node_idx"].reshape(1).to(device),
        "current_time": item["current_time"].to(device),
        "E": int(src_edge_feat.shape[0]),
    }
    return ctx, src_edge_feat


def explain_samples(model, device, sample_idx, collate, test_dataset, bg_by_E, rng, args, split_dim):
    """Compute mean |SHAP| per feature dim for the given samples (one class)."""
    feats, ctxs = [], []
    for idx in sample_idx:
        ctx, feat = build_context(test_dataset, collate, int(idx), device)
        ctxs.append(ctx)
        feats.append(feat)

    agg = None
    agg_count = 0
    import shap

    for j in range(len(feats)):
        E = feats[j].shape[0]
        pool = bg_by_E.get(E, [])
        if not pool:
            pool = [feats[j]]
        S = min(args.shap_background_samples, len(pool))
        bg_idx = rng.choice(len(pool), size=S, replace=False)
        bg = torch.stack([pool[i] for i in bg_idx])
        # When the (per-#edges) pool holds only this sample's own features, using it as
        # the SHAP baseline makes (x - baseline) = 0, so GradientExplainer returns ~0
        # importances. Fall back to a mean-row baseline (deviation from the average edge row).
        if S == 1 and torch.equal(bg[0], feats[j]):
            bg = feats[j].mean(dim=0, keepdim=True).expand(1, E, -1).contiguous()
        X = feats[j].reshape(1, E, -1)

        sm = SADShapModel(model, device)
        sm.set_context(ctxs[j])
        explainer = shap.GradientExplainer(sm, bg)
        sv = explainer.shap_values(X, nsamples=args.shap_nsamples)
        shap_arr = np.abs(np.asarray(sv)[0])  # [E, D]
        if agg is None:
            agg = shap_arr.sum(0)
        else:
            agg += shap_arr.sum(0)
        agg_count += E

    if agg is None:
        return None
    per_feature = agg / float(agg_count)  # [D]

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
        "num_samples_used": len(feats),
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


def compute_importances(model, device, test_dataset, collate, args, m_pred, m_label, split_dim):
    """Select correctly classified samples per class and compute SHAP summaries."""
    pos = np.arange(len(m_label))
    correct = (m_pred >= args.pred_threshold).astype(np.int64) == m_label
    correct_normal = pos[np.logical_and(m_label == 0, correct)]
    correct_anomaly = pos[np.logical_and(m_label == 1, correct)]

    rng = np.random.RandomState(args.seed)

    # Always include all correctly classified anomalies, fill the rest with correctly
    # classified normals up to --shap_max_samples.
    selected_anomaly = correct_anomaly[: args.shap_max_samples]
    n_normal = max(0, args.shap_max_samples - len(selected_anomaly))
    selected_normal = rng.permutation(correct_normal)[:n_normal] if n_normal > 0 else np.zeros(0, dtype=np.int64)

    all_idx = np.concatenate([selected_normal, selected_anomaly]).astype(np.int64)

    # shared background pool built from all selected samples (both classes)
    bg_by_E = {}
    D = None
    for idx in all_idx:
        _, feat = build_context(test_dataset, collate, int(idx), device)
        bg_by_E.setdefault(feat.shape[0], []).append(feat)
        if D is None:
            D = feat.shape[1]
    if D is None:
        D = 0

    emb_end = min(split_dim, D) if D else split_dim
    imp = {
        "enabled": True,
        "limit": args.shap_max_samples,
        "split_dim": split_dim,
        "feature_dim": D,
        "split": {"embedding": [0, emb_end], "statistic": [emb_end, D]},
    }
    if len(all_idx) > 0:
        if len(selected_normal) > 0:
            cls0 = explain_samples(model, device, selected_normal, collate, test_dataset, bg_by_E, rng, args, split_dim)
            if cls0 is not None:
                cls0["num_correct_normals"] = int(len(correct_normal))
                imp["class_normal"] = cls0
        if len(selected_anomaly) > 0:
            cls1 = explain_samples(model, device, selected_anomaly, collate, test_dataset, bg_by_E, rng, args, split_dim)
            if cls1 is not None:
                cls1["num_correct_anomalies"] = int(len(correct_anomaly))
                imp["class_ano"] = cls1
        cls_all = explain_samples(model, device, all_idx, collate, test_dataset, bg_by_E, rng, args, split_dim)
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
    device = get_device(args)
    set_seed(getattr(config, "seed", args.seed))

    raw_mode = args.raw_mode or _mlp_ks_raw_data_set(
        getattr(config, "data_set", "")) is not None

    split_list = [getattr(config, "train_split", 0.7), getattr(config, "val_split", 0.15), getattr(config, "test_split", 0.15)]
    test_dataset = dataset.DygDataset(config, "test", split_list=split_list)
    collate = dataset.Collate(config)

    config.input_dim = dataset.get_feature_dim(config)
    model = TGAT(config, device)

    encoder = None
    if raw_mode:
        enc = get_raw_feature_encoder(config, args)
        if enc is None:
            print("[skip raw mode] no raw features/encoder available for %s" % config.data_set)
            return
        encoder, raw_feats = enc
        test_dataset.edge_features = raw_feats
    model.load_state_dict(torch.load(mod_path, map_location=device))
    if raw_mode:
        model = EncTGAT(encoder, model)
    model.to(device).eval()
    # Evaluation uses no drop-edge augmentation: run the pure forward (also avoids the
    # empty augmented-graph edge case hit by single-sample SHAP contexts).
    getattr(model, "tgat", model).augmentation = False

    roc_auc, pr_auc, m_pred, m_label = eval_test(model, test_dataset, collate, config, args, device)

    result = {
        "checkpoint": os.path.basename(model_dir),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    if args.compute_importances:
        if raw_mode:
            split_dim = args.split_dim if args.split_dim is not None else \
                int(encoder.raw_dim) - raw_stat_feature_count(getattr(config, "data_set", ""), args)
        else:
            split_dim = args.split_dim if args.split_dim is not None else \
                (int(config.input_dim) - raw_stat_feature_count(getattr(config, "data_set", ""), args)
                 if "stats" in getattr(config, "data_set", "") else int(config.input_dim))
        result["importance"] = compute_importances(
            model, device, test_dataset, collate, args, m_pred, m_label, split_dim
        )

    with open(os.path.join(model_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[ok] %s -> roc_auc=%.4f pr_auc=%.4f (result.json written)" % (os.path.basename(model_dir), roc_auc, pr_auc))


def main():
    args = parse_args()
    if args.dirs:
        model_dirs = [d if os.path.isdir(d) else os.path.join(args.checkpoint_root, d) for d in args.dirs]
    else:
        model_dirs = [
            os.path.join(args.checkpoint_root, d)
            for d in sorted(os.listdir(args.checkpoint_root))
            if os.path.isdir(os.path.join(args.checkpoint_root, d))
        ]
    if not model_dirs:
        print("no model dirs found in %s" % args.checkpoint_root)
        return
    for d in model_dirs:
        evaluate_model_dir(d, args)


if __name__ == "__main__":
    main()
