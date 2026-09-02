#!/usr/bin/env python3
"""Evaluate trained DyGLib node-classification checkpoints on the test split.

Run from the DyGLib directory:

    # score all saved models in ./checkpoints (ROC-AUC / PR-AUC only)
    python test.py

    # also compute gradient SHAP feature importances (embedding vs statistic dims)
    python test.py --compute_importances

    # explain one specific saved model dir
    python test.py --dirs node_classification_TGN_seed0 --compute_importances

For each model a result.json is written into the model's checkpoint dir with
roc_auc / pr_auc and, if requested, per-class SHAP importances split into
embedding ([0:split_dim]) and statistic ([split_dim:]) shares.

SHAP target: the classifier input - the *aggregated* source-node temporal
embedding (the encoder output consumed by the MLP classifier). This is the
well-defined differentiable input for TGAT / TGN / DyGFormer. Candidate edge
features only reach TGN memory through non-differentiable in-place updates, so
edge-feature gradient attribution is not supported.
"""

import argparse
import json
import os
import types

import numpy as np
import torch
import torch.nn as nn

import sklearn.metrics

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.DyGFormer import DyGFormer
from models.modules import MLPClassifier
from utils.utils import get_neighbor_sampler
from utils.feature_encoder import FeatureEncoder
from utils.DataLoader import get_node_classification_data


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DyGLib node-classification checkpoints on the test split"
    )
    p.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    p.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        help="explicit model dir(s); default: all dirs in checkpoint_root",
    )
    p.add_argument("--model_file", type=str, default="model.pt")
    p.add_argument("--config_file", type=str, default="config.json")
    p.add_argument(
        "--importance",
        "--compute_importances",
        dest="compute_importances",
        action="store_true",
        help="compute SHAP feature importances (optional, off by default)",
    )
    p.add_argument("--shap_max_samples", type=int, default=2000)
    p.add_argument("--shap_background_samples", type=int, default=50)
    p.add_argument("--shap_nsamples", type=int, default=30)
    p.add_argument(
        "--split_dim",
        type=int,
        default=None,
        help="feature split: embedding=[0:split_dim], statistic=[split_dim:]; "
        "default node feat dim minus statistic columns for *_stats",
    )
    p.add_argument(
        "--raw_data_dir",
        type=str,
        default="../raw_data/",
        help="directory with the raw CSVs and the saved *_mlp_ks_*.pt encoders",
    )
    p.add_argument(
        "--encoder_file",
        type=str,
        default="",
        help="encoder .pt within raw_data_dir (default: <data_set>.pt)",
    )
    p.add_argument("--pred_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(model_dir, filename):
    with open(os.path.join(model_dir, filename)) as f:
        return types.SimpleNamespace(**json.load(f))


def raw_data_set(data_set):
    for suffix in (
        "_mlp_ks_64",
        "_mlp_ks_16",
        "_mlp_llr_64",
        "_mlp_ks_8",
        "_mlp_combined_64",
        "_mlp_llr",
        "_mlp_ks",
    ):
        if data_set.endswith(suffix):
            return data_set[: -len(suffix)]
    return None


def raw_stat_feature_count(data_set, args, default=10):
    raw_name = raw_data_set(data_set)
    if raw_name is None or not args.raw_data_dir:
        return default
    csv_path = os.path.join(args.raw_data_dir, "%s.csv" % raw_name)
    if not os.path.exists(csv_path):
        return default
    with open(csv_path) as f:
        tokens = f.readline().strip().split(",")
    return max(0, len(tokens) - 5)


MEMORY_MODELS = ("JODIE", "DyRep", "TGN")


# ---------------------------------------------------------------------- model


def build_model(cfg, node_features, edge_features, train_data, sampler, device):
    if cfg.model_name == "TGAT":
        backbone = TGAT(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=sampler,
            time_feat_dim=cfg.time_feat_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            device=device,
        )
    elif cfg.model_name in MEMORY_MODELS:
        stats = compute_src_dst_node_time_shifts(
            train_data.src_node_ids,
            train_data.dst_node_ids,
            train_data.node_interact_times,
        )
        backbone = MemoryModel(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=sampler,
            time_feat_dim=cfg.time_feat_dim,
            model_name=cfg.model_name,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            src_node_mean_time_shift=stats[0],
            src_node_std_time_shift=stats[1],
            dst_node_mean_time_shift_dst=stats[2],
            dst_node_std_time_shift=stats[3],
            device=device,
        )
    elif cfg.model_name == "DyGFormer":
        backbone = DyGFormer(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=sampler,
            time_feat_dim=cfg.time_feat_dim,
            channel_embedding_dim=cfg.channel_embedding_dim,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            max_input_sequence_length=cfg.max_input_sequence_length,
            device=device,
        )
    else:
        raise ValueError("Unsupported model_name %s" % cfg.model_name)
    classifier = MLPClassifier(input_dim=node_features.shape[1], dropout=cfg.dropout)
    return nn.Sequential(backbone, classifier)


def load_weights(model, model_dir, args, device):
    state = torch.load(os.path.join(model_dir, args.model_file), map_location=device)
    model.load_state_dict(state)
    model.to(device)


# ----------------------------------------------------------------- evaluation


def forward_embeddings(model, cfg, src, dst, times, edge_ids, device):
    if cfg.model_name in MEMORY_MODELS:
        emb, _ = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=src,
            dst_node_ids=dst,
            node_interact_times=times,
            edge_ids=edge_ids,
            edges_are_positive=True,
            num_neighbors=cfg.num_neighbors,
        )
    elif cfg.model_name in ("TGAT", "CAWN", "TCL"):
        emb, _ = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=src,
            dst_node_ids=dst,
            node_interact_times=times,
            num_neighbors=cfg.num_neighbors,
        )
    elif cfg.model_name == "GraphMixer":
        emb, _ = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=src,
            dst_node_ids=dst,
            node_interact_times=times,
            num_neighbors=cfg.num_neighbors,
            time_gap=cfg.time_gap,
        )
    elif cfg.model_name == "DyGFormer":
        emb, _ = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=dst, node_interact_times=times
        )
    else:
        raise ValueError(cfg.model_name)
    return emb


def evaluate_test(model, cfg, test, device):
    model.eval()
    if cfg.model_name in MEMORY_MODELS:
        model[0].memory_bank.node_memories.data.zero_()
        model[0].memory_bank.node_last_updated_times.data.zero_()

    preds, labels = [], []
    with torch.no_grad():
        for s in range(0, len(test.src_node_ids), cfg.batch_size):
            e = min(s + cfg.batch_size, len(test.src_node_ids))
            emb = forward_embeddings(
                model,
                cfg,
                test.src_node_ids[s:e],
                test.dst_node_ids[s:e],
                test.node_interact_times[s:e],
                test.edge_ids[s:e],
                device,
            )
            preds.append(model[1](x=emb).squeeze(dim=-1).sigmoid().cpu().numpy())
            labels.append(test.labels[s:e])

    m_pred = np.concatenate(preds)
    m_label = np.concatenate(labels)
    keep = m_label >= 0
    m_pred, m_label = m_pred[keep], m_label[keep].astype(np.int64)

    roc = float("nan")
    try:
        roc = float(sklearn.metrics.roc_auc_score(m_label, m_pred))
    except ValueError:
        pass
    pr = float(sklearn.metrics.average_precision_score(m_label, m_pred))
    return m_pred, m_label, roc, pr


def sample_embedding(model, cfg, src, dst, times, edge_ids, device):
    model.eval()
    if cfg.model_name in MEMORY_MODELS:
        model[0].memory_bank.node_memories.data.zero_()
        model[0].memory_bank.node_last_updated_times.data.zero_()
    emb = forward_embeddings(
        model,
        cfg,
        np.asarray([src]),
        np.asarray([dst]),
        np.asarray([times]),
        np.asarray([edge_ids]),
        device,
    )
    return emb[0].detach().cpu().numpy()


# --------------------------------------------------------- SHAP importances


def load_raw_encoder(data_set_name, args):
    """Load the saved *_mlp_ks_*.pt FeatureEncoder plus the RAW ``*_stats`` features.

    Returns (encoder, raw_node_features, raw_edge_features) or None when the
    data_set is not an encoded one / the encoder is missing.
    """
    if raw_data_set(data_set_name) is None or not args.raw_data_dir:
        return None
    enc_path = args.encoder_file or os.path.join(
        args.raw_data_dir, "%s.pt" % data_set_name
    )
    if not os.path.exists(enc_path):
        print("[raw mode] encoder not found: %s" % enc_path)
        return None
    raw_name = raw_data_set(data_set_name)
    try:
        node_raw, edge_raw, *_ = get_node_classification_data(
            dataset_name=raw_name, val_ratio=0.15, test_ratio=0.15, min_feature_dim=0
        )
        encoder = FeatureEncoder.from_checkpoint(enc_path, edge_raw[1:])
    except Exception as exc:
        print("[raw mode] skipped for %s: %s" % (data_set_name, exc))
        return None
    return encoder, node_raw, edge_raw


class EdgeRawShapModel(nn.Module):
    """SAD-style per-sample wrapper: logit w.r.t. the RAW edge-feature vector of the
    candidate interaction, encoded on the fly by the saved FeatureEncoder."""

    def __init__(self, model, cfg, encoder, device):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.ctx = None

    def set_context(self, src, dst, times, edge_id):
        self.ctx = (src, dst, times, int(edge_id))

    def _set_edge_features(self, feats):
        """Replace every edge-feature table the ensemble reads from.

        Memory models (TGN / DyRep) copy ``edge_raw_features`` into the graph-
        attention embedding module at construction, so assigning only the top
        level attribute leaves the actually-consumed tensor untouched and the
        SHAP gradients empty.  Update all copies.
        """
        self.model[0].edge_raw_features = feats
        for module in self.model[0].modules():
            if module is not self.model[0] and hasattr(module, "edge_raw_features"):
                module.edge_raw_features = feats

    def forward(self, X):
        src, dst, times, eid = self.ctx
        self.encoder = self.encoder.to(self.device)
        enc = self.encoder(X)  # [B, latent_dim]
        base = self.model[0].edge_raw_features
        outs = []
        for b in range(X.shape[0]):
            feats = torch.cat([base[:eid], enc[b : b + 1], base[eid + 1 :]], dim=0)
            self._set_edge_features(feats)
            if self.cfg.model_name in MEMORY_MODELS:
                self.model[0].memory_bank.node_memories.data.zero_()
                self.model[0].memory_bank.node_last_updated_times.data.zero_()
            emb = forward_embeddings(
                self.model,
                self.cfg,
                np.asarray([src]),
                np.asarray([dst]),
                np.asarray([times]),
                np.asarray([eid]),
                self.device,
            )
            self._set_edge_features(base)
            outs.append(self.model[1](x=emb))
        logits = torch.cat(outs, dim=0)
        # The candidate's own edge feature is, for these models, only consumed
        # through the neighbour aggregation of the *next* interaction, so the
        # gradient w.r.t. X can legitimately be empty for a sample. SHAP's
        # autograd.grad() raises in that case ("not used in the graph"); the
        # zero-coefficient term keeps X functionally present so the explainer
        # assigns a correct zero attribution instead of crashing.
        return logits + X.sum(dim=1, keepdim=True) * 0.0


def compute_importances_raw(
    model, cfg, test, m_pred, m_label, raw_tup, split_dim, args, device
):
    import shap

    encoder, _raw_node, raw_edge = raw_tup
    raw_name = raw_data_set(cfg.dataset_name)
    raw_dim = int(encoder.raw_dim)

    imp = {
        "enabled": True,
        "limit": args.shap_max_samples,
        "target": "raw_%s_via_encoder" % raw_name,
        "split_dim": split_dim,
        "feature_dim": raw_dim,
        "split": {
            "embedding": [0, min(split_dim, raw_dim)],
            "statistic": [min(split_dim, raw_dim), raw_dim],
        },
    }

    pos = np.arange(len(m_label))
    correct = (m_pred >= args.pred_threshold).astype(np.int64) == m_label
    correct_normal = pos[np.logical_and(m_label == 0, correct)]
    correct_anomaly = pos[np.logical_and(m_label == 1, correct)]

    selected_anomaly = correct_anomaly[: args.shap_max_samples]
    n_normal = max(0, args.shap_max_samples - len(selected_anomaly))
    rng = np.random.RandomState(args.seed)
    selected_normal = (
        rng.permutation(correct_normal)[:n_normal]
        if n_normal > 0
        else np.zeros(0, dtype=np.int64)
    )

    sel_n = selected_normal.astype(np.int64)
    sel_a = selected_anomaly.astype(np.int64)

    def raw_vec(sample_id):
        return raw_edge[test.edge_ids[sample_id]].astype(np.float32)

    def explain_class(cls_idx, bg_idx):
        if len(cls_idx) == 0:
            return None
        acc = None
        cnt = 0
        bg_rows = [raw_vec(int(j)) for j in bg_idx] if len(bg_idx) else []
        bg_rows = bg_rows or [raw_vec(int(cls_idx[0]))]
        bg = (
            torch.from_numpy(np.stack(bg_rows)[: args.shap_background_samples])
            .float()
            .to(device)
        )
        for sample_id in cls_idx:
            i = int(sample_id)
            src, dst = test.src_node_ids[i], test.dst_node_ids[i]
            ts, eid = test.node_interact_times[i], test.edge_ids[i]
            X = torch.from_numpy(raw_vec(i)[None]).float().to(device)
            sm = EdgeRawShapModel(model, cfg, encoder, device)
            sm.set_context(src, dst, ts, eid)
            explainer = shap.GradientExplainer(sm, [bg])
            sv = explainer.shap_values([X], nsamples=1)
            arr = np.asarray(sv)[0]
            if arr.ndim == 2:
                arr = arr[0]
            acc = arr if acc is None else acc + np.abs(arr)
            cnt += 1
        per_feature = np.abs(acc) / cnt
        d = summarize(per_feature, split_dim)
        return d

    if sel_n.size:
        d = explain_class(sel_n, sel_a)
        if d is not None:
            d["num_samples_used"] = int(len(sel_n))
            d["num_correct_normals"] = int(len(correct_normal))
            imp["class_normal"] = d
    if sel_a.size:
        d = explain_class(sel_a, sel_n)
        if d is not None:
            d["num_samples_used"] = int(len(sel_a))
            d["num_correct_anomalies"] = int(len(correct_anomaly))
            imp["class_ano"] = d
    if len(sel_n) or len(sel_a):
        d = explain_class(
            np.concatenate([sel_n, sel_a]), np.concatenate([sel_n, sel_a])
        )
        if d is not None:
            d["num_samples_used"] = int(len(sel_n)) + int(len(sel_a))
            d["num_correct_samples"] = int(len(correct_normal)) + int(
                len(correct_anomaly)
            )
            imp["class_all"] = d
    return imp


class ClassifierShapModel(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, X):
        return self.classifier(X)


def aggregate_shap(cls_emb, bg_emb, classifier, args):
    import shap

    cls_emb = np.asarray(cls_emb)
    bg_emb = np.asarray(bg_emb)
    if cls_emb.ndim != 2 or cls_emb.shape[0] == 0 or cls_emb.shape[1] == 0:
        return None
    if bg_emb.ndim != 2 or bg_emb.shape[0] == 0 or bg_emb.shape[1] != cls_emb.shape[1]:
        bg_emb = cls_emb
    X = torch.from_numpy(cls_emb.astype(np.float32))
    bg = torch.from_numpy(bg_emb.astype(np.float32))
    device = next(classifier.parameters()).device
    X, bg = X.to(device), bg.to(device)
    if len(bg) > args.shap_background_samples:
        rng = np.random.RandomState(args.seed)
        bg = bg[rng.choice(len(bg), size=args.shap_background_samples, replace=False)]
    explainer = shap.GradientExplainer(ClassifierShapModel(classifier), [bg])
    sv = explainer.shap_values([X], nsamples=min(args.shap_nsamples, X.shape[0]))
    return np.abs(np.asarray(sv)).mean(axis=0)


def summarize(per_feature, split_dim):
    n_emb = min(len(per_feature), max(0, split_dim))
    embedding_part = per_feature[:n_emb]
    statistic_part = per_feature[n_emb:]

    def _stats(a):
        return (
            (float(a.min()), float(a.max()), float(a.std()))
            if len(a)
            else (0.0, 0.0, 0.0)
        )

    emin, emax, estd = _stats(embedding_part)
    smin, smax, sstd = _stats(statistic_part)
    embedding_importance = float(embedding_part.mean()) if len(embedding_part) else 0.0
    statistic_importance = float(statistic_part.mean()) if len(statistic_part) else 0.0
    total = embedding_importance + statistic_importance
    if total > 0:
        emb_share, stat_share = (
            embedding_importance / total,
            statistic_importance / total,
        )
    else:
        emb_share, stat_share = 0.0, 0.0
    return {
        "embedding_importance": embedding_importance,
        "statistic_importance": statistic_importance,
        "embedding_min": emin,
        "embedding_max": emax,
        "embedding_std": estd,
        "statistic_min": smin,
        "statistic_max": smax,
        "statistic_std": sstd,
        "embedding_share": round(emb_share, 4),
        "statistic_share": round(stat_share, 4),
        "score": "embedding: %.0f%%, statistic: %.0f%%"
        % (emb_share * 100, stat_share * 100),
        "mean_abs_shap_per_feature": per_feature.tolist(),
    }


def compute_importances(model, cfg, test, m_pred, m_label, split_dim, args, device):
    raw = load_raw_encoder(cfg.dataset_name, args)
    if raw is not None:
        raw_encoder = raw[0]
        if args.split_dim is not None:
            split_dim = args.split_dim
        else:
            split_dim = int(raw_encoder.raw_dim)
            if "stats" in cfg.dataset_name:
                split_dim = max(
                    0, split_dim - raw_stat_feature_count(cfg.dataset_name, args)
                )
        return compute_importances_raw(
            model, cfg, test, m_pred, m_label, raw, split_dim, args, device
        )

    pos = np.arange(len(m_label))
    correct = (m_pred >= args.pred_threshold).astype(np.int64) == m_label

    correct_normal = pos[np.logical_and(m_label == 0, correct)]
    correct_anomaly = pos[np.logical_and(m_label == 1, correct)]

    selected_anomaly = correct_anomaly[: args.shap_max_samples]
    n_normal = max(0, args.shap_max_samples - len(selected_anomaly))
    rng = np.random.RandomState(args.seed)
    selected_normal = (
        rng.permutation(correct_normal)[:n_normal]
        if n_normal > 0
        else np.zeros(0, dtype=np.int64)
    )

    sel_n = selected_normal.astype(np.int64)
    sel_a = selected_anomaly.astype(np.int64)

    def emb(sample_id):
        return sample_embedding(
            model,
            cfg,
            test.src_node_ids[sample_id],
            test.dst_node_ids[sample_id],
            test.node_interact_times[sample_id],
            test.edge_ids[sample_id],
            device,
        )

    emb_n = np.stack([emb(j) for j in sel_n]) if len(sel_n) else np.zeros((0, 0))
    emb_a = np.stack([emb(j) for j in sel_a]) if len(sel_a) else np.zeros((0, 0))
    if emb_n.shape[0] and emb_a.shape[0]:
        emb_all = np.concatenate([emb_n, emb_a])
    elif emb_n.shape[0]:
        emb_all = emb_n
    elif emb_a.shape[0]:
        emb_all = emb_a
    else:
        emb_all = np.zeros((0, 0))

    D = emb_n.shape[1] if emb_n.shape[0] else (emb_a.shape[1] if emb_a.shape[0] else 0)
    emb_end = min(split_dim, D) if D else split_dim
    imp = {
        "enabled": True,
        "limit": args.shap_max_samples,
        "split_dim": split_dim,
        "feature_dim": D,
        "split": {"embedding": [0, emb_end], "statistic": [emb_end, D]},
        "target": "aggregated_embedding",
    }

    classifier = model[1]

    def explain(cls_emb, bg_emb, counts):
        per_feature = aggregate_shap(cls_emb, bg_emb, classifier, args)
        if per_feature is None:
            return None
        d = summarize(per_feature, split_dim)
        d.update(counts)
        d["num_samples_used"] = int(cls_emb.shape[0])
        return d

    if emb_n.shape[0]:
        imp["class_normal"] = explain(
            emb_n, emb_a, {"num_correct_normals": int(len(correct_normal))}
        )
    if emb_a.shape[0]:
        imp["class_ano"] = explain(
            emb_a, emb_n, {"num_correct_anomalies": int(len(correct_anomaly))}
        )
    if emb_all.shape[0]:
        imp["class_all"] = explain(
            emb_all,
            emb_all,
            {
                "num_correct_samples": int(len(correct_normal))
                + int(len(correct_anomaly))
            },
        )
    return imp


# ---------------------------------------------------------------------- main


def evaluate_model(model_dir, args):
    cfg_path = os.path.join(model_dir, args.config_file)
    if not (
        os.path.exists(cfg_path)
        and os.path.exists(os.path.join(model_dir, args.model_file))
    ):
        print("[skip] %s missing model/config" % model_dir)
        return

    cfg = load_config(model_dir, args.config_file)
    device = get_device()
    set_seed(getattr(cfg, "seed", args.seed))

    node_features, edge_features, full_data, train_data, _val_data, test_data = (
        get_node_classification_data(
            dataset_name=cfg.dataset_name,
            val_ratio=0.15,
            test_ratio=0.15,
            min_feature_dim=getattr(cfg, "min_feat_dim", 0),
        )
    )

    sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=getattr(cfg, "sample_neighbor_strategy", "recent"),
        time_scaling_factor=getattr(cfg, "time_scaling_factor", 0.0),
        seed=1,
    )

    model = build_model(cfg, node_features, edge_features, train_data, sampler, device)
    load_weights(model, model_dir, args, device)
    model.to(device).eval()

    m_pred, m_label, roc, pr = evaluate_test(model, cfg, test_data, device)

    result = {
        "checkpoint": os.path.basename(model_dir),
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "seed": int(getattr(cfg, "seed", args.seed)),
        "roc_auc": roc,
        "pr_auc": pr,
    }

    if args.compute_importances:
        if args.split_dim is not None:
            split_dim = args.split_dim
        else:
            split_dim = int(node_features.shape[1])
            if "stats" in cfg.dataset_name:
                split_dim = max(
                    0, split_dim - raw_stat_feature_count(cfg.dataset_name, args)
                )
        result["importance"] = compute_importances(
            model, cfg, test_data, m_pred, m_label, split_dim, args, device
        )

    with open(os.path.join(model_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(
        "[ok] %s -> roc_auc=%.4f pr_auc=%.4f (result.json written)"
        % (os.path.basename(model_dir), roc, pr)
    )


def main():
    args = parse_args()
    if args.dirs:
        model_dirs = [
            d if os.path.isdir(d) else os.path.join(args.checkpoint_root, d)
            for d in args.dirs
        ]
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
        evaluate_model(d, args)


if __name__ == "__main__":
    main()
