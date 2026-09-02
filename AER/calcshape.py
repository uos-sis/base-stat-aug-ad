#!/usr/bin/env python3
"""SHAP importance for the saved AER best models (per seed, then mean +- std).

Mirrors the evaluation semantics of the GraphSAGE torch_test.py / SAD test.py
SHAP pipelines: for each saved seed model the test split is scored, only
*correctly classified* events are kept (all correct anomalies first, then
correct normals up to --shap_max_samples), and the edge-feature path is
explained per event with an expected-gradients estimator (equivalent to
shap.GradientExplainer).  Importances are aggregated per class
(normal / anomaly / all) into an embedding vs statistic split (--split_dim).

Outputs (per dataset-feets, in --outdir):

  result.json                     per-class embedding/statistic importance + shares, +- std over seeds
  feature_shap_seeds.csv          mean |SHAP| per feature dim, per class, +- std over seeds
  group_shares.csv                embedding vs statistic share per class (mean +- std over seeds)
  shap_seed<seed>_class<name>.npy raw |SHAP| values (n_samples, step, D) per seed and class

Examples:
  python calcshape.py -d mooc --feets stats --seeds 0 1 2 3
  python calcshape.py -d reddit --feets statsdim --shap_max_samples 500 \
      --shap_nsamples 20 --shap_background_samples 40
"""

import os
import sys
import glob

# ----------------------------------------------------------------------------
# bootstrap: run from within the AER project directory (graph/graph.so and the
# model paths are relative to it) and make its modules importable.
# ----------------------------------------------------------------------------
AER_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(AER_DIR):
    raise SystemExit("AER directory not found: {}".format(AER_DIR))
os.chdir(AER_DIR)
sys.path.insert(0, AER_DIR)

# remove any compiled graph modules from cache
if "graph" in sys.modules:
    del sys.modules["graph"]
if "graph.graph" in sys.modules:
    del sys.modules["graph.graph"]

HistoryFinder = None
try:
    import importlib.util

    graph_file_path = os.path.join(AER_DIR, "graph", "graph.py")
    if os.path.exists(graph_file_path):
        spec = importlib.util.spec_from_file_location("graph_module", graph_file_path)
        graph_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_module)
        HistoryFinder = graph_module.HistoryFinder
    else:
        raise ImportError("graph.py file not found")
except Exception as exc:
    print("Error importing HistoryFinder: {}".format(exc))
    sys.exit(1)

from parserpara import *
from utils import *
from module_sample import RAPAD
from loaddata import loadDropoutData, DATA_DIR
from sklearn.metrics import roc_auc_score, average_precision_score
import json

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

_INTER_GRAPH = None


class SHAPPredictor(nn.Module):
    """Differentiable re-implementation of the model's forward path starting
    from the (B, step, D) neighbour-edge features X.

    The structural tensors (s_vector, s_partner, d_partner) are computed once
    from the graph and kept fixed; only the edge-feature path is explained.
    """

    def __init__(self, rap, s_vector, s_partner, d_partner, encoder=None):
        super(SHAPPredictor, self).__init__()
        self.rap = rap
        self.encoder = encoder
        self.register_buffer("s_vector", s_vector.detach())
        self.register_buffer("s_partner", s_partner.detach())
        self.register_buffer("d_partner", d_partner.detach())

    def forward(self, X):
        if self.encoder is not None:
            X = self.encoder(X)  # raw statsdim features -> reduced mlp features
        edge_features = self.rap.corr_encoder.trainable_embedding_feat(X)  # (B, step, D)

        base = self.s_partner.shape[0]
        n = X.shape[0]
        if n != base:
            if base == 0 or n % base != 0:
                raise RuntimeError(
                    "SHAPPredictor: fixed structural batch {} is not expandable to {}".format(base, n)
                )
            k = n // base
            s_vector = self.s_vector.repeat_interleave(k, dim=0)
            s_partner = self.s_partner.repeat_interleave(k, dim=0)
            d_partner = self.d_partner.repeat_interleave(k, dim=0)
        else:
            s_vector = self.s_vector
            s_partner = self.s_partner
            d_partner = self.d_partner

        hk_center_s = self.rap.edge_cent(s_partner.mean(dim=-2))
        hk_user = self.rap.edge_s(s_vector)
        edge_feature_s = self.rap.edge_sd(torch.cat([hk_user, hk_center_s], dim=-1))

        hk_center_d = self.rap.edge_d(d_partner.mean(dim=-2))
        edge_feature_d = torch.cat([hk_center_d], dim=-1)

        edge_feature = torch.cat([edge_feature_s, edge_feature_d, edge_features], dim=-1)
        edge_feature = self.rap.edge_com(edge_feature)

        xr = edge_feature.permute([1, 0, 2])  # [step, B, dim]
        encoded = self.rap.encoder_model(xr)[0]
        encoded = self.rap.dropout(encoded.permute([1, 0, 2]))
        encoded = self.rap.projector(encoded)
        encoded = self.rap.pooler(encoded, agg="mean")
        score = self.rap.out_sequential(encoded).squeeze(-1)
        return score


def expected_gradients_shap(predictor, x, background, steps=20):
    """Expected-gradients SHAP (same estimator as shap.GradientExplainer).

    Args:
        x: (B, step, D) tensor of explained events (on model device)
        background: (K, step, D) tensor of background events
    Returns:
        (B, step, D) numpy array of SHAP values
    """
    b, s, d = x.shape
    k = background.shape[0]
    xb = background.detach().unsqueeze(0)  # (1, K, S, D)
    xx = x.detach().unsqueeze(1)           # (B, 1, S, D)
    grads = None
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False  # native RNN impl supports backward in eval mode
    try:
        for a in torch.linspace(0.0, 1.0, steps, device=x.device):
            xa = (xb + a * (xx - xb)).reshape(b * k, s, d).requires_grad_(True)
            y = predictor(xa)
            g = torch.autograd.grad(y.sum(), xa)[0]
            g = g.detach().reshape(b, k, s, d)
            grads = g if grads is None else grads + g
    finally:
        torch.backends.cudnn.enabled = prev_cudnn
    grads = grads / steps                          # (B, K, S, D)
    shap_vals = ((xx - xb) * grads).mean(dim=1)  # (B, S, D)
    return shap_vals.detach().cpu().numpy()


class FeatureEncoder(nn.Module):
    """raw stats features -> reduced mlp_ks features, differentiable w.r.t. raw.

    Reconstructs the transform used to build the *_stats_mlp_ks_* datasets
    (same as GraphSAGE graphsage/feature_encoder.py)::

        reduced = ( MLP( StandardScaler(raw) ) - latent_mu ) / latent_sd

    so SHAP can attribute the RAW input features (embedding + statistics) of a
    statsdim model instead of the opaque mlp-compressed vector.
    """

    def __init__(self, encoder_state, fit_feats):
        super(FeatureEncoder, self).__init__()
        self.register_buffer("w0", torch.tensor(np.asarray(encoder_state["0.weight"], dtype=np.float32)))
        self.register_buffer("b0", torch.tensor(np.asarray(encoder_state["0.bias"], dtype=np.float32)))
        self.has_hidden = "3.weight" in encoder_state
        if self.has_hidden:
            self.register_buffer("w3", torch.tensor(np.asarray(encoder_state["3.weight"], dtype=np.float32)))
            self.register_buffer("b3", torch.tensor(np.asarray(encoder_state["3.bias"], dtype=np.float32)))
            self.latent_dim = int(self.w3.shape[0])
        else:
            self.latent_dim = int(self.w0.shape[0])
        self.raw_dim = int(self.w0.shape[1])

        fit_feats = np.asarray(fit_feats, dtype=np.float64)
        if fit_feats.ndim == 1:
            fit_feats = fit_feats[None, :]
        if fit_feats.shape[1] != self.raw_dim:
            raise ValueError(
                "raw feature dim %d != encoder input dim %d" % (fit_feats.shape[1], self.raw_dim)
            )

        raw_mean = fit_feats.mean(axis=0)
        raw_std = fit_feats.std(axis=0)
        raw_scale = np.where(raw_std < 1e-6, 1.0, raw_std)

        scaled = ((fit_feats - raw_mean) / raw_scale).astype(np.float32)
        with torch.no_grad():
            z = self._mlp(torch.from_numpy(scaled)).numpy().astype(np.float64)
        lat_mu = z.mean(axis=0)
        lat_sd = z.std(axis=0)
        lat_scale = np.where(lat_sd < 1e-6, 1e-6, lat_sd)

        self.register_buffer("raw_mean", torch.tensor(raw_mean, dtype=torch.float32))
        self.register_buffer("raw_scale", torch.tensor(raw_scale, dtype=torch.float32))
        self.register_buffer("lat_mu", torch.tensor(lat_mu, dtype=torch.float32))
        self.register_buffer("lat_scale", torch.tensor(lat_scale, dtype=torch.float32))
        self.eval()

    @staticmethod
    def from_checkpoint(encoder_path, fit_feats):
        state = torch.load(encoder_path, map_location="cpu")
        if not (isinstance(state, dict) and "0.weight" in state and "0.bias" in state):
            raise ValueError("encoder %s does not look like a FeatureEncoder state dict" % encoder_path)
        return FeatureEncoder(state, fit_feats)

    def _mlp(self, x):
        h = x @ self.w0.t() + self.b0
        if not self.has_hidden:
            return h
        h = torch.relu(h)
        return h @ self.w3.t() + self.b3

    def forward(self, x):
        x = x.float()
        is_zero = x.detach().abs().sum(dim=-1, keepdim=True) == 0
        scaled = (x - self.raw_mean) / self.raw_scale
        out = (self._mlp(scaled) - self.lat_mu) / self.lat_scale
        return out.masked_fill(is_zero.expand_as(out), 0.0)


def load_raw_encoder(dataset):
    """Load the raw-mode encoder + full-range raw stats feature table.

    The encoder scaler/latent statistics are fitted on the ORIGINAL stats
    features (features_stats.npy), which reproduces the model's mlp features
    exactly; the returned feature TABLE is the contrastive variant
    (features_cont_full_stats.npy) so it covers the full edge-index range
    (original + synthetic edges) used by the model inference table.
    """
    fit_path = os.path.join(DATA_DIR, dataset, "features_stats.npy")
    cont_path = os.path.join(DATA_DIR, dataset, "AER", "features_cont_full_stats.npy")
    if not os.path.isfile(fit_path):
        print("note: raw stats features not found at {} -> raw mode disabled".format(fit_path))
        return None, None
    enc_dir = "../raw_data"
    candidates = sorted(glob.glob(os.path.join(enc_dir, "{}_stats_mlp_ks*.pt".format(dataset))))
    if not candidates:
        print("note: no encoder found in {} -> raw mode disabled".format(enc_dir))
        return None, None
    table_path = cont_path if os.path.isfile(cont_path) else fit_path
    print("raw mode: fit {} | table {} | encoder {}".format(fit_path, table_path, candidates[0]))
    fit_feats = np.load(fit_path)
    encoder = FeatureEncoder.from_checkpoint(candidates[0], fit_feats)
    raw_feats = np.load(table_path)
    if raw_feats.shape[1] != fit_feats.shape[1]:
        raise ValueError(
            "raw feature dim mismatch: fit {} vs table {}".format(fit_feats.shape[1], raw_feats.shape[1])
        )
    return encoder, raw_feats, len(fit_feats)


def build_rap(n_feat, e_feat, args, device):
    rap = RAPAD(
        n_feat,
        e_feat,
        tf_matrix=None,
        drop_out=args.drop_out,
        num_neighbors=args.n_degree,
        get_checkpoint_path=None,
        step=args.step,
        mask_num=args.mask,
    )
    rap.corr_encoder.graph = _INTER_GRAPH
    rap.to(device)
    nodetime2emb_maps = dict()
    if e_feat is not None:
        for idx in range(len(e_feat)):
            nodetime2emb_maps[str(idx)] = e_feat[idx]
    rap.corr_encoder.init_edge2emb(nodetime2emb_maps)
    rap.update_ngh_finder(_INTER_GRAPH)
    return rap


def checkpoint_candidates(dataset, feets, seed, model_root, n0, n1, step):
    runtime_id = "{}-mask {}-Neighbors {}-{}-step {}-{}-seed{}".format(
        dataset, "[0, 0]", n0, n1, step, feets, seed
    )
    flat = os.path.join(model_root, runtime_id + "best-model.pth")
    if os.path.isfile(flat):
        return flat
    dpath = os.path.join(model_root, runtime_id, "best-model.pth")
    if os.path.isfile(dpath):
        return dpath
    return None


def predict_test(rap, src, dst, ts, eidx, chunk=256, progress=True):
    """Sigmoid test scores for all test events (chunked, with progress bar)."""
    scores = []
    total = len(src)
    bar = tqdm(range(0, total, chunk), desc="predict", unit="chunk", disable=not progress)
    for i in bar:
        sl = slice(i, i + chunk)
        sc = rap.contrast(src[sl], dst[sl], ts[sl], e_idx_l=eidx[sl], test=True)[0]
        scores.append(sc.detach().cpu().numpy().reshape(-1))
    return np.concatenate(scores)


def make_groups(d_feat, feets, embed_dim, allow_statsdim=False):
    groups = [""] * d_feat
    ok_feets = feets == "stats" or (feets == "statsdim" and allow_statsdim)
    if ok_feets and embed_dim > 0 and d_feat > embed_dim:
        groups = ["embedding" if i < embed_dim else "statistic" for i in range(d_feat)]
    return groups


def group_stats(a):
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    return float(a.min()), float(a.max()), float(a.std())


def main():
    global _INTER_GRAPH
    args, sys_argv = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AER SHAP evaluation requires a CUDA-enabled GPU.")
    device = torch.device("cuda")
    set_random_seed(args.seed)

    if args.shap_max_samples == 2000 and args.n_samples is not None:
        args.shap_max_samples = args.n_samples
    if args.shap_background_samples == 40 and args.n_background is not None:
        args.shap_background_samples = args.n_background

    embed_dim_defaults = {"mooc": 4, "reddit": 172, "wikipedia": 172}
    embed_dim = args.embedding_dim if args.embedding_dim > 0 else embed_dim_defaults.get(args.data, 0)

    out_dir = os.path.join(args.outdir, "{}-{}".format(args.data, args.feets))
    os.makedirs(out_dir, exist_ok=True)
    print("output dir:", out_dir)
    print("device:", device)

    # ------------------------------------------------------------------
    # load data + history graph (same as main.py)
    # ------------------------------------------------------------------
    train_val_data, test_data, full_adj_list, e_feat, n_feat, max_idx, _ = loadDropoutData(
        args.data,
        args.data_mode,
        trainRatio=args.train_ratio,
        valRatio=args.val_ratio,
        feets=args.feets,
    )
    _INTER_GRAPH = HistoryFinder()
    _INTER_GRAPH.init_off_set(full_adj_list)
    print("loaded {} test events".format(len(test_data)))

    if e_feat is None:
        raise ValueError("e_feat is None - SHAP on edge features needs an e_feat file")

    d_feat = e_feat.shape[1]
    print("edge feature dim:", d_feat, "| step:", args.step, "| embedding dim:", embed_dim)

    test_src = test_data[:, 0].astype(np.int64)
    test_dst = test_data[:, 1].astype(np.int64)
    test_ts = test_data[:, 2].astype(np.int64)
    test_eidx = test_data[:, 3].astype(np.int64)
    test_label = test_data[:, 4].astype(np.int64)

    # raw mode: attribute the raw stats features for statsdim models so the
    # embedding/statistic split is well-defined (mirrors GraphSAGE --raw_mode)
    raw_mode = args.raw_mode or args.feets == "statsdim"
    encoder = None
    raw_feats = None
    n_orig = None
    feat_table = e_feat
    if raw_mode:
        encoder, raw_feats, n_orig = load_raw_encoder(args.data)
        if encoder is not None:
            # validate that encoder(raw) reproduces the stored mlp features; otherwise
            # the reconstructed encoder does not match the trained model input
            probe = np.arange(1, min(len(e_feat), len(raw_feats), 101))
            with torch.no_grad():
                recon = encoder(torch.from_numpy(raw_feats[probe].astype(np.float32))).numpy()
            err = float(np.abs(recon - e_feat[probe]).max())
            ref = float(np.abs(e_feat[probe]).std()) or 1.0
            print("raw mode validation: max |encoder(raw) - stored| = {:.4e} (std {:.4e})".format(err, ref))
            if err > max(0.05, 0.1 * ref):
                print("  -> reconstruction mismatch, disabling raw mode")
                encoder, raw_feats = None, None
        if encoder is None:
            raw_mode = False
        else:
            d_raw = raw_feats.shape[1]
            print("  raw mode active: attributing raw stats features, dim {}".format(d_raw))
            feat_table = raw_feats

    rap = build_rap(n_feat, e_feat, args, device)

    def compute_inputs(indices):
        src_b = test_src[indices]
        dst_b = test_dst[indices]
        ts_b = test_ts[indices]
        eid_b = test_eidx[indices]
        subgraphs = rap.ngh_finder.get_interaction(
            src_b, dst_b, ts_b, num_neighbors=rap.num_neighbors, step=rap.step, e_idx_l=eid_b
        )
        neighbor_eid = subgraphs[0][1]  # list of arrays (len num_neighbors[0]+step)
        L = len(neighbor_eid[0])
        x = np.stack(
            [np.stack([feat_table[int(e)] for e in arr[L - rap.step:]]) for arr in neighbor_eid]
        ).astype(np.float32)
        with torch.no_grad():
            _, s_vec, desti = rap.corr_encoder(src_b, dst_b, eid_b, ts_b, subgraphs, test=True)
        s_vector, s_partner, _ = s_vec
        d_partner, _ = desti
        s_vector = torch.as_tensor(s_vector, device=device)
        s_partner = torch.as_tensor(s_partner, device=device)
        d_partner = torch.as_tensor(d_partner, device=device)
        return torch.as_tensor(x, device=device), s_vector, s_partner, d_partner

    def neighbor_attributable(indices):
        """Mask over `indices`: True if all last-step neighbor edge ids < n_orig.

        In raw mode only edges with id < n_orig (= original stats rows) are
        reproducible by the encoder; events whose neighborhood contains synthetic
        edges (id >= n_orig) cannot be attributed faithfully and are skipped.
        """
        src_b = test_src[indices]
        dst_b = test_dst[indices]
        ts_b = test_ts[indices]
        eid_b = test_eidx[indices]
        subgraphs = rap.ngh_finder.get_interaction(
            src_b, dst_b, ts_b, num_neighbors=rap.num_neighbors, step=rap.step, e_idx_l=eid_b
        )
        neighbor_eid = subgraphs[0][1]
        L = len(neighbor_eid[0])
        ok = np.array([bool(np.all(arr[L - rap.step:] < n_orig)) for arr in neighbor_eid])
        return ok

    # feature dims used for the group split (raw dim in raw mode)
    group_dim = raw_feats.shape[1] if raw_feats is not None else d_feat
    groups = make_groups(group_dim, args.feets, embed_dim, allow_statsdim=raw_mode)

    # ----------------------------------------------------------------------
    # per-seed loop: score -> select correctly classified -> per-class SHAP
    # ----------------------------------------------------------------------
    per_seed = {}
    seed_metrics = {}
    used_seeds = []
    for s in args.seeds:
        ckpt = checkpoint_candidates(
            args.data, args.feets, s, args.model_root, args.n_degree[0], args.n_degree[1], args.step
        )
        if ckpt is None:
            print("no checkpoint for seed {} -> skip".format(s))
            continue
        print("=== seed {} <== {} ===".format(s, ckpt))
        rap.load_state_dict(torch.load(ckpt, map_location=device))
        rap.eval()

        prob = predict_test(rap, test_src, test_dst, test_ts, test_eidx, progress=not args.no_progress)
        prob = 1.0 / (1.0 + np.exp(-prob))  # sigmoid, like GraphSAGE/SAD
        correct = (prob >= args.pred_threshold).astype(np.int64) == test_label
        pred = (prob >= args.pred_threshold).astype(np.int64)
        if len(np.unique(test_label)) > 1:
            seed_metrics[s] = {
                "roc_auc": float(roc_auc_score(test_label, prob)),
                "pr_auc": float(average_precision_score(test_label, prob)),
                "acc": float((pred == test_label).mean()),
            }
        else:
            seed_metrics[s] = {"roc_auc": None, "pr_auc": None, "acc": float((pred == test_label).mean())}
        pos_arr = np.arange(len(test_label))
        correct_normal = pos_arr[np.logical_and(test_label == 0, correct)]
        correct_anomaly = pos_arr[np.logical_and(test_label == 1, correct)]

        rng = np.random.RandomState(args.seed * 1000 + s)

        if raw_mode:
            # ---- raw mode: skip events with synthetic neighbors, keep the count ----
            # target = the count the non-raw selection would have produced
            target = min(args.shap_max_samples, len(correct_normal) + len(correct_anomaly))
            all_arr = np.arange(len(test_label))
            all_anom = all_arr[test_label == 1]
            all_norm = all_arr[test_label == 0]
            candidates = np.concatenate(
                [
                    correct_anomaly,
                    rng.permutation(correct_normal),
                    rng.permutation(np.setdiff1d(all_anom, correct_anomaly)),
                    rng.permutation(np.setdiff1d(all_norm, correct_normal)),
                ]
            ).astype(np.int64)
            selected = []
            examined = 0
            chunk = 128
            i = 0
            while len(selected) < target and i < len(candidates):
                c = candidates[i : i + chunk]
                ok = neighbor_attributable(c)
                examined += int(len(c))
                selected.extend(c[ok].tolist())
                i += chunk
            sel = np.asarray(selected[:target], dtype=np.int64)
            n_skipped = examined - len(sel)
            sel_normal = sel[test_label[sel] == 0]
            sel_anomaly = sel[test_label[sel] == 1]
            sel = np.concatenate([sel_normal, sel_anomaly])
            n_norm, n_anom = len(sel_normal), len(sel_anomaly)
            if len(sel) == 0:
                print("  no attributable samples for seed {} -> skip".format(s))
                continue
            print(
                "  explain {} events ({} normal / {} anomaly), background {} "
                "(raw filter: {} skipped with synthetic neighbors, wanted {})".format(
                    len(sel), n_norm, n_anom, min(args.shap_background_samples, len(sel)),
                    n_skipped, target,
                )
            )
            if len(sel) < target:
                print("  WARNING: fewer than {} clean events available -> using {}".format(target, len(sel)))
        else:
            sel_anomaly = correct_anomaly[: args.shap_max_samples]
            n_normal = max(0, args.shap_max_samples - len(sel_anomaly))
            sel_normal = rng.permutation(correct_normal)[:n_normal] if n_normal > 0 else np.zeros(
                0, dtype=np.int64
            )
            sel = np.concatenate([sel_normal, sel_anomaly]).astype(np.int64)
            n_norm, n_anom = len(sel_normal), len(sel_anomaly)
            if len(sel) == 0:
                print("  no correctly classified samples for seed {} -> skip".format(s))
                continue
            print(
                "  explain {} events ({} correct normal / {} correct anomaly), background {}".format(
                    len(sel), n_norm, n_anom, min(args.shap_background_samples, len(sel))
                )
            )

        pd.DataFrame(
            {
                "src": test_src[sel],
                "dst": test_dst[sel],
                "ts": test_ts[sel],
                "e_idx": test_eidx[sel],
                "label": test_label[sel],
            }
        ).to_csv(os.path.join(out_dir, "events_sample_seed{}.csv".format(s)), index=False)

        x_sel, s_vector, s_partner, d_partner = compute_inputs(sel)
        if len(used_seeds) == 0:
            # sanity check: wrapper must reproduce the real model predictions
            nchk = min(len(sel), 64)
            predictor = SHAPPredictor(
                rap, s_vector[:nchk], s_partner[:nchk], d_partner[:nchk], encoder=encoder
            ).to(device)
            predictor.eval()
            with torch.no_grad():
                pred_wrap = predictor(x_sel[:nchk])
            ref = rap.contrast(
                test_src[sel[:nchk]], test_dst[sel[:nchk]], test_ts[sel[:nchk]],
                e_idx_l=test_eidx[sel[:nchk]], test=True,
            )[0]
            max_diff = float((pred_wrap - ref).abs().max())
            print("  sanity check: |wrapper - model| max = {:.3e}".format(max_diff))
            if max_diff > 1e-2:
                print("  WARNING: wrapper does not reproduce the model exactly - SHAP may be unreliable")

        bg_count = min(args.shap_background_samples, len(sel))
        pool = x_sel  # background pool = features of all selected events

        classes = {}
        for cname, idx in (
            ("normal", np.arange(n_norm)),
            ("anomaly", np.arange(n_norm, n_norm + n_anom)),
            ("all", np.arange(len(sel))),
        ):
            if len(idx) == 0:
                continue
            per_sample = []
            bar = (
                range(len(idx))
                if args.no_progress
                else tqdm(range(len(idx)), desc="seed{} class{}".format(s, cname), unit="evt")
            )
            for j in bar:
                pos = int(idx[j])
                pi = SHAPPredictor(
                    rap, s_vector[pos : pos + 1], s_partner[pos : pos + 1], d_partner[pos : pos + 1],
                    encoder=encoder,
                ).to(device)
                pi.eval()
                bg = pool[rng.choice(len(pool), size=bg_count, replace=False)]
                sv = expected_gradients_shap(pi, x_sel[pos : pos + 1], bg, steps=args.shap_nsamples)
                per_sample.append(np.mean(np.abs(np.asarray(sv)[0]), axis=0))  # (D,)
            classes[cname] = {
                "per_feature": np.mean(np.asarray(per_sample), axis=0),
                "n": len(idx),
            }
            sv_stack = np.stack(per_sample)
            np.save(os.path.join(out_dir, "shap_seed{}_class{}.npy".format(s, cname)), sv_stack)

        per_seed[s] = {"classes": classes, "n_normal": n_norm, "n_anomaly": n_anom}
        used_seeds.append(s)

    if not used_seeds:
        print("no models loaded -> nothing to aggregate")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # aggregation over seeds (per class, mean +- std)
    # ----------------------------------------------------------------------
    seeds = sorted(used_seeds)
    groups = make_groups(group_dim, args.feets, embed_dim, allow_statsdim=raw_mode)
    has_split = any(g == "statistic" for g in groups)
    if not has_split:
        print("WARNING: no embedding/statistic column split available for {} (dim {}, split {}) "
              "-> group shares not computed".format(args.feets, group_dim, embed_dim))
    class_keys = {"normal": "class_normal", "anomaly": "class_ano", "all": "class_all"}

    result = {
        "dataset": args.data,
        "feets": args.feets,
        "split_dim": embed_dim,
        "feature_dim": group_dim,
        "raw_mode": raw_mode,
        "has_split": has_split,
        "seeds": seeds,
    }
    # test-split metrics (mean +- std over the scored seeds)
    metric_vals = {k: [seed_metrics[s][k] for s in seeds if seed_metrics.get(s, {}).get(k) is not None] for k in ("roc_auc", "pr_auc", "acc")}
    result["auc"] = {
        k: {
            "mean": float(np.mean(v)) if v else None,
            "std": float(np.std(v)) if v else None,
            "per_seed": dict(zip([s for s in seeds if seed_metrics.get(s, {}).get(k) is not None], v)) if v else {},
        }
        for k, v in metric_vals.items()
    }
    feat_rows = []
    share_rows = []

    for cname, key in class_keys.items():
        present = [s for s in seeds if cname in per_seed[s]["classes"]]
        if not present:
            continue
        mat = np.stack([per_seed[s]["classes"][cname]["per_feature"] for s in present])  # (nS, D)
        mean_pf = mat.mean(axis=0)
        std_pf = mat.std(axis=0)

        entry = {
            "n_samples_mean": float(np.mean([per_seed[s]["classes"][cname]["n"] for s in present])),
            "mean_abs_shap_per_feature_mean": mean_pf.tolist(),
            "mean_abs_shap_per_feature_std": std_pf.tolist(),
        }
        if has_split:
            emb_part = mean_pf[:embed_dim]
            stat_part = mean_pf[embed_dim:]
            emb_imp = float(emb_part.mean()) if len(emb_part) > 0 else 0.0
            stat_imp = float(stat_part.mean()) if len(stat_part) > 0 else 0.0
            emb_shares, stat_shares = [], []
            for pf in mat:
                e = pf[:embed_dim]
                st = pf[embed_dim:]
                ei = float(e.mean()) if len(e) > 0 else 0.0
                si = float(st.mean()) if len(st) > 0 else 0.0
                tot = ei + si
                emb_shares.append(ei / tot if tot > 0 else 0.0)
                stat_shares.append(si / tot if tot > 0 else 0.0)
            e_min, e_max, e_std = group_stats(emb_part)
            st_min, st_max, st_std = group_stats(stat_part)
            entry.update(
                {
                    "embedding_importance": round(emb_imp, 6),
                    "statistic_importance": round(stat_imp, 6),
                    "embedding_min": e_min,
                    "embedding_max": e_max,
                    "embedding_std": e_std,
                    "statistic_min": st_min,
                    "statistic_max": st_max,
                    "statistic_std": st_std,
                    "embedding_share_mean": round(float(np.mean(emb_shares)), 4),
                    "embedding_share_std": round(float(np.std(emb_shares)), 4),
                    "statistic_share_mean": round(float(np.mean(stat_shares)), 4),
                    "statistic_share_std": round(float(np.std(stat_shares)), 4),
                }
            )
        result[key] = entry
        for i in range(group_dim):
            feat_rows.append(
                {
                    "feature_dim": i,
                    "group": groups[i],
                    "class": cname,
                    "mean_abs_shap": mean_pf[i],
                    "std_abs_shap": std_pf[i],
                }
            )
        if has_split:
            share_rows.append(
                {
                    "class": cname,
                    "group": "embedding",
                    "share_mean": entry["embedding_share_mean"],
                    "share_std": entry["embedding_share_std"],
                }
            )
            share_rows.append(
                {
                    "class": cname,
                    "group": "statistic",
                    "share_mean": entry["statistic_share_mean"],
                    "share_std": entry["statistic_share_std"],
                }
            )

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(feat_rows).to_csv(os.path.join(out_dir, "feature_shap_seeds.csv"), index=False)
    pd.DataFrame(share_rows).to_csv(os.path.join(out_dir, "group_shares.csv"), index=False)
    print(pd.DataFrame(share_rows).to_string(index=False))

    # ----------------------------------------------------------------------
    # plots (top-20 bar per class)
    # ----------------------------------------------------------------------
    for cname, key in class_keys.items():
        if key not in result:
            continue
        mean_pf = np.asarray(result[key]["mean_abs_shap_per_feature_mean"])
        std_pf = np.asarray(result[key]["mean_abs_shap_per_feature_std"])
        order = np.argsort(mean_pf)[::-1]
        topk = min(20, group_dim)
        plt.figure(figsize=(9, 6))
        plt.barh(
            np.arange(topk)[::-1], mean_pf[order[:topk]], xerr=std_pf[order[:topk]], capsize=3
        )
        plt.yticks(np.arange(topk)[::-1], order[:topk].tolist())
        plt.xlabel("mean |SHAP| (avg over seeds)")
        plt.title("{} - {} - {} - top {} features".format(args.data, args.feets, cname, topk))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "bar_top{}_class{}.png".format(topk, cname)))
        plt.close()

    print("done -> {}".format(out_dir))


if __name__ == "__main__":
    main()