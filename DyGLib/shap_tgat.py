#!/usr/bin/env python3
"""Edge-feature SHAP importances for the trained TGAT node-classification
checkpoints, post-event (history-inverted) formulation.

Why a dedicated wrapper: TGAT gathers edge features only from strictly older
neighbors (find_neighbors_before < t), so the candidate edge's own features
never influence its own src/dst embeddings. The wrapper below inverts that
history-gating by swapping in a ``PeekNeighborSampler`` that re-inserts the
candidate (v, e, t) as the most-recent first-hop neighbour of both its
endpoints. Everything else --- edge-table gather, key/value projection,
attention, merge layer, MLP --- is already differentiable w.r.t. the rebuilt
edge-feature table, so gradients flow from logit to the input X.

Run from the DyGLib directory (mirrors tgat_test_all.sh):

    python shap_tgat.py --dirs checkpoints/node_classification_TGAT_wikipedia_stats_seed0

Writes/updates result.json in each checkpoint dir.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from models.TGAT import TGAT
from utils.shap_common import (
    PeekNeighborSampler,
    RecordingNeighborSampler,
    get_device,
    load_checkpoint,
    load_raw_encoder,
    patched_edge_row,
    run_importance_scan,
    set_seed,
    write_result,
)


class TGATShapWrapper(nn.Module):
    """f(X) = MLP_classifier(src_embedding after injecting the candidate edge).

    X can be either the raw CSV features (when a saved FeatureEncoder exists)
    or the processed .npy rows; the wrapper applies the (frozen) encoder and
    patches the backbone's edge-feature table row `eid` before the forwards.
    """

    def __init__(self, model, cfg, encoder, device):
        super(TGATShapWrapper, self).__init__()
        self.model = model
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.ctx = None
        # usage instrumentation: which edge-feature rows the current forward reads
        self.used_edges = set()
        self._attrib_eid = None

    def set_context(self, ctx):
        """ctx: dict with keys src, dst, time, eid, memory (unused here)."""
        self.ctx = ctx

    def reset_usage(self):
        """Forget previously recorded edge usage (call once per sample)."""
        self.used_edges = set()

    def set_attrib_edge(self, eid):
        """Row of `edge_raw_features` that will be replaced by the SHAP input X.
        None means the candidate edge (predicted edge) itself."""
        self._attrib_eid = None if eid is None else int(eid)

    @property
    def attrib_eid(self):
        if self._attrib_eid is not None:
            return self._attrib_eid
        if self.ctx is None:
            return None
        return int(self.ctx["eid"])

    def _encode(self, x):
        if self.encoder is not None:
            return self.encoder(x)
        return x

    def forward(self, X):
        if self.ctx is None:
            raise ValueError("set_context() must be called before forward()")
        src, dst = int(self.ctx["src"]), int(self.ctx["dst"])
        t = float(self.ctx["time"])
        eid = int(self.ctx["eid"])
        attrib = self.attrib_eid
        num_neighbors = int(getattr(self.cfg, "num_neighbors", 20))

        base_table = self.model[0].edge_raw_features
        base_sampler = self.model[0].neighbor_sampler
        peek_map = {
            (src, t): (dst, eid, t),
            (dst, t): (src, eid, t),
        }

        outs = []
        for b in range(X.shape[0]):
            enc_b = self._encode(X[b : b + 1])
            table = patched_edge_row(base_table, attrib, enc_b)
            rec = RecordingNeighborSampler(PeekNeighborSampler(base_sampler, peek_map))
            try:
                self.model[0].edge_raw_features = table
                # install the history-inverted, usage-recording sampler
                self.model[0].set_neighbor_sampler(rec)
                emb, _ = self.model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=np.asarray([src]),
                    dst_node_ids=np.asarray([dst]),
                    node_interact_times=np.asarray([t]),
                    num_neighbors=num_neighbors,
                )
                outs.append(self.model[1](x=emb))
            finally:
                self.model[0].edge_raw_features = base_table
                self.model[0].set_neighbor_sampler(base_sampler)
                self.used_edges |= rec.usage
        # the predicted edge itself always participated (post-event inversion)
        self.used_edges.add(eid)
        return torch.cat(outs, dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="TGAT edge-feature SHAP importances")
    p.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    p.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        help="explicit model dir(s); default: all dirs in checkpoint_root",
    )
    p.add_argument("--model_file", type=str, default="model.pt")
    p.add_argument("--config_file", type=str, default="config.json")
    p.add_argument("--shap_max_samples", type=int, default=100)
    p.add_argument("--shap_background_samples", type=int, default=50)
    p.add_argument("--shap_nsamples", type=int, default=30)
    p.add_argument("--split_dim", type=int, default=None)
    p.add_argument(
        "--raw_data_dir", type=str, default="../raw_data/"
    )
    p.add_argument("--encoder_file", type=str, default="")
    p.add_argument("--pred_threshold", type=float, default=0.5)
    p.add_argument(
        "--no_edge_shap",
        action="store_true",
        default=False,
        help="skip the per-used-edge importance scan (candidate summary only)",
    )
    p.add_argument(
        "--max_edges_per_sample",
        type=int,
        default=40,
        help="max number of used edges to attribute per explained sample "
        "(candidate always first; 0 = all)",
    )
    p.add_argument(
        "--edge_report_topk",
        type=int,
        default=10,
        help="how many edges to print per class",
    )
    p.add_argument(
        "--edge_csv_out",
        type=str,
        default="",
        help="optional path to write the per-edge importance table as CSV",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

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

    for model_dir in model_dirs:
        cfg_path = os.path.join(model_dir, args.config_file)
        if not (
            os.path.exists(cfg_path)
            and os.path.exists(os.path.join(model_dir, args.model_file))
        ):
            print("[skip] %s missing model/config" % model_dir)
            continue

        set_seed(args.seed)
        run = load_checkpoint(model_dir, args)
        cfg = run["cfg"]
        model = run["model"]
        test_data = run["test_data"]
        device = run["device"]

        raw_tup = load_raw_encoder(cfg.dataset_name, args)
        encoder = raw_tup[0] if raw_tup is not None else None

        wrapper = TGATShapWrapper(model, cfg, encoder, device)
        try:
            import shap  # noqa: F401  (availability check)
        except ImportError:
            print("[skip] %s python module 'shap' not installed" % model_dir)
            continue

        print("====================")
        print("SHAP %s: %s" % (cfg.model_name, os.path.basename(model_dir)))
        print("====================")
        importance = run_importance_scan(args, wrapper, model, cfg, test_data)
        write_result(model_dir, importance)
        if importance is not None:
            groups = ["class_normal", "class_ano", "class_all"]
            for g in groups:
                d = importance.get(g)
                if d:
                    print(
                        "[ok] %s (%s): %s"
                        % (os.path.basename(model_dir), g, d["score"])
                    )
        else:
            print("[skip] %s: empty importance" % model_dir)

        if not getattr(args, "no_edge_shap", False):
            from utils.shap_common import edge_usage_report

            wrapper.reset_usage()
            edge_usage_report(args, wrapper, model, cfg, test_data)


if __name__ == "__main__":
    main()
