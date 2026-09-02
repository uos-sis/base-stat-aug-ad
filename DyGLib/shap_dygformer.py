#!/usr/bin/env python3
"""Edge-feature SHAP importances for the trained DyGFormer node-classification
checkpoints, post-event (history-inverted) formulation.

Why a dedicated wrapper: DyGFormer builds a per-node temporal sequence of
first-hop neighbors strictly older than t, and ``pad_sequences`` places the
target node at sequence position 0 *with the dummy edge id 0*
(DyGFormer.py:243), which forces the candidate edge's feature row to the
all-zero padding row.  The wrapper inverts this by temporarily substituting
the real edge id into that source slot, so the first edge-channel patch of the
source sequence carries the candidate's own features. Everything downstream
(gather, patching, per-channel projection, transformer, mean pooling, output
head) is already differentiable w.r.t. the rebuilt edge-feature table.

Run from the DyGLib directory (mirrors dygformer_test_all.sh):

    python shap_dygformer.py --dirs checkpoints/node_classification_DyGFormer_wikipedia_stats_seed0

Writes/updates result.json in each checkpoint dir.
"""

import argparse
import os
import types

import numpy as np
import torch
import torch.nn as nn

from utils.shap_common import (
    get_device,
    load_checkpoint,
    load_raw_encoder,
    patched_edge_row,
    run_importance_scan,
    set_seed,
    write_result,
)


class DyGFormerShapWrapper(nn.Module):
    """f(X) = MLP_classifier(src embedding with the candidate edge in the
    first patch of the source sequence)."""

    def __init__(self, model, cfg, encoder, device):
        super(DyGFormerShapWrapper, self).__init__()
        self.model = model
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.ctx = None
        # usage instrumentation: which edge-feature rows enter the sequence
        self.used_edges = set()
        self._attrib_eid = None

        backbone = model[0]
        # remember the stock method and install the interception wrapper
        self._orig_pad_sequences = backbone.pad_sequences
        self._patch_pending = False
        self._patch_eid = 0
        _wrapper = self
        _orig = self._orig_pad_sequences

        def _interceptor(
            node_ids,
            node_interact_times,
            nodes_neighbor_ids_list,
            nodes_edge_ids_list,
            nodes_neighbor_times_list,
            patch_size,
            max_input_sequence_length,
        ):
            return _wrapper._intercept_pad_impl(
                _orig,
                node_ids,
                node_interact_times,
                nodes_neighbor_ids_list,
                nodes_edge_ids_list,
                nodes_neighbor_times_list,
                patch_size,
                max_input_sequence_length,
            )

        # the model calls ``self.pad_sequences(...)`` directly, so a plain
        # closure assigned to the attribute is enough (no extra ``self`` binding)
        backbone.pad_sequences = _interceptor

    def reset_usage(self):
        """Forget previously recorded edge usage (call once per sample)."""
        self.used_edges = set()

    def set_attrib_edge(self, eid):
        """Row of `edge_raw_features` whose SHAP input X is; None -> candidate."""
        self._attrib_eid = None if eid is None else int(eid)

    @property
    def attrib_eid(self):
        if self._attrib_eid is not None:
            return self._attrib_eid
        if self.ctx is None:
            return None
        return int(self.ctx["eid"])

    # ------------------------------------------------------------------ pad hook
    def _intercept_pad_impl(
        self,
        _orig_pad_sequences,
        node_ids,
        node_interact_times,
        nodes_neighbor_ids_list,
        nodes_edge_ids_list,
        nodes_neighbor_times_list,
        patch_size,
        max_input_sequence_length,
    ):
        """Stock pad_sequences, but on the first (source) call the target slot
        keeps the *real* edge id instead of the dummy 0; also record every
        typed edge id that enters the sequence feature gather."""
        (
            padded_nodes_neighbor_ids,
            padded_nodes_edge_ids,
            padded_nodes_neighbor_times,
        ) = _orig_pad_sequences(
            node_ids,
            node_interact_times,
            nodes_neighbor_ids_list,
            nodes_edge_ids_list,
            nodes_neighbor_times_list,
            patch_size,
            max_input_sequence_length,
        )
        if self._patch_pending:
            padded_nodes_edge_ids[:, 0] = self._patch_eid
            self._patch_pending = False
        self.used_edges.update(
            int(x) for x in padded_nodes_edge_ids.ravel() if int(x) > 0
        )
        return (
            padded_nodes_neighbor_ids,
            padded_nodes_edge_ids,
            padded_nodes_neighbor_times,
        )

    # ---------------------------------------------------------------- SHAP iface
    def set_context(self, ctx):
        """ctx: dict with keys src, dst, time, eid, memory (unused here)."""
        self.ctx = ctx

    def _encode(self, x):
        if self.encoder is not None:
            return self.encoder(x)
        return x

    def forward(self, X):
        if self.ctx is None:
            raise ValueError("set_context() must be called before forward()")
        src = int(self.ctx["src"])
        dst = int(self.ctx["dst"])
        t = float(self.ctx["time"])
        eid = int(self.ctx["eid"])
        attrib = self.attrib_eid

        base_table = self.model[0].edge_raw_features
        outs = []
        for b in range(X.shape[0]):
            enc_b = self._encode(X[b : b + 1])
            table = patched_edge_row(base_table, attrib, enc_b)
            # the candidate edge always occupies sequence slot 0 of the source
            self._patch_eid = eid
            self._patch_pending = True
            try:
                self.model[0].edge_raw_features = table
                emb, _ = self.model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=np.asarray([src]),
                    dst_node_ids=np.asarray([dst]),
                    node_interact_times=np.asarray([t]),
                )
                outs_b = self.model[1](x=emb)
            finally:
                self.model[0].edge_raw_features = base_table
                self._patch_pending = False
            outs.append(outs_b)
        self.used_edges.add(eid)
        return torch.cat(outs, dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="DyGFormer edge-feature SHAP importances")
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
    get_device()

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

        try:
            import shap  # noqa: F401
        except ImportError:
            print("[skip] %s python module 'shap' not installed" % model_dir)
            continue

        wrapper = DyGFormerShapWrapper(model, cfg, encoder, device)

        print("====================")
        print("SHAP %s: %s" % (cfg.model_name, os.path.basename(model_dir)))
        print("====================")
        importance = run_importance_scan(args, wrapper, model, cfg, test_data)
        write_result(model_dir, importance)
        if importance is not None:
            for g in ["class_normal", "class_ano", "class_all"]:
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
