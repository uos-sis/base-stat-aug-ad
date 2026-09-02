#!/usr/bin/env python3
"""Edge-feature SHAP importances for the trained TGN node-classification
checkpoints, post-event (history-inverted) formulation.

Why a dedicated wrapper: TGN updates its node memories *in place* on a
``requires_grad=False`` parameter (MemoryBank.node_memories) and only after a
whole batch, so the candidate edge's own features influence neither its own
prediction nor a differentiable path out of the bank.  Two inversions are
combined here to reconstruct the "state right after the candidate event":

1. Memory-path inversion (one-step, non-BPTT): we stream the test split, take
   the memory snapshot *before* the candidate's batch, then build the message
   exactly as ``MemoryModel.compute_new_node_raw_messages`` does
   (``cat(src_mem, dst_mem, time_enc(delta), edge_features)``) and apply ONE
   forward ``GRUCell`` on a *clone* of the bank --- so the updated memory, and
   therefore the src embedding, is differentiable w.r.t. the edge features.
2. Neighbour-path inversion: like TGAT, a ``PeekNeighborSampler`` re-inserts
   the candidate edge as the most-recent first-hop neighbor of both endpoints,
   so its feature row also flows through the attention keys/values.

Run from the DyGLib directory (mirrors test.sh for TGN):

    python shap_tgn.py --dirs checkpoints/node_classification_TGN_wikipedia_stats_seed0

Writes/updates result.json in each checkpoint dir.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from utils.shap_common import (
    PeekNeighborSampler,
    RecordingNeighborSampler,
    collect_memory_snapshots,
    get_device,
    load_checkpoint,
    load_raw_encoder,
    patched_edge_row,
    run_importance_scan,
    select_samples,
    set_seed,
    write_result,
)


class TGNShapWrapper(nn.Module):
    """f(X) = MLP_classifier(src embedding after the differentiable one-step
    candidate update induced by the edge features).

    Streams are handled outside: ``set_context`` receives the pre-event memory
    snapshot (node_memories, node_last_updated_times) collected per batch by
    ``collect_memory_snapshots``.  The message is rebuilt exactly like
    ``MemoryModel.compute_new_node_raw_messages`` and applied with a single
    forward ``GRUCell`` pass on a clone of the bank, which keeps the whole
    source-to-logit path differentiable w.r.t. ``X`` without any recurrent
    BPTT (the model itself detaches the bank between batches).
    """

    def __init__(self, model, cfg, encoder, device):
        super(TGNShapWrapper, self).__init__()
        self.model = model
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.ctx = None
        # usage instrumentation: which edge-feature rows this forward reads
        self.used_edges = set()
        self._attrib_eid = None

    def set_context(self, ctx):
        """ctx: dict with keys src, dst, time, eid, memory=(mem, last_times)."""
        self.ctx = ctx

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

    def _encode(self, x):
        if self.encoder is not None:
            return self.encoder(x)
        return x

    def forward(self, X):
        if self.ctx is None:
            raise RuntimeError("set_context() must be called before forward()")
        if self.ctx["memory"] is None:
            raise RuntimeError(
                "TGN wrapper needs a pre-event memory snapshot; "
                "pass snapshots= into run_importance_scan()"
            )

        src = int(self.ctx["src"])
        dst = int(self.ctx["dst"])
        t = float(self.ctx["time"])
        eid = int(self.ctx["eid"])
        attrib = self.attrib_eid
        mem_snapshot, last_snapshot = self.ctx["memory"]
        num_neighbors = int(getattr(self.cfg, "num_neighbors", 20))

        backbone = self.model[0]
        base_table = backbone.edge_raw_features
        base_sampler = backbone.embedding_module.neighbor_sampler
        # the message path always consumes the *candidate* edge's features,
        # not the attributed row (unless they coincide)
        base_candidate_row = base_table[eid].unsqueeze(0)
        peek_map = {
            (src, t): (dst, eid, t),
            (dst, t): (src, eid, t),
        }

        mem_base = mem_snapshot.float()
        last_base = last_snapshot.float()

        outs = []
        for b in range(X.shape[0]):
            enc_b = self._encode(X[b : b + 1])
            table = patched_edge_row(base_table, attrib, enc_b)
            rec = RecordingNeighborSampler(PeekNeighborSampler(base_sampler, peek_map))
            try:
                backbone.edge_raw_features = table
                backbone.embedding_module.edge_raw_features = table
                backbone.embedding_module.neighbor_sampler = rec

                # ---- differentiable one-step memory update on a bank clone ----
                src_idx, dst_idx = src, dst
                src_delta = torch.tensor(
                    [t], dtype=torch.float32, device=self.device
                ) - last_base[src_idx].unsqueeze(0)
                dst_delta = torch.tensor(
                    [t], dtype=torch.float32, device=self.device
                ) - last_base[dst_idx].unsqueeze(0)
                time_src = backbone.time_encoder(src_delta.unsqueeze(dim=1)).reshape(
                    1, -1
                )
                time_dst = backbone.time_encoder(dst_delta.unsqueeze(dim=1)).reshape(
                    1, -1
                )

                # message edge row: the candidate's own features (fixed) unless we
                # are attributing the candidate row itself
                msg_feat = enc_b if attrib == eid else base_candidate_row

                post = mem_base.clone()
                msg_src = torch.cat(
                    [
                        post[src_idx].unsqueeze(0),
                        post[dst_idx].unsqueeze(0),
                        time_src,
                        msg_feat,
                    ],
                    dim=1,
                )
                msg_dst = torch.cat(
                    [
                        post[dst_idx].unsqueeze(0),
                        post[src_idx].unsqueeze(0),
                        time_dst,
                        msg_feat,
                    ],
                    dim=1,
                )
                gru_cell = backbone.memory_updater.memory_updater
                updated_src = gru_cell(msg_src, post[src_idx : src_idx + 1])
                updated_dst = gru_cell(msg_dst, post[dst_idx : dst_idx + 1])
                # rebuild the post-event memory WITHOUT in-place slice writes
                # (in-place writes bump the autograd version counters of the views
                # captured while building msg_src / msg_dst, which breaks the
                # backward pass); a concat keeps gradient flowing to the messages
                row_lo, row_hi = sorted([src_idx, dst_idx])
                cell_lo = updated_src if src_idx < dst_idx else updated_dst
                cell_hi = updated_dst if src_idx < dst_idx else updated_src
                post = torch.cat(
                    [
                        post[:row_lo],
                        cell_lo,
                        post[row_lo + 1 : row_hi],
                        cell_hi,
                        post[row_hi + 1 :],
                    ],
                    dim=0,
                )

                emb = backbone.embedding_module.compute_node_temporal_embeddings(
                    node_memories=post,
                    node_ids=np.asarray([src]),
                    node_interact_times=np.asarray([t]),
                    current_layer_num=int(backbone.num_layers),
                    num_neighbors=num_neighbors,
                )
                outs.append(self.model[1](x=emb))
            finally:
                backbone.edge_raw_features = base_table
                backbone.embedding_module.edge_raw_features = base_table
                backbone.embedding_module.neighbor_sampler = base_sampler
                self.used_edges |= rec.usage
        # the candidate edge is always consumed by the message path
        self.used_edges.add(eid)
        return torch.cat(outs, dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="TGN edge-feature SHAP importances")
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

        wrapper = TGNShapWrapper(model, cfg, encoder, device)

        # which samples will be explained (needed for the memory snapshots)
        from utils.shap_common import evaluate_model

        m_pred, m_label, _roc, _pr = evaluate_model(model, cfg, test_data, device)
        sel_n, sel_a = select_samples(m_pred, m_label, args)
        selected = (
            set(int(i) for i in np.concatenate([sel_n, sel_a]))
            if (sel_n.size or sel_a.size)
            else set()
        )

        print("====================")
        print("SHAP %s: %s" % (cfg.model_name, os.path.basename(model_dir)))
        print("====================")
        if selected:
            snapshots = collect_memory_snapshots(
                model, cfg, test_data, selected, device
            )
        else:
            snapshots = {}

        importance = run_importance_scan(
            args, wrapper, model, cfg, test_data, snapshots=snapshots
        )
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
            edge_usage_report(args, wrapper, model, cfg, test_data, snapshots=snapshots)


if __name__ == "__main__":
    main()
