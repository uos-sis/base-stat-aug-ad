"""Shared building blocks for the per-model edge-feature SHAP scripts.

In the node-classification setup the candidate edge's own features never enter
its own prediction (TGAT/DyGFormer gather edge features only from strictly
older neighbors via ``find_neighbors_before``; TGN commits candidate messages
in-place and uses them only from the *next* batches on).  A plain
"perturb the candidate row, run the stock forward" SHAP model is therefore
~0 by construction.

The agreed correction (post-event probe) re-inserts the candidate edge into
its own computation through a *minimal differentiable inversion* of that
history-gating; each model wrapper in shap_tgat.py / shap_tgn.py /
shap_dygformer.py implements its own inversion and exposes a uniform
``forward(X) -> logits`` contract. This module provides the shared drivers.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from utils.DataLoader import get_node_classification_data
from utils.utils import get_neighbor_sampler

from test import (
    MEMORY_MODELS,
    build_model,
    forward_embeddings,
    load_config,
    load_raw_encoder as test_load_raw_encoder,
    load_weights,
    raw_stat_feature_count,
    summarize,
)


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_raw_encoder(data_set_name, args):
    """test.load_raw_encoder + move the FeatureEncoder to the evaluation device.

    The encoder is constructed on CPU (map_location="cpu"); the SHAP wrappers
    feed it device tensors, so its buffers must follow the model device or the
    ``(x - raw_mean) / raw_scale`` normalization raises a cuda/cpu mismatch.
    """
    raw_tup = test_load_raw_encoder(data_set_name, args)
    if raw_tup is not None:
        raw_tup[0].to(get_device())
    return raw_tup


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model_dir, args):
    """Load config + frozen (backbone, classifier) model exactly like test.py.

    Returns a dict with the training artifacts needed by the explainers.
    """
    cfg = load_config(model_dir, args.config_file)
    device = get_device()
    set_seed(getattr(cfg, "seed", args.seed))

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = (
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
    model = build_model(  # this model is the [backbone, classifier] nn.Sequential
        cfg, node_raw_features, edge_raw_features, train_data, sampler, device
    )
    load_weights(model, model_dir, args, device)
    model.to(device).eval()

    return dict(
        cfg=cfg,
        model=model,
        device=device,
        node_features=node_raw_features,
        edge_features=edge_raw_features,
        full_data=full_data,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )


def predict_split(model, cfg, data, device, reset_memory=True):
    """Evaluate on ``data`` in the exact order; returns (preds, labels)."""
    if reset_memory and cfg.model_name in MEMORY_MODELS:
        # full re-init: zero memories/last-updated-times and drop stale raw
        # messages from a previous pass (otherwise a later pass processes them
        # with later timestamps and asserts "update memory to time in the past")
        model[0].memory_bank.__init_memory_bank__()

    preds, labels = [], []
    with torch.no_grad():
        for s in range(0, len(data.src_node_ids), cfg.batch_size):
            e = min(s + cfg.batch_size, len(data.src_node_ids))
            emb = forward_embeddings(
                model,
                cfg,
                data.src_node_ids[s:e],
                data.dst_node_ids[s:e],
                data.node_interact_times[s:e],
                data.edge_ids[s:e],
                device,
            )
            preds.append(model[1](x=emb).squeeze(dim=-1).sigmoid().cpu().numpy())
            labels.append(data.labels[s:e])
    return np.concatenate(preds), np.concatenate(labels)


def evaluate_model(model, cfg, data, device):
    """One-shot eval mirroring test.py; returns (preds, labels, roc, pr)."""
    import sklearn.metrics

    m_pred, m_label = predict_split(model, cfg, data, device)
    keep = m_label >= 0
    m_pred, m_label = m_pred[keep], m_label[keep].astype(np.int64)

    roc = float("nan")
    try:
        roc = float(sklearn.metrics.roc_auc_score(m_label, m_pred))
    except ValueError:
        pass
    pr = float(sklearn.metrics.average_precision_score(m_label, m_pred))
    return m_pred, m_label, roc, pr


def select_samples(model_pred, model_label, args):
    """Exact same sample selection as test.py::compute_importances_*."""
    pos = np.arange(len(model_label))
    correct = (model_pred >= args.pred_threshold).astype(np.int64) == model_label
    correct_normal = pos[np.logical_and(model_label == 0, correct)]
    correct_anomaly = pos[np.logical_and(model_label == 1, correct)]

    selected_anomaly = correct_anomaly[: args.shap_max_samples]
    n_normal = max(0, args.shap_max_samples - len(selected_anomaly))
    rng = np.random.RandomState(args.seed)
    if n_normal > 0:
        selected_normal = rng.permutation(correct_normal)[:n_normal]
    else:
        selected_normal = np.zeros(0, dtype=np.int64)
    return selected_normal.astype(np.int64), selected_anomaly.astype(np.int64)


def collect_memory_snapshots(model, cfg, data, selected, device):
    """Stream ``data`` in evaluation order and store, for every selected index,
    the memory-bank state (node_memories, node_last_updated_times) *before* the
    batch that contains the index is embedded.

    All edges in one test batch are embedded using the same pre-batch memory
    (memory updates / history messages happen at the end of the forward), so a
    single snapshot per batch is correct for every selected sample inside it.

    Returns dict {sample_index: (node_memories, node_last_updated_times)}.
    """
    assert cfg.model_name in MEMORY_MODELS, "memory snapshots need a memory model"
    selected = set(int(i) for i in selected)
    # full re-init (memories, last-updated-times AND stale raw messages)
    model[0].memory_bank.__init_memory_bank__()

    snapshots = {}
    with torch.no_grad():
        for s in range(0, len(data.src_node_ids), cfg.batch_size):
            e = min(s + cfg.batch_size, len(data.src_node_ids))
            batch_has_selected = any(i in selected for i in range(s, e))
            snap = None
            if batch_has_selected:
                base = model[0].memory_bank
                snap = (
                    base.node_memories.detach().clone(),
                    base.node_last_updated_times.detach().clone(),
                )
            forward_embeddings(
                model,
                cfg,
                data.src_node_ids[s:e],
                data.dst_node_ids[s:e],
                data.node_interact_times[s:e],
                data.edge_ids[s:e],
                device,
            )
            if snap is not None:
                mem, last = snap
                for i in range(s, e):
                    if i in selected:
                        snapshots[i] = (mem.clone(), last.clone())
    return snapshots


class RecordingNeighborSampler(object):
    """Wrap any NeighborSampler (or PeekNeighborSampler) and record, during a
    forward, the exact set of edge ids that flow into the model's neighbour
    gathers (``get_historical_neighbors`` -> ``edge_raw_features[edge_ids]``
    in TGAT / TGN graph attention).

    This is the ``which edge features were used`` instrumentation: every id in
    ``usage`` corresponds to one row of the edge-feature table that actually
    participated in the prediction.
    """

    def __init__(self, base_sampler, usage=None):
        self.base = base_sampler
        self.usage = set() if usage is None else usage

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_historical_neighbors(self, node_ids, node_interact_times, num_neighbors=20):
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = (
            self.base.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=num_neighbors,
            )
        )
        self.usage.update(
            int(x) for x in np.asarray(nodes_edge_ids).ravel() if int(x) > 0
        )
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_all_first_hop_neighbors(self, *args, **kwargs):
        return self.base.get_all_first_hop_neighbors(*args, **kwargs)


class PeekNeighborSampler(object):
    """Deterministic wrapper around a NeighborSampler that injects a single
    historical edge into the returned neighbor set for a given
    ``(node_id, interact_time)`` query.

    This is the history-gating inversion used by the TGAT / TGN wrappers: the
    candidate edge becomes the most-recent first-hop neighbor, so its feature
    row flows into the attention keys/values of the very nodes we score.
    """

    def __init__(self, base_sampler, peek_map):
        self.base = base_sampler
        # peek_map: {(node_id, interact_time): (neighbor_id, edge_id, time)}
        self.peek_map = dict(peek_map)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_historical_neighbors(self, node_ids, node_interact_times, num_neighbors=20):
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = (
            self.base.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=num_neighbors,
            )
        )
        for idx, (nid, tint) in enumerate(
            zip(*(np.asarray(x) for x in (node_ids, node_interact_times)))
        ):
            key = (int(nid), float(tint))
            if key in self.peek_map:
                neighbor_id, edge_id, edge_time = self.peek_map[key]
                nodes_neighbor_ids[idx, -1] = neighbor_id
                nodes_edge_ids[idx, -1] = edge_id
                nodes_neighbor_times[idx, -1] = edge_time
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_all_first_hop_neighbors(self, *args, **kwargs):
        return self.base.get_all_first_hop_neighbors(*args, **kwargs)


def patched_edge_row(base_table, edge_id, new_row):
    """(N+1, d) tensor with row ``edge_id`` replaced by ``new_row`` (1, d)."""
    return torch.cat([base_table[:edge_id], new_row, base_table[edge_id + 1 :]], dim=0)


def edge_input_table(model, raw_edge, encoder, device):
    """Per-edge feature vectors in *SHAP input space*:
    - raw CSV rows when a saved *_mlp_ks_*.pt encoder exists,
    - the processed .npy rows (what the models were trained on) otherwise.
    Returns a Tensor of shape (num_edges + 1, feat_dim).
    """
    if raw_edge is not None:
        return torch.from_numpy(np.asarray(raw_edge)).float().to(device)
    return model[0].edge_raw_features.detach()


def resolve_split_dim(cfg, args, feature_dim):
    if args.split_dim is not None:
        return int(args.split_dim)
    if "stats" in cfg.dataset_name:
        return max(0, int(feature_dim) - raw_stat_feature_count(cfg.dataset_name, args))
    return int(feature_dim)


def run_importance_scan(args, wrapper, model, cfg, data, snapshots=None):
    """Execute the full importance scan for one model dir.

    Parameters
    ----------
    wrapper : nn.Module
        a per-model SHAP wrapper with ``set_context(dict)`` and
        ``forward(X) -> logits`` (X in SHAP input space).
    model : nn.Sequential (backbone, classifier)
    data  : test-split Data
    snapshots : optional dict {index -> (memories, last_times)} for memory models
    """
    import shap

    device = get_device()
    m_pred, m_label, roc, pr = evaluate_model(model, cfg, data, device)
    sel_n, sel_a = select_samples(m_pred, m_label, args)

    raw_tup = load_raw_encoder(cfg.dataset_name, args)
    encoder = raw_tup[0] if raw_tup is not None else None
    raw_edge = raw_tup[2] if raw_tup is not None else None

    edge_table = edge_input_table(model, raw_edge, encoder, device)
    feature_dim = int(edge_table.shape[1])
    split_dim = resolve_split_dim(cfg, args, feature_dim)

    def feature_row(sample_id):
        return edge_table[int(data.edge_ids[sample_id])]

    def context_for(i):
        return dict(
            src=int(data.src_node_ids[i]),
            dst=int(data.dst_node_ids[i]),
            time=float(data.node_interact_times[i]),
            eid=int(data.edge_ids[i]),
            memory=(snapshots[i] if snapshots is not None else None),
        )

    def explain_class(cls_idx, bg_idx, counts):
        if len(cls_idx) == 0:
            return None

        def bg_for(sample_i):
            # never use the explained sample's own feature vector as part of its
            # background: if x equals a bg row, GradientExplainer's
            # expected-gradient attributions collapse to zero.
            rows = [feature_row(int(j)) for j in bg_idx if int(j) != int(sample_i)]
            if not rows:
                rows = [feature_row(int(j)) for j in cls_idx if int(j) != int(sample_i)]
            if not rows:
                rows = [feature_row(int(sample_i))]
            return torch.stack(rows[: args.shap_background_samples]).float().to(device)

        acc = None
        cnt = 0
        for sample_id in cls_idx:
            i = int(sample_id)
            bg = bg_for(i)
            x = feature_row(i).float().to(device).unsqueeze(0)
            wrapper.reset_usage()
            wrapper.set_attrib_edge(int(data.edge_ids[i]))
            wrapper.set_context(context_for(i))
            explainer = shap.GradientExplainer(wrapper, [bg])
            sv = explainer.shap_values([x], nsamples=1)
            arr = np.asarray(sv)[0]
            if arr.ndim == 2:
                arr = arr[0]
            acc = np.abs(arr) if acc is None else acc + np.abs(arr)
            cnt += 1
        per_feature = np.abs(acc) / cnt
        d = summarize(per_feature, split_dim)
        d.update(counts)
        d["num_samples_used"] = int(len(cls_idx))
        return d

    pos = np.arange(len(m_label))
    correct = (m_pred >= args.pred_threshold).astype(np.int64) == m_label
    correct_normal = pos[np.logical_and(m_label == 0, correct)]
    correct_anomaly = pos[np.logical_and(m_label == 1, correct)]

    imp = {
        "enabled": True,
        "limit": args.shap_max_samples,
        "target": "edge_features_post_event",
        "split_dim": split_dim,
        "feature_dim": feature_dim,
        "split": {
            "embedding": [0, min(split_dim, feature_dim)],
            "statistic": [min(split_dim, feature_dim), feature_dim],
        },
    }

    if sel_n.size:
        imp["class_normal"] = explain_class(
            sel_n,
            sel_a,
            {
                "num_correct_normals": int(len(correct_normal)),
                "roc_auc": roc,
                "pr_auc": pr,
            },
        )
    if sel_a.size:
        imp["class_ano"] = explain_class(
            sel_a,
            sel_n,
            {
                "num_correct_anomalies": int(len(correct_anomaly)),
                "roc_auc": roc,
                "pr_auc": pr,
            },
        )
    if sel_n.size or sel_a.size:
        all_idx = np.concatenate([sel_n, sel_a]) if sel_a.size else sel_n
        imp["class_all"] = explain_class(
            all_idx,
            all_idx,
            {
                "num_correct_samples": int(len(correct_normal))
                + int(len(correct_anomaly))
            },
        )
    return imp


def write_result(model_dir, importance, **meta):
    """Append the ``shap`` key to the checkpoint's result.json."""
    import json

    result_path = os.path.join(model_dir, "result.json")
    result = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
    result["shap"] = importance if importance is not None else {"enabled": False}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)


def edge_usage_report(args, wrapper, model, cfg, data, snapshots=None, device=None):
    """Discover the *exact* edge-feature rows that feed each prediction in the
    explained groups and aggregate a gradient-SHAP importance value per used
    edge.

    For every explained sample the wrapper is asked to run one forward with
    ``torch.no_grad()`` while its usage recorder is armed; that yields the
    precise set of edge ids consumed by the model for that sample.  Each
    distinct used edge is then attributed with the same GradientExplainer
    machinery the candidate edge receives, by switching the wrapper's
    attribution row to that edge id.

    ``result.json`` is NOT touched here --- the per-edge numbers are printed
    and (optional) dumped to a CSV so the existing embedding/statistic split
    summary stays the only ``shap`` entry in result.json.

    Returns a dict ``{group_name: {...report...}}``.
    """
    import shap

    device = device or get_device()

    m_pred, m_label, _roc, _pr = evaluate_model(model, cfg, data, device)
    sel_n, sel_a = select_samples(m_pred, m_label, args)

    raw_tup = load_raw_encoder(cfg.dataset_name, args)
    encoder = raw_tup[0] if raw_tup is not None else None
    raw_edge = raw_tup[2] if raw_tup is not None else None
    edge_table = edge_input_table(model, raw_edge, encoder, device)

    def feature_row(sample_id):
        return edge_table[int(data.edge_ids[sample_id])]

    def context_for(i):
        return dict(
            src=int(data.src_node_ids[i]),
            dst=int(data.dst_node_ids[i]),
            time=float(data.node_interact_times[i]),
            eid=int(data.edge_ids[i]),
            memory=(snapshots[i] if snapshots is not None else None),
        )

    def probe_used(i):
        """One no_grad forward through the wrapper while its usage recorder is
        live; the set of edge ids gathered is the ground truth of "which edge
        features were used for this prediction"."""
        wrapper.reset_usage()
        wrapper.set_context(context_for(i))
        wrapper.set_attrib_edge(int(data.edge_ids[i]))
        x = feature_row(i).float().to(device).unsqueeze(0)
        with torch.no_grad():
            wrapper(x)
        used = tuple(sorted(int(u) for u in wrapper.used_edges))
        if not used:
            used = (int(data.edge_ids[i]),)
        return used

    def bg_for(i, target_eid):
        rows = [
            feature_row(int(j))
            for j in np.concatenate([sel_n, sel_a])
            if int(j) != int(i) and int(data.edge_ids[j]) != int(target_eid)
        ]
        if not rows:
            rows = [
                feature_row(int(j))
                for j in [i]
                if int(data.edge_ids[j]) != int(target_eid)
            ]
            if not rows:
                rows = [feature_row(i)]
        return torch.stack(rows[: args.shap_background_samples]).float().to(device)

    groups = []
    if sel_n.size:
        groups.append(("class_normal", sel_n))
    if sel_a.size:
        groups.append(("class_ano", sel_a))
    if sel_n.size or sel_a.size:
        groups.append(
            ("class_all", np.concatenate([sel_n, sel_a]) if sel_a.size else sel_n)
        )

    max_edges = int(getattr(args, "max_edges_per_sample", 40) or 0)

    report = {}
    for gname, cls_idx in groups:
        usage = {}  # edge_id -> number of samples that used it
        attr = {}  # edge_id -> {'cnt': int, 'sum': float}
        per_sample_edges = []
        for sid in cls_idx:
            i = int(sid)
            used = probe_used(i)
            per_sample_edges.append(len(used))
            for u in used:
                usage[u] = usage.get(u, 0) + 1
            # attribute the used edges, candidate first
            cand = int(data.edge_ids[i])
            targets = [cand] + [u for u in used if u != cand]
            if max_edges > 0:
                targets = targets[:max_edges]
            bg = bg_for(i, cand)
            for e in targets:
                x = edge_table[e].float().to(device).unsqueeze(0)
                wrapper.set_attrib_edge(e)
                explainer = shap.GradientExplainer(wrapper, [bg])
                sv = np.asarray(explainer.shap_values([x], nsamples=1))[0]
                if sv.ndim == 2:
                    sv = sv[0]
                a = float(np.abs(sv).sum())
                st = attr.setdefault(e, {"cnt": 0, "sum": 0.0})
                st["cnt"] += 1
                st["sum"] += a
            wrapper.set_attrib_edge(int(data.edge_ids[i]))

        edges = []
        for e in sorted(usage, key=lambda e: (-usage[e], e)):
            st = attr.get(e, {"cnt": 0, "sum": 0.0})
            edges.append(
                {
                    "edge_id": int(e),
                    "used_in_samples": int(usage[e]),
                    "attributed_in": int(st["cnt"]),
                    "mean_abs_shap_total": float(st["sum"] / st["cnt"])
                    if st["cnt"]
                    else 0.0,
                }
            )
        report[gname] = {
            "num_samples": int(len(cls_idx)),
            "distinct_edges_used": int(len(usage)),
            "mean_edges_per_sample": float(np.mean(per_sample_edges))
            if per_sample_edges
            else 0.0,
            "edges": edges,
        }
        sample_tag = gname.replace("class_", "")
        print(
            "[edge-usage] %-12s %4d samples, %4d distinct used edges, "
            "mean edges/sample %.2f"
            % (
                sample_tag,
                len(cls_idx),
                len(usage),
                float(np.mean(per_sample_edges)) if per_sample_edges else 0.0,
            )
        )
        for row in edges[: getattr(args, "edge_report_topk", 10)]:
            print(
                "  edge[%d] used_in=%d attributed_in=%d mean|SHAP|total=%.6g"
                % (
                    row["edge_id"],
                    row["used_in_samples"],
                    row["attributed_in"],
                    row["mean_abs_shap_total"],
                )
            )

    csv_path = getattr(args, "edge_csv_out", None)
    if csv_path:
        with open(csv_path, "w") as f:
            f.write("class,edge_id,used_in_samples,attributed_in,mean_abs_shap_total\n")
            for gname, r in report.items():
                for row in r["edges"]:
                    f.write(
                        "%s,%d,%d,%d,%.8g\n"
                        % (
                            gname,
                            row["edge_id"],
                            row["used_in_samples"],
                            row["attributed_in"],
                            row["mean_abs_shap_total"],
                        )
                    )
        print("[edge-usage] csv written -> %s" % csv_path)
    return report
