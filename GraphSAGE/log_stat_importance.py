#!/usr/bin/env python3
"""Log the per-dimension statistic SHAP importance to wandb (step = stat dim idx).

For every checkpoints dir with a result.json containing `importance.class_all`,
the *statistic* feature block of `mean_abs_shap_per_feature` is logged to wandb
as one run: each step is the statistic-feature dimension index (0..n-1) and the
metric is that dimension's absolute SHAP importance (raw value and percent of
the total attribution).

Example:
    python log_stat_importance.py                        # GraphSAGE ./checkpoints
    python log_stat_importance.py --checkpoint_root ../SAD/checkpoints
    python log_stat_importance.py --dry_run              # print what would run
"""
from __future__ import print_function

import argparse
import json
import os

import wandb

DEFAULT_CHECKPOINT_ROOT = "./checkpoints"
DEFAULT_PROJECT = "cna2026_importance"
DEFAULT_RAW_DATA_DIR = "../raw_data/"
_ENCODED_SUFFIXES = ("_mlp_ks_64", "_mlp_ks_16", "_mlp_llr_64", "_mlp_ks_8", "_mlp_combined_64", "_mlp_llr", "_mlp_ks")


def raw_stat_count(data_set, raw_data_dir):
    """Number of statistic columns in the raw csv (header tokens - 5), or None."""
    raw_name = data_set
    for suffix in _ENCODED_SUFFIXES:
        if raw_name.endswith(suffix):
            raw_name = raw_name[: -len(suffix)]
            break
    csv_path = os.path.join(raw_data_dir, "%s.csv" % raw_name)
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path) as f:
        n_tokens = len(f.readline().strip().split(","))
    return max(0, n_tokens - 5)


def load_runs(root, raw_data_dir=DEFAULT_RAW_DATA_DIR):
    runs = []
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if not os.path.isdir(path):
            continue
        rp, cp = os.path.join(path, "result.json"), os.path.join(path, "config.json")
        if not (os.path.isfile(rp) and os.path.isfile(cp)):
            continue
        try:
            config = json.load(open(cp))
            result = json.load(open(rp))
        except Exception:
            continue
        importance = result.get("importance", {})
        class_all = importance.get("class_all")
        if class_all is None:
            continue
        per_feature = class_all.get("mean_abs_shap_per_feature")
        if not per_feature:
            continue
        split = importance.get("split")
        split_dim = int((split.get("embedding") or [0, 0])[-1])
        feat_dim = int((split.get("statistic") or [split_dim, len(per_feature)])[-1])
        per_feature = [float(v) for v in per_feature[:feat_dim]]

        # keep the statistic feature count identical across all runs (the raw
        # statistic column count); fall back to the stored split when unavailable
        data_set = config.get("data_set", "")
        n_stat = raw_stat_count(data_set, raw_data_dir)
        if n_stat is None:
            n_stat = max(0, feat_dim - split_dim)
        stat_block = per_feature[feat_dim - n_stat:] if n_stat <= feat_dim else []
        if not stat_block:
            continue
        runs.append({
            "name": os.path.basename(path),
            "path": path,
            "data_set": data_set,
            "model": config.get("model_name") or config.get("mode", "model"),
            "seed": config.get("seed"),
            "split_dim": split_dim,
            "feat_dim": feat_dim,
            "stat_block": stat_block,
            "stat_total": sum(stat_block),
        })
    return runs


def main():
    p = argparse.ArgumentParser(description="Log statistic-dimension importance to wandb")
    p.add_argument("--checkpoint_root", type=str, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--raw_data_dir", type=str, default=DEFAULT_RAW_DATA_DIR,
                   help="raw csv dir used to count the (shared) statistics dimension")
    p.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))
    p.add_argument("--dry_run", action="store_true",
                   help="organize runs and just print the plan, do not log to wandb")
    args = p.parse_args()

    runs = load_runs(args.checkpoint_root, args.raw_data_dir)
    if not runs:
        print("no runs with class_all importances under %s" % args.checkpoint_root)
        return

    print("plan: %d run(s) to log to project '%s'" % (len(runs), args.project))
    for r in runs:
        print("  %s (data_set=%s, seed=%s, stat dims=%d)"
              % (r["name"], r["data_set"], r["seed"], len(r["stat_block"])))

    if args.dry_run:
        return

    for r in runs:
        with wandb.init(
            project=args.project,
            entity=args.entity,
            config={
                "checkpoint": r["name"],
                "data_set": r["data_set"],
                "model": r["model"],
                "seed": r["seed"],
                "split_dim": r["split_dim"],
                "feat_dim": r["feat_dim"],
                "num_stat_dims": len(r["stat_block"]),
            },
            name=r["name"],
            mode=args.wandb_mode,
            reinit=True,
        ):
            for dim, value in enumerate(r["stat_block"]):
                wandb.log({
                    "stat_importance": value,
                    "stat_importance_pct": 100.0 * value / r["stat_total"],
                }, step=int(dim))
            wandb.summary["num_stat_dims"] = len(r["stat_block"])
            wandb.summary["stat_total"] = r["stat_total"]


if __name__ == "__main__":
    main()