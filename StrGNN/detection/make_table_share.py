#!/usr/bin/env python3
"""Table of class_all embedding/statistic share importance for StrGNN, per
feature-config (adapted from GraphSAGE/make_table_share.py).

Layout (one row per dataset+variant):

    Dataset | Model | +Stats: Embed, Stat | +Stats+Encoded: Embed, Stat

For each of the three feature configurations (original, *_stats,
*_stats_mlp_ks_*) the *share importance* reported in
`checkpoints/<run>/result.json` -> `importance.class_all`
(``embedding_share`` / ``statistic_share``, already in fraction) is averaged
over the seeded runs and shown as mean +/- std in percent.

Example:
    python make_table_share.py                 # uses ./checkpoints (StrGNN)
    python make_table_share.py --checkpoint_root /path/to/checkpoints
"""
from __future__ import print_function

import argparse
import json
import math
import os
import re

DEFAULT_CHECKPOINT_ROOT = "./checkpoints"
BASE_DATASETS = ("reddit", "wikipedia", "mooc")


def mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return None, None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return mu, math.sqrt(var)


def classify(data_set, base_datasets=BASE_DATASETS):
    """Return (base, category) with category in original|stats|encoded, else (None, None)."""
    for base in base_datasets:
        if data_set == base:
            return base, "original"
        if data_set == base + "_stats":
            return base, "stats"
        if re.match(r"^%s_stats_mlp_ks_\d+$" % re.escape(base), data_set):
            return base, "encoded"
    return None, None


def model_label(config):
    graph = config.get("graph", "")
    mode = "acc" if graph.startswith("acc_") else ("sta" if graph.startswith("sta_") else "model")
    return "StrGNN-%s" % mode


def load_runs(root):
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
        class_all = result.get("importance", {}).get("class_all")
        if class_all is None:
            continue
        emb_share = class_all.get("embedding_share")
        if emb_share is None:
            continue
        data = config.get("split") or config.get("data_set", "")
        base, category = classify(data)
        if base is None:
            continue
        runs.append({
            "base": base,
            "category": category,
            "model": model_label(config),
            "seed": config.get("seed"),
            "emb_share_pct": float(emb_share) * 100.0,
            "stat_share_pct": float(class_all.get("statistic_share", 1.0 - float(emb_share))) * 100.0,
        })
    return runs


def fmt_ms(ms):
    mu, sd = ms
    if mu is None:
        return "--"
    return "$%.1f\\pm%.1f$" % (mu, sd if sd is not None else 0.0)


def build_table(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["base"], r["model"]), []).append(r)

    rows = []
    for (base, model), rs in groups.items():
        cells = {}
        for cat in ("stats", "encoded"):
            sub = [r for r in rs if r["category"] == cat]
            cells[cat + "_emb"] = mean_std([r["emb_share_pct"] for r in sub])
            cells[cat + "_stat"] = mean_std([r["stat_share_pct"] for r in sub])
        rows.append({
            "base": base,
            "model": model,
            "n": {c: len([r for r in rs if r["category"] == c]) for c in ("original", "stats", "encoded")},
            **cells,
        })

    per_dataset = {}
    for row in rows:
        per_dataset.setdefault(row["base"], []).append(row)
    ds_order = {b: i for i, b in enumerate(BASE_DATASETS)}
    for k in per_dataset:
        per_dataset[k].sort(key=lambda r: r["model"])

    cols = ["stats_emb", "stats_stat", "encoded_emb", "encoded_stat"]

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Share of embedding vs statistic SHAP importance "
          "(class\\_all, mean$\\pm$std over seeds, in \\%).}")
    print("\\label{tab:share_importance}")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Model} & "
          "\\multicolumn{2}{c}{+Stats} & \\multicolumn{2}{c}{+Stats +Encoded} \\\\")
    print(" & & Emb & Stat & Emb & Stat \\\\")
    print("\\midrule")
    n_groups = 0
    for base in sorted(per_dataset, key=lambda b: ds_order.get(b, 1e9)):
        ds_rows = per_dataset[base]
        for i, row in enumerate(ds_rows):
            cell_str = " & ".join([fmt_ms(row[c]) for c in cols])
            if i == 0:
                print("\\multirow{%d}{*}{%s} & %s & %s \\\\" % (len(ds_rows), base, row["model"], cell_str))
            else:
                print(" & %s & %s \\\\" % (row["model"], cell_str))
        n_groups += 1
        if n_groups < len(per_dataset):
            print("\\addlinespace")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n# runs used per config:")
    for ds in sorted(per_dataset, key=lambda b: ds_order.get(b, 1e9)):
        for row in per_dataset[ds]:
            print("  %-9s %-10s original=%d +stats=%d +encoded=%d" %
                  (row["base"], row["model"], row["n"]["original"], row["n"]["stats"], row["n"]["encoded"]))


def main():
    p = argparse.ArgumentParser(description="Share-importance table (class_all) per "
                                            "feature configuration")
    p.add_argument("--checkpoint_root", type=str, default=DEFAULT_CHECKPOINT_ROOT)
    args = p.parse_args()

    runs = load_runs(args.checkpoint_root)
    if not runs:
        print("no runs with class_all importances under %s" % args.checkpoint_root)
        return
    build_table(runs)


if __name__ == "__main__":
    main()