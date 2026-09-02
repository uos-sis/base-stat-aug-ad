#!/usr/bin/env python3
"""Table of embedding vs statistic SHAP share importance, per feature-config.

For every DyGLib node-classification checkpoint with a ``result.json`` carrying
the new edge-feature SHAP entry (``result["shap"]["class_all"]``), the
``embedding_share`` / ``statistic_share`` (already fractions) are averaged over
the seeded runs and printed as mean +/- std in percent, laid out like the
GraphSAGE ``make_table_share.py``.

Layout (one row per dataset+model, models from the checkpoints: TGAT, TGN,
DyGFormer):

    Dataset | Model | Original | +Stats | +Stats +Encoded     (Emb, Stat pairs)

Only the ``shap`` key is read (the newer edge-feature post-event scan);
configs without a run yet show ``--``.

Example:
    python make_shap_table.py                             # ./checkpoints
    python make_shap_table.py --checkpoint_root checkpoints
"""

from __future__ import print_function

import argparse
import json
import math
import os
import re

DATASETS = ("reddit", "wikipedia", "mooc")
_CONFIG = "config.json"
_RESULT = "result.json"


def mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return None, None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return mu, math.sqrt(var)


def classify(data_set):
    """Return (dataset, category) for a dataset_name, or (None, None)."""
    for ds in DATASETS:
        if data_set == ds:
            return ds, "original"
        if data_set == ds + "_stats":
            return ds, "stats"
        if re.match(r"^%s_stats_mlp_ks_\d+$" % re.escape(ds), data_set):
            return ds, "encoded"
    return None, None


def load_runs(root):
    runs = []
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if not os.path.isdir(path):
            continue
        rp, cp = os.path.join(path, _RESULT), os.path.join(path, _CONFIG)
        if not (os.path.isfile(rp) and os.path.isfile(cp)):
            continue
        try:
            config = json.load(open(cp))
            result = json.load(open(rp))
        except Exception:
            continue
        class_all = result.get("shap", {}).get("class_all")
        if class_all is None:
            continue
        emb_share = class_all.get("embedding_share")
        if emb_share is None:
            continue
        data_set = config.get("dataset_name") or config.get("data_set", "")
        base, category = classify(data_set)
        if base is None:
            continue
        runs.append(
            {
                "base": base,
                "category": category,
                "model": config.get("model_name") or config.get("model", "model"),
                "seed": config.get("seed"),
                "emb_share_pct": float(emb_share) * 100.0,
                "stat_share_pct": float(
                    class_all.get("statistic_share", 1.0 - float(emb_share))
                )
                * 100.0,
            }
        )
    return runs


def fmt_ms(ms):
    mu, sd = ms
    if mu is None:
        return "--"
    return "$%.1f\\pm%.1f$" % (mu, (sd if sd is not None else 0.0))


def build_table(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["base"], r["model"]), []).append(r)

    rows = []
    for (base, model), rs in groups.items():
        cells = {}
        for cat in ("original", "stats", "encoded"):
            sub = [r for r in rs if r["category"] == cat]
            cells[cat + "_emb"] = mean_std([r["emb_share_pct"] for r in sub])
            cells[cat + "_stat"] = mean_std([r["stat_share_pct"] for r in sub])
        rows.append(
            {
                "base": base,
                "model": model,
                "n": {
                    c: len([r for r in rs if r["category"] == c])
                    for c in ("original", "stats", "encoded")
                },
                **cells,
            }
        )

    per_dataset = {}
    for row in rows:
        per_dataset.setdefault(row["base"], []).append(row)
    ds_order = {b: i for i, b in enumerate(DATASETS)}
    for k in per_dataset:
        per_dataset[k].sort(key=lambda r: r["model"])

    cols = [
        "original_emb",
        "original_stat",
        "stats_emb",
        "stats_stat",
        "encoded_emb",
        "encoded_stat",
    ]

    print("\\begin{table}[htbp]")
    print("\\centering")
    print(
        "\\caption{Share of embedding vs statistic SHAP importance "
        "(edge-feature scan, class\\_all, mean$\\pm$std over seeds, in \\%).}"
    )
    print("\\label{tab:shap_share}")
    print("\\begin{tabular}{llcccccc}")
    print("\\toprule")
    print(
        "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Model} & "
        "\\multicolumn{2}{c}{Original} & \\multicolumn{2}{c}{+Stats} & "
        "\\multicolumn{2}{c}{+Stats +Encoded} \\\\"
    )
    print(" & & Emb & Stat & Emb & Stat & Emb & Stat \\\\")
    print("\\midrule")
    n_groups = 0
    for base in DATASETS:
        ds_rows = per_dataset.get(base, [])
        if not ds_rows:
            continue
        for i, row in enumerate(ds_rows):
            cell_str = " & ".join([fmt_ms(row[c]) for c in cols])
            if i == 0:
                print(
                    "\\multirow{%d}{*}{%s} & %s & %s \\\\"
                    % (len(ds_rows), base, row["model"], cell_str)
                )
            else:
                print(" & %s & %s \\\\" % (row["model"], cell_str))
        n_groups += 1
        if n_groups < len(per_dataset):
            print("\\addlinespace")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n# runs used per config:")
    for base in sorted(per_dataset, key=lambda b: ds_order.get(b, 1e9)):
        for row in per_dataset[base]:
            print(
                "  %-9s %-10s original=%d +stats=%d +encoded=%d"
                % (
                    row["base"],
                    row["model"],
                    row["n"]["original"],
                    row["n"]["stats"],
                    row["n"]["encoded"],
                )
            )


def main():
    p = argparse.ArgumentParser(
        description="Build the SHAP share-importance LaTeX table"
    )
    p.add_argument(
        "--checkpoint_root",
        nargs="+",
        default=["./checkpoints"],
        help="folder(s) containing one sub-directory per run "
        "(config.json + result.json); results are merged",
    )
    args = p.parse_args()

    runs = []
    for root in args.checkpoint_root:
        runs.extend(load_runs(root))
    if not runs:
        print(
            "no runs with edge-feature SHAP shares under %s"
            % " ".join(args.checkpoint_root)
        )
        return
    print("found %d runs with class_all SHAP shares" % len(runs))
    build_table(runs)


if __name__ == "__main__":
    main()
