#!/usr/bin/env python3
"""Generate the LaTeX ROC-AUC table for the DyGLib node classifiers.

Same layout and styling as the GraphSAGE / SAD ``make_table.py``: for each run
directory under --checkpoint_root (config.json with dataset_name/model_name/seed
and result.json with roc_auc) the runs are classified by

    dataset prefix  : reddit | wikipedia | mooc
    category       : original | +stats (dataset_name *_stats)
                     | +stats+encoded (dataset_name *_stats_mlp_ks_<n>)

Rows are the models found in the checkpoints (TGAT, TGN, DyGFormer). For every
(dataset, model) group the mean +/- std ROC-AUC is computed over seeds for the
three categories, plus the per-seed max gain of the augmented categories vs the
original dataset (same seed).  The best category per row is bold and the Max
Gain column is highlighted with a green tint scaled by the strongest gain.

Example:
    python make_roc_table.py                              # ./checkpoints
    python make_roc_table.py --checkpoint_root checkpoints
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


def mean_std(values):
    values = [float(v) for v in values]
    if not values:
        return None, None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return mu, math.sqrt(var)


def fmt(v, as_pct=False):
    if v is None or math.isnan(v):
        return "--"
    if as_pct:
        return "+%.1f\\%%" % (v * 100.0)
    return "%.4f" % v


def compute_metric(run_dir):
    rp = os.path.join(run_dir, _RESULT)
    if not os.path.exists(rp):
        return None
    try:
        with open(rp) as f:
            result = json.load(f)
    except Exception:
        return None
    if "roc_auc" not in result:
        return None
    try:
        return float(result["roc_auc"])
    except (TypeError, ValueError):
        return None


def load_runs(root):
    runs = []
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if not os.path.isdir(path):
            continue
        cfgp = os.path.join(path, _CONFIG)
        if not os.path.exists(cfgp):
            continue
        try:
            with open(cfgp) as f:
                config = json.load(f)
        except Exception:
            continue
        ds = config.get("dataset_name") or config.get("data_set", "")
        dataset, category = classify(ds)
        if dataset is None:
            continue
        runs.append(
            {
                "dir": path,
                "dataset": dataset,
                "category": category,
                "model": config.get("model_name") or config.get("model", "model"),
                "seed": config.get("seed"),
                "metric": compute_metric(path),
            }
        )
    return runs


def make_table(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["dataset"], r["model"]), []).append(r)

    rows = []
    for (dataset, model), rs in sorted(groups.items()):
        cats = {"original": [], "stats": [], "encoded": []}
        by_seed = {"original": {}, "stats": {}, "encoded": {}}
        for r in rs:
            if r["metric"] is None:
                continue
            cats[r["category"]].append(r["metric"])
            by_seed[r["category"]][r["seed"]] = r["metric"]

        cells = {c: mean_std(cats[c]) for c in cats}

        # per-seed relative gains vs the original dataset (same seed)
        seeds = (
            set(by_seed["original"]) & set(by_seed["stats"]) & set(by_seed["encoded"])
        )
        gains = {"stats": [], "encoded": []}
        for s in seeds:
            base = by_seed["original"][s]
            if base in (None, 0.0):
                continue
            gains["stats"].append(by_seed["stats"][s] - base)
            gains["encoded"].append(by_seed["encoded"][s] - base)

        stat_gain = mean_std(gains["stats"])[0]
        enc_gain = mean_std(gains["encoded"])[0]
        paired = [
            (g_s, g_e)
            for g_s, g_e in zip(gains["stats"], gains["encoded"])
            if g_s is not None and g_e is not None
        ]
        avg_max_gain = (
            mean_std([max(a, b) for a, b in paired]) if paired else (None, None)
        )

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "orig": cells["original"],
                "stats": cells["stats"],
                "enc": cells["encoded"],
                "stat_gain": (stat_gain, mean_std(gains["stats"])[1]),
                "enc_gain": (enc_gain, mean_std(gains["encoded"])[1]),
                "avg_max_gain": avg_max_gain,
                "n_seeds": len(cats["original"]),
            }
        )

    per_dataset = {}
    for row in rows:
        per_dataset.setdefault(row["dataset"], []).append(row)

    # green highlight: scale 0..50% opacity with the average max gain
    pos_gains = [
        r["avg_max_gain"][0]
        for r in rows
        if r["avg_max_gain"][0] is not None and not math.isnan(r["avg_max_gain"][0])
    ]
    max_gain_all = max(pos_gains) if pos_gains and max(pos_gains) > 0 else None

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Performance comparison of anomaly detection models across "
        "feature configurations (ROC-AUC in %, mean$\\pm$std).}"
    )
    lines.append("\\label{tab:roc_auc}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append(
        "Dataset & Model & Original & +Stats & +Stats +Encoded & Max Gain \\\\"
    )
    lines.append("\\midrule")

    n_groups = 0
    for dataset in DATASETS:
        ds_rows = per_dataset.get(dataset, [])
        if not ds_rows:
            continue
        n_model = len(ds_rows)
        for i, row in enumerate(ds_rows):
            cells = []
            means = {c: row[c][0] for c in ("orig", "stats", "enc")}
            valid = [m for m in means.values() if m is not None]
            best = max(valid) if valid else None
            for c in ("orig", "stats", "enc"):
                mu, sd = row[c]
                if mu is None:
                    cells.append("--")
                else:
                    body = "$%.2f\\pm%.2f$" % (
                        mu * 100.0,
                        (sd if sd is not None else 0.0) * 100.0,
                    )
                    cells.append(
                        body
                        if (best is None or mu < best - 1e-9)
                        else "$\\mathbf{%.2f}\\pm\\mathbf{%.2f}$"
                        % (mu * 100.0, (sd if sd is not None else 0.0) * 100.0)
                    )
            gain_mean, gain_std = row["avg_max_gain"]
            if gain_mean is not None and not math.isnan(gain_mean):
                sign = "+" if gain_mean >= 0 else ""
                body = "$%s%.2f\\pm%.2f$" % (
                    sign,
                    gain_mean * 100.0,
                    (gain_std if gain_std is not None else 0.0) * 100.0,
                )
                if gain_mean >= 0 and max_gain_all is not None:
                    intensity = max(0.0, 50.0 * (gain_mean / max_gain_all))
                    cells.append("\\cellcolor{green!%.0f}%s" % (intensity, body))
                else:
                    cells.append(body)
            else:
                cells.append("--")
            if i == 0:
                row_str = (
                    ("\\multirow{%d}{*}{%s}" % (n_model, dataset))
                    + " & "
                    + row["model"]
                    + " & "
                    + " & ".join(cells)
                    + " \\\\"
                )
            else:
                row_str = " & " + row["model"] + " & " + " & ".join(cells) + " \\\\"
            lines.append(row_str)
        n_groups += 1
        if n_groups < len(per_dataset):
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    print("\n".join(lines))
    print("\n# per-category gain (mean+/-std over seeds, vs original):")
    for row in rows:
        print(
            "  %-9s %-10s +Stats %s  +Stats+Encoded %s"
            % (
                row["dataset"],
                row["model"],
                fmt(row["stat_gain"][0], as_pct=True),
                fmt(row["enc_gain"][0], as_pct=True),
            )
        )


def main():
    p = argparse.ArgumentParser(description="Build the ROC-AUC LaTeX table")
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
        print("no usable runs found under %s" % " ".join(args.checkpoint_root))
        return
    print("found %d classified runs" % len(runs))
    make_table(runs)


if __name__ == "__main__":
    main()
