#!/usr/bin/env python3
"""Generate LaTeX tables for the AER test results.

Reads the result.json files written by calcshape.py (results/shap/<dataset>-<feets>/)
and produces:

  1. an AUC table: per dataset the test-split ROC-AUC for raw / stats / statsdim
     (mean +- std over seeds) plus the max-gain column (best(stats, statsdim) - raw),
  2. a SHAP table: statistic vs embedding share (%) of the |SHAP| attribution
     (class_all) for the stats and statsdim feature sets.

Example:
    python make_table.py                 # uses ./results/shap
    python make_table.py -o table.tex
"""
from __future__ import print_function

import argparse
import json
import math
import os

DATASETS = ("reddit", "wikipedia", "mooc")
FEATS = ("raw", "stats", "statsdim")


def load_result(results_root, dataset, feets):
    path = os.path.join(results_root, "{}-{}".format(dataset, feets), "result.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def mean_std(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return mu, math.sqrt(var)


def fmt(v, as_pct=False, signed=False):
    if v is None or math.isnan(v):
        return "--"
    if as_pct:
        return ("+%.1f\\%%" if signed else "%.1f\\%%") % (v * 100.0)
    return "%.4f" % v


def auc_str(result):
    if result is None or "auc" not in result:
        return "--"
    mu, sd = mean_std(result["auc"].get("roc_auc", {}).get("per_seed", {}).values())
    if mu is None:
        return "--"
    return "{:.4f}$\\pm${:.4f}".format(mu, sd or 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="./results/shap", help="results/shap dir")
    p.add_argument("-o", "--output", default=None, help="write LaTeX to this file (default: stdout)")
    args = p.parse_args()

    lines = []

    # ------------------------------------------------------------------ AUC
    lines.append("% AER test-split ROC-AUC, mean +- std over seeds")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Dataset & raw & stats & statsdim & max gain \\")
    lines.append(r"\hline")
    for ds in DATASETS:
        raw = auc_from_result(load_result(args.results, ds, "raw"))
        stats = auc_from_result(load_result(args.results, ds, "stats"))
        statsdim = auc_from_result(load_result(args.results, ds, "statsdim"))

        raw_mu = _auc_mean(load_result(args.results, ds, "raw"))
        stats_mu = _auc_mean(load_result(args.results, ds, "stats"))
        statsdim_mu = _auc_mean(load_result(args.results, ds, "statsdim"))
        candidates = [v for v in (stats_mu, statsdim_mu) if v is not None]
        gain = (max(candidates) - raw_mu) if (candidates and raw_mu is not None) else None

        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                ds, raw, stats, statsdim, fmt(gain, as_pct=True, signed=True)
            )
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    # ------------------------------------------------------------------ SHAP table
    lines.append("")
    lines.append("Statistic-feature share of the mean |SHAP| (class all):")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\hline")
    lines.append(r"dataset & stats & statsdim \\")
    lines.append(r"\hline")
    for ds in DATASETS:
        row = [ds]
        for feats in ("stats", "statsdim"):
            res = load_result(args.results, ds, feats)
            share = None
            if res is not None and "class_all" in res:
                share = res["class_all"].get("statistic_share_mean")
            row.append(fmt(share, as_pct=True))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print("wrote", args.output)
    else:
        print(text)


def _auc_metrics(res):
    if res is None or "auc" not in res:
        return None
    return res["auc"].get("roc_auc", {})


def _auc_mean(res):
    m = _auc_metrics(res)
    if not m or "per_seed" not in m:
        return None
    vals = [v for v in m["per_seed"].values() if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def auc_from_result(res):
    m = _auc_metrics(res)
    if not m:
        return "--"
    mu = m.get("mean")
    sd = m.get("std")
    if mu is None:
        return "--"
    return "{:.3f}$\\pm${:.3f}".format(mu, sd or 0.0)


if __name__ == "__main__":
    main()