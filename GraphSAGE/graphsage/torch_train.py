from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from graphsage.snapshots_data import SnapshotDataset
from graphsage.torch_model import GraphSAGEModel


class EarlyStopMonitor(object):
    """Mirror of SAD/utils.py:EarlyStopMonitor."""

    def __init__(self, max_round=10, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round


def parse_data_set(args):
    if args.data_set:
        return args.data_set
    meta = os.path.join(args.data_dir, args.data_set) + ".json"
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f).get("data_set", "unknown")
    return "unknown"


def log_metrics(labels, preds):
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds)
    keep = labels >= 0
    labels, preds = labels[keep], preds[keep]
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    roc = float(roc_auc_score(labels, preds))
    pr = float(average_precision_score(labels, preds))
    return roc, pr


def evaluate(model, dataset, ids, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for k in ids:
            snap = dataset.snapshots[k]
            x = snap.x.to(device)
            edge_index = snap.edge_index.to(device)
            user_idx = snap.user_idx.to(device)
            logits = model(x, edge_index, user_idx)
            preds.append(logits.sigmoid().cpu().numpy().flatten())
            labels.append(snap.labels.flatten())
    roc, pr = log_metrics(np.concatenate(labels), np.concatenate(preds))
    return roc, pr


def build_config_dict(args, dataset, auc, split_name, epoch):
    cfg = vars(args)
    cfg["seed"] = int(cfg["seed"])
    cfg["feat_dim"] = int(dataset.feat_dim)
    cfg["num_snapshots"] = int(dataset.num_snapshots)
    cfg["split"] = dataset.split_counts()
    cfg["max_degree"] = int(dataset.max_degree)
    cfg["auc"] = float(auc)
    cfg["best_split"] = split_name
    cfg["best_epoch"] = int(epoch)
    cfg["model_file"] = "model.pt"
    cfg["config_file"] = "config.json"
    return cfg


def save_best_model(model, config_dict, checkpoint_dir, split_file):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, split_file)
    current_best = None
    if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        try:
            with open(os.path.join(checkpoint_dir, "config.json")) as f:
                current_best = json.load(f)
        except Exception:
            current_best = None
    if current_best is not None and "auc" in current_best and \
            config_dict["auc"] <= float(current_best["auc"]) - 1e-9 and \
            config_dict["best_split"] == current_best.get("best_split"):
        return  # not strictly better than the model already in this run dir
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print("[checkpoint] saved best %s model to %s (auc=%.4f)" %
          (config_dict["best_split"], checkpoint_dir, config_dict["auc"]))


def main():
    p = argparse.ArgumentParser(description="Train PyG GraphSAGE on JODIE snapshots")
    p.add_argument("--data_set", type=str, required=True,
                  help="dataset name; snapshots are read from <data_dir>/<data_set>.snapshots.npz "
                       "/ <data_dir>/<data_set>.json")
    p.add_argument("--data_dir", type=str, default="./data/",
                  help="directory produced by prepare_jodie_snapshots.py")
    p.add_argument("--model_name", type=str, default="GraphSAGE")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--dim_1", type=int, default=128)
    p.add_argument("--dim_2", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning_rate", type=float, default=0.0005)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--early_stop", type=int, default=100,
                  help="stop after N epochs without improvement (0 disables)")
    p.add_argument("--shuffle_train", action="store_true", default=False)
    p.add_argument("--gpu", type=int, default=0, help="cuda device id, -1 for cpu")
    p.add_argument("--use_wandb", action="store_true", default=True)
    p.add_argument("--wandb_project", type=str, default="CNA2026_GraphSAGE")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online",
                  choices=("online", "offline", "disabled"))
    p.add_argument("--no_progress_bar", action="store_true", default=False)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available()
                        else "cuda:%d" % args.gpu)
    args.data_set = parse_data_set(args)
    snapshot_prefix = os.path.join(args.data_dir, args.data_set)

    print("Loading snapshots..")
    dataset = SnapshotDataset(snapshot_prefix + ".snapshots.npz")
    print("Done loading snapshots (train=%d val=%d test=%d, feat_dim=%d)." %
          (len(dataset.train_ids), len(dataset.val_ids), len(dataset.test_ids), dataset.feat_dim))

    if not args.wandb_run_name:
        args.wandb_run_name = (
            "%s_%s_dim_%d-%d_lr%s_bs-snapshot_seed%s" %
            (args.data_set, args.model_name, args.dim_1, args.dim_2,
             args.learning_rate, args.seed))
    run_dir = os.path.join(args.checkpoint_dir,
                        str(args.wandb_run_name).replace("/", "_").replace("\\", "_"))
    os.makedirs(run_dir, exist_ok=True)

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    model = GraphSAGEModel(dataset.feat_dim, args.dim_1, args.dim_2, args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    val_enabled = len(dataset.val_ids) > 0
    best_auc = [0, 0, 0]
    last_log = None
    max_val_auc, max_test_auc = 0.0, 0.0
    early_stopper = EarlyStopMonitor(max_round=args.early_stop) if args.early_stop > 0 else None

    for epoch in range(args.epochs):
        model.train()
        train_ids_iter = list(dataset.train_ids)
        total_loss = 0.0
        if args.shuffle_train:
            random.shuffle(train_ids_iter)
        if not args.no_progress_bar:
            train_ids_iter = tqdm(train_ids_iter, desc="Epoch %d/%d" % (epoch + 1, args.epochs))
        train_preds, train_labels = [], []
        for k in train_ids_iter:
            snap = dataset.snapshots[k]
            x = snap.x.to(device)
            edge_index = snap.edge_index.to(device)
            user_idx = snap.user_idx.to(device)
            y = snap.y.to(device)
            optimizer.zero_grad()
            logits = model(x, edge_index, user_idx)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), y)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            train_preds.append(logits.sigmoid().detach().cpu().numpy().flatten())
            train_labels.append(snap.labels.flatten())

        train_roc, train_pr = log_metrics(np.concatenate(train_labels), np.concatenate(train_preds))
        if val_enabled:
            val_roc, val_pr = evaluate(model, dataset, dataset.val_ids, device)
        else:
            val_roc, val_pr = np.nan, np.nan
        test_roc, test_pr = evaluate(model, dataset, dataset.test_ids, device)

        log = {
            "train/loss": float(total_loss) / max(len(dataset.train_ids), 1),
            "train/roc-auc": train_roc,
            "train/pr-auc": train_pr,
            "val/roc-auc": val_roc,
            "val/pr-auc": val_pr,
            "test/roc-auc": test_roc,
            "test/pr-auc": test_pr,
        }
        last_log = log
        max_test_auc = max(max_test_auc, test_roc if not np.isnan(test_roc) else 0.0)
        if val_enabled:
            max_val_auc = max(max_val_auc, val_roc if not np.isnan(val_roc) else 0.0)
        if run is not None:
            wandb.log(log, step=epoch)

        print("Epoch %03d | train/loss=%.5f train/roc-auc=%.5f train/pr-auc=%.5f "
              "val/roc-auc=%.5f val/pr-auc=%.5f test/roc-auc=%.5f test/pr-auc=%.5f" %
              (epoch + 1, log["train/loss"], log["train/roc-auc"], log["train/pr-auc"],
               log["val/roc-auc"], log["val/pr-auc"], log["test/roc-auc"], log["test/pr-auc"]))

        if val_enabled:
            if val_roc > best_auc[1]:
                best_auc = [epoch, val_roc, test_roc]
                cfg = build_config_dict(args, dataset, val_roc, "val", epoch)
                save_best_model(model, cfg, run_dir, "model.pt")
        else:
            if test_roc > best_auc[2]:
                best_auc = [epoch, np.nan, test_roc]
                cfg = build_config_dict(args, dataset, test_roc, "test", epoch)
                save_best_model(model, cfg, run_dir, "model.pt")

        if early_stopper is not None:  # re-init each epoch to mirror TF loop step-by-epoch
            if early_stopper.early_stop_check(val_roc if val_enabled else test_roc):
                print("No improvment over {} epochs, stop training".format(early_stopper.max_round))
                break

    print("\n best auc: epoch={}, val={}, test={}".format(best_auc[0], best_auc[1], best_auc[2]))

    if run is not None:
        wandb.summary["max/test_auc"] = float(max_test_auc)
        wandb.summary["best/epoch"] = int(best_auc[0])
        if last_log is not None:
            wandb.summary["last/test/auc_roc"] = float(last_log["test/roc-auc"])
        if val_enabled:
            wandb.summary["max/val_auc"] = float(max_val_auc)
            wandb.summary["best/val_auc"] = float(best_auc[1])
        wandb.summary["best/test_auc"] = float(best_auc[2])
        wandb.finish()


if __name__ == "__main__":
    main()
