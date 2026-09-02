import sys
import os

# Add the current directory to the path to ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Remove any compiled graph modules from cache
if "graph" in sys.modules:
    del sys.modules["graph"]
if "graph.graph" in sys.modules:
    del sys.modules["graph.graph"]

# Import with error handling and alternative loading
HistoryFinder = None
try:
    # Try direct file import first to avoid module conflicts
    import importlib.util

    graph_file_path = os.path.join(os.path.dirname(__file__), "graph", "graph.py")

    if os.path.exists(graph_file_path):
        spec = importlib.util.spec_from_file_location("graph_module", graph_file_path)
        graph_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_module)
        HistoryFinder = graph_module.HistoryFinder
        print("Successfully imported HistoryFinder using direct file import")
    else:
        print(f"Graph file not found at {graph_file_path}")
        raise ImportError("graph.py file not found")

except Exception as e:
    print(f"Error importing HistoryFinder: {e}")
    print("Please ensure:")
    print("1. The graph/graph.py file exists")
    print("2. The C++ library is compiled: cd graph && g++ -shared -fPIC -o graph.so graph.cpp")
    print("3. No conflicting .so files in the graph directory")
    sys.exit(1)

if HistoryFinder is None:
    print("Failed to import HistoryFinder")
    sys.exit(1)

from parserpara import *
import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
from module_sample import RAPAD
import random
import os
from loaddata import *
import torch
import numpy as np

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
DROP_OUT = args.drop_out
DATA = args.data
LEARNING_RATE = args.lr
SEED = args.seed
raps = args.raps
rapd = args.rapd
STEP = args.step
DATA_MODE = args.data_mode
FEETS = args.feets
WANDB = args.wandb_logging
if WANDB:
    wandb.init(
        project="anomaly-detection",
        name=f"AER-new-{DATA}-tr{args.train_ratio}-vr{args.val_ratio}-seed{SEED}-{DATA_MODE}-{FEETS}-run",
        config=vars(args),
    )
mask_num = args.mask


if not torch.cuda.is_available():
    raise RuntimeError(
        "AER must be trained on GPU but CUDA is not available. "
        "Run with a CUDA-enabled torch (e.g. the `aer` conda env)."
    )
device = torch.device("cuda")
print(f"[device] training on GPU: {torch.cuda.get_device_name(device)}")
set_random_seed(SEED)
if __name__ == "__main__":
    logger, get_checkpoint_path, best_model_path = set_up_logger(
        args, sys_argv, mask_num, NUM_NEIGHBORS, STEP, DATA, feets=FEETS
    )
    if (DATA == "mooc") or (DATA == "reddit") or (DATA == "wikipedia"):
        train_val_data, test_data, full_adj_list, e_feat, n_feat, max_idx, dataset = (
            loadDropoutData(
                DATA, DATA_MODE, trainRatio=args.train_ratio, valRatio=args.val_ratio, feets=FEETS
            )
        )
    elif DATA == "amazon_filter":
        train_val_data, test_data, _, full_adj_list, e_feat, n_feat, max_idx = loadAmazonData(DATA)
    else:
        train_val_data, test_data, full_adj_train, full_adj_list, e_feat, n_feat, max_idx = (
            loadAnomalData(
                DATA, DATA_MODE, trainRatio=args.train_ratio, valRatio=args.val_ratio, feets=FEETS
            )
        )

    inter_history = HistoryFinder()
    inter_history.init_off_set(full_adj_list)

    rap = RAPAD(
        n_feat,
        e_feat,
        tf_matrix=None,
        drop_out=DROP_OUT,
        num_neighbors=NUM_NEIGHBORS,
        get_checkpoint_path=get_checkpoint_path,
        step=STEP,
        mask_num=mask_num,
    )
    rap.corr_encoder.graph = inter_history
    rap.to(device)
    nodetime2emb_maps = dict()
    if e_feat is not None:
        for idx in range(len(e_feat)):
            nodetime2emb_maps[str(idx)] = e_feat[idx]

    rap.corr_encoder.init_edge2emb(nodetime2emb_maps)
    rap.update_ngh_finder(inter_history)

    if e_feat is not None:
        max_edge_id = -1
        for part in (train_val_data[0], train_val_data[1], train_val_data[2], test_data):
            part = np.asarray(part)
            if part.ndim == 2 and part.size > 0 and part.shape[1] > 3:
                max_edge_id = max(max_edge_id, int(part[:, 3].max()))
        if max_edge_id >= len(e_feat):
            raise ValueError(
                "edge-feature array has {} rows but the data contains edge id {}. "
                "Features and edge csv are out of sync. Re-run preprocessing.py and "
                "generateNeg.py with the same --feets to regenerate consistent files.".format(
                    len(e_feat), max_edge_id
                )
            )

    optimizer = torch.optim.Adam(rap.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    # train and val (model selection by pooled val auc, test logged per epoch for diagnostics)
    best_val_auc, best_epoch, best_test_auc = train_val(
        train_val_data,
        rap,
        BATCH_SIZE,
        NUM_EPOCH,
        criterion,
        optimizer,
        inter_history,
        logger,
        test_data=test_data,
        best_model_path=best_model_path,
        patience=args.patience,
    )

    # load best model (selected by pooled val auc) for final test evaluation
    if os.path.exists(best_model_path):
        logger.info(
            "Loading best model from {} (best epoch {})".format(best_model_path, best_epoch)
        )
        rap.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.info(
            "No best model checkpoint found at {} - using final model state".format(best_model_path)
        )

    # test on best model
    test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = (
        test_data[:, 0],
        test_data[:, 1],
        test_data[:, 2],
        test_data[:, 3],
        test_data[:, 4],
    )
    rap.update_ngh_finder(
        inter_history
    )  # remember that testing phase should always use the full neighbor finder
    test_acc, test_ap, test_f1, test_auc, test_pr_auc = eval_one_epoch(
        rap, test_src_l, test_dst_l, test_ts_l, test_label_l, val_e_idx_l=test_e_idx_l
    )
    if WANDB:
        wandb.log(
            {
                "test/acc": test_acc,
                "test/ap": test_ap,
                "test/f1": test_f1,
                "test/auc": test_auc,
                "test/pr_auc": test_pr_auc,
            }
        )
        wandb.run.summary["best_val_auc"] = best_val_auc
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_test_auc_at_best_val"] = best_test_auc
        wandb.run.summary["final_test_acc"] = test_acc
        wandb.run.summary["final_test_auc"] = test_auc
        wandb.run.summary["final_test_ap"] = test_ap
        wandb.run.summary["final_test_f1"] = test_f1
        wandb.run.summary["final_test_pr_auc"] = test_pr_auc
    logger.info(
        "Best epoch {} | best val auc {:.6f} | test auc at best val {:.6f}".format(
            best_epoch, best_val_auc, best_test_auc
        )
    )
    logger.info(
        "Final test statistics (best model): -- acc: {}, auc: {}, ap: {}, f1: {}".format(
            test_acc, test_auc, test_ap, test_f1
        )
    )

    logger.info("Saving RAP model")
    torch.save(rap.state_dict(), best_model_path)
    logger.info("RAP models saved")
