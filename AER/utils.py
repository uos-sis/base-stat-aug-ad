import numpy as np
import json
import torch
import os
import random
from collections import defaultdict


def load_project_config(config_filename: str = "config.json"):
    """
    Load project configuration from a JSON file.

    Args:
        config_filename: Name of the configuration file (default: "config.json")

    Returns:
        Dictionary containing the loaded configuration

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        json.JSONDecodeError: If the configuration file is not valid JSON
    """
    try:
        # Get the root directory of the repository (2 levels up from current file)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        config_path = os.path.join(repo_root, config_filename)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as json_file:
            config = json.load(json_file)

        return config

    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val, epoch_num):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        elif curr_val <= self.last_best:
            self.num_round += 1
        # else:
        #     self.num_round += 1
        self.epoch_count = epoch_num
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        src_list = np.concatenate(src_list)
        dst_list = np.concatenate(dst_list)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers


def create_stratified_split(g_df, seed=42):
    """
    Build a user-level train/val/test split that separates anomaly users (users
    with at least one label==1 interaction) from normal (label==0 only) users,
    shuffles each group separately and then interleaves them evenly (quotient
    method) so that every contiguous slice of the returned array contains users of
    both classes. This guarantees that any later train/val/test ratio slicing
    yields both classes, keeping the AUC computable on every split. The ordering
    stays random (no chronological sorting), only the stratification is added.

    Args:
        g_df: DataFrame with at least the columns "u" (source user id) and
              "label" (interaction label).
        seed: fixed random seed so splits are reproducible across machines.

    Returns:
        np.ndarray of source user ids, evenly mixing anomaly and normal users.
    """
    import random as _random

    u_all = np.asarray(g_df["u"].values)
    label_all = np.asarray(g_df["label"].values)
    anomaly = sorted(set(int(u) for u in u_all[label_all == 1]))
    normal = sorted(set(int(u) for u in u_all) - set(anomaly))
    rng = _random.Random(seed)
    rng.shuffle(anomaly)
    rng.shuffle(normal)

    num_a, num_n = len(anomaly), len(normal)
    total = num_a + num_n
    merged = []
    a_i = n_i = 0
    for i in range(total):
        if (i + 1) * num_a // total > i * num_a // total:
            merged.append(anomaly[a_i])
            a_i += 1
        else:
            merged.append(normal[n_i])
            n_i += 1
    print(
        "stratified split: {} anomaly users, {} normal users -> both classes in every slice".format(
            num_a, num_n
        )
    )
    return np.array(merged)
