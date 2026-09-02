import pandas as pd
import numpy as np
import os
from collections import defaultdict
import random
import utils

DATA_DIR = "./data"

FEETS_SUFFIX = {"raw": "", "stats": "_stats", "statsdim": "_statsdim"}


def check_edge_feat_coverage(max_edge_id, e_feat, context=""):
    """Fail early with a clear message if the edge-feature array cannot cover all edge ids."""
    if e_feat is None:
        return
    if max_edge_id >= len(e_feat):
        raise ValueError(
            f"[{context}] edge-feature array has {len(e_feat)} rows but edge ids go up to "
            f"{max_edge_id}. Missing edge ids: {max(0, len(e_feat))}..{max_edge_id}. "
            f"This means the features and the edge csv are out of sync. "
            f"Re-run `preprocessing.py --feets <feets>` and `generateNeg.py --feets <feets>` "
            f"so that {context} csv and features are generated from the same source."
        )


def generate_data(src_l, dst_l, e_idx_l, ts_l, label_l, max_idx):
    if len(src_l) == 0:
        return np.empty((0, 5), dtype=np.int64)
    full_list = [[] for _ in range(src_l.max() + 1)]
    for src, dst, eidx, ts, label in zip(src_l, dst_l, e_idx_l, ts_l, label_l):
        full_list[src].append([dst, ts, eidx, label])
    src_data = []
    count = 0
    l1000 = 0
    l500 = 0
    lo = 0
    for src in range(len(full_list)):
        ut = full_list[src]
        newlist = sorted(ut, key=lambda x: x[1], reverse=False)

        for idx in range(3, len(newlist)):
            src_data.append(
                [
                    src,
                    int(newlist[idx][0]),
                    int(newlist[idx][1]),
                    int(newlist[idx][2]),
                    newlist[idx][3],
                ]
            )
    return np.array(src_data)


import ast


# LOAD THE NETWORK
def load_data_amazon(src_l, dst_l, e_idx_l, label_l, ts_l, predict_l):
    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = 0
    y_true_labels = []
    e_idx_sequence = []

    for user, item, t, label, edge_id, p in zip(src_l, dst_l, ts_l, label_l, e_idx_l, predict_l):
        time = int(t)
        if start_timestamp > time:
            start_timestamp = time
        p = int(p)
        if p == 0:
            continue
        user_sequence.append(int(user))
        item_sequence.append(int(item))
        timestamp_sequence.append(time)
        y_true_labels.append(int(label))
        e_idx_sequence.append(edge_id)
    timestamp_sequence = np.array(timestamp_sequence)
    user_sequence_id = user_sequence
    item_sequence_id = item_sequence

    user_sequence_id = np.array(user_sequence_id)
    item_sequence_id = np.array(item_sequence_id)
    y_true_labels = np.array(y_true_labels)
    e_idx_sequence = np.array(e_idx_sequence)
    return user_sequence_id, item_sequence_id, timestamp_sequence, y_true_labels, e_idx_sequence


# LOAD THE NETWORK
def load_data(datapath, df, time_scaling=True):
    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = 20210707
    y_true_labels = []
    e_idx_sequence = []
    f = open(datapath, "r")
    f.readline()
    dataset = []
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list
        ls = l.strip().split(",")
        dataset.append([ls[0], ls[1], ls[2], ls[3]])
    f.close()
    random.shuffle(dataset)

    for cnt, ls in enumerate(dataset):
        user_sequence.append(int(ls[0]))
        item_sequence.append(int(ls[1]))
        if start_timestamp > float(ls[2]):
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]))
        y_true_labels.append(int(ls[3]))  # label = 1 at state change, 0 otherwise
        e_idx_sequence.append(cnt + 1)
    timestamp_sequence = np.array(timestamp_sequence) - start_timestamp
    print("Formating user sequence")
    nodeid = 1
    user2id = {}
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]
    print("Formating item sequence")
    item2id = {}
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating dropping")
    userdrop = {}
    for uidx, itemidx, t, y in zip(
        user_sequence_id, item_sequence_id, timestamp_sequence, y_true_labels
    ):
        if y == 1:
            userdrop[uidx] = (itemidx, t)

    user_sequence_id = np.array(user_sequence_id)
    item_sequence_id = np.array(item_sequence_id)
    y_true_labels = np.array(y_true_labels)
    e_idx_sequence = np.array(e_idx_sequence)
    return (
        user_sequence_id,
        item_sequence_id,
        userdrop,
        timestamp_sequence,
        y_true_labels,
        e_idx_sequence,
        user2id,
        item2id,
    )


def loadAmazonData(DATA, trainRatio=0.6):
    # Load data and sanity check
    g_df = pd.read_csv("./data/amazon_filter_rp3_hf20_rid_small.csv")
    src_or = g_df.u.values
    dst_or = g_df.i.values
    e_idx_or = g_df.idx.values
    label_or = g_df.label.values
    ts_or = g_df.ts.values

    predict_or = g_df.predict.values

    src_l, dst_l, ts_l, label_l, e_idx_l = load_data_amazon(
        src_or, dst_or, e_idx_or, label_or, ts_or, predict_or
    )

    e_feat = None
    n_feat = None

    max_idx = max(src_or.max(), dst_or.max())

    val_idx, test_idx = src_l.max() * trainRatio, src_l.max() * 0.7
    valid_train_flag = src_l <= val_idx

    train_src_l = src_l[valid_train_flag]

    # define the new nodes sets for testing inductiveness of the model
    total_node_set = set(src_l)
    train_node_set = set(train_src_l)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    new_node_set = total_node_set - train_node_set

    # select validation and test dataset
    valid_val_flag = (src_l <= test_idx) * (src_l > val_idx)
    valid_test_flag = src_l > test_idx
    is_new_node_edge = np.array([(a in new_node_set) for a in src_l])

    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    # validation and test with edges
    val_src_l = src_l[nn_val_flag]
    val_dst_l = dst_l[nn_val_flag]
    val_ts_l = ts_l[nn_val_flag]
    val_e_idx_l = e_idx_l[nn_val_flag]
    val_label_l = label_l[nn_val_flag]

    test_src_l = src_l[nn_test_flag]
    test_dst_l = dst_l[nn_test_flag]
    test_ts_l = ts_l[nn_test_flag]
    test_e_idx_l = e_idx_l[nn_test_flag]
    test_label_l = label_l[nn_test_flag]

    src_data_train = generate_data(
        train_src_l, train_dst_l, train_e_idx_l, train_ts_l, train_label_l, max_idx
    )
    if len(val_src_l) == 0:
        src_data_val = np.empty((0, 5), dtype=np.int64)
    else:
        src_data_val = generate_data(
            val_src_l, val_dst_l, val_e_idx_l, val_ts_l, val_label_l, max_idx
        )

    if len(test_src_l) == 0:
        src_data_test = np.empty((0, 5), dtype=np.int64)
    else:
        src_data_test = generate_data(
            test_src_l, test_dst_l, test_e_idx_l, test_ts_l, test_label_l, max_idx
        )
    print("train val test size", len(src_data_train), len(src_data_val), len(src_data_test))

    td_true = []
    td_false = []
    for td in src_data_train:
        if td[-1] == 1:
            td_true.append(td)
        else:
            td_false.append(td)
    if len(td_true) == 0:
        for td in src_data_val:
            if td[-1] == 1:
                td_true.append(td)
    train_val_data = (td_true, td_false, src_data_val)
    test_data = src_data_test
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_or, dst_or, e_idx_or, ts_or):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))

    degree = []
    for src in src_l:
        degree.append(len(full_adj_list[src]))
    degree = []
    for src in dst_l:
        degree.append(len(full_adj_list[src]))
    return train_val_data, test_data, [], full_adj_list, e_feat, n_feat, max_idx


def loadAnomalData(DATA, data_mode="real_synth", trainRatio=0.6, valRatio=0.1, feets="raw"):
    data_dir = DATA_DIR
    suffix = FEETS_SUFFIX[feets]
    # Load data and sanity check
    src_l, dst_l, userdrop, ts_l, label_l, e_idx_l, user2id, item2id = load_data(
        f"{data_dir}/{DATA}/{DATA}{suffix}.csv", DATA
    )
    e_feat = None
    n_feat = None

    if os.path.isfile(f"{data_dir}/ml_{DATA}{suffix}.npy"):
        e_feat = np.load(f"{data_dir}/ml_{DATA}{suffix}.npy")
    max_idx = max(src_l.max(), dst_l.max())
    print(src_l.max(), dst_l.max() - src_l.max())
    print(len(e_idx_l), sum(label_l))
    val_idx = int(src_l.max() * trainRatio)
    test_idx = int(src_l.max() * (trainRatio + valRatio))
    print(val_idx, test_idx)
    valid_train_flag = src_l <= val_idx

    train_src_l = src_l[valid_train_flag]

    # define the new nodes sets for testing inductiveness of the model
    total_node_set = set(src_l)
    train_node_set = set(train_src_l)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    new_node_set = total_node_set - train_node_set

    # select validation and test dataset
    valid_val_flag = (src_l <= test_idx) * (src_l > val_idx)
    valid_test_flag = src_l > test_idx
    is_new_node_edge = np.array([(a in new_node_set) for a in src_l])

    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    # validation and test with edges
    val_src_l = src_l[nn_val_flag]
    val_dst_l = dst_l[nn_val_flag]
    val_ts_l = ts_l[nn_val_flag]
    val_e_idx_l = e_idx_l[nn_val_flag]
    val_label_l = label_l[nn_val_flag]

    test_src_l = src_l[nn_test_flag]
    test_dst_l = dst_l[nn_test_flag]
    test_ts_l = ts_l[nn_test_flag]
    test_e_idx_l = e_idx_l[nn_test_flag]
    test_label_l = label_l[nn_test_flag]

    src_data_train = generate_data(
        train_src_l, train_dst_l, train_e_idx_l, train_ts_l, train_label_l, max_idx
    )
    src_data_val = generate_data(val_src_l, val_dst_l, val_e_idx_l, val_ts_l, val_label_l, max_idx)
    src_data_test = generate_data(
        test_src_l, test_dst_l, test_e_idx_l, test_ts_l, test_label_l, max_idx
    )
    print("train val test size", len(src_data_train), len(src_data_val), len(src_data_test))

    # ---- synthetic anomaly augmentation ----
    aug_src_l = aug_dst_l = aug_e_idx_l = aug_ts_l = None
    if data_mode in ("real_synth", "synth"):
        cont_full_path = f"{data_dir}/{DATA}/AER/{DATA}_cont_full{suffix}.csv"
        if not os.path.isfile(cont_full_path):
            raise FileNotFoundError(
                f"cont_full.csv not found at {cont_full_path}. "
                f"Run `python generateNeg.py --dataset {DATA}` first to create synthetic anomalies, "
                f"or use --data_mode real."
            )
        g_df_new = pd.read_csv(cont_full_path)
        aug_src_l = g_df_new.u.values
        aug_dst_l = g_df_new.i.values
        aug_e_idx_l = g_df_new.idx.values
        aug_ts_l = g_df_new.ts.values
        aug_label_l = g_df_new.label.values
        if len(aug_src_l) == 0:
            raise ValueError(
                f"cont_full.csv at {cont_full_path} is empty. "
                f"generateNeg.py produced no synthetic anomalies (early-exit triggered). "
                f"Check generateNeg.py output or use --data_mode real."
            )
        max_idx = max(max_idx, aug_src_l.max(), aug_dst_l.max())
        src_data_aug = generate_data(
            aug_src_l, aug_dst_l, aug_e_idx_l, aug_ts_l, aug_label_l, max_idx
        )
        print(f"loaded {len(src_data_aug)} synthetic interactions (data_mode={data_mode})")
        if data_mode == "synth":
            src_data_train = src_data_aug
        else:  # real_synth
            if len(src_data_train) > 0 and len(src_data_aug) > 0:
                src_data_train = np.concatenate([src_data_train, src_data_aug])
            elif len(src_data_aug) > 0:
                src_data_train = src_data_aug
        print("train size after augmentation:", len(src_data_train))

    td_true = []
    td_false = []
    for td in src_data_train:
        if td[-1] == 1:
            td_true.append(td)
        else:
            td_false.append(td)
    if len(td_true) == 0 and len(src_data_val) > 0:
        for td in src_data_val:
            if td[-1] == 1:
                td_true.append(td)
                break
    train_val_data = (td_true, td_false, src_data_val)
    test_data = src_data_test
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    # extend full_adj_list with synthetic edges (needed for HistoryFinder)
    if aug_src_l is not None:
        for src, dst, eidx, ts in zip(aug_src_l, aug_dst_l, aug_e_idx_l, aug_ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))
    M = dst_l.max() - src_l.max() + 1
    N = src_l.max() + 1

    degree = []
    for src in src_l:
        degree.append(len(full_adj_list[src]))
    print(
        "source degree mean {},median {}, max {}, min {}".format(
            np.mean(degree), np.median(degree), max(degree), min(degree)
        )
    )

    degree = []
    for src in dst_l:
        degree.append(len(full_adj_list[src]))
    print(
        "dest degree mean {},median {}, max {}, min {}".format(
            np.mean(degree), np.median(degree), max(degree), min(degree)
        )
    )

    return train_val_data, test_data, [], full_adj_list, e_feat, n_feat, max_idx


def loadDropoutData(
    DATA,
    data_mode="real_synth",
    trainRatio=0.6,
    valRatio=0.1,
    augType=None,
    e_feat=None,
    feets="raw",
):
    data_dir = DATA_DIR
    suffix = FEETS_SUFFIX[feets]
    g_df = pd.read_csv(f"{data_dir}/{DATA}/{DATA}{suffix}.csv")
    # rename the columns
    g_df.columns = ["u", "i", "ts", "label"]
    g_df["idx"] = list(range(len(g_df)))

    e_feat = None
    n_feat = None
    if DATA in ("reddit", "mooc", "wikipedia"):
        e_feat = np.load(f"{data_dir}/{DATA}/features{suffix}.npy")
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    label_l = g_df.label.values
    ts_l = g_df.ts.values
    max_idx = max(src_l.max(), dst_l.max())
    split_path = f"{data_dir}/{DATA}/{DATA}_split{suffix}.npy"
    if os.path.isfile(split_path):
        print(f"load {split_path}")
        src_data = np.load(split_path)
        val_idx, test_idx = int(len(src_data) * trainRatio), int(
            len(src_data) * (trainRatio + valRatio)
        )
        src_train, src_val, src_test = (
            src_data[:val_idx],
            src_data[val_idx:test_idx],
            src_data[test_idx:],
        )
    else:
        print("Random splitting data (stratified: anomaly and normal users interleaved)")
        src_data = utils.create_stratified_split(g_df)
        os.makedirs(f"{data_dir}/{DATA}", exist_ok=True)
        np.save(split_path, src_data)
        print("Please run again")
        exit()

    full_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts, label in zip(src_l, dst_l, e_idx_l, ts_l, label_l):
        full_list[src].append((dst, ts, eidx, label))

    train_data = []
    for src in src_train:
        ut = full_list[src]
        newlist = sorted(ut, key=lambda x: x[1], reverse=False)
        if len(newlist) == 0:
            continue
        train_data.append(
            [src, int(newlist[-1][0]), int(newlist[-1][1]), int(newlist[-1][2]), newlist[-1][3]]
        )

    val_data = []
    for src in src_val:
        ut = full_list[src]
        newlist = sorted(ut, key=lambda x: x[1], reverse=False)
        if len(newlist) == 0:
            continue
        val_data.append(
            [src, int(newlist[-1][0]), int(newlist[-1][1]), int(newlist[-1][2]), newlist[-1][3]]
        )
    val_data = np.array(val_data)

    test_data = []
    for src in src_test:
        ut = full_list[src]
        newlist = sorted(ut, key=lambda x: x[1], reverse=False)
        if len(newlist) == 0:
            continue
        test_data.append(
            [src, int(newlist[-1][0]), int(newlist[-1][1]), int(newlist[-1][2]), newlist[-1][3]]
        )

    test_data = np.array(test_data)
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    degree = []
    for src in src_l:
        degree.append(len(full_adj_list[src]))

    degree = []
    for src in dst_l:
        degree.append(len(full_adj_list[src]))
    if data_mode in ("real_synth", "synth"):
        cont_full_path = f"{data_dir}/{DATA}/AER/{DATA}_cont_full{suffix}.csv"
        if not os.path.isfile(cont_full_path):
            raise FileNotFoundError(
                f"cont_full.csv not found at {cont_full_path}. "
                f"Run `python generateNeg.py --dataset {DATA}` first to create synthetic anomalies, "
                f"or use --data_mode real."
            )
        print("load contrastive data")
        if e_feat is not None:
            e_feat = np.load(f"{data_dir}/{DATA}/AER/features_cont_full{suffix}.npy")
        g_df_new = pd.read_csv(cont_full_path)
        src_l2 = g_df_new.u.values
        dst_l2 = g_df_new.i.values
        e_idx_l2 = g_df_new.idx.values
        label_l2 = g_df_new.label.values
        ts_l2 = g_df_new.ts.values
        if len(src_l2) == 0:
            raise ValueError(
                f"cont_full.csv at {cont_full_path} is empty. "
                f"generateNeg.py produced no synthetic anomalies (early-exit triggered). "
                f"Check generateNeg.py output or use --data_mode real."
            )
        max_idx = max(src_l2.max(), dst_l2.max())
        cont_dict = dict()
        for src, dst, eidx, ts, label in zip(src_l2, dst_l2, e_idx_l2, ts_l2, label_l2):
            if src not in cont_dict:
                cont_dict[src] = [(dst, ts, eidx, label)]
            else:
                cont_dict[src].append((dst, ts, eidx, label))
        aug_data = []
        for src in cont_dict:
            ut = cont_dict[src]
            newlist = sorted(ut, key=lambda x: x[1], reverse=False)
            if len(newlist) == 0:
                continue
            aug_data.append(
                [
                    src,
                    int(newlist[-1][0]),
                    int(newlist[-1][1]),
                    int(newlist[-1][2]),
                    newlist[-1][3],
                ]
            )
        print(f"loaded {len(aug_data)} synthetic interactions")

        full_adj_list_cont = [[] for _ in range(max_idx + 1)]
        for src in range(len(full_adj_list)):
            full_adj_list_cont[src] = full_adj_list[src]
        for src, dst, eidx, ts in zip(src_l2, dst_l2, e_idx_l2, ts_l2):
            full_adj_list_cont[src].append((dst, eidx, ts))
            full_adj_list_cont[dst].append((src, eidx, ts))

        full_adj_list = full_adj_list_cont
        src_l = list(src_l) + list(src_l2)
        dst_l = list(dst_l) + list(dst_l2)
        e_idx_l = list(e_idx_l) + list(e_idx_l2)
        ts_l = list(ts_l) + list(ts_l2)

        if data_mode == "synth":
            train_data = aug_data
        else:  # real_synth
            train_data = train_data + aug_data

    td_true = []
    td_false = []
    for td in train_data:
        if td[-1] == 1:
            td_true.append(td)
        else:
            td_false.append(td)

    train_val_data = (td_true, td_false, val_data)

    check_edge_feat_coverage(max(e_idx_l), e_feat, f"{DATA} (feets={feets})")

    return (
        train_val_data,
        test_data,
        full_adj_list,
        e_feat,
        n_feat,
        max_idx,
        (np.array(src_l), np.array(dst_l), np.array(e_idx_l), np.array(ts_l)),
    )
