import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
from utils import EarlyStopMonitor
import logging
import time
import random
import os
import wandb
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def train_val(train_val_data, model, bs, epochs, criterion, optimizer,
              inter_history, logger, test_data=None, best_model_path=None,
              patience=10):
    # unpack the data, prepare for the training
    train_true, train_false, val_data = train_val_data
    if val_data is None or len(val_data) == 0:
        val_src_l = val_dst_l = val_ts_l = val_e_idx_l = val_label_l = np.array([])
    else:
        val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = (
            val_data[:,0], val_data[:,1], val_data[:,2], val_data[:,3], val_data[:,4]
        )
    # unpack test data (only for diagnostic logging, never used for selection)
    if test_data is None or len(test_data) == 0:
        test_src_l = test_dst_l = test_ts_l = test_e_idx_l = test_label_l = np.array([])
    else:
        test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = (
            test_data[:,0], test_data[:,1], test_data[:,2], test_data[:,3], test_data[:,4]
        )
    model.update_ngh_finder(inter_history)
    train_radio = len(train_false)/len(train_true)

    early_stopper = EarlyStopMonitor(max_round=patience, higher_better=True)
    best_val_auc = -1.0
    best_epoch = -1
    best_test_auc = float('nan')

    for epoch in range(epochs):
        if train_radio<0.5:
            train_true_sample = random.sample(train_true, int(len(train_false)))
        else:
            train_true_sample = train_true

        if train_radio>2:
            train_false_sample = random.sample(train_false, int(len(train_true)))
        else:
            train_false_sample = train_false
        train_data = train_false_sample+train_true_sample
        random.shuffle(train_data)
        train_data = np.array(train_data)
        train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data[:,0],train_data[:,1],train_data[:,2],train_data[:,3],train_data[:,4]

        num_instance = len(train_src_l)
        num_batch = math.ceil(num_instance / bs)
        idx_list = np.arange(num_instance)
        acc, ap, f1, auc, m_loss, all_label = [], [], [], [], [],[]
        logger.info('start {} epoch'.format(epoch))
        logger.info('num of training instances: {}'.format(num_instance))
        logger.info('num of batches per epoch: {}'.format(num_batch))

        edge_num = 0
        zero_partner_num = 0
        zero_partner_or_num = 0
        for k in tqdm(range(num_batch)):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]
            size = min(len(label_l_cut),len(src_l_cut))
            if size<2:
                continue
            try:
                optimizer.zero_grad()
                model.train()
            except:
                print('fail optimizer')
            score,kl_loss =model.contrast(src_l_cut, dst_l_cut, ts_l_cut, e_idx_l=e_l_cut)  # the core training code
            edge_num += len(src_l_cut)
            device = score.device
            true_label = np.array(label_l_cut)
            label_l_cut = torch.FloatTensor(label_l_cut).to(device)
            loss = criterion(score, label_l_cut)
            loss.backward()
            optimizer.step()

        # ---- evaluation every epoch ----
        # validation (used for model selection)
        if val_src_l.size == 0:
            logger.info('epoch {}: no validation data, skip evaluation'.format(epoch))
            continue
        val_acc, val_ap, val_f1, val_auc, val_pr_auc = eval_one_epoch(
            model, val_src_l, val_dst_l, val_ts_l, val_label_l, val_e_idx_l
        )
        # test (diagnostic only, never used for selection)
        if test_src_l.size > 0:
            test_acc, test_ap, test_f1, test_auc, test_pr_auc = eval_one_epoch(
                model, test_src_l, test_dst_l, test_ts_l, test_label_l,
                val_e_idx_l=test_e_idx_l
            )
        else:
            test_acc = test_ap = test_f1 = test_auc = test_pr_auc = float('nan')

        log_payload = {
            "val/acc": val_acc,
            "val/ap": val_ap,
            "val/f1": val_f1,
            "val/auc": val_auc,
            "val/pr_auc": val_pr_auc,
            "test/acc": test_acc,
            "test/ap": test_ap,
            "test/f1": test_f1,
            "test/auc": test_auc,
            "test/pr_auc": test_pr_auc,
            "epoch": epoch,
        }
        if wandb.run is not None:
            wandb.log(log_payload)
        logger.info(
            'epoch {}: val auc {:.6f} | test auc {:.6f} | val ap {:.6f} | val f1 {:.6f}'.format(
                epoch, val_auc, test_auc, val_ap, val_f1
            )
        )

        # ---- best model selection by pooled val auc ----
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_test_auc = test_auc
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)
                logger.info('epoch {}: new best val auc {:.6f} -> saved to {}'.format(
                    epoch, best_val_auc, best_model_path))

        # ---- early stopping on val auc ----
        if early_stopper.early_stop_check(val_auc, epoch):
            logger.info('early stopping triggered at epoch {} (best epoch {}, best val auc {:.6f})'.format(
                epoch, early_stopper.best_epoch, best_val_auc))
            break

    logger.info('training finished. best epoch {}: best val auc {:.6f} | test auc at best epoch {:.6f}'.format(
        best_epoch, best_val_auc, best_test_auc))
    return best_val_auc, best_epoch, best_test_auc



