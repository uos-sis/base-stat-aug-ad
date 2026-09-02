import torch
import torch.nn.functional as F
import datasets as dataset
import torch.utils.data
import sklearn
import numpy as np
from option import args
from model.tgat import TGAT
from utils import EarlyStopMonitor
from tqdm import tqdm
import datetime, os
import json
import wandb
import random


# torch.use_deterministic_algorithms(True)
def criterion(prediction_dict, labels, model, config):
    for key, value in prediction_dict.items():
        if key != "root_embedding" and key != "group" and key != "dev":
            prediction_dict[key] = value[labels > -1]

    labels = labels[labels > -1]

    # Fix: Convert labels to float for binary cross entropy
    labels = labels.float()

    logits = prediction_dict["logits"]

    loss_classify = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss_classify = torch.mean(loss_classify)

    loss = loss_classify.clone()
    loss_anomaly = torch.tensor([0.0]).to(logits.device)
    loss_supc = torch.tensor([0.0]).to(logits.device)

    alpha = config.anomaly_alpha  # 1e-1
    beta = config.supc_alpha  # 1e-3

    if config.mode == "sad":
        loss_anomaly = model.gdn.dev_loss(
            torch.squeeze(labels),
            torch.squeeze(prediction_dict["anom_score"]),
            torch.squeeze(prediction_dict["time"]),
        )
        loss_supc = model.suploss(
            prediction_dict["root_embedding"], prediction_dict["group"], prediction_dict["dev"]
        )
        loss += alpha * loss_anomaly + beta * loss_supc

    return loss, loss_classify, loss_anomaly, loss_supc


def eval_epoch(dataset, model, config, device):
    m_loss, m_pred, m_label = [], [], []
    m_dev = []
    with torch.no_grad():
        model.eval()
        for batch_sample in dataset:
            x = model(
                batch_sample["src_edge_feat"].to(device),
                batch_sample["src_edge_to_time"].to(device),
                batch_sample["src_center_node_idx"].to(device),
                batch_sample["src_neigh_edge"].to(device),
                batch_sample["src_node_features"].to(device),
                batch_sample["current_time"].to(device),
                batch_sample["labels"].to(device),
            )
            y = batch_sample["labels"].to(device).float()  # Add .float() here

            dev_score = x["dev"].cpu().numpy().flatten()
            m_loss = np.concatenate(
                (m_loss, criterion(x, y, model, config)[1].cpu().numpy().flatten())
            )

            pred_score = x["logits"].sigmoid().cpu().numpy().flatten()
            y_np = y.cpu().numpy().flatten()
            m_pred = np.concatenate((m_pred, pred_score))
            m_label = np.concatenate((m_label, y_np))
            m_dev = np.concatenate((m_dev, dev_score))

    try:
        auc_roc = sklearn.metrics.roc_auc_score(m_label, m_pred)
        pr_auc = sklearn.metrics.average_precision_score(m_label, m_pred)
    except ValueError:
        auc_roc, pr_auc = float("nan"), float("nan")
    return auc_roc, np.mean(m_loss), m_dev, m_label, pr_auc


config = args
run = None
seed = config.seed
if not config.wandb_run_name:
    config.wandb_run_name = (
        f"neu_{config.data_set}_mode-{config.mode}_bs{config.batch_size}"
        f"_lr{config.learning_rate}_mask{config.mask_ratio}"
        f"_a{config.anomaly_alpha}_s{config.supc_alpha}"
        f"_split{config.train_split}_{config.val_split}_{config.test_split}_seed{seed}"
    )

torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if config.use_wandb:
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=vars(config),
    )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# log file name set
now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# log_base_path = f"{os.getcwd()}/train_log"
# file_list = os.listdir(log_base_path)
# max_num = [0] # [int(fl.split("_")[0]) for fl in file_list if len(fl.split("_"))>2] + [-1]
# log_base_path = f"{log_base_path}/{max(max_num)+1}_{now_time}"
# log and path
# get_checkpoint_path = lambda epoch: f'{log_base_path}saved_checkpoints/{args.data_set}-{args.mode}-{args.module_type}-{args.mask_ratio}-{epoch}.pth'
# logger = logger_config(log_path=f'{log_base_path}/log.txt', logging_name='gdn')
# logger.info(config)
# split_list = [0.5, 0.0, 0.5]
split_list = [config.train_split, config.val_split, config.test_split]
assert abs(sum(split_list) - 1.0) < 1e-8, f"Splits must sum to 1.0, got {split_list}"
val_enabled = split_list[1] > 0.0
# dataset_train = dataset.DygDataset(config, 'train')
# dataset_valid = dataset.DygDataset(config, 'valid')
# dataset_test = dataset.DygDataset(config, 'test')
dataset_train = dataset.DygDataset(config, "train", split_list=split_list)
dataset_test = dataset.DygDataset(config, "test", split_list=split_list)
if val_enabled:
    dataset_valid = dataset.DygDataset(config, "valid", split_list=split_list)
else:
    dataset_valid = None

gpus = None if config.gpus == 0 else config.gpus

collate_fn = dataset.Collate(config)

config.input_dim = dataset.get_feature_dim(config)
backbone = TGAT(config, device)
model = backbone.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


train_sampler = None
if config.train_drop_rate > 0:
    train_sampler = dataset.RandomDropSampler(dataset_train, config.train_drop_rate)
loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=config.batch_size,
    shuffle=False,
    # shuffle=True,
    num_workers=config.num_data_workers,
    pin_memory=True,
    sampler=train_sampler,
    collate_fn=collate_fn.dyg_collate_fn,
)
if val_enabled:
    loader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        shuffle=False,
        # shuffle=True,
        num_workers=config.num_data_workers,
        collate_fn=collate_fn.dyg_collate_fn,
    )
else:
    loader_valid = None

loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=config.batch_size,
    shuffle=False,
    # shuffle=True,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn,
)
global_step = 0
max_val_auc, max_test_auc = 0.0, 0.0
early_stopper = EarlyStopMonitor(max_round=config.patience, tolerance=0.0)
best_auc = [0, 0, 0]


def save_best_model(model, config, auc_score, split_name, epoch):
    run_dir = os.path.join(
        config.checkpoint_dir,
        str(config.wandb_run_name).replace("/", "_").replace("\\", "_"),
    )
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model.pt")
    current_best = None
    if os.path.exists(os.path.join(run_dir, "config.json")):
        try:
            with open(os.path.join(run_dir, "config.json")) as f:
                current_best = json.load(f)
        except Exception:
            current_best = None
    if (
        current_best is not None
        and "auc" in current_best
        and auc_score <= float(current_best["auc"]) - 1e-9
        and split_name == current_best.get("best_split")
    ):
        print(
            f"\n[checkpoint] skip (auc={auc_score:.4f} not better than {float(current_best['auc']):.4f})"
        )
        return run_dir
    torch.save(model.state_dict(), model_path)
    cfg = dict(vars(config))
    cfg.update(
        {
            "auc": float(auc_score),
            "best_split": split_name,
            "best_epoch": int(epoch),
            "model_file": "model.pt",
            "config_file": "config.json",
        }
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"\n[checkpoint] saved best {split_name} model to {run_dir} (auc={auc_score:.4f})")
    return run_dir


for epoch in range(config.n_epochs):
    ave_loss = 0
    count_flag = 0
    m_loss, auc = [], []
    loss_anomaly_list = []
    loss_class_list = []
    loss_supc_list = []
    train_pred_list = np.array([])
    train_label_list = np.array([])
    with tqdm(total=len(loader_train), disable=config.no_process_bar) as t:
        for batch_sample in loader_train:
            count_flag += 1
            t.set_description("Epoch %i" % epoch)
            optimizer.zero_grad()
            model.train()
            x = model(
                batch_sample["src_edge_feat"].to(device),
                batch_sample["src_edge_to_time"].to(device),
                batch_sample["src_center_node_idx"].to(device),
                batch_sample["src_neigh_edge"].to(device),
                batch_sample["src_node_features"].to(device),
                batch_sample["current_time"].to(device),
                batch_sample["labels"].to(device),
            )
            y = batch_sample["labels"].to(device)
            dev_score = x["dev"]
            loss, loss_classify, loss_anomaly, loss_supc = criterion(x, y, model, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            global_step += 1
            if config.use_wandb:
                pass  # metrics are logged at epoch level (step = epoch)

            # get training results
            with torch.no_grad():
                model = model.eval()
                m_loss.append(loss.item())
                pred_score = x["logits"].sigmoid()

                train_pred_list = np.concatenate(
                    (train_pred_list, pred_score.cpu().numpy().flatten())
                )
                train_label_list = np.concatenate(
                    (train_label_list, batch_sample["labels"].cpu().numpy().flatten())
                )

            loss_class_list.append(loss_classify.detach().clone().cpu().numpy().flatten())
            loss_anomaly_list.append(loss_anomaly.detach().clone().cpu().numpy().flatten())
            loss_supc_list.append(loss_supc.detach().clone().cpu().numpy().flatten())

            t.set_postfix(loss=np.mean(m_loss))
            t.update(1)

    train_mask = train_label_list >= 0
    try:
        train_auc = float(
            sklearn.metrics.roc_auc_score(train_label_list[train_mask], train_pred_list[train_mask])
        )
        train_pr_auc = float(
            sklearn.metrics.average_precision_score(
                train_label_list[train_mask], train_pred_list[train_mask]
            )
        )
    except Exception:
        train_auc, train_pr_auc = float("nan"), float("nan")
    if val_enabled:
        val_auc, val_loss, val_m_dev, val_m_label, val_pr_auc = eval_epoch(
            loader_valid, model, config, device
        )
    else:
        val_auc, val_loss, val_pr_auc = np.nan, np.nan, np.nan
    test_auc, test_loss, test_m_dev, test_m_label, test_pr_auc = eval_epoch(
        loader_test, model, config, device
    )

    max_test_auc = max(max_test_auc, test_auc)

    if val_enabled:
        max_val_auc = max(max_val_auc, val_auc)
        if val_auc > best_auc[1]:
            best_auc = [epoch, val_auc, test_auc]
            save_best_model(model, config, val_auc, "val", epoch)
    else:
        if test_auc > best_auc[2]:
            best_auc = [epoch, np.nan, test_auc]
            save_best_model(model, config, test_auc, "test", epoch)
    train_loss = float(np.mean(m_loss)) if len(m_loss) > 0 else float("nan")
    if config.use_wandb:
        metrics = {
            "train/auc_roc": float(train_auc),
            "train/pr_auc": float(train_pr_auc),
            "train/loss": train_loss,
            "test/auc_roc": float(test_auc),
            "test/pr_auc": float(test_pr_auc),
        }
        if val_enabled:
            metrics["val/auc_roc"] = float(val_auc)
            metrics["val/pr_auc"] = float(val_pr_auc)
        wandb.log(metrics, step=epoch)

    print(f"\n epoch: {epoch}")
    print(
        f"train mean loss:{np.mean(m_loss)}, class loss: {np.mean(loss_class_list)}, anomaly loss: {np.mean(loss_anomaly_list)}, sup loss: {np.mean(loss_supc_list)}"
    )
    if val_enabled:
        print(f"val mean loss:{val_loss}, val auc:{val_auc}")
    print(f"test mean loss:{test_loss}, test auc:{test_auc}")
    # logger.info('\n epoch: {}'.format(epoch))
    # logger.info(f'train mean loss:{np.mean(m_loss)}, class loss: {np.mean(loss_class_list)}, anomaly loss: {np.mean(loss_anomaly_list)}, sup loss: {np.mean(loss_supc_list)}')
    # logger.info('val mean loss:{}, val auc:{}'.format(val_loss, val_auc))
    # logger.info('test mean loss:{}, test auc:{}'.format(test_loss, test_auc))
    # logger.info('val pr_auc:{}, test pr_auc:{}'.format(val_pr_auc, test_pr_auc))

    monitor_auc = val_auc if val_enabled else test_auc
    if config.patience > 0 and early_stopper.early_stop_check(monitor_auc):
        print(
            "No improvement over {} epochs of the {} ROC-AUC, stop training".format(
                early_stopper.max_round, "val" if val_enabled else "test"
            )
        )
        # print(f'Loading the best model at epoch {early_stopper.best_epoch}')
        # best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        # model.load_state_dict(torch.load(best_model_path))
        # print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        # model.eval()
        break
    else:
        # torch.save(model.state_dict(), get_checkpoint_path(epoch))
        pass

    # 记录下score的结果
#     dev_score_list = np.concatenate((dev_score_list, val_m_dev, test_m_dev))
#     dev_label_list = np.concatenate((dev_label_list, val_m_label, test_m_label))
#     output_file = './graph_dev_score.txt'
#     with open(output_file, 'w') as fout:
#         for i, (score, label) in enumerate(zip( dev_score_list, dev_label_list)):
#             fout.write(f'{i}\t')
#             fout.write(f'{score}\t')
#             fout.write(f'{label}\n')
#             pass
#         pass
#     pass

# logger.info(f'\n max_val_auc: {max_val_auc}, max_test_auc: {max_test_auc}')
# logger.info('\n best auc: epoch={}, val={}, test={}'.format(best_auc[0], best_auc[1], best_auc[2]))
print(f"\n max_val_auc: {max_val_auc}, max_test_auc: {max_test_auc}")
print("\n best auc: epoch={}, val={}, test={}".format(best_auc[0], best_auc[1], best_auc[2]))


if config.use_wandb:
    wandb.summary["last/train/loss_mean"] = float(np.mean(m_loss))
    if val_enabled:
        wandb.summary["last/val/loss"] = float(val_loss)
        wandb.summary["last/val/auc_roc"] = float(val_auc)
    wandb.summary["last/test/loss"] = float(test_loss)
    wandb.summary["last/test/auc_roc"] = float(test_auc)
    wandb.summary["best/epoch"] = int(best_auc[0])
    if val_enabled:
        wandb.summary["best/val_auc"] = float(best_auc[1])
        wandb.summary["max/val_auc"] = float(max_val_auc)
    wandb.summary["best/test_auc"] = float(best_auc[2])

    wandb.summary["max/test_auc"] = float(max_test_auc)
    wandb.finish()
