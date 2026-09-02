import torch
import numpy as np
import sys, copy, math, time, pdb, json
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os
import os.path
import random
import argparse
import pickle
try:
    import wandb
except ImportError:
    wandb = None
sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
from os import path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Anomaly detection')
# general settings
parser.add_argument('--data-name', default='USAir', help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.815,
                    help='ratio of test links')
parser.add_argument('--window', type=int, default=5,
                    help='window size')
parser.add_argument('--graph', default='acc_digg.npy')
parser.add_argument('--split', default='digg0.1')
parser.add_argument('--gpu', default='1', help='gpu number')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
# hyperparameters
parser.add_argument('--hidden', type=int, default=128, help='MLP hidden size')
parser.add_argument('--latent-dim', default='32-32-32-1',
                    help='DGCNN latent dims, dash separated')
parser.add_argument('--num-epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=50, help='batch size')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--sortpooling-k', type=float, default=0.6,
                    help='sortpooling k (ratio0 or absolute)')
parser.add_argument('--no-dropout', action='store_true', default=False,
                    help='disable dropout in the classifier')
parser.add_argument('--no-progress-bar', action='store_true', default=False,
                    help='disable tqdm progress bars')
parser.add_argument('--edge-proj-dim', type=int, default=0,
                    help='learned edge-feature projection dim (0 = no projection)')
parser.add_argument('--dense-dim', type=int, default=256,
                    help='final graph embedding / MLP input dimension')
parser.add_argument('--mlp-dropout', type=float, default=0.5,
                    help='dropout rate of the MLP classifier')
# wandb
parser.add_argument('--wandb-project', default=None, help='wandb project name')
parser.add_argument('--wandb-entity', default=None, help='wandb entity')
parser.add_argument('--wandb-mode', default='online', help='wandb mode (offline/online/disabled)')
parser.add_argument('--wandb-run-name', default=None, help='wandb run name')
# checkpointing
parser.add_argument('--checkpoint-dir', default='checkpoints', help='root directory to save the best val ROC checkpoint model checkpoints')
parser.add_argument('--patience', type=int, default=10,
                    help='early stopping patience in val ROC-AUC (0 disables early stopping)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

cmd_args.seed = args.seed
cmd_args.no_progress_bar = args.no_progress_bar
cmd_args.edge_proj_dim = args.edge_proj_dim
cmd_args.dense_dim = args.dense_dim
cmd_args.mlp_dropout = args.mlp_dropout
random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

if args.train_name is None:
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))

    net = np.load('data_sta/'+args.graph, allow_pickle=True)

    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
    #Sample train and test links
    f = np.load('data_sta/'+args.split+'.npz')
    train_pos_id, train_neg_id, test_pos_id, test_neg_id = f['train_pos_id'], f['train_neg_id'], f['test_pos_id'], f['test_neg_id']
    train_pos, train_neg, test_pos, test_neg = f['train_pos'], f['train_neg'], f['test_pos'], f['test_neg']
    if 'val_pos_id' in f.files:
        val_pos_id, val_neg_id = f['val_pos_id'], f['val_neg_id']
        val_pos, val_neg = f['val_pos'], f['val_neg']
    else:
        val_pos_id = val_neg_id = np.zeros((0,), dtype=np.int64)
        val_pos = val_neg = np.zeros((2, 0), dtype=np.int64)
else:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    max_idx = max(np.max(train_idx), np.max(test_idx))
    net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    #Sample negative train and test links
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    train_pos, train_neg, test_pos, test_neg = sample_dyn(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)


'''Train and apply classifier'''
A = net.copy()  # the observed network
# A[test_pos[0], test_pos[1]] = 0  # mask test links
# A[test_pos[1], test_pos[0]] = 0  # mask test links

node_information = None
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

efeat_path = 'data_sta/'+args.graph.replace('.npy', '_efeat.npy')
edge_feat = np.load(efeat_path, allow_pickle=True) if path.exists(efeat_path) else None
cache_name = args.split+'h'+str(args.hop) + ('f' if edge_feat is not None else '') + 'v'
if not path.exists('data_sta/'+cache_name):
    train_graphs, val_graphs, test_graphs, max_n_label = dyn_links2subgraphs(A, args.window, train_pos_id, train_pos, train_neg_id, train_neg, val_pos_id, val_pos, val_neg_id, val_neg, test_pos_id, test_pos, test_neg_id, test_neg, args.hop, args.max_nodes_per_hop, node_information, edge_feat)
    print(('# train: %d, # val: %d, # test: %d' % (len(train_graphs), len(val_graphs), len(test_graphs))))
    with open('data_sta/'+cache_name, 'wb') as f:
        pickle.dump([train_graphs, val_graphs, test_graphs, max_n_label], f, protocol=4)
else:
    with open('data_sta/'+cache_name, 'rb') as f:
        train_graphs, val_graphs, test_graphs, max_n_label = pickle.load(f)
        print(('# train: %d, # val: %d, # test: %d' % (len(train_graphs), len(val_graphs), len(test_graphs))))



# DGCNN configurations
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = args.sortpooling_k
cmd_args.latent_dim = [int(x) for x in args.latent_dim.split('-')]
cmd_args.hidden = args.hidden
cmd_args.out_dim = 0
cmd_args.dropout = not args.no_dropout
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = args.num_epochs
cmd_args.learning_rate = args.learning_rate
cmd_args.batch_size = args.batch_size
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
cmd_args.edge_feat_dim = 0
for g_list_i in train_graphs:
    for g in g_list_i:
        if g.edge_features is not None:
            cmd_args.edge_feat_dim = g.edge_features.shape[1]
            break
    if cmd_args.edge_feat_dim:
        break
if cmd_args.edge_feat_dim:
    print(('edge feature dim: %d' % cmd_args.edge_feat_dim))
cmd_args.window = 5
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    A = []
    for i in train_graphs:
        A.append(i[-1])
    for i in val_graphs:
        A.append(i[-1])
    for i in test_graphs:
        A.append(i[-1])
    num_nodes_list = sorted([g.num_nodes for g in A])
    cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print(('k used in SortPooling is: ' + str(cmd_args.sortpooling_k)))

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_auc = 0
best_val_auc = 0
best_test_at_val = None
has_val = len(val_graphs) > 0
run_name = args.wandb_run_name or (args.graph.replace('.npy', '') + '_seed' + str(args.seed))
ckpt_folder = os.path.join(args.checkpoint_dir, run_name)
ckpt_path = os.path.join(ckpt_folder, 'model.pt')
wandb_run = None
if args.wandb_project and wandb is not None:
    wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                           mode=args.wandb_mode, name=run_name, config=vars(args))

patience_counter = 0
best_epoch = -1


def save_checkpoint(epoch):
    os.makedirs(ckpt_folder, exist_ok=True)
    torch.save(classifier.state_dict(), ckpt_path)
    cfg = dict(vars(args))
    cfg.update({
        'best_epoch': int(epoch),
        'best_val_roc_auc': float(best_val_auc),
        'test_roc_auc_at_best_val': float(best_test_at_val[2]) if best_test_at_val is not None else None,
        'model_file': 'model.pt',
        'config_file': 'config.json',
    })
    with open(os.path.join(ckpt_folder, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    print(('save model %s' % ckpt_path))


for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f roc_auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2])))

    classifier.eval()
    if has_val:
        val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
        if not cmd_args.printAUC:
            val_loss[2] = 0.0
        print(('\033[96maverage val of epoch %d: loss %.5f acc %.5f roc_auc %.5f avg_precision %.5f precision-recall auc %.5f\033[0m' % (epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4])))
    test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    print(('\033[93maverage test of epoch %d: loss %.5f acc %.5f roc_auc %.5f avg_precision %.5f precision-recall auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4])))

    improved = False
    if has_val:
        if val_loss[2] > best_val_auc:
            best_val_auc = val_loss[2]
            best_test_at_val = test_loss
            best_epoch = epoch
            improved = True
    elif test_loss[2] > best_auc:
        best_auc = test_loss[2]
        best_epoch = epoch
        improved = True

    if improved:
        patience_counter = 0
        save_checkpoint(epoch)
    else:
        patience_counter += 1

    if wandb_run is not None:
        log_dict = {
            'epoch': epoch,
            'train/loss': avg_loss[0],
            'train/acc': avg_loss[1],
            'train/roc_auc': avg_loss[2],
        }
        if has_val:
            log_dict['val/loss'] = val_loss[0]
            log_dict['val/acc'] = val_loss[1]
            log_dict['val/roc_auc'] = val_loss[2]
            log_dict['val/avg_precision'] = val_loss[3]
            log_dict['val/pr_auc'] = val_loss[4]
        log_dict['test/loss'] = test_loss[0]
        log_dict['test/acc'] = test_loss[1]
        log_dict['test/roc_auc'] = test_loss[2]
        log_dict['test/avg_precision'] = test_loss[3]
        log_dict['test/pr_auc'] = test_loss[4]
        wandb.log(log_dict)

    if args.patience > 0 and patience_counter >= args.patience:
        print(('early stopping at epoch %d' % epoch))
        break

if os.path.exists(ckpt_path):
    classifier.load_state_dict(torch.load(ckpt_path))
    print(('loaded best checkpoint %s' % ckpt_path))

if has_val:
    print('best_val_auc = ', best_val_auc)
    if best_test_at_val is not None:
        print(('best test metrics at best val AUC: loss %.5f acc %.5f roc_auc %.5f avg_precision %.5f precision-recall auc %.5f' % (best_test_at_val[0], best_test_at_val[1], best_test_at_val[2], best_test_at_val[3], best_test_at_val[4])))
else:
    print('best_auc = ', best_auc)

if wandb_run is not None:
    if has_val:
        wandb.summary['best_val_auc'] = best_val_auc
        if best_test_at_val is not None:
            wandb.summary['test_roc_auc_at_best_val'] = best_test_at_val[2]
    else:
        wandb.summary['best_auc'] = best_auc
    wandb.summary['best/epoch'] = int(best_epoch)
    wandb.finish()

# with open('acc_results.txt', 'a+') as f:
#     f.write(str(test_loss[1]) + '\n')

# if cmd_args.printAUC:
#     with open('auc_results.txt', 'a+') as f:
#         f.write(str(test_loss[2]) + '\n')

