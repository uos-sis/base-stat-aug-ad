import argparse
import sys
import torch


def get_node_classification_args(is_evaluation: bool = False):
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',
                        choices=['wikipedia', 'reddit', 'mooc',
                                 'wikipedia_stats', 'wikipedia_stats_mlp_ks_64',
                                 'reddit_stats', 'reddit_stats_mlp_ks_64',
                                 'mooc_stats', 'mooc_stats_mlp_ks_8'])
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer'])
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--min_feat_dim', type=int, default=0,
                        help='minimal node/edge feature dimension after zero-padding; 0 keeps the dataset native feature size (use 172 for the classic DyGLib fixed-width models)')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', '--epochs', type=int, default=100, help='number of epochs ("epochs" alias matches GraphSAGE)')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping (0 disables)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--seed', type=int, default=0, help='base random seed (each run i uses seed + i)')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='kept for backwards compatibility; the test set is now evaluated every --test_every epochs')
    parser.add_argument('--val_every', type=int, default=1, help='run the validation evaluation every N epochs (0 = never)')
    parser.add_argument('--test_every', type=int, default=1, help='run the test evaluation every N epochs (0 = never)')
    parser.add_argument('--neighbor_cache_size', type=int, default=2000000,
                        help='max cached first-hop neighbor lookups when sampling strategy is "recent"; 0 disables the cache')
    parser.add_argument('--no_neighbor_cache', dest='neighbor_cache_size', action='store_const', const=0,
                        default=2000000, help='disable the first-hop neighbor sampling cache')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='root directory to save the best val ROC-AUC model checkpoints (config.json + model.pt)')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='log metrics to wandb (see --wandb_mode)')
    parser.add_argument('--wandb_project', type=str, default='CNA2026_DyGLib2')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online', choices=('online', 'offline', 'disabled'))

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    if not torch.cuda.is_available() and not is_evaluation:
        raise RuntimeError(
            'DyGLib training requires a GPU, but CUDA is not available. '
            'Aborting instead of silently training on CPU.')

    assert args.dataset_name.split('_stats')[0] in ['wikipedia', 'reddit', 'mooc'], \
        f'Wrong value for dataset_name {args.dataset_name}!'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.load_best_configs:
        load_node_classification_best_configs(args=args)

    return args


def load_node_classification_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node classification task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 30
        args.dropout = 0.5
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        base_name = args.dataset_name
        for suffix in ('_stats_mlp_ks_64', '_stats_mlp_ks_8', '_stats'):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break
        if base_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif base_name in ['mooc']:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if base_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
