import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser("RAP-AD")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="data sources to use, try wikipedia or reddit",
        default="mooc",
    )
    parser.add_argument(
        "--n_degree",
        nargs="*",
        default=["32", "32"],
        help="[l, s], history length and partner size",
    )
    parser.add_argument(
        "-s", "--step", type=int, help="number of interactions used for prediction", default=5
    )
    parser.add_argument(
        "--raps", type=str, help="using source node encoding (on/None)", default="on"
    )
    parser.add_argument(
        "--rapd", type=str, help="using destination node encoding (on/None)", default="on"
    )
    parser.add_argument("--mask", nargs="*", default=[0, 0], help="mask numbers")
    parser.add_argument(
        "-m",
        "--data_mode",
        type=str,
        default="real_synth",
        choices=["real", "real_synth", "synth"],
        help="real: only original data; "
        "real_synth: original + synthetic anomalies in train; "
        "synth: only synthetic anomalies in train. "
        "val/test are always original.",
    )

    # general training hyper-parameters
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epochs")
    parser.add_argument("--bs", type=int, default=32, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--drop_out", type=float, default=0.1, help="dropout probability for all dropout layers"
    )
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience in epochs (val auc)"
    )

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for all randomized algorithms"
    )
    # Falg for new improved features
    parser.add_argument(
        "--feets",
        type=str,
        default="raw",
        choices=["raw", "stats", "statsdim"],
        help="using new improved features (raw,stats,statsdim)",
    )
    parser.add_argument(
        "--wandb-logging",
        action="store_true",
        default=False,
        help="enable wandb logging (default: off)",
    )

    # ---- SHAP evaluation arguments (used by calcshape.py) ----
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3],
        help="seeds whose saved best models are explained and aggregated",
    )
    parser.add_argument(
        "--shap_max_samples",
        type=int,
        default=2000,
        help="max # correctly classified test events to explain (like SAD/GraphSAGE: "
        "correct anomalies first, then correct normals)",
    )
    parser.add_argument(
        "--shap_background_samples",
        type=int,
        default=40,
        help="number of background samples per explained event",
    )
    parser.add_argument(
        "--shap_nsamples",
        type=int,
        default=20,
        help="number of integration steps for the expected-gradients estimator",
    )
    parser.add_argument(
        "--pred_threshold",
        type=float,
        default=0.5,
        help="classification threshold on sigmoid(score) used to pick "
        "correctly classified events",
    )
    parser.add_argument("--no_progress", action="store_true", help="disable tqdm progress bars")
    parser.add_argument(
        "--raw_mode",
        action="store_true",
        help="statsdim: attribute the RAW stats features (embedding + statistics) "
        "instead of the mlp-compressed vector; auto-enabled for --feets statsdim",
    )
    # legacy aliases kept for compatibility with older run scripts
    parser.add_argument(
        "--n_samples", type=int, default=None, help="deprecated: use --shap_max_samples"
    )
    parser.add_argument(
        "--n_background", type=int, default=None, help="deprecated: use --shap_background_samples"
    )
    parser.add_argument(
        "--model_root",
        type=str,
        default="./saved_models",
        help="directory containing the saved best-model.pth files",
    )
    parser.add_argument(
        "--outdir", type=str, default="./results/shap", help="output directory for SHAP results"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=-1,
        help="number of embedding feature columns (for stats group split); "
        "default is inferred per dataset",
    )
    parser.add_argument(
        "--integrated_grads_only",
        action="store_true",
        help="use the internal expected-gradients SHAP implementation "
        "instead of shap.GradientExplainer",
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv
