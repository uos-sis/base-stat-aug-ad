# Evaluate trained SAD checkpoints on the test split (mirrors GraphSAGE/test.sh).
# Runs from the SAD directory (anomaly_detection/SAD).  Each command writes a
# result.json next to the model dir; use `--compute_importances` for the
# embedding vs statistic SHAP importance split.  Checkpoints named `*_mlp_ks_*`
# automatically evaluate in raw mode (raw features embedded on the fly).

# python test.py --dirs mooc_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0
# python test.py --dirs mooc_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1
# python test.py --dirs mooc_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2
# python test.py --dirs mooc_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3

# python test.py --dirs mooc_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0 --compute_importances
# python test.py --dirs mooc_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1 --compute_importances
# python test.py --dirs mooc_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2 --compute_importances
# python test.py --dirs mooc_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3 --compute_importances
#
# python test.py --dirs mooc_stats_mlp_ks_8_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0 --compute_importances
# python test.py --dirs mooc_stats_mlp_ks_8_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1 --compute_importances
# python test.py --dirs mooc_stats_mlp_ks_8_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2 --compute_importances
# python test.py --dirs mooc_stats_mlp_ks_8_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3 --compute_importances

# python test.py --dirs reddit_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0
# python test.py --dirs reddit_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1
# python test.py --dirs reddit_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2
# python test.py --dirs reddit_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3

# python test.py --dirs reddit_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0 --compute_importances
# python test.py --dirs reddit_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1 --compute_importances
# python test.py --dirs reddit_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2 --compute_importances
# python test.py --dirs reddit_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3 --compute_importances
#
# python test.py --dirs reddit_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0 --compute_importances
python test.py --dirs reddit_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1 --compute_importances
python test.py --dirs reddit_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2 --compute_importances
python test.py --dirs reddit_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3 --compute_importances

# python test.py --dirs wikipedia_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0
# python test.py --dirs wikipedia_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1
# python test.py --dirs wikipedia_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2
# python test.py --dirs wikipedia_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3

# python test.py --dirs wikipedia_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0 --compute_importances
# python test.py --dirs wikipedia_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1 --compute_importances
# python test.py --dirs wikipedia_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2 --compute_importances
# python test.py --dirs wikipedia_stats_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3 --compute_importances
#
# python test.py --dirs wikipedia_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed0 --compute_importances
# python test.py --dirs wikipedia_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed1 --compute_importances
# python test.py --dirs wikipedia_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed2 --compute_importances
# python test.py --dirs wikipedia_stats_mlp_ks_64_mode-sad_bs256_lr0.0005_mask0.5_a0.1_s0.005_splittvtest0.7_0.15_0.15seed3 --compute_importances
