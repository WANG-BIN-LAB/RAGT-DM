import numpy as np
import torch
from datetime import datetime
from config import Config
from dataset import dataset_factory, load_all_data
from model import BrainNetworkTransformer
from components import optimizers_factory, lr_scheduler_factory, logger_factory
from train import Train
from sklearn.model_selection import StratifiedKFold
import itertools
import warnings
def model_training(cfg, device, train_idx=None, test_idx=None):
    """
    Train model for single fold in cross validation
    """
    cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
    dataloaders = dataset_factory(cfg, train_idx=train_idx, test_idx=test_idx)
    logger = logger_factory(cfg)
    # Initialize model, optimizer, scheduler
    model = BrainNetworkTransformer(cfg).to(device)
    optimizers = optimizers_factory(model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
    # Start training
    training = Train(cfg, model, optimizers, lr_schedulers, dataloaders, logger, device)
    return training.train()

def run_cv(cfg, device):
    """
    Run 10-fold stratified cross validation
    Returns:
        mean_dict: mean metrics of 10 folds
        std_dict: std metrics of 10 folds
    """
    time_series, _, _, labels, site, _, _ = load_all_data(cfg)
    # Stratified K-fold split by site
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    all_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(time_series.numpy(), site.numpy())):
        print(f'\n===== Fold {fold_idx + 1}/{cfg.n_folds} =====')
        # Set fold seed for reproducibility
        fold_seed = cfg.seed + fold_idx
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        # Train current fold
        fold_result = model_training(cfg, device, train_idx, test_idx)
        all_results.append(fold_result)
        print(f'Fold {fold_idx + 1} Done | Test Acc: {fold_result["test_accuracy"]:.4f}')
    # Calculate mean and std of metrics
    if all_results:
        metrics = [
            "train_accuracy", "test_accuracy", "test_auc",
            "test_sensitivity", "test_specificity",
            "micro_f1", "micro_recall", "micro_precision"
        ]
        result_dict = {metric: [res[metric] for res in all_results] for metric in metrics}
        mean_dict = {metric: np.mean(result_dict[metric]) for metric in metrics}
        std_dict = {metric: np.std(result_dict[metric]) for metric in metrics}
        return mean_dict, std_dict
    return None, None

def init_result_file(output_file="results.txt"):
    """Initialize result file with header"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("Multi-Parameter Grid Search Results (Real-time Save)\n")
        f.write("=" * 120 + "\n")
        header = (f"{'Layers':<6} {'Heads':<6} {'TopK':<6} {'LocalN':<6} "
                  f"{'Test Acc (Mean±Std)':<25} {'Test AUC (Mean±Std)':<25} {'F1 (Mean±Std)':<25}\n")
        f.write(header)
        f.write("-" * 120 + "\n")

def append_single_result(params, mean, std, output_file="results.txt"):
    """Append single group of results to file"""
    layers, heads, topk, local_n = params
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{layers:<6d} {heads:<6d} {topk:<6.2f} {local_n:<6d} "
                f"{mean['test_accuracy']:.4f}±{std['test_accuracy']:.4f}   "
                f"{mean['test_auc']:.4f}±{std['test_auc']:.4f}   "
                f"{mean['micro_f1']:.4f}±{std['micro_f1']:.4f}\n")
        f.write(f"  └─ Detailed | Layers={layers} Heads={heads} TopK={topk:.2f} LocalNeighbor={local_n}\n")
        for metric in mean.keys():
            f.write(f"     {metric:<15}: {mean[metric]:.7f} ± {std[metric]:.7f}\n")
        f.write("-" * 120 + "\n")

def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # -------------------------- Grid Search Params -------------------------- #
    num_layers_list = [1]
    nhead_list = [4]
    global_topk_ratios = [0.3]
    local_neighbor_nums = [9]
    # Generate param combinations
    param_combinations = list(itertools.product(
        num_layers_list, nhead_list,
        global_topk_ratios, local_neighbor_nums
    ))
    total_combinations = len(param_combinations)
    # Initialize result file
    init_result_file()
    print(f"\n===== Total Param Combinations: {total_combinations} | Results saved to results.txt =====")
    # Start grid search
    for idx, params in enumerate(param_combinations):
        layers, heads, topk_ratio, local_n = params
        # Update config
        cfg.model.num_layers = layers
        cfg.model.nhead = heads
        cfg.model.global_topk_ratio = topk_ratio
        cfg.model.local_neighbor_num = local_n
        # Print current params
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{total_combinations}] Training Params:")
        print(f" Layers: {layers} | Heads: {heads}")
        print(f"  Global TopK: {topk_ratio:.2f} | Local Neighbor: {local_n}")
        print('='*70)
        # Run cross validation
        mean, std = run_cv(cfg, device)
        if mean is not None:
            append_single_result(params, mean, std)
            print(f"✅ Done | Test Acc = {mean['test_accuracy']:.4f} ± {std['test_accuracy']:.4f}")
        else:
            print(f"❌ Training Failed, Skip")
    print("\n🎉 All Grid Search Finished!")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()