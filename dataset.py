import torch
import torch.utils.data as utils
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

def load_all_data(cfg):
    """
    Load full dataset from .npy file
    Args:
        cfg: config object
    Returns:
        time_series, node_feature, labels_onehot, labels, sites, ages, sexes
    """
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    # Convert to torch tensor
    time_series = torch.from_numpy(data['timeseires']).float()
    node_feature = torch.from_numpy(data['corr']).float()
    labels = torch.from_numpy(data['label']).long()
    # Load auxiliary features
    site = data.get('site', np.zeros(len(labels), dtype=np.int64))
    ages = data.get('ages', np.zeros(len(labels), dtype=np.float32))
    sexes = data.get('sexs', np.zeros(len(labels), dtype=np.int64))
    # Preprocess labels and sites
    labels_onehot = F.one_hot(labels.to(torch.int64))
    sites = torch.from_numpy(site).long() - 1  # Map to [0, 1, ..., 18]
    ages = torch.from_numpy(ages).float()
    sexes = torch.from_numpy(sexes).long()
    return time_series, node_feature, labels_onehot, labels, sites, ages, sexes

def get_dataloader_by_indices(cfg, time_series, node_feature, labels_onehot,
                              sites, ages, sexes, train_idx, test_idx):
    """
    Create train/test dataloader by given indices (for cross validation)
    """
    # Split data by indices
    train_data = (time_series[train_idx], node_feature[train_idx], labels_onehot[train_idx],
                  sites[train_idx], ages[train_idx], sexes[train_idx])
    test_data = (time_series[test_idx], node_feature[test_idx], labels_onehot[test_idx],
                 sites[test_idx], ages[test_idx], sexes[test_idx])
    # Calculate training steps
    train_length = len(train_idx)
    cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
    cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs
    # Build dataset and dataloader
    train_dataset = utils.TensorDataset(*train_data)
    test_dataset = utils.TensorDataset(*test_data)
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size,
        shuffle=True, drop_last=cfg.dataset.drop_last
    )
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.test_batch_size, shuffle=False
    )
    return [train_dataloader, test_dataloader]

def dataset_factory(cfg, train_idx=None, test_idx=None):
    """
    Dataset factory: support both CV mode and debug mode
    """
    time_series, node_feature, labels_onehot, labels, sites, ages, sexes = load_all_data(cfg)
    # Cross validation mode (main mode)
    if train_idx is not None and test_idx is not None:
        return get_dataloader_by_indices(
            cfg, time_series, node_feature, labels_onehot,
            sites, ages, sexes, train_idx, test_idx
        )
    # Debug mode (random split)
    length = time_series.shape[0]
    train_length = int(length * 0.9)
    indices = torch.randperm(length).tolist()
    train_indices = indices[:train_length]
    test_indices = indices[train_length:]
    return get_dataloader_by_indices(
        cfg, time_series, node_feature, labels_onehot,
        sites, ages, sexes, train_indices, test_indices
    )