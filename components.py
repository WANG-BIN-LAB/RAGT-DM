import logging
import sys
import torch
from torch.optim import Adam

# Empty LR scheduler to keep fixed learning rate (no decay, no warmup)
class LRScheduler:
    def __init__(self, lr_scheduler=None, warmup_steps=0, warmup_from=0.0):
        self.lr = None

    # Empty update: keep learning rate fixed during training
    def update(self, optimizer, step):
        self.lr = optimizer.param_groups[0]['lr']

def optimizers_factory(model, optimizer_configs):
    """
    Create optimizer for model (only support Adam)
    Args:
        model: neural network model
        optimizer_configs: configs including lr and weight_decay
    Returns:
        list: wrapped optimizer list
    """
    if optimizer_configs.name == 'Adam':
        optimizer = Adam(
            model.parameters(),
            lr=optimizer_configs.lr,
            weight_decay=optimizer_configs.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_configs.name}")
    return [optimizer]

def lr_scheduler_factory(lr_configs, cfg):
    """
    Create empty lr scheduler to maintain fixed learning rate
    Returns:
        list: wrapped lr scheduler list
    """
    lr_scheduler = LRScheduler()
    return [lr_scheduler]

def logger_factory(cfg):
    """
    Create logger for training process
    Returns:
        logger: configured logger
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    logger.propagate = False
    return logger