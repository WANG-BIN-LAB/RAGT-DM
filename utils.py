import numpy as np
import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """
    Calculate top-k accuracy
    Args:
        output: model output logits
        target: ground truth label
    Returns:
        list: top-k accuracy values
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

class TotalMeter:
    """Meter to track average and sum of metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update_with_weight(self, val, weight=1):
        """Update meter with weight (batch size)"""
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count if self.count > 0 else 0

def count_params(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def isfloat(x):
    """Check if string is float"""
    try:
        float(x)
        return True
    except ValueError:
        return False

def continus_mixup_data(*xs, y=None, site=None, age=None, sex=None, alpha=1.0, device='cuda'):
    """
    Mixup augmentation for continuous data
    Returns:
        mixed data and labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    # Mix inputs
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    # Mix labels
    y = lam * y + (1 - lam) * y[index]
    return *new_xs, y, site[index], age[index], sex[index]