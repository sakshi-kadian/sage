import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """
    Calculates how often the chosen reward is strictly greater than the rejected reward.
    Returns a percentage (0.0 to 1.0).
    """
    correct = (chosen_rewards > rejected_rewards).sum().item()
    total = chosen_rewards.size(0)
    return correct / total
