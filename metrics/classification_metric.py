import torch
import torch.nn as nn

from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        bs = target.shape[0]
        _, predicted = torch.max(pred, dim=1)
        correct = (predicted == target).sum()
        return correct / bs