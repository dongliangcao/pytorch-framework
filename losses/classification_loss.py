import torch.nn as nn
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * F.cross_entropy(pred, target)
