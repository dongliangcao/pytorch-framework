import torch.nn as nn
import timm
from utils.registry import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
class ResNet(nn.Module):
    def __init__(self, model_name, in_channels=3):
        super().__init__()
        assert 'resnet' in model_name, f'Invalid resnet model name: {model_name}'
        self.net = timm.create_model(model_name, pretrained=False, in_chans=in_channels)

    def forward(self, img):
        return self.net(img)
