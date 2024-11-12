from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MNISTDataset(Dataset):
    def __init__(self, root, train, transform):
        self.dataset = MNIST(root=root, train=train, transform=transform, download=True)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
