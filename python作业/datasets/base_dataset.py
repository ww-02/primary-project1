from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass