import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        pass