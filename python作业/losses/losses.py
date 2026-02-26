import torch.nn as nn

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.loss_fn(pred, target)