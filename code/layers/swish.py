import torch
from torch import nn


class Swish(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        return x * torch.sigmoid(self.scale * x)

    def extra_repr(self):
        return ('num_features={}'.format(self.num_features))
