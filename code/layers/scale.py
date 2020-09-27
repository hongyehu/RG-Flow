import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        # Zero init to avoid value explosion
        self.scale = nn.Parameter(torch.zeros(num_features))

        # For debug
        self.register_buffer('saved_mean', torch.zeros(num_features))
        self.register_buffer('saved_var', torch.ones(num_features))

    def forward(self, x):
        with torch.no_grad():
            self.saved_mean = x.mean(dim=0)
            self.saved_var = x.var(dim=0)
        return self.scale * x

    def extra_repr(self):
        return ('num_features={}'.format(self.num_features))
