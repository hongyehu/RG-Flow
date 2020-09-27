# AutoRegressive flow
# forward() is fast, inverse() is slow

import torch
from torch import nn

from .flow import Flow


class ARFlow(Flow):
    def __init__(self, s_layers, t_layers, prior=None):
        super().__init__(prior)
        assert len(s_layers) == len(t_layers)
        self.s_layers = nn.ModuleList(s_layers)
        self.t_layers = nn.ModuleList(t_layers)

    def forward(self, x):
        # x.shape = (B, C*W*H)
        ldj = x.new_zeros(x.shape[0])
        for s_layer, t_layer in zip(self.s_layers, self.t_layers):
            s = s_layer(x)
            t = t_layer(x)
            x = x * torch.exp(s) + t
            ldj_ = s.sum(dim=1)
            ldj = ldj + ldj_
        return x, ldj

    def inverse(self, z):
        # z.shape = (B, C*W*H)
        core_size = z.shape[1]
        inv_ldj = z.new_zeros(z.shape[0])
        for s_layer, t_layer in reversed(
                list(zip(self.s_layers, self.t_layers))):
            mask_z = torch.zeros_like(z)
            for i in range(core_size):
                s = s_layer(mask_z)
                t = t_layer(mask_z)
                if i == core_size - 1:
                    z = (z - t) * torch.exp(-s)
                else:
                    mask_z[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            inv_ldj_ = -s.sum(dim=1)
            inv_ldj = inv_ldj + inv_ldj_
        return z, inv_ldj


class ARFlowReshape(ARFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        shape = x.shape    # (B*num_RG_blocks, C, K, K)
        x = x.view(shape[0], -1)    # (B*num_RG_blocks, C*K*K)
        z, ldj = super().forward(x)
        z = z.view(shape)
        return z, ldj

    def inverse(self, z):
        shape = z.shape    # (B*num_RG_blocks, C, K, K)
        z = z.view(shape[0], -1)    # (B*num_RG_blocks, C*K*K)
        x, inv_ldj = super().forward(z)
        x = x.view(shape)
        return x, inv_ldj
