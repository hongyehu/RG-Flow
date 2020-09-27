# Used for ARFlow

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils import weight_norm

from .scale import Scale
from .swish import Swish


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, width, bias, exclusive):
        super().__init__(in_channels * width, out_channels * width, bias)
        self.exclusive = exclusive

        mask = torch.ones([width] * 2)
        if self.exclusive:
            mask = 1 - torch.triu(mask)
        else:
            mask = torch.tril(mask)
        mask = torch.cat([mask] * in_channels, dim=1)
        mask = torch.cat([mask] * out_channels, dim=0)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return (super().extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))


class MaskedResNet(nn.Module):
    def __init__(self, num_res_blocks, channels, width, final_scale,
                 final_tanh):
        assert channels[0] == channels[-1]

        super().__init__()

        self.res_blocks = nn.ModuleList([
            self.build_res_block(channels, width)
            for _ in range(num_res_blocks)
        ])

        if final_scale:
            self.scale = Scale(channels[-1] * width)
        else:
            self.scale = None

        if final_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

    def build_linear(self, in_channels, out_channels, width, exclusive):
        linear = MaskedLinear(in_channels,
                              out_channels,
                              width,
                              bias=True,
                              exclusive=exclusive)

        # 2.81 is the gain for swish
        bound = sqrt(2.81 * 3 / (in_channels * width))
        init.uniform_(linear.weight, -bound, bound)

        # Correction to Xavier initialization
        # We cannot make a row in weight to be all 0,
        # otherwise weight_norm() will produce nan
        linear.weight.data *= torch.sqrt(
            (linear.mask + 1e-7) /
            (linear.mask.mean(dim=1, keepdim=True) + 1e-7))

        init.zeros_(linear.bias)

        # TODO: weight norm with mask
        # linear = weight_norm(linear)

        return linear

    def build_res_block(self, channels, width):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(Swish(channels[i] * width))
            layers.append(
                self.build_linear(channels[i],
                                  channels[i + 1],
                                  width,
                                  exclusive=(i == 0)))
        return nn.Sequential(*layers)

    def forward(self, x):
        for res_block in self.res_blocks:
            x = (x + res_block(x)) / sqrt(2)
        if self.scale:
            x = self.scale(x)
        if self.tanh:
            x = self.tanh(x)
        return x
