from torch import nn

import utils

from .flow import Flow


class HierarchyBijector(Flow):
    def __init__(self, indexI, indexJ, layers, prior=None):
        super().__init__(prior)
        assert len(layers) == len(indexI)
        assert len(layers) == len(indexJ)
        self.layers = nn.ModuleList(layers)
        self.indexI = indexI
        self.indexJ = indexJ

    def forward(self, x):
        # dim(x) = (B, C, H, W)
        batch_size = x.shape[0]
        ldj = x.new_zeros(batch_size)
        for layer, indexI, indexJ in zip(self.layers, self.indexI,
                                         self.indexJ):
            x, x_ = utils.dispatch(indexI, indexJ, x)
            # dim(x_) = (B, C, num_RG_blocks, K*K)
            x_ = utils.stackRGblock(x_)
            # dim(x_) = (B*num_RG_blocks, C, K, K)

            x_, log_prob = layer.forward(x_)
            ldj = ldj + log_prob.view(batch_size, -1).sum(dim=1)

            x_ = utils.unstackRGblock(x_, batch_size)
            x = utils.collect(indexI, indexJ, x, x_)

        return x, ldj

    def inverse(self, z):
        batch_size = z.shape[0]
        inv_ldj = z.new_zeros(batch_size)
        for layer, indexI, indexJ in reversed(
                list(zip(self.layers, self.indexI, self.indexJ))):
            z, z_ = utils.dispatch(indexI, indexJ, z)
            z_ = utils.stackRGblock(z_)

            z_, log_prob = layer.inverse(z_)
            inv_ldj = inv_ldj + log_prob.view(batch_size, -1).sum(dim=1)

            z_ = utils.unstackRGblock(z_, batch_size)
            z = utils.collect(indexI, indexJ, z, z_)

        return z, inv_ldj
