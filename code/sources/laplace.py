import torch

from .source import Source


class Laplace(Source):
    def __init__(self, nvars, scale=1):
        super().__init__(nvars)
        self.register_buffer(
            'scale', torch.tensor(scale, dtype=torch.get_default_dtype()))

    def sample(self, batch_size):
        shape = [batch_size] + self.nvars
        # uniform_(a, b) is in [a, b), so we need eps
        finfo = torch.finfo(self.scale.dtype)
        u = self.scale.new_empty(shape).uniform_(finfo.eps - 1, 1)
        out = self.scale * u.sign() * torch.log1p(-u.abs())
        return out

    def log_prob(self, x):
        out = -torch.abs(x) / self.scale - torch.log(2 * self.scale)
        out = out.view(out.shape[0], -1).sum(dim=1)
        return out
