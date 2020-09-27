from torch import nn


class Source(nn.Module):
    def __init__(self, nvars):
        super().__init__()
        self.nvars = nvars

    def sample(self, batch_size):
        raise NotImplementedError(str(type(self)))

    def log_prob(self, x):
        raise NotImplementedError(str(type(self)))
