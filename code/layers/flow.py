from torch import nn


class Flow(nn.Module):
    def __init__(self, prior=None):
        super().__init__()
        self.prior = prior

    def forward(self, x):
        raise NotImplementedError(str(type(self)))

    def inverse(self, z):
        raise NotImplementedError(str(type(self)))

    def sample(self, batch_size, prior=None):
        if prior is None:
            prior = self.prior
        assert prior is not None
        z = prior.sample(batch_size)
        logp = prior.log_prob(z)
        x, logp_ = self.inverse(z)
        return x, logp - logp_

    def log_prob(self, x):
        z, logp = self.forward(x)
        if self.prior is not None:
            logp = logp + self.prior.log_prob(z)
        return logp
