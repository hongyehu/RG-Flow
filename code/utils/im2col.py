from math import sqrt


def dispatch(i, j, x):
    # i, j are indices of elements being processed
    # dim(i) = (num_RG_blocks, K*K)
    # dim(x_) = (B, C, num_RG_blocks, K*K)
    x_ = x[:, :, i, j].reshape(x.shape[0], x.shape[1], i.shape[0], i.shape[1])
    return x, x_


def collect(i, j, x, x_):
    x = x.clone()
    x[:, :, i, j] = x_.reshape(x.shape[0], x.shape[1], i.shape[0], i.shape[1])
    return x


def stackRGblock(x):
    # x should be dispatched
    # dim(x) = (B, C, num_RG_blocks, K*K)
    # -> (B, num_RG_blocks, C, K*K)
    # -> (B*num_RG_blocks, C, K, K)
    _, C, _, KK = x.shape
    K = int(sqrt(KK))
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, C, K, K)
    return x


def unstackRGblock(x, batch_size):
    # dim(x) = (B*num_RG_blocks, C, K, K)
    # -> (B, num_RG_blocks, C, K*K)
    # -> (B, C, num_RG_blocks, K*K)
    _, C, KH, KW = x.shape
    x = x.reshape(batch_size, -1, C, KH * KW)
    x = x.permute(0, 2, 1, 3)
    return x
