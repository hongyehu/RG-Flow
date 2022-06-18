#!/usr/bin/env python3

import torch

import utils
from args import args
from main import build_mera

args.device = torch.device('cuda')


def corrcoef(x):
    n = x.shape[1]
    x = x - x.mean(dim=1, keepdims=True)
    c = x @ x.t() / (n - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.sqrt(d)
    c = c / stddev.expand_as(c)
    c = c / stddev.expand_as(c).t()

    return c


def main():
    last_epoch = utils.get_last_checkpoint_step()
    print('Checkpoint found: {}'.format(last_epoch))
    flow = build_mera()
    utils.load_checkpoint(last_epoch, flow)

    train_set, _, _ = utils.load_dataset()
    train_loader = torch.utils.data.DataLoader(train_set,
                                               args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    zs = []
    for batch_idx, (x, _) in enumerate(train_loader):
        if batch_idx >= 10:
            break
        print(batch_idx)

        x = x.to(args.device)
        z, _ = flow.forward(x)
        z = z[:, :, ::8, ::8].reshape(z.shape[0], -1)
        zs.append(z)
    z = torch.cat(zs)
    corr = corrcoef(z.T)
    print(corr)
    torch.save(corr, 'corr.pth')


if __name__ == '__main__':
    with torch.no_grad():
        main()
