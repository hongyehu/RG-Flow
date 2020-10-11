#!/usr/bin/env python3

from math import sqrt

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.distributions.laplace import Laplace

import utils
from args import args
from main import build_mera

T_low = 0.2    # Effective temperature for low-level latent variables
T_high = 0.8    # Effective temperature for high-level latent variables
level_cutoff = 1    # Cutoff level (\lambda in the paper)


def main():
    flow = build_mera()
    last_epoch = utils.get_last_checkpoint_step()
    utils.load_checkpoint(last_epoch, flow)
    flow.train(False)

    shape = (16, args.nchannels, args.L, args.L)
    prior_low = Laplace(torch.tensor(0.), torch.tensor(T_low / sqrt(2)))
    z = prior_low.sample(shape)
    prior_high = Laplace(torch.tensor(0.), torch.tensor(T_high / sqrt(2)))
    z_high = prior_high.sample(shape)
    k = 2**level_cutoff
    z[:, :, ::k, ::k] = z_high[:, :, ::k, ::k]
    z = z.to(args.device)

    with torch.no_grad():
        x, _ = flow.inverse(z)

    samples = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    samples = 1 / (1 + np.exp(-samples))

    fig, axes = plt.subplots(4, 4, figsize=(4, 4), sharex=True, sharey=True)
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.imshow(samples[j * 4 + i])
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('./mix_T.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
