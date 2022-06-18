#!/usr/bin/env python3

import os
import time
import traceback
from math import log, sqrt

import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

import layers
import sources
import utils
from args import args
from utils import my_log
from utils_support import my_tight_layout, plot_samples_np

torch.backends.cudnn.benchmark = True


def get_prior(temperature=1):
    if args.prior == 'gaussian':
        prior = sources.Gaussian([args.nchannels, args.L, args.L],
                                 scale=temperature)
    elif args.prior == 'laplace':
        # Set scale = 1/sqrt(2) to make var = 1
        prior = sources.Laplace([args.nchannels, args.L, args.L],
                                scale=temperature / sqrt(2))
    else:
        raise ValueError('Unknown prior: {}'.format(args.prior))
    prior = prior.to(args.device)
    return prior


def build_rnvp(nchannels, kernel_size, nlayers, nresblocks, nmlp, nhidden):
    core_size = nchannels * kernel_size**2
    widths = [core_size] + [nhidden] * nmlp + [core_size]
    net = layers.RNVP(
        [
            layers.ResNetReshape(
                nresblocks,
                widths,
                final_scale=True,
                final_tanh=True,
            ) for _ in range(nlayers)
        ],
        [
            layers.ResNetReshape(
                nresblocks,
                widths,
                final_scale=True,
                final_tanh=False,
            ) for _ in range(nlayers)
        ],
        nchannels,
        kernel_size,
    )
    return net


def build_arflow(nchannels, kernel_size, nlayers, nresblocks, nmlp, nhidden):
    assert nhidden % kernel_size**2 == 0
    channels = [nchannels] + [nhidden // kernel_size**2] * nmlp + [nchannels]
    width = kernel_size**2
    net = layers.ARFlowReshape(
        [
            layers.MaskedResNet(
                nresblocks,
                channels,
                width,
                final_scale=True,
                final_tanh=True,
            ) for _ in range(nlayers)
        ],
        [
            layers.MaskedResNet(
                nresblocks,
                channels,
                width,
                final_scale=True,
                final_tanh=False,
            ) for _ in range(nlayers)
        ],
    )
    return net


def build_mera():
    prior = get_prior()

    _layers = []
    for i in range(args.depth):
        if args.subnet == 'rnvp':
            _layers.append(
                build_rnvp(
                    args.nchannels,
                    args.kernel_size,
                    args.nlayers_list[i],
                    args.nresblocks_list[i],
                    args.nmlp_list[i],
                    args.nhidden_list[i],
                ))
        elif args.subnet == 'ar':
            _layers.append(
                build_arflow(
                    args.nchannels,
                    args.kernel_size,
                    args.nlayers_list[i],
                    args.nresblocks_list[i],
                    args.nmlp_list[i],
                    args.nhidden_list[i],
                ))
        else:
            raise ValueError('Unknown subnet: {}'.format(args.subnet))

    flow = layers.MERA(_layers, args.L, args.kernel_size, prior)
    flow = flow.to(args.device)

    return flow


def do_plot(flow, epoch_idx):
    flow.train(False)

    # When using multiple GPUs, each GPU samples batch_size / device_count
    sample, _ = flow.sample(args.batch_size // args.device_count)
    my_log('plot min {:.3g} max {:.3g} mean {:.3g} std {:.3g}'.format(
        sample.min().item(),
        sample.max().item(),
        sample.mean().item(),
        sample.std().item(),
    ))
    sample, _ = utils.logit_transform(sample, inverse=True)
    sample = torch.clamp(sample, 0, 1)
    sample = sample.permute(0, 2, 3, 1).detach().cpu().numpy()

    fig, axes = plot_samples_np(sample)

    fig.suptitle('{}/{}/epoch{}'.format(args.data, args.net_name, epoch_idx))
    my_tight_layout(fig)
    plot_filename = '{}/epoch{}.pdf'.format(args.plot_filename, epoch_idx)
    utils.ensure_dir(plot_filename)
    fig.savefig(plot_filename, bbox_inches='tight')
    fig.clf()
    plt.close()

    flow.train(True)


def main():
    start_time = time.time()

    utils.init_out_dir()
    last_epoch = utils.get_last_checkpoint_step()
    if last_epoch >= args.epoch:
        exit()
    if last_epoch >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_epoch))
    else:
        utils.clear_log()
    utils.print_args()

    flow = build_mera()
    flow.train(True)
    my_log('nparams in each RG layer: {}'.format(
        [utils.get_nparams(layer) for layer in flow.layers]))
    my_log('Total nparams: {}'.format(utils.get_nparams(flow)))

    # Use multiple GPUs
    if args.cuda and torch.cuda.device_count() > 1:
        flow = utils.data_parallel_wrap(flow)

    params = [x for x in flow.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    if last_epoch >= 0:
        utils.load_checkpoint(last_epoch, flow, optimizer)

    train_set, _, _ = utils.load_dataset()
    train_loader = torch.utils.data.DataLoader(train_set,
                                               args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))

    my_log('Training...')
    start_time = time.time()
    for epoch_idx in range(last_epoch + 1, args.epoch + 1):
        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.to(args.device)
            x, ldj_logit = utils.logit_transform(x)
            log_prob = flow.log_prob(x)
            loss = -(log_prob + ldj_logit) / (args.nchannels * args.L**2)
            loss_mean = loss.mean()
            loss_std = loss.std()

            utils.check_nan(loss_mean)

            loss_mean.backward()
            if args.clip_grad:
                clip_grad_norm_(params, args.clip_grad)
            optimizer.step()

            if args.print_step and batch_idx % args.print_step == 0:
                bit_per_dim = (loss_mean.item() + log(256)) / log(2)
                my_log(
                    'epoch {} batch {} bpp {:.8g} loss {:.8g} +- {:.8g} time {:.3f}'
                    .format(
                        epoch_idx,
                        batch_idx,
                        bit_per_dim,
                        loss_mean.item(),
                        loss_std.item(),
                        time.time() - start_time,
                    ))

        if (args.out_filename and args.save_epoch
                and epoch_idx % args.save_epoch == 0):
            state = {
                'flow': flow.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state,
                       '{}_save/{}.state'.format(args.out_filename, epoch_idx))

            if epoch_idx > 0 and (epoch_idx - 1) % args.keep_epoch != 0:
                os.remove('{}_save/{}.state'.format(args.out_filename,
                                                    epoch_idx - 1))

        if (args.plot_filename and args.plot_epoch
                and epoch_idx % args.plot_epoch == 0):
            with torch.no_grad():
                do_plot(flow, epoch_idx)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
