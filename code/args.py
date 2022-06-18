import argparse
import os
from math import log2

import torch

parser = argparse.ArgumentParser()

group = parser.add_argument_group('dataset parameters')
group.add_argument(
    '--data',
    type=str,
    default='celeba32',
    choices=['celeba32', 'mnist32', 'cifar10', 'chair600', 'msds1', 'msds2'],
    help='dataset name',
)
group.add_argument(
    '--data_path',
    type=str,
    default='./data',
    help='dataset path',
)
group.add_argument(
    '--nchannels',
    type=int,
    default=3,
    help='number of channels',
)
group.add_argument(
    '--L',
    type=int,
    default=32,
    help='edge length of images',
)

group = parser.add_argument_group('network parameters')
group.add_argument(
    '--prior',
    type=str,
    default='laplace',
    choices=['gaussian', 'laplace'],
    help='prior of latent variables',
)
group.add_argument(
    '--subnet',
    type=str,
    default='rnvp',
    choices=['rnvp', 'ar'],
    help='type of subnet in an RG block',
)
group.add_argument(
    '--kernel_size',
    type=int,
    default=4,
    help='edge length of an RG block',
)
group.add_argument(
    '--nlayers',
    type=str,
    default='8,6,4,2',
    help='number of subnet layers in an RG block',
)
group.add_argument(
    '--nresblocks',
    type=str,
    default='4',
    help='number of residual blocks in a subnet layer',
)
group.add_argument(
    '--nmlp',
    type=str,
    default='2',
    help='number of MLP hidden layers in an residual block',
)
group.add_argument(
    '--nhidden',
    type=str,
    default='512',
    help='width of MLP hidden layers',
)
group.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float64'],
    help='dtype',
)

group = parser.add_argument_group('optimizer parameters')
group.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='batch size',
)
group.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='learning rate',
)
group.add_argument(
    '--weight_decay',
    type=float,
    default=5e-2,
    help='weight decay',
)
group.add_argument(
    '--epoch',
    type=int,
    default=500,
    help='number of epoches',
)
group.add_argument(
    '--clip_grad',
    type=float,
    default=1,
    help='global norm to clip gradients, 0 for disabled',
)

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance',
)
group.add_argument(
    '--print_step',
    type=int,
    default=1,
    help='number of batches to print log, 0 for disabled',
)
group.add_argument(
    '--save_epoch',
    type=int,
    default=1,
    help='number of epochs to save network weights, 0 for disabled',
)
group.add_argument(
    '--keep_epoch',
    type=int,
    default=10,
    help='number of epochs to keep saved network weights, 0 for disabled',
)
group.add_argument(
    '--plot_epoch',
    type=int,
    default=1,
    help='number of epochs to plot samples, 0 for disabled',
)
group.add_argument(
    '--cuda',
    type=str,
    default='',
    help='IDs of GPUs to use, empty for disabled',
)
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs',
)
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='./saved_model',
    help='directory for output, empty for disabled',
)

args = parser.parse_args()

if args.dtype == 'float32':
    torch.set_default_tensor_type(torch.FloatTensor)
elif args.dtype == 'float64':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    raise ValueError(f'Unknown dtype: {args.dtype}')

if args.cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = torch.device('cuda')
    args.device_count = len(args.cuda.split(','))
else:
    args.device = torch.device('cpu')
    args.device_count = 1


def get_net_name():
    net_name = ''
    if args.prior != 'gaussian':
        net_name += '{prior}_'
    if args.subnet != 'rnvp':
        net_name += '{subnet}_'
    if args.kernel_size != 4:
        net_name += 'ks{kernel_size}_'
    net_name += 'nl{nlayers}_nr{nresblocks}_nm{nmlp}_nh{nhidden}'
    net_name = net_name.format(**vars(args))
    return net_name


args.net_name = get_net_name()

if args.out_dir:
    args.out_filename = os.path.join(
        args.out_dir,
        args.data,
        args.net_name,
        'out{out_infix}'.format(**vars(args)),
    )
    args.plot_filename = os.path.join(
        args.out_dir,
        args.data,
        args.net_name,
        'epoch_sample',
    )
else:
    args.out_filename = None
    args.plot_filename = None

args.depth = int(log2(args.L / args.kernel_size) + 1) * 2


def str_to_int_list(s, depth):
    if ',' in s:
        out = []
        for x in s.split(','):
            x = int(x)
            out += [x, x]
        return out
    else:
        return [int(s)] * depth


args.nlayers_list = str_to_int_list(args.nlayers, args.depth)
args.nresblocks_list = str_to_int_list(args.nresblocks, args.depth)
args.nmlp_list = str_to_int_list(args.nmlp, args.depth)
args.nhidden_list = str_to_int_list(args.nhidden, args.depth)
assert args.depth == len(args.nlayers_list)
assert args.depth == len(args.nresblocks_list)
assert args.depth == len(args.nmlp_list)
assert args.depth == len(args.nhidden_list)
