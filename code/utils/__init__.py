import os
from glob import glob

import numpy as np
import torch
from torch import nn

from args import args

from .data_utils import get_data_batch, load_dataset, logit_transform
from .im2col import collect, dispatch, stackRGblock, unstackRGblock


def check_nan(x):
    assert torch.isnan(x).sum().item() == 0
    return x


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.out_filename:
        return
    ensure_dir(args.out_filename)
    if args.save_epoch:
        ensure_dir(args.out_filename + '_save/')


def clear_log():
    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def my_log(s):
    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + '\n')
    if not args.no_stdout:
        print(s)


def print_args(print_fn=my_log):
    for k, v in args._get_kwargs():
        print_fn(f'{k} = {v}')
    print_fn('')


def parse_checkpoint_name(filename):
    filename = os.path.basename(filename)
    filename = filename.replace('.state', '')
    step = int(filename)
    return step


def get_last_checkpoint_step():
    if not (args.out_filename and args.save_epoch):
        return -1
    filename_list = glob(f'{args.out_filename}_save/*.state')
    if not filename_list:
        return -1
    step = max(parse_checkpoint_name(x) for x in filename_list)
    return step


def load_checkpoint(epoch, flow, optimizer=None):
    state = torch.load(f'{args.out_filename}_save/{epoch}.state',
                       map_location=args.device)

    if state.get('flow'):
        flow_state = state['flow']
    else:
        flow_state = state

    keys = list(flow_state.keys())
    if isinstance(flow, nn.DataParallel):
        for key in keys:
            if not key.startswith('module.net.'):
                flow_state['module.net.' + key] = flow_state[key]
    else:
        for key in keys:
            if key.startswith('module.net.'):
                flow_state[key[11:]] = flow_state[key]

    flow.load_state_dict(flow_state, strict=False)

    if optimizer is not None and state.get('optimizer'):
        # Learning rate is not saved in the state
        optimizer.load_state_dict(state['optimizer'])


def clip(x, threshold=1e-4):
    x = x.clone()
    x[x.abs() < threshold] = 0
    return x


def get_nparams(net):
    return sum(
        int(np.prod(p.shape)) for p in net.parameters() if p.requires_grad)


# When using multiple GPUs, change function name to `forward' for nn.DataParallel
class DataParallelWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.default_func_name = 'forward'
        self.func_name = self.default_func_name

    def forward(self, *args, **kwargs):
        return getattr(self.net, self.func_name)(*args, **kwargs)


def data_parallel_wrap(net):
    net = nn.DataParallel(DataParallelWrapper(net))
    wrapper = net.module

    def wrap_func(func_name):
        def func(*args, **kwargs):
            wrapper.func_name = func_name
            out = net(*args, **kwargs)
            wrapper.func_name = wrapper.default_func_name
            return out

        return func

    net.sample = wrap_func('sample')
    net.log_prob = wrap_func('log_prob')

    return net
